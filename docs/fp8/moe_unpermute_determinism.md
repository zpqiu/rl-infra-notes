# MoE Unpermute 非确定性分析

问题背景是在 NeMo-RL 中进行 on-policy GRPO 实验时发现importance sampling ratio 不为 1，即 MCore 在进行两次前向时结果不一致，原始的 issue: [NVIDIA-NeMo/RL#2255](https://github.com/NVIDIA-NeMo/RL/issues/2255). 经过定位发现和 `moe_permute_fusion=False`  有关。

下面是MoE forward pass 中 `moe_permute_fusion=False` 导致 `scatter_add_` 非确定性的根因分析，以及 TE fused kernel 如何通过 gather-reduce 模式避免此问题。

> **分析基于以下版本：**
>
> | 库 | Commit | 
> |---|---|
> | Megatron-LM | `17a67b9a9` |
> | TransformerEngine | `6638fefb` (v0.1+1653) | 

---

## 1. 问题描述

当 `moe_permute_fusion=False` 时，MoE 模型的 forward pass 产生非确定性结果。根因是 `unpermute()` 中的 `scatter_add_` 操作：在 top-K routing 下，多个 expert 的输出需要累加回同一个 token 位置，而 `scatter_add_` 的 CUDA 实现对同一地址的并发浮点加法顺序不确定。

## 2. MoE Forward 数据流

```
Original tokens [num_tokens, hidden]
        │
        ▼  permute()     ── 按 expert 分组：token → sorted by expert assignment
Permuted tokens [num_permuted_tokens, hidden]
        │
        ▼  Expert FFN    ── 每个 expert 处理自己的 token chunk
Expert output [num_permuted_tokens, hidden]
        │
        ▼  unpermute()   ── 结果散射回原始 token 位置并累加
Output tokens [num_tokens, hidden]
```

关键：top-K routing 时同一个 token 被路由到 K 个 expert。`permute()` 会将该 token **复制** K 份到不同位置，`unpermute()` 需要将 K 个 expert 的输出 **求和** 回同一个位置。

## 3. 非确定性代码路径（Megatron-LM PyTorch）

### 3.1 `permute()` — 确定性，无问题

`Megatron-LM/megatron/core/transformer/moe/moe_utils.py:382-428`

```python
# 按 routing_map 排序，将 token 按 expert 分组
sorted_indices = routing_map.argsort(dim=-1, descending=True, stable=True)
# 用 index_select gather token 到新位置
permuted_input = tokens.index_select(0, sorted_indices)   # L426
```

`index_select` 是一对一的 gather 操作——每个输出位置只从一个输入位置读取。**确定性。**

### 3.2 `unpermute()` — 非确定性根因

`Megatron-LM/megatron/core/transformer/moe/moe_utils.py:431-530`

```python
def unpermute(permuted_tokens, sorted_indices, restore_shape, probs=None,
              routing_map=None, fused=False, drop_and_pad=False, pad_offsets=None):
    if fused:
        return fused_unpermute(...)   # → TE 路径，见下文第 5 节

    _, hidden = restore_shape

    if probs is not None:
        # 将 router probs 应用到 expert 输出上（加权）
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)  # L510

    # 创建全零输出
    output_tokens = torch.zeros(restore_shape, dtype=permuted_tokens.dtype,
                                device=permuted_tokens.device)      # L513

    if torch.are_deterministic_algorithms_enabled():
        # 确定性路径：index_add_
        output_tokens.index_add_(0, sorted_indices, permuted_tokens)  # L524
    else:
        # 默认路径：scatter_add_ ← 非确定性！
        output_tokens.scatter_add_(
            0,
            sorted_indices.unsqueeze(1).expand(-1, hidden),           # L527-528
            permuted_tokens,
        )
    return output_tokens.to(dtype=input_dtype)
```

### 3.3 为什么 `scatter_add_` 非确定性

`sorted_indices` 中 **存在重复值**——top-K routing 下同一个 token 被多个 expert 处理。例如 token 5 被 expert 0 和 expert 3 选中：

```
permuted_tokens[42] = expert_0 对 token_5 的输出
permuted_tokens[91] = expert_3 对 token_5 的输出
sorted_indices[42] = 5
sorted_indices[91] = 5

scatter_add_ 需要：output[5] += permuted_tokens[42] + permuted_tokens[91]
```

`scatter_add_` 的 CUDA 实现为每个 **源行** 启动一个线程，多个线程并发对 `output[5]` 执行浮点加法。由于浮点加法不满足结合律：

```
Run 1:  0.0 + a + b = x    （thread 42 先执行）
Run 2:  0.0 + b + a = x'   （thread 91 先执行）
x ≠ x'                     （浮点舍入误差 ~1e-7 for FP32）
```

CUDA 线程调度顺序跨 run 不确定 → 结果不确定。

### 3.4 三种配置的行为

| 配置 | 代码路径 | 确定性？ |
|---|---|---|
| `moe_permute_fusion=True` | TE `fused_unpermute()` | 是 |
| `moe_permute_fusion=False` + `torch.use_deterministic_algorithms(True)` | `index_add_()` | 是 |
| `moe_permute_fusion=False`（默认） | **`scatter_add_()`** | **否** |

## 4. TE Fused Kernel：确定性的 Gather-Reduce 设计

### 4.1 调用链

```
Megatron moe_utils.py: fused_unpermute
  └── megatron.core.extensions.transformer_engine: moe_unpermute        # alias
      └── transformer_engine.pytorch.permutation: moe_unpermute         # autograd Function
          └── _moe_unpermute_mask_map.forward                           # L339
              └── triton_permutation.unpermute_with_mask_map             # PyTorch wrapper
                  └── _unpermute_kernel                                  # Triton kernel
```

### 4.2 `row_id_map` 数据结构

TE 不使用 Megatron 的 `sorted_indices`，而是构建一个专用的 `row_id_map` 张量。

`TE/transformer_engine/pytorch/triton/permutation.py:24-118` — `make_row_id_map()`

输入 `routing_map` shape `[num_tokens, num_experts]`，输出 `row_id_map` shape `[num_tokens, num_experts * 2 + 1]`。

以注释中的例子说明（5 tokens, 3 experts）：

```
routing_map:
  [[1, 1, 0],    # token 0 → expert 0, 1
   [1, 0, 1],    # token 1 → expert 0, 2
   [0, 0, 1],    # token 2 → expert 2
   [1, 1, 0],    # token 3 → expert 0, 1
   [0, 0, 0]]    # token 4 → not routed
```

经过 3-pass Triton kernel 后：

```
row_id_map (最终):
  token 0: [3, 0, _, 1, 0, _, 2]   → 路由到 2 个 expert，permuted 行号 3 和 0，expert idx 1 和 0
  token 1: [5, 1, _, 2, 0, _, 2]   → 路由到 2 个 expert，permuted 行号 5 和 1
  token 2: [6, _, _, 2, _, _, 1]   → 路由到 1 个 expert
  token 3: [4, 2, _, 1, 0, _, 2]   → 路由到 2 个 expert
  token 4: [_, _, _, _, _, _, 0]   → 未路由

布局：[dst_row_0..dst_row_{K-1}, expert_idx_0..expert_idx_{K-1}, n_routed]
       ├── 前 num_experts 列 ──┤├── 中 num_experts 列 ──────────┤├─ 1 ─┤
```

关键设计：**每个 token 自己知道它被路由到了哪些 expert，以及对应的 permuted 行号**。

### 4.3 `_unpermute_kernel` — Gather-Reduce 核心

`TE/transformer_engine/common/triton/permutation.py:314-414`

```python
@triton.jit
def _unpermute_kernel(...):
    # Grid: (num_tokens, cdiv(hidden_size, BLOCK_SIZE))
    # 每个 token 一个 program ← 关键！
    pid_t = tl.program_id(0)    # 当前处理的 output token
    pid_h = tl.program_id(1)    # hidden dim 的 block 分片

    accumulator = tl.zeros((BLOCK_SIZE,), dtype=compute_type)  # FP32 局部累加器

    # 从 row_id_map 读取：这个 token 被路由到了几个 expert？
    n_routed = tl.load(row_id_map_ptr + pid_t * ... + num_experts * 2 * ...)

    for idx in tl.range(n_routed):
        # 从 row_id_map 读取：第 idx 个 expert 对应的 permuted 行号
        src_row = tl.load(row_id_map_ptr + pid_t * ... + idx * ...).to(tl.int64)

        # 读取该 expert 的输出
        inp = tl.load(input_ptr + src_row * stride_input_token + ...)
        inp = inp.to(compute_type)          # 上转 FP32

        if WITH_MERGING_PROBS:
            # 从 row_id_map 读取 expert idx，查找 router prob
            expert_idx = tl.load(row_id_map_ptr + ... + (num_experts + idx) * ...)
            merging_prob = tl.load(merging_probs_ptr + pid_t * ... + expert_idx * ...)
            inp *= merging_prob

        accumulator += inp                  # 顺序累加，无竞争

    # 单次写入输出
    tl.store(output_ptr + pid_t * stride_output_token + ..., accumulator.to(data_type))
```

### 4.4 Scatter vs Gather：并行化轴的反转

| | PyTorch `scatter_add_` | TE `_unpermute_kernel` |
|---|---|---|
| **Grid 维度** | 每个 **源行**（permuted token）一个线程 | 每个 **目标行**（original token）一个 program |
| **累加方式** | 多线程并发写同一输出地址 | 单 program 顺序读所有贡献源 |
| **模式** | Scatter（write-side conflict） | **Gather-reduce**（read-side，no conflict） |
| **确定性** | 否 — 线程调度顺序不确定 | **是** — 单一 owner，顺序循环 |
| **精度** | 取决于输入 dtype | 强制 FP32 累加（`compute_type = tl.float32`） |

核心思路：**反转并行化的轴**。`scatter_add_` 从源视角出发（每个源行一个线程），导致多线程竞争同一输出；TE kernel 从目标视角出发（每个输出 token 一个 program），自己 gather 所有贡献然后顺序累加。单一 owner = 无竞争 = 确定性。

### 4.5 Permute Kernel 对比

`TE/transformer_engine/common/triton/permutation.py:196-293` — `_permute_kernel`

permute 方向相反：从 original token 分发到 permuted positions。TE 同样用 per-token program：

```python
pid_t = tl.program_id(0)     # 当前 original token
src_row = pid_t.to(tl.int64)
inp = tl.load(input_ptr + src_row * ...)

n_routed = tl.load(row_id_map_ptr + pid_t * ... + num_experts * 2 * ...)
for idx in tl.range(n_routed):
    dst_row = tl.load(row_id_map_ptr + pid_t * ... + idx * ...).to(tl.int64)
    tl.store(output_ptr + dst_row * ..., inp)    # 写到不同目标行，无冲突
```

permute 是 scatter-write（一对多），但每个目标行只被一个源写入（routing 保证），所以天然确定。unpermute 是多对一（多个 expert 输出 → 一个 token），才有累加冲突的问题。

## 5. `row_id_map` 构建：3-Pass Triton Pipeline

`TE/transformer_engine/common/triton/permutation.py:82-192`

将稀疏的 `routing_map [num_tokens, num_experts]` 转换为稠密的 `row_id_map [num_tokens, num_experts * 2 + 1]`，分三步：

| Pass | Kernel | 作用 | Grid |
|---|---|---|---|
| 1 | `_row_id_map_pass_1_kernel` | 对每个 expert 列，分 block 做 cumsum | `(num_experts, cdiv(num_tokens, 1024))` |
| 2 | `_row_id_map_pass_2_kernel` | 跨 block 前缀和 → 全局 row index；非路由位置设为 -1 | 同上 |
| 3 | `_row_id_map_pass_3_kernel` | 每个 token 内：将稀疏 expert 列压缩为稠密 [dst_rows | expert_idxs | n_routed] | `(num_tokens,)` |

Pass 3 内部使用 bitonic argsort（`_argsort`）将有效条目排到前面，无效的 -1 排到后面。

## 6. Backward 路径

`_moe_unpermute_mask_map` 的 backward（`TE/transformer_engine/pytorch/permutation.py:396-510`）：

- **无 probs 时**：unpermute 的反向就是 permute（scatter → gather 互为反操作），直接调 `permute_with_mask_map`
- **有 probs 时**：需要额外计算 `probs_grad`，调 `unpermute_with_mask_map_bwd_with_merging_probs`（专门的 backward kernel）

两条路径都使用 TE 的 per-token Triton kernel，保持确定性。
