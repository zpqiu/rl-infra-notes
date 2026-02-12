# MoE Router Dtype 分析（Qwen3 MoE）— 训练 & 推理

> **分析基于以下版本：**
>
> | 库 | 版本 / Tag | Commit | 日期 |
> |---|---|---|---|
> | vLLM | v0.12.0 | `4fd9d6a8` | 2025-12-02 |
> | Megatron-LM (Core) | core_v0.15.0rc7+688 | `b47c376f` | 2026-01-27 |
> | Megatron-Bridge | v0.2.0rc6+477 | `56200ade` | 2026-01-30 |

---

# Part I — Megatron 训练侧

## 1. 关键文件

| 组件 | 文件路径 | 关键行 |
|------|----------|--------|
| Router 基类 | `Megatron-LM/megatron/core/transformer/moe/router.py` | L28-101 |
| TopKRouter | `Megatron-LM/megatron/core/transformer/moe/router.py` | L500-562 |
| Router GEMM (autograd) | `Megatron-LM/megatron/core/transformer/moe/moe_utils.py` | L1099-1199 |
| topk + softmax/sigmoid | `Megatron-LM/megatron/core/transformer/moe/moe_utils.py` | L561-679 |
| unpermute（probs 加权） | `Megatron-LM/megatron/core/transformer/moe/moe_utils.py` | L333-410 |
| MoE Layer 调用流程 | `Megatron-LM/megatron/core/transformer/moe/moe_layer.py` | L357-409 |
| AlltoAll Dispatcher | `Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py` | L592-849 |
| GroupedMLP Expert | `Megatron-LM/megatron/core/transformer/moe/experts.py` | L65-278 |
| TransformerConfig | `Megatron-LM/megatron/core/transformer/transformer_config.py` | `moe_router_dtype` |
| Qwen3 MoE 配置 | `Megatron-Bridge/src/megatron/bridge/models/qwen/qwen_provider.py` | L362-396 |
| Qwen3Next 配置 | `Megatron-Bridge/src/megatron/bridge/models/qwen/qwen_provider.py` | L433-454 |
| 权重映射 | `Megatron-Bridge/src/megatron/bridge/models/qwen/qwen3_moe_bridge.py` | L86 |

## 2. Router 权重初始化

```python
# router.py:53-54 — 始终以 fp32 创建
self.weight = torch.nn.Parameter(
    torch.empty((self.config.num_moe_experts, self.config.hidden_size), dtype=torch.float32)
)

# router.py:67-77 — reset_parameters: 初始化后转为 params_dtype
def reset_parameters(self):
    if self.config.perform_initialization:
        self.config.init_method(self.weight)          # 在 fp32 中执行 init_method
    self.weight.data = self.weight.data.to(dtype=self.config.params_dtype)  # 转为 bf16
```

- Qwen3 MoE: `params_dtype=bf16`, `add_bias_linear=False`（无 bias）
- 权重最终存储为 **bf16**，不受 FP8 量化影响（FP8 只作用于 expert FFN）

## 3. Router Gating（dtype 决定逻辑）

```python
# router.py:79-101
def gating(self, input: torch.Tensor):
    router_dtype = input.dtype                          # 默认：跟随输入 dtype
    if self.config.moe_router_dtype == 'fp32':
        router_dtype = torch.float32                    # 显式覆盖
    elif self.config.moe_router_dtype == 'fp64':
        router_dtype = torch.float64
    logits = router_gating_linear(input, self.weight, self.bias, router_dtype)
    return logits
```

`router_dtype` 优先级：`moe_router_dtype` 配置 > `input.dtype`

## 4. Router GEMM（RouterGatingLinearFunction）

```python
# moe_utils.py:1105-1142
class RouterGatingLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight, bias, router_dtype):
        ctx.router_dtype = router_dtype
        ctx.input_dtype = inp.dtype          # 保存原始 dtype 用于 backward
        ctx.weight_dtype = weight.dtype

        if te_general_gemm is not None and router_dtype != torch.float64:
            output = te_general_gemm(weight, inp, router_dtype, layout="TN", bias=bias)
        elif bias is None:
            output = torch.mm(inp.to(router_dtype), weight.to(router_dtype).t())
        else:
            output = torch.addmm(bias.to(router_dtype), inp.to(router_dtype), weight.to(router_dtype).t())

        return output    # dtype = router_dtype，不转回 params_dtype
```

关键点：
- 输入和权重在 GEMM 前**显式 cast 到 `router_dtype`**
- `torch.mm` 输出 dtype = 两个操作数的 dtype = `router_dtype`
- **forward 输出不做任何转回**，直接返回 `router_dtype` 的 logits

Backward 中梯度**转回原始 dtype**：
```python
# moe_utils.py:1164-1175
grad_input = grad_input[0].to(ctx.input_dtype)    # 转回 input 的 dtype
grad_weight = grad_weight[0].to(ctx.weight_dtype)  # 转回 weight 的 dtype
```

## 5. TopK + Softmax

```python
# moe_utils.py:662-668
# Qwen3 MoE: score_function="softmax", use_pre_softmax=False
if score_function == "softmax":
    if use_pre_softmax:
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
        probs, top_indices = compute_topk(scores, topk, ...)
    else:  # ← Qwen3 MoE 走这个分支
        scores, top_indices = compute_topk(logits, topk, ...)           # topk 在 logits 原始 dtype 上执行
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)      # softmax 强制 fp32
                      .type_as(logits)                                   # 转回 logits 的 dtype
```

`.type_as(logits)` 的效果取决于 logits 的 dtype：
- `moe_router_dtype=None` → logits=bf16 → probs **截断回 bf16**
- `moe_router_dtype='fp32'` → logits=fp32 → probs **保持 fp32**

## 6. Expert 内 Probs 加权（GroupedMLP）

```python
# experts.py:112-115 — activation_func_with_probs
def activation_func_with_probs(x, probs):
    dtype = x.dtype                            # fc1_output 的 dtype = bf16
    res = self.activation_func(x) * probs      # bf16 * probs_dtype → 广播提升
    return res.to(dtype)                        # 转回 bf16

# experts.py:256-258 — 在 fc1 和 fc2 之间调用
intermediate_parallel = self.activation_func_with_probs(
    fc1_output, permuted_probs.unsqueeze(-1)   # probs 在此加权
)
fc2_output = gg.ops.gmm(intermediate_parallel, w2, ...)  # 输入已是 bf16
```

## 7. Token Combine（AlltoAll Dispatcher）

```python
# token_dispatcher.py:779-783 — combine_preprocess（仅 TP>1 时执行）
hidden_states = reduce_scatter_to_sequence_parallel_region(
    hidden_states.to(self.probs.dtype),       # 转为 probs 的 dtype 做归约
    group=self.tp_group,
).to(hidden_states.dtype)                      # 转回 hidden 的 dtype

# token_dispatcher.py:833-840 — combine_postprocess
output = unpermute(
    permutated_local_input_tokens,
    self.reversed_local_input_permutation_mapping,
    restore_shape=self.hidden_shape_before_permute,
    routing_map=self.routing_map,
    # 注意：此处 **没有传 probs 参数**，不做加权
    # AlltoAll 的 probs 加权在 expert 内部完成（见第 6 节）
)
```

## 8. 两条路径完整 Dtype 对比

Qwen3 MoE 配置：`params_dtype=bf16`, `pre_softmax=False`, `score_function="softmax"`, dispatcher=AlltoAll, experts=GroupedMLP, `topk=8`

| 阶段 | 代码位置 | `moe_router_dtype=None`（默认） | `moe_router_dtype='fp32'` |
|------|----------|------|------|
| weight 存储 | `router.py:73` | bf16 | bf16 |
| router_dtype 决定 | `router.py:95-99` | `input.dtype` → bf16 | `torch.float32` → fp32 |
| Router GEMM | `moe_utils.py:1134-1135` | bf16 × bf16 → **bf16** | bf16→fp32 × bf16→fp32 → **fp32** |
| logits | | **bf16** | **fp32** |
| topk 选择 | `moe_utils.py:667` | 在 **bf16** 上选 topk | 在 **fp32** 上选 topk |
| softmax 计算 | `moe_utils.py:668` | `softmax(dtype=fp32)` → fp32 | `softmax(dtype=fp32)` → fp32 |
| `.type_as(logits)` | `moe_utils.py:668` | fp32 → **bf16**（截断） | fp32 → **fp32**（无损） |
| probs 输出 | | **bf16** | **fp32** |
| permute + AlltoAll | `token_dispatcher.py:637-671` | probs 保持 **bf16** | probs 保持 **fp32** |
| Expert 内加权 | `experts.py:114` | `act(x_bf16) * probs_bf16` → **bf16** | `act(x_bf16) * probs_fp32` → **fp32** |
| `.to(dtype)` 转回 | `experts.py:115` | → **bf16** | fp32 → **bf16** |
| fc2 GEMM 输入 | `experts.py:259` | **bf16** | **bf16** |
| reduce_scatter 精度 | `token_dispatcher.py:779-783` | `hidden.to(bf16)` → **bf16** 归约 | `hidden.to(fp32)` → **fp32** 归约 |
| reduce_scatter 转回 | `token_dispatcher.py:779-783` | → **bf16** | fp32 → **bf16** |
| unpermute（combine） | `token_dispatcher.py:833-840` | 无 probs 加权，**bf16** | 无 probs 加权，**bf16** |
| 最终输出 | | **bf16** | **bf16** |

## 9. 核心差异总结

1. **topk 精度**：默认路径在 bf16（~3.3 位尾数）上做 topk，对 128/512 个专家，接近的 logit 值可能导致专家选择不稳定；fp32 路径（~7.2 位尾数）更可靠。

2. **`.type_as(logits)` 的隐式截断**：同一行代码 `softmax(..., dtype=fp32).type_as(logits)`，默认路径把 fp32 softmax 结果截断回 bf16，丢失了 softmax 的精度优势。

3. **Expert 加权精度**：fp32 路径的 `activation(x) * probs` 乘法在 fp32 中执行（广播提升），虽然最终 `.to(dtype)` 转回 bf16，但乘法本身精度更高。

4. **通信归约精度**：fp32 路径在 reduce_scatter 中以 fp32 累加再转回 bf16，避免多卡归约时的 bf16 精度损失。

5. **最终输出一致**：两条路径最终输出均为 bf16，差异仅在 routing 中间计算的精度。

## 10. Qwen3 变体配置对比

```python
# qwen_provider.py:362-396
class Qwen3MoEModelProvider:
    num_moe_experts: int = 128
    moe_router_topk: int = 8
    # moe_router_dtype 未设置 → None → 默认走 bf16 路径

# qwen_provider.py:433-454
class Qwen3NextModelProvider(Qwen3MoEModelProvider):
    num_moe_experts: int = 512
    moe_router_topk: int = 10
    moe_router_dtype: str = "fp32"    # 显式 fp32，512 专家需要更高精度
```

Qwen3Next 的 512 专家变体显式启用了 `moe_router_dtype='fp32'`，说明 NVIDIA 团队认为大量专家时 bf16 routing 精度不够。

---

# Part II — vLLM 推理侧

## 11. 关键文件

| 组件 | 文件路径 | 关键行 |
|------|----------|--------|
| Qwen3 MoE Block | `vllm/model_executor/models/qwen3_moe.py` | L121-210 |
| ReplicatedLinear (gate) | `vllm/model_executor/layers/linear.py` | L296-367 |
| LinearBase (params_dtype) | `vllm/model_executor/layers/linear.py` | L243-279 |
| UnquantizedLinearMethod | `vllm/model_executor/layers/linear.py` | L196-240 |
| Gate GEMM 实际调用 | `vllm/model_executor/layers/utils.py` | L99-105 |
| FusedMoE.select_experts | `vllm/model_executor/layers/fused_moe/layer.py` | L1519-1636 |
| fused_topk (Python) | `vllm/model_executor/layers/fused_moe/fused_moe.py` | L1101-1130 |
| topk_softmax (C++ 入口) | `vllm/_custom_ops.py` | L1977-1986 |
| topk_softmax (CUDA kernel) | `csrc/moe/topk_softmax_kernels.cu` | L675-705 |
| FusedMoE.forward_impl | `vllm/model_executor/layers/fused_moe/layer.py` | L1889-1985 |

## 12. Router 模块初始化

```python
# qwen3_moe.py:178-184
self.gate = ReplicatedLinear(
    config.hidden_size,      # e.g. 2560
    config.num_experts,      # e.g. 128
    bias=False,
    quant_config=quant_config,  # ← 传入了模型的 quant_config！
    prefix=f"{prefix}.gate",
)
```

**未显式指定 `params_dtype`**，走 `LinearBase.__init__` 默认逻辑：

```python
# linear.py:275-276
if params_dtype is None:
    params_dtype = torch.get_default_dtype()  # vllm 根据 --dtype 设置，通常 bf16/fp16
```

**与其他 MoE 模型的关键差异**：Qwen3 MoE **传入了 `quant_config`**，大多数模型显式传 `quant_config=None`：

| 模型 | gate 的 `quant_config` | gate 的 `params_dtype` |
|------|----------------------|----------------------|
| **Qwen3 MoE** | **`quant_config`（跟随模型）** | 默认 (bf16) |
| Qwen2 MoE | `None` | 默认 |
| DeepSeek-V2 | `None` | 默认 |
| Mixtral | `None` | 显式指定 |
| MiniMax-Text-01 | `None` | **`torch.float32`** |
| Nemotron-H | `None` | **`torch.float32`** |
| ERNIE 4.5 MoE | `None` | **`torch.float32`** |

## 13. Gate GEMM 计算

**Forward 路径**：

```python
# qwen3_moe.py:198
router_logits, _ = self.gate(hidden_states)   # hidden_states: [num_tokens, hidden_dim]
```

对于非量化场景，最终调用：

```python
# linear.py:240 → utils.py:99-105
def default_unquantized_gemm(layer, x, weight, bias):
    return torch.nn.functional.linear(x, weight, bias)
    # x: bf16, weight: bf16 → output: bf16
```

**没有任何 dtype cast 逻辑** — 与 Megatron 的 `RouterGatingLinearFunction` 不同，vLLM 就是一个普通的 `F.linear`，输入什么 dtype，输出什么 dtype。

## 14. TopK + Softmax

Router logits 进入 `FusedMoE.select_experts`（`layer.py:1609-1616`），Qwen3 MoE 走 `fused_topk` 路径：

```python
# layer.py:1610-1616 — 无 grouped_topk，无 bias correction
topk_weights, topk_ids, token_expert_indices = fused_topk(
    hidden_states=hidden_states,
    gating_output=router_logits,   # bf16（来自 gate GEMM）
    topk=self.top_k,              # 8
    renormalize=self.renormalize,  # config.norm_topk_prob
)
```

`fused_topk` 内部（`fused_moe.py:1101-1130`）：

```python
# 输出 topk_weights 始终分配为 FP32
topk_weights = torch.empty(M, topk, dtype=torch.float32, ...)
topk_ids = torch.empty(M, topk, dtype=torch.int32, ...)

# 调用 CUDA kernel — gating_output 可以是 float/half/bf16
ops.topk_softmax(topk_weights, topk_ids, token_expert_indices,
                 gating_output, renormalize)
```

## 15. topk_softmax CUDA Kernel 内部

`csrc/moe/topk_softmax_kernels.cu:675-705` 按输入 dtype 分发：

```cpp
if (gating_output.scalar_type() == at::ScalarType::Float) {
    dispatch_topk_softmax_launch<float>(...);
} else if (gating_output.scalar_type() == at::ScalarType::Half) {
    dispatch_topk_softmax_launch<__half>(...);
} else if (gating_output.scalar_type() == at::ScalarType::BFloat16) {
    dispatch_topk_softmax_launch<__nv_bfloat16>(...);
}
```

kernel 内部**所有 softmax/topk 计算都先转为 FP32**：

```cpp
// topk_softmax_kernels.cu:55-63
template <typename T>
__device__ __forceinline__ float toFloat(T value) {
    if constexpr (std::is_same_v<T, float>)          return value;
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) return __bfloat162float(value);
    else if constexpr (std::is_same_v<T, __half>)        return __half2float(value);
}

// topk_softmax_kernels.cu:91,107,122 — 读取 gating_output 后立即转 float
const float val = toFloat(input[idx]);
```

输出的 `topk_weights` 是 FP32（`torch.float32`）。

## 16. Routing Method

```python
# qwen3_moe.py:175
routing_method_type=RoutingMethodType.Renormalize
```

语义：先 TopK 选择 expert，再对选出的 weights 做 softmax 归一化。在 `topk_softmax` CUDA kernel 内部一步完成。

## 17. vLLM 推理侧完整 Dtype 流

```
hidden_states (BF16, 来自上一层)
    │
    ▼
[Gate GEMM]  F.linear(hidden_states, gate_weight)
    │  weight: bf16（非量化）/ 可能 fp8（若 quant_config 未跳过 gate）
    │  计算: bf16 × bf16 → bf16
    ▼
router_logits (BF16)              ← 精度瓶颈
    │
    ▼
[topk_softmax CUDA kernel]
    │  内部: toFloat() 转 fp32 → softmax fp32 → topk fp32
    ▼
topk_weights (FP32), topk_ids (INT32)
    │
    ▼
[FusedMoE Expert GEMM]  Triton kernel, FP32 accumulation
    ▼
final_hidden_states (BF16)
```

## 18. 影响 Router Dtype 的配置

| 配置项 | 作用域 | 影响 |
|--------|--------|------|
| `--dtype` (vllm 启动参数) | `torch.get_default_dtype()` | 决定 gate 权重和 GEMM 的 dtype |
| `--quantization` | quant_config | 若为 FP8 且 gate 未被排除，gate **可能被量化** |
| quant config 的 `ignored_layers` | 按 prefix 排除 | 可显式排除 gate 层不被量化 |
| `config.norm_topk_prob` | renormalize | 控制 topk_softmax 是否做归一化 |

---

# Part III — 训练 vs 推理对比

## 19. 完整对比表

Qwen3 MoE（128 experts, topk=8, `params_dtype=bf16`），对比 Megatron 训练（默认 `moe_router_dtype=None`）和 vLLM 推理（默认配置）。

| 阶段 | Megatron 训练（默认） | Megatron 训练（`moe_router_dtype='fp32'`） | vLLM 推理（默认） |
|------|------|------|------|
| **Gate 权重存储** | bf16 | bf16 | bf16 |
| **Gate 量化保护** | N/A（FP8 不影响 router 权重） | N/A | **无**（`quant_config` 透传，FP8 可能量化 gate） |
| **Router GEMM 方式** | `RouterGatingLinearFunction`（自定义 autograd） | 同左 | `F.linear`（普通 PyTorch） |
| **GEMM 前 dtype cast** | 显式 `.to(router_dtype)` | 显式 `.to(fp32)` | **无 cast** |
| **GEMM 计算 dtype** | **bf16** | **fp32** | **bf16** |
| **logits dtype** | **bf16** | **fp32** | **bf16** |
| **TopK 选择 dtype** | **bf16** | **fp32** | **bf16**（kernel 内转 fp32 后选取） |
| **Softmax 计算** | `softmax(dtype=fp32)` → fp32 | `softmax(dtype=fp32)` → fp32 | CUDA kernel 内 fp32 |
| **Softmax → probs 转换** | `.type_as(logits)` → **bf16 截断** | `.type_as(logits)` → **fp32 无损** | 直接输出 **fp32** |
| **topk_weights 最终 dtype** | **bf16** | **fp32** | **fp32** |
| **Expert 加权位置** | Expert 内（fc1 后 `act(x) * probs`） | 同左 | FusedMoE Triton kernel 内 |
| **Expert 加权精度** | bf16 × bf16 → **bf16** | bf16 × fp32 → **fp32** → bf16 | fp32 weights 进入 kernel |
| **Router dtype 配置** | `TransformerConfig.moe_router_dtype` | 同左 | **无等效配置** |
| **最终输出** | bf16 | bf16 | bf16 |

## 20. 关键差异分析

### 20.1 Gate GEMM 精度控制

- **Megatron** 有专门的 `RouterGatingLinearFunction`，在 GEMM 前**显式 cast**输入和权重到 `router_dtype`，提供了 bf16→fp32 提升的能力。
- **vLLM** 用普通的 `F.linear`，**没有任何 dtype cast 逻辑**。Gate GEMM 始终在权重/输入的原始 dtype 下执行。

### 20.2 量化场景下的 Gate 保护

- **Megatron** 的 router 是独立模块（`Router` 类），FP8 量化只作用于 `GroupedMLP` 的 expert FFN，router 权重**天然不受影响**。
- **vLLM** 的 Qwen3 MoE **透传了 `quant_config` 给 gate**（`qwen3_moe.py:182`），在 FP8 量化场景下 gate 层有可能被量化（取决于具体 quant method 是否处理 `ReplicatedLinear`）。大多数其他模型（Qwen2-MoE, DeepSeek-V2, Mixtral 等）显式传 `quant_config=None` 避免此问题。

### 20.3 TopK 选择精度

- **Megatron 默认**：TopK 在 bf16 logits 上执行，128 个专家的 logit 差异可能被 bf16 的低尾数精度（~3.3 位）抹平。
- **vLLM**：虽然 logits 也是 bf16，但 CUDA kernel 内部**先转 fp32 再做 topk**，topk 选择本身的精度高于 Megatron 默认路径。
- **Megatron `fp32` 路径**：logits 本身就是 fp32，topk 在 fp32 上执行，精度最高。

### 20.4 Softmax → Probs 的截断

- **Megatron 默认**：`softmax(dtype=fp32).type_as(logits)` 将 fp32 结果**截断回 bf16**，probs 精度损失。
- **Megatron fp32**：`.type_as(logits)` 中 logits 是 fp32，probs 保持 fp32。
- **vLLM**：`topk_weights` 输出始终是 fp32，**不存在截断问题**。这一点 vLLM 比 Megatron 默认路径更好。

### 20.5 配置灵活性

- **Megatron** 提供 `moe_router_dtype` 配置，一个参数控制整条 routing 路径的精度。
- **vLLM** 没有等效的统一配置。要提升 gate 精度，需要在模型代码中手动修改（如加 `params_dtype=torch.float32` 和 `quant_config=None`，并在 forward 中 cast 输入）。

## 21. 总结

```
                     Gate GEMM        TopK选择        Softmax→Probs     配置能力
                     ─────────        ────────        ─────────────     ────────
Megatron (默认)       bf16             bf16            fp32→bf16截断     有 moe_router_dtype
Megatron (fp32)       fp32 ✓           fp32 ✓          fp32 无损 ✓       ↑
vLLM (默认)           bf16             bf16→fp32(kernel) fp32 无损 ✓     无统一配置
```

vLLM 的 topk_softmax kernel 在 kernel 内部做了 fp32 提升，使得 softmax 和 topk 选择精度优于 Megatron 默认路径；但 Gate GEMM 本身仍在 bf16 下执行且无配置可调，加上 gate 可能被 FP8 量化，**整体 routing 精度保护弱于 Megatron 的 `moe_router_dtype='fp32'` 路径**。
