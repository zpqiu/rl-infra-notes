# DeepSeek-V4 模型结构学习笔记

按 `model.py` 中类与函数的**定义顺序**梳理。每节包含三块：
- **论文位置**：对应技术报告章节/公式
- **代码定位**：`model.py` / `kernel.py` 行号
- **关键理解**：我们讨论过的要点

---

## 0. 总览

### 0.1 模型规格（DeepSeek-V4-Pro，来自 `config.json`）

| 项目 | 值 | 说明 |
|---|---|---|
| 总层数 | `n_layers=43` + `n_mtp_layers=1` | 43 个 transformer block + 1 个 MTP |
| `dim` | 4096 | 主隐藏维度 |
| `vocab_size` | 129280 | 词表 |
| `n_heads` | 64 | Q head 数（K/V 共享 → MQA） |
| `head_dim` | 512 | 每个 Q/K/V head 维度 |
| `rope_head_dim` | 64 | partial RoPE 的旋转维度 |
| `q_lora_rank` | 1024 | Q 低秩投影中间维（共享给 indexer） |
| `o_lora_rank` | 1024 | 输出 grouped low-rank 投影维度 |
| `o_groups` | 8 | 输出投影的 head 分组数 |
| `n_routed_experts` | 256 | MoE 路由专家数 |
| `n_activated_experts` | 6 | 每 token 激活专家数 |
| `n_shared_experts` | 1 | 共享专家 |
| `n_hash_layers` | 3 | 前 3 层用 hash 路由 |
| `hc_mult` | 4 | Hyper-Connections 残差扩展倍数 |
| `index_n_heads` | 64 | indexer head 数 |
| `index_head_dim` | 128 | indexer head 维度 |
| `index_topk` | 512 | CSA 选 top-512 压缩块 |
| `window_size` | 128 | SWA 窗口大小 |
| `original_seq_len` | 65536 | 训练上下文长度（YaRN 用） |
| `rope_factor` | 16 | YaRN 外推因子 |
| `rope_theta` | 10000 | 纯 SWA 层的 RoPE base |
| `compress_rope_theta` | 160000 | CSA/HCA 层的 RoPE base |
| `compress_ratios` | `[0,0,4,128,4,128,...,4,0]` | 每层注意力类型（44 项）|

### 0.2 整体架构（Figure 2）

```
Embedding → repeat hc_mult 份 → N × Block → ParallelHead → logits
                                                         ↘
                                                          MTP Block → MTP loss
```

每个 Block 内部（`Block.forward`，model.py:689-701）：

```
residual = x
x, post, comb = hc_pre(x, hc_attn_*)        ← Pre-Block Mixing
x = attn_norm(x); x = attn(x)                ← Hybrid Attention (CSA / HCA / SWA)
x = hc_post(x, residual, post, comb)         ← Post-Block Mixing

residual = x
x, post, comb = hc_pre(x, hc_ffn_*)
x = ffn_norm(x); x = ffn(x, input_ids)       ← DeepSeekMoE
x = hc_post(x, residual, post, comb)
```

### 0.3 三种层类型（由 `compress_ratios[layer_id]` 决定）

| ratio | 层类型 | compressor | indexer | KV 来源 | 论文图 |
|---|---|---|---|---|---|
| 0 | 纯 SWA | ✗ | ✗ | 最近 128 token | — |
| 4 | CSA | ✓ (overlap) | ✓ | SWA + top-512 压缩块 | Figure 3 |
| 128 | HCA | ✓ (no overlap) | ✗ | SWA + 全部压缩块 | Figure 4 |

`config.json` 的 `compress_ratios`：`[0, 0, 4, 128, 4, 128, ..., 4, 128, 4, 0]`，前两层和最后一层是纯 SWA（含 MTP 层），中间 41 层 CSA/HCA 交错。

---

## 1. 量化与 Linear（model.py:108-180）

### 1.1 三种 dtype 路径

`linear()` 函数（L108-120）根据 weight dtype 分发：

```python
if weight.dtype == torch.float4_e2m1fn_x2:    # FP4 expert weight
    x, s = act_quant(x, block_size, scale_fmt, scale_dtype)  # x → FP8
    return fp4_gemm(x, s, weight, weight.scale, scale_dtype)
elif weight.dtype == torch.float8_e4m3fn:     # FP8 backbone
    x, s = act_quant(x, block_size, scale_fmt, scale_dtype)
    return fp8_gemm(x, s, weight, weight.scale, scale_dtype)
else:
    return F.linear(x, weight)                 # BF16 fallback
```

### 1.2 Linear 类的三种存储格式（L123-152）

| dtype | weight 形状 | scale 形状 | 适用 |
|---|---|---|---|
| FP4 | `[out, in/2]` (`float4_e2m1fn_x2`，2 个 FP4 packed) | `[out, in/32]` `e8m0` | MoE expert 权重 |
| FP8 | `[out, in]` `e4m3fn` | `[out/128, in/128]` `e8m0` | 主干所有 Linear |
| BF16 | `[out, in]` | None | RMSNorm、HC 参数等敏感模块 |

### 1.3 张量并行变体

- `ColumnParallelLinear`：切 `out_features`，无需 all-reduce
- `RowParallelLinear`：切 `in_features`，需要 all-reduce 求和

### 1.4 量化精度（来自论文 §3.4 + 代码细节）

| 模块 | 精度 |
|---|---|
| Embedding / RMSNorm 权重 | fp32 |
| 主干 Linear 权重 | FP8 + ue8m0 per-block scale |
| MoE Expert 权重 | **FP4** + ue8m0 scale |
| KV 非 RoPE 维 | FP8（per-64 块）|
| KV RoPE 维 | BF16（精度敏感）|
| Indexer KV | Hadamard 旋转 + FP4 |
| HC 参数 | FP32 |

---

## 2. RMSNorm（model.py:183-196）

```python
var = x.float().square().mean(-1, keepdim=True)
x = x * torch.rsqrt(var + eps)
return (self.weight * x).to(dtype)
```

- 在 fp32 内做计算，最后 cast 回原 dtype
- weight 存为 fp32 参数（checkpoint 是 bf16 加载时转 fp32）

---

## 3. RoPE 与 YaRN（model.py:199-244）

### 3.1 RoPE 是什么

位置编码 `θ_t,i = t · 1/base^(2i/d)`：
- 高频维（i 小）：转得快，64K 内已转过千百圈
- 低频维（i 大）：转得慢，可能 64K 内没转完一圈

V4 是 **partial RoPE**：只对最后 `rope_head_dim=64` 维做旋转。

### 3.2 YaRN 是什么

YaRN **不是另一种位置编码**，而是 RoPE 长度外推的频率修正方案。

V4 训练上下文 64K，目标推理 1M，`factor=16`。

```python
# precompute_freqs_cis (L199-229)
freqs = 1.0 / (base ** (arange(0, dim, 2) / dim))
if original_seq_len > 0:                                  # ← YaRN 触发开关
    low, high = find_correction_range(beta_fast, beta_slow, dim, base, original_seq_len)
    smooth = 1 - linear_ramp_factor(low, high, dim // 2)
    freqs = freqs / factor * (1 - smooth) + freqs * smooth
                  ↑                          ↑
               缩小后的频率                原始频率
```

**分维度处理**：
- 高频维（在 64K 内 ≥32 圈）：保持原 freq
- 低频维（在 64K 内 ≤1 圈）：除以 factor=16
- 中间维：线性 ramp 平滑过渡

参数：`beta_fast=32`、`beta_slow=1`。

### 3.3 V4 的两套 RoPE 配置

| | 纯 SWA 层 | CSA / HCA 层 |
|---|---|---|
| `base` | 10000 | **160000**（NTK 预放大 16×） |
| `original_seq_len` | 0（关闭 YaRN） | 65536（开启 YaRN） |
| 看的位置范围 | 最近 128 | 最远 1M |
| 频率表存量 | 1 份 | 1 份 |

注意 `@lru_cache(2)` 装饰 `precompute_freqs_cis`——刚好缓存这两种配置。

### 3.4 一致性约束：CSA/HCA 层里 SWA K 也要走 YaRN

**关键事实**：在 CSA/HCA 层里，**Q、SWA K、压缩 K 三者用同一份 `freqs_cis`**：

```python
# Attention.__init__ (L480-482)
self.register_buffer("freqs_cis", freqs_cis, ...)

# Attention.forward
apply_rotary_emb(q[..., -rd:], freqs_cis)               # Q
apply_rotary_emb(kv[..., -rd:], freqs_cis)              # SWA K

# Compressor.forward (复用)
apply_rotary_emb(kv[..., -rd:], self.freqs_cis[...])    # 压缩 K
```

**为什么必须一致**：RoPE 的相对位置性质 `Q·K = q·R_{s-t}·k` 要求两边用同一频率表。Q 是单一向量，被同时用于跟 SWA K 和压缩 K 做点积——所以 K 必须用相同频率。

### 3.5 inverse RoPE

```python
# Attention.forward L534
apply_rotary_emb(o[..., -rd:], freqs_cis, inverse=True)
```

由于 V4 是 Shared-KV（V=K），attention 输出会带着 K 的"绝对位置成分"。反旋转用 query 位置的共轭把这部分抹掉，留下纯相对位置贡献。

---

## 4. SWA 窗口与 ring buffer（model.py:254-265, 268-276）

### 4.1 SWA cache 是 ring buffer

```python
# Attention.forward L530
self.kv_cache[:bsz, start_pos % win] = kv.squeeze(1)
```

第 `t` 个 token 永远写到第 `t % win` 号物理槽。写满 128 后绕回覆盖。

### 4.2 `get_window_topk_idxs` 的三个分支

```python
# decode 稳态（start_pos >= win-1）
start_pos %= window_size
matrix = cat([arange(start_pos+1, win), arange(0, start_pos+1)])
# 例：start_pos=10000 → [17, 18, ..., 127, 0, 1, ..., 16]
# 这是 [step-127..step] mod 128 的物理槽位序列
```

**注意**：返回的索引顺序对 attention 没影响（softmax 与顺序无关），位置信息已经焊在 K 的 RoPE 里。

### 4.3 `get_compress_topk_idxs`（HCA 用）

返回从 0 开始到当前位置之前的所有完整压缩块索引（带 `+ offset`），prefill 时还会把"未来块"标 -1。**HCA 因为没有 indexer，所有合法压缩块都进入 attention**。

---

## 5. Compressor（model.py:279-377）

### 5.1 它在做什么

把 hidden states `H ∈ R^{n×d}` 压成 `C^Comp ∈ R^{n/m × c}`，每条 entry 代表 m 个 token 的加权聚合。

### 5.2 两种模式

| | CSA Compressor | HCA Compressor |
|---|---|---|
| `compress_ratio` | 4 | 128 |
| `overlap` | True | False |
| `coff = 1+overlap` | 2 | 1 |
| `wkv` 输出 | `[..., 2·head_dim]` (含 C^a, C^b) | `[..., head_dim]` (单 C) |
| 压缩条目 i 覆盖 | 2m=8 个 token（与邻块重叠半数） | m=128 个 token（不重叠）|
| 论文公式 | (11)-(12) | (22)-(23) |

### 5.3 prefill 路径（`start_pos == 0`）

```python
kv = wkv(x).unflatten(1, (-1, ratio))           # [b, s/m, m, 2d]
score = wgate(x).unflatten(1, (-1, ratio)) + ape  # 加 B^{a/b}
if overlap:
    kv    = overlap_transform(kv, 0)             # [b, s/m, 2m, d]
    score = overlap_transform(score, -inf)
kv = (kv * score.softmax(dim=2)).sum(dim=2)     # 公式 (12)
```

**`overlap_transform` 干的事**：构造每个压缩块的"前块 C^a + 当前块 C^b"拼接结构。第一块 i=0 的前半填 0/-inf（masked）。

### 5.4 decode 路径（`start_pos > 0`）

用 `kv_state`/`score_state` 作为 ring buffer 缓存最近 m 个 token；每 `m` 步触发一次实际压缩并写入 cache：
- `should_compress = (start_pos + 1) % ratio == 0`
- 触发时：把 buffer 里的 `[前块 C^a; 当前块 C^b]` 做 softmax 加权求和

### 5.5 写入 KV cache 时的 RoPE

```python
if start_pos == 0:
    freqs_cis = self.freqs_cis[:cutoff:ratio]   # 0, m, 2m, ...
else:
    freqs_cis = self.freqs_cis[start_pos + 1 - ratio]  # = m·k = block 第一个 token 位置
apply_rotary_emb(kv[..., -rd:], freqs_cis)
```

→ **压缩块 k 的 RoPE 位置 = `m·k`，即块第一个 token 在原始序列中的位置**。这让压缩 K 和 SWA K 处于同一坐标系。

### 5.6 量化路径

- 主 Compressor（rotate=False）：非 RoPE 维做 `act_quant`（FP8）
- Indexer 的 Compressor（rotate=True）：先 Hadamard 旋转 + `fp4_act_quant`

---

## 6. Indexer（model.py:380-433）

CSA 专属。它的作用是**为每个 query 选 top-512 个最相关的压缩块**。

### 6.1 与主 Attention 的共享与差异

| | 主 Attention | Indexer |
|---|---|---|
| 用途 | 算实际输出 | 选 top-k |
| head 数 | 64 (MQA) | 64 (MQA) |
| head_dim | 512 | 128 |
| K/V 精度 | FP8 + BF16 | FP4 + Hadamard |
| Q 共享 | `c^Q = q_norm(wq_a(x))` ───── 共享 ───── 这边 `wq_b` 投到 64×128 |
| 打分 | softmax(QK/√d) | ReLU(qk) · per-head 权重 |
| 输出 | attn output | top-k 索引 |

### 6.2 评分公式（论文公式 13-17）

```python
q = wq_b(qr).unflatten(...)                  # [b, s, n_h, 128]
apply_rotary_emb(q[..., -64:], freqs_cis)
q = rotate_activation(q)                      # Hadamard 旋转
fp4_act_quant(q, ..., inplace=True)

self.compressor(x, start_pos)                 # 写 indexer 自己的 K^IComp cache

weights = self.weights_proj(x) * (softmax_scale * n_heads ** -0.5)
index_score = einsum("bshd,btd->bsht", q, kv_cache)
index_score = (index_score.relu_() * weights.unsqueeze(-1)).sum(dim=2)
# index_score : [b, s, t]   每 query 对每个块一个标量分

topk_idxs = index_score.topk(min(512, end_pos // ratio), dim=-1)[1]
topk_idxs += offset           # 转换到主 KV cache 的物理坐标
```

### 6.3 三个关键设计点

**(a) 共享 c^Q**：Indexer 不重新算 `wq_a`，直接接收主 attention 的 `qr`，节省一次 d→1024 的投影。

**(b) ReLU 而非 softmax**：保留绝对幅值，不相关块直接 0。top-k 选择更干净，FP4 算术下也更稳。

**(c) MQA 极致版**：Indexer 的 K cache 没有 head 维（`[B, T/4, 128]`）——所有 64 个 indexer head 共享同一份 K。

---

## 7. Attention 主类（model.py:436-543）

### 7.1 MLA 骨架

```python
# Q：低秩投影
qr = q = q_norm(wq_a(x))                       # [b, s, q_lora_rank=1024] = c^Q
q = wq_b(q).unflatten(-1, (n_h, head_dim))     # [b, s, 64, 512]
q *= rsqrt(q.square().mean(-1, keepdim=True) + eps)  # inline RMSNorm
apply_rotary_emb(q[..., -rd:], freqs_cis)

# KV：单次低秩投影（Shared KV）
kv = kv_norm(wkv(x))                           # [b, s, 512]   ← 没有 head 维！
apply_rotary_emb(kv[..., -rd:], freqs_cis)
act_quant(kv[..., :-rd], 64, ..., inplace=True)  # 非 RoPE 维 FP8

# 输出：grouped low-rank
o = o.view(bsz, seqlen, n_local_groups, -1)
o = einsum("bsgd,grd->bsgr", o, wo_a.weight.view(...))  # 分组压缩
x = wo_b(o.flatten(2))                                   # 回到 d
```

### 7.2 KV cache 布局

```
kv_cache: [B, win + max_seq_len/m, head_dim]
              ↑       ↑
          SWA 段     压缩段
        128 个槽   max_seq_len/m 个槽
        ring buffer 写入  线性写入
```

**关键**：`Compressor.kv_cache` 是 `Attention.kv_cache[:, win:]` 的 view（L491）——**两段在物理上共享同一块连续内存**。

### 7.3 SWA 与压缩段的索引拼接

```python
topk_idxs = get_window_topk_idxs(win, ...)    # SWA 索引 [0, win)
if compress_ratio:
    offset = kv.size(1) if start_pos == 0 else win
    if indexer is not None:
        compress_topk_idxs = indexer(x, qr, start_pos, offset)        # 已加 offset
    else:
        compress_topk_idxs = get_compress_topk_idxs(ratio, ..., offset)
    topk_idxs = cat([topk_idxs, compress_topk_idxs], dim=-1)
```

#### prefill：offset = seqlen
prefill 时 sparse_attn 的 KV 是 `cat([原始 token, 压缩条目], dim=1)`：

```
kv [B, seqlen + seqlen/m, d]
   |←─ 原始 ─→|←── 压缩 ──→|
   0          seqlen=offset
```

#### decode：offset = win = 128
decode 时 sparse_attn 的 KV 是 `kv_cache` 直接：

```
kv_cache [B, 128 + max/m, d]
         |←SWA→|←─ 压缩 ──→|
         0     128=offset
```

无论哪种情况，`topk_idxs` 都是"SWA 索引 + (压缩索引 + offset)"的拼接，物理上指向同一个 KV tensor。

### 7.4 SWA 与压缩段的"逻辑重叠"

SWA 窗的最近 128 个 token 也会被包进压缩块（因为压缩 buffer 接收所有 token）。这是**有意设计**：

1. **K 向量本身不同**：SWA K 是单 token + 主分支算子；压缩 K 是 m 个 token softmax 聚合 + compressor 算子。geometrically distinct。
2. **softmax 自适应分配**：模型学习何时偏好 SWA（局部精确）、何时偏好压缩（远距离覆盖）。
3. **必要性**：因果性要求 query 不能看到包含自己 token 的压缩块（论文 §2.3.3）——SWA 正好填这个洞。

### 7.5 Attention sink

```python
self.attn_sink = nn.Parameter(torch.empty(n_local_heads, dtype=torch.float32))
```

每 head 一个可学习"垃圾桶"logit `z'_h`，传给 sparse_attn kernel。在 softmax 分母里加 `exp(z'_h)`，让 attention 输出可以接近 0（不强制权重和为 1）。论文公式 (27)。

---

## 8. sparse_attn kernel（kernel.py:276-368）

### 8.1 它是 V4 唯一的 attention kernel

SWA、CSA、HCA 三种层全都用同一个 `sparse_attn`，区别只在喂的 `topk_idxs` 不同。

### 8.2 接口

```python
sparse_attn_kernel_(
    q          : [b, m, h, d]   BF16
    kv         : [b, n,    d]   BF16    ← MQA：没有 head 维！
    o          : [b, m, h, d]   BF16
    attn_sink  : [h]            FP32
    topk_idxs  : [b, m, topk]   INT32   ← -1 = padding/未来
)
```

### 8.3 FlashAttention 风格的 online softmax

```python
for t in num_blocks (each block = 64 indices):
    # gather 64 个 KV
    kv_shared[i] = kv[idxs[i]]  if idxs[i]!=-1 else 0
    acc_s[i,j] = 0              if idxs[j]!=-1 else -inf
    
    # QK^T
    T.gemm(q_shared, kv_shared, acc_s, transpose_B=True)
    acc_s *= scale
    
    # online softmax 状态更新
    scores_max_prev = scores_max
    scores_max = max(scores_max, max(acc_s, dim=1))
    scores_scale = exp(scores_max_prev - scores_max)
    acc_s = exp(acc_s - scores_max)
    sum_exp = sum_exp * scores_scale + sum(acc_s)
    
    # PV 累加
    acc_o *= scores_scale
    T.gemm(acc_s, kv_shared, acc_o)

# attention sink（仅加分母）
sum_exp += exp(attn_sink - scores_max)
acc_o /= sum_exp
```

### 8.4 几个关键设计点

| 维度 | 说明 |
|---|---|
| **Shared KV** | `kv` 单 tensor，K 和 V 复用同一份 |
| **MQA** | `kv` 没有 head 维，64 个 head 共享 K/V |
| **-1 哨兵** | 同时承担 mask 和"超出 topk 边界"的角色 |
| **head padding** | `n_local_heads<16` 时 pad 到 16 让 GEMM 维度对齐 |
| **每 CTA 一 query** | grid 维度 `(m, b)`，decode 时 m=1 完美并行 |
| **Layout** | Q 是 `[b, m, h, d]`，方便 decode 时单 query 取 `[h, d]` 到 shared mem |

### 8.5 RoPE 在哪里？

**Kernel 里没有 RoPE**——它只做"按索引的 attention"。RoPE 在 kernel 外部：
- 调用前：q 和 kv 的 last 64 dims 已加 RoPE
- 调用后：output 的 last 64 dims 加 inverse RoPE

---

## 9. 块的位置语义（重要）

### 9.1 压缩块 K 的 RoPE 位置 = 块第一个 token 的原始位置

```python
# Compressor.forward (L362-368)
freqs_cis = self.freqs_cis[block_index * m]   # 不是 block_index！
```

→ 块 k 在 RoPE 几何里"占据"原始序列位置 `m·k`。

### 9.2 同一坐标系

- Q at step t            → RoPE 位置 t
- SWA K at step s        → RoPE 位置 s
- 压缩 K of block k      → RoPE 位置 `m·k`
- Indexer K of block k   → RoPE 位置 `m·k`

→ 所有 K 活在同一条原始序列时间轴上。`topk_idxs` 里的整数只是物理 cache 槽位，**位置信息完全 bake 在 K 的 RoPE 旋转里**。

---

## 10. Gate（model.py:546-584）

### 10.1 两种路由模式

```python
self.hash = layer_id < args.n_hash_layers   # 前 3 层 hash 路由

if self.hash:
    self.tid2eid = Parameter(empty(vocab_size, n_activated, int32), requires_grad=False)
    self.bias = None
else:
    self.bias = Parameter(empty(n_routed_experts, fp32))
```

### 10.2 forward 流程

```python
scores = linear(x.float(), self.weight.float())    # [B*S, 256]
scores = F.softplus(scores).sqrt()                  # sqrtsoftplus
original_scores = scores

if self.bias is not None:
    scores = scores + self.bias                     # 仅影响 top-k 选择
if self.hash:
    indices = self.tid2eid[input_ids]               # 查表
else:
    indices = scores.topk(self.topk, dim=-1)[1]     # 学习选择

weights = original_scores.gather(1, indices)       # 注意：gather 自原始分数
weights /= weights.sum(dim=-1, keepdim=True)
weights *= self.route_scale                         # × 1.5
```

### 10.3 Hash routing 关键点

- `tid2eid: [vocab_size=129280, n_activated=6]` 是预计算的查找表，**不参与梯度**
- 每个 token ID → 6 个固定 expert ID（永不改变）
- **路由决策硬编码，路由权重仍然是学的**（gather 自学习的 `weight` 矩阵 score）

**为什么前 3 层用 hash？**
1. 早期层 hidden state ≈ embedding，学习路由信号弱
2. Hash 表预计算时已保证 256 个 expert 负载均衡（≈129280×6/256≈3030 token/expert），训练 0 开销
3. 强制 token-级专家化，与后层语义级路由形成层次化分工
4. 替代 V3 的 dense FFN（V3 前几层是 dense 的），同时降低激活参数

### 10.4 aux-loss-free balancing（learned 层）

```python
scores = scores + self.bias    # bias 进入 top-k 排名
weights = original_scores.gather(1, indices)   # 但权重用未加 bias 的分数
```

bias 通过训练时根据 expert 负载动态调整：
- 处理多了 → bias↓ → 选中概率↓
- 处理少了 → bias↑ → 选中概率↑

→ 不需要额外 aux loss，自然均衡。

---

## 11. Expert + MoE（model.py:587-645）

### 11.1 Expert：SwiGLU FFN

```python
class Expert:
    w1: dim → inter_dim   # gate
    w2: inter_dim → dim   # down
    w3: dim → inter_dim   # up
    
    def forward(x, weights):
        gate = w1(x).float()
        up = w3(x).float()
        if swiglu_limit > 0:           # = 10.0，clamp 防爆
            up = clamp(up, -10, 10)
            gate = clamp(gate, max=10)
        x = silu(gate) * up
        if weights is not None:
            x = weights * x
        return w2(x.to(dtype))
```

### 11.2 MoE 调度

```python
# MoE.forward (L630-645)
weights, indices = self.gate(x, input_ids)
counts = bincount(indices.flatten(), minlength=n_experts).tolist()

for i in range(experts_start_idx, experts_end_idx):   # 当前 rank 负责的 expert
    if counts[i] == 0:
        continue
    expert = self.experts[i]
    idx, top = where(indices == i)
    y[idx] += expert(x[idx], weights[idx, top, None])

if world_size > 1:
    dist.all_reduce(y)
y += self.shared_experts(x)    # 共享 expert，所有 token 都过
```

- TP 切分：每个 rank 持有 `n_routed_experts // world_size` 个 expert
- `where(indices == i)` 找出所有要去 expert i 的 token
- 共享 expert 单独处理一次（无路由）

---

## 12. mHC: Manifold-Constrained Hyper-Connections

论文 §2.2，公式 (1)-(8)。代码：`Block.hc_pre/hc_post`（L674-687）+ `hc_split_sinkhorn`（kernel.py:371-427）。

### 12.1 思想

把残差流从 `[d]` 扩到 `[n_hc, d]`（n_hc=4），每层前后做参数化变换：

```
X_{l+1} = B_l X_l + C_l F_l(A_l X_l)
```

- `A_l ∈ R^{1×n_hc}`：把 4 份压成 1 份送入该层
- `B_l ∈ R^{n_hc×n_hc}`：约束在 Birkhoff polytope（双随机矩阵流形）
- `C_l ∈ R^{n_hc×1}`：把层输出扩回 4 份

### 12.2 代码对应

```python
# hc_pre: x [b,s,hc,d] → y [b,s,d] 进入 attn/ffn
mixes = F.linear(x.flatten(2), hc_fn) * rsqrt
pre, post, comb = hc_split_sinkhorn(mixes, hc_scale, hc_base, hc_mult, sinkhorn_iters)
# pre  = sigmoid(...)             ← A_l (公式 6)
# post = 2·sigmoid(...)            ← C_l (公式 7)
# comb = Sinkhorn(softmax(...))    ← B_l (公式 8, 投影到双随机)

y = sum(pre.unsqueeze(-1) * x, dim=hc)   # A_l · X_l

# hc_post:
out = post * F(...) + sum(comb * residual)   # = C_l F_l(...) + B_l X_l
```

### 12.3 Sinkhorn-Knopp 迭代

`hc_split_sinkhorn` kernel 把 raw mix 投影到 Birkhoff polytope：
1. softmax(comb) + eps（行归一化 + 正性）
2. 列归一化
3. 重复 `sinkhorn_iters=20` 次行/列归一化
4. 收敛到双随机矩阵（行和列都为 1）

**为什么要双随机**：保证 `‖B_l‖_2 ≤ 1`（spectral norm 有界），残差非扩张，深度堆叠数值稳定。

---

## 13. Block（model.py:648-701）

完整层逻辑，前面已展示。每层有两次 hc_pre/hc_post 包夹（一次包 attn，一次包 ffn）。

参数：
- `hc_attn_fn / hc_ffn_fn`：动态参数化的 W^{pre,post,res}（论文公式 3-5）
- `hc_attn_base / hc_ffn_base`：静态偏置 S
- `hc_attn_scale / hc_ffn_scale`：门控因子 α

---

## 14. ParallelHead 与 MTPBlock（model.py:704-767）

### 14.1 ParallelHead

```python
def forward(self, x, hc_fn, hc_scale, hc_base, norm):
    x = self.hc_head(x, hc_fn, hc_scale, hc_base)   # HC 4 → 1 份
    logits = F.linear(x[:, -1].float(), self.weight)  # vocab 分片
    if world_size > 1:
        all_gather(logits)
    return logits
```

`hc_head` 是简化版的 hc_pre：只用 sigmoid（无 Sinkhorn），把 4 份合一份做 logits。

### 14.2 MTPBlock

继承 `Block`，多了：
- `e_proj`：embedding 投影
- `h_proj`：hidden 投影
- `enorm / hnorm / norm`
- 自己的 `hc_head_*` 参数

forward：把上一步 hidden 和下一个 token embedding 融合，再跑一遍 Block，最后过 head 出 logits（next-next token 预测）。

---

## 15. Transformer（model.py:770-810）

```python
def forward(self, input_ids, start_pos=0):
    h = self.embed(input_ids)
    h = h.unsqueeze(2).repeat(1, 1, self.hc_mult, 1)   # 扩 4 份残差流
    for layer in self.layers:
        h = layer(h, start_pos, input_ids)
    logits = self.head(h, self.hc_head_fn, self.hc_head_scale, self.hc_head_base, self.norm)
    return logits
```

构造时设全局变量 `world_size, rank, default_dtype, scale_fmt, scale_dtype`，所有子模块通过这些全局变量决定形状和量化路径。

---

## 16. 推理框架与 KV cache 管理（论文 §3.6）

### 16.1 异构 KV cache

V4 的 KV cache 不再统一管理，而是分成两类：
- **State Cache**：SWA KV + 压缩 buffer 的"未压缩尾巴"——按请求分配定长 block
- **KV Cache**：CSA/HCA 的压缩条目 + indexer KV——多 block 池化管理

### 16.2 块大小对齐

每个 cache block 的 token 数取 `lcm(m, m') = lcm(4, 128) = 128`，保证：
- CSA 压缩比 4 → 128/4 = 32 条 CSA 条目/block
- HCA 压缩比 128 → 128/128 = 1 条 HCA 条目/block

### 16.3 On-disk KV cache

shared-prefix 请求复用：把压缩 KV 序列化到磁盘，命中时直接加载。SWA KV 提供三种策略（Full / Periodic / Zero）权衡存储与重算。

---

## 17. 速查：论文概念 ↔ 代码位置

| 论文概念 | model.py / kernel.py 位置 |
|---|---|
| Figure 2 整体结构 | `Transformer` / `Block` |
| mHC 公式 (1)-(8) | `Block.hc_pre/hc_post` + `hc_split_sinkhorn` |
| CSA 公式 (9)-(17) | `Attention(ratio=4)` + `Compressor(overlap=True)` + `Indexer` |
| HCA 公式 (20)-(26) | `Attention(ratio=128)` + `Compressor(overlap=False)` |
| Shared KV MQA | `wkv` 单 tensor，`sparse_attn` kernel kv 无 head 维 |
| Grouped Output Projection | `wo_a`(分组) → `wo_b` |
| Partial RoPE + inverse | `apply_rotary_emb(o, ..., inverse=True)` |
| Attention Sink (公式 27) | `self.attn_sink` |
| SWA 附加分支 | `get_window_topk_idxs` + `kv_cache[:win]` |
| Sqrt(Softplus) 路由 | `Gate.forward` (`F.softplus(scores).sqrt()`) |
| Hash routing 前 3 层 | `Gate.hash` + `tid2eid` |
| Aux-loss-free bias | `Gate.bias` 仅加给 top-k |
| YaRN 长上下文 | `precompute_freqs_cis` + `beta_fast/slow` |
| KV cache 异构布局 | `Attention.kv_cache` (SWA \| 压缩) + `Compressor.kv_state` tail |
| FP4 expert | `Expert(dtype=float4_e2m1fn_x2)` + `fp4_gemm` |
| Birkhoff 投影 | `hc_split_sinkhorn` 的 20 次 Sinkhorn 迭代 |

---

## 18. 三种层类型完整对比

```
┌──────────────────────┬─────────────────┬────────────────────┬────────────────────┐
│                      │ ratio=0 (SWA)   │ ratio=4 (CSA)      │ ratio=128 (HCA)    │
├──────────────────────┼─────────────────┼────────────────────┼────────────────────┤
│ Compressor           │ ✗               │ ✓ (overlap=True)   │ ✓ (overlap=False)  │
│ Indexer              │ ✗               │ ✓                  │ ✗                  │
│ KV cache 大小 (1M)   │ 128             │ 128 + 256K         │ 128 + 8K           │
│ topk SWA 来源        │ get_window..    │ get_window..       │ get_window..       │
│ topk 压缩来源        │ —               │ Indexer (top-512)  │ get_compress.. 全选 │
│ 每 query 看几个 KV   │ 128             │ 640                │ ≈8200              │
│ RoPE base            │ 10000 (无 YaRN) │ 160000 (+YaRN)     │ 160000 (+YaRN)     │
│ 本质                  │ 局部窗口         │ 真稀疏              │ 全压缩 dense        │
│ 论文 Figure          │ —               │ Figure 3           │ Figure 4           │
└──────────────────────┴─────────────────┴────────────────────┴────────────────────┘
```

---

## 19. 待深挖问题列表

讨论中标记为"接下来可以挖"但还没展开的点，留作后续学习：

1. **Inverse RoPE 数学推导**：为什么 V=K 时 output 会"带着位置"，反旋转如何抵消。
2. **Compressor 的 `kv_state` ring buffer 在 prefill→decode 之间怎么衔接**：边界处理细节。
3. **HC `hc_split_sinkhorn` kernel 的数值稳定性**：Sinkhorn 迭代收敛性、`eps=1e-6` 的作用。
4. **MTP 块的 next-token 预测路径**：怎么把上一步 hidden 和下一个 token 融合。
5. **`fp4_gemm` 的 FP4×FP8 混合精度实现**：scale 的 per-32 vs per-128 对齐策略。
6. **Hadamard 旋转**（`rotate_activation`）的精度收益：为什么 indexer 走 FP4 也需要先旋转。
7. **`@lru_cache(2)` 的两份缓存**到底是哪两份。
8. **YaRN 在 base=160000 下的实际频率分布**：哪些维度被缩放、哪些保留。
9. **训练时 Hash routing 表 `tid2eid` 的具体生成算法**。
10. **Multi-Token Prediction (MTP) 的训练目标**：跟 V3 完全一致还是有调整。

---

## 附录 A：术语对照

| 缩写 | 全称 | 含义 |
|---|---|---|
| MLA | Multi-head Latent Attention | 低秩 Q + 共享 KV 的 attention |
| CSA | Compressed Sparse Attention | 轻压缩 + Top-k 稀疏 |
| HCA | Heavily Compressed Attention | 重压缩 + dense over compressed |
| SWA | Sliding Window Attention | 局部窗口注意力 |
| MQA | Multi-Query Attention | 所有 head 共享 K/V |
| MoE | Mixture-of-Experts | 专家混合 |
| MTP | Multi-Token Prediction | 多步预测 |
| mHC | Manifold-Constrained Hyper-Connections | 流形约束的 HC |
| HC | Hyper-Connections | 多份残差流 |
| YaRN | Yet another RoPE extensioN | RoPE 长度外推 |
| RoPE | Rotary Position Embedding | 旋转位置编码 |
| FP4 / FP8 | floating-point 4/8 bit | 低精度浮点 |
| TP | Tensor Parallelism | 张量并行 |
| EP | Expert Parallelism | 专家并行 |
