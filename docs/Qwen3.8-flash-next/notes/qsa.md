# Qwen Sparse Attention（QSA）

备查笔记。对应 [ARCHITECTURE.md](../ARCHITECTURE.md) §5，以及之后关于 Indexer 投影/RoPE、RoPE 原理、block 打分、Gated Attention、`[:, -T:, :]` 与 \(T\) 的问答。

实现：`Qwen4ExpTextQSAIndexer` + `Qwen4ExpTextAttention`（后者继承 `Qwen3_5Attention`）。每 4 层里 1 层 QSA；进层前同样 GR Read `[B,T,2560]`，出来 `o_proj` 再 GR Write。

---

## 1. 它是什么

GDN 把历史压进固定白板（有损）。QSA 在全上下文上做一次 **稀疏但精确** 的查找。

不是每个 query 看全部 \(T\) 个 token（\(O(T^2)\)），而是大约只看 **2048 个 token**（512 block × 4，再加 tail）。选人不是逐 token top-k，而是：

1. 连续 4 个 token 收成 1 个 micro-block（K 平均）
2. block 级打分，最多留 512 块
3. 展开回 token，跑真正的注意力

两段式，权重不共享：

```
hidden [B, T, 2560]
  ├─ Indexer（轻量）→ 0/1 mask（每个 query 允许看哪些 KV）
  └─ Gated Attention（GQA）
         mask = 因果 ∧ indexer
```

Indexer **不产出 hidden**，只产出 mask。

| | Indexer（选人） | Gated Attention（干活） |
|---|---|---|
| Q heads | 4 | 24 |
| K/V | 1 个共享 K（MQA） | 2 KV（GQA） |
| head dim | 128 | 256 |
| 作用 | block 打分 | softmax 加权 V、输出门 |

这一层仍存 KV cache（2 头 × 256，随上下文涨）；计算只在约 2048 个位置上做。Indexer 另外 cache 所有 token 的 raw K（128 维），给后续 query 打分。

---

## 2. Indexer：怎么投影、怎么 RoPE

### 投影

一条无 bias 线性层：

```
index_qk_proj: 2560 → (4+1)×128 = 640
```

```
hidden [B, T, 2560]
  → qk [B, T, 640]
  → q:     [B, T, 4, 128]
  → raw_k: [B, T, 128]        # squeeze 掉唯一的 K head
```

只对 Q 做 RMSNorm（沿 128）。Token 级 K 不 Norm、不 RoPE，原样进 indexer 的 K 缓存。

### RoPE：Q 现在转，K 晚点转

同一套主干 `cos/sin`。Partial RoPE：`cos` 最后一维 **64**，Indexer head 128，只转前 64，后 64 不动。

- **Q**：`full_cos/sin` 是整段历史；当前 query 用 `full[:, -T:, :]`，按 **自己所在位置** 转。
- **K**：token 级不转。4 个 raw_k 平均 → `k_layernorm` → 用 block **第一个 token** 的位置转。

先转再平均 ≠ 平均再转。未旋转空间里平均是内容摘要，再贴块级位置。Q 必须带当前位置，才能和「带块起点的 K」做相对位置匹配。

和后面 Gated Attention 的 RoPE 独立：那边 Q/K 都是 **每个 token 自己的位置** 立刻转。

---

## 3. RoPE 是怎么做的（原理）

给 Q/K 乘依赖位置的旋转，点积带上 **相对位置**，而不是加绝对位置向量。只转 Q/K，不转 V。

二维：位置 \(m\) 的 q、位置 \(n\) 的 k 各转自己的角，点积只跟转角差 \(\theta_n-\theta_m\) 有关。

高维：\(d\) 维两两一组，每组一个频率（低频管长距离，高频管近）。Qwen 用 `rotate_half`：向量切成 \(a|b\)，

```
RoPE(x) = x * cos + rotate_half(x) * sin
rotate_half(a|b) = (-b | a)
```

即第 \(j\) 维和第 \(j+d/2\) 维一对。效果：

\[
\mathrm{RoPE}(q,m)^\top \mathrm{RoPE}(k,n) = q^\top R_{n-m}\, k
\]

Flash-Next **Partial RoPE**：注意力 head 256 只转前 64（`partial_rotary_factor=0.25`）。图/视频还有 3D MRoPE（频率拆给 T/H/W）。

---

## 4. 4 个 token 收成 block、分数怎么算

对位置 \(t\) 的 query，可见一般是 `0..t`（再去掉 padding）。从 **序列开头** 每 4 个切一块；末尾凑不满 4 的当 **tail**（不打分，必留）。

例：\(t=9\)

```
[0 1 2 3] [4 5 6 7] [8 9]
  block0    block1    tail
```

每个完整 block：4 个未 RoPE 的 raw_k 平均 → RMSNorm → 用块起点位置 RoPE。MQA，一条 K。`num_complete_blocks = 可见长度 // 4`，512 是 top-k 上限不是块数。

分数（该 query 的 Q 为 `[4,128]`，block K 为 `[B块,128]`）：

\[
s_b=\frac{1}{\sqrt{128}}\sum_{h=1}^{4}\mathrm{ReLU}(q_h\cdot k_b)
\]

ReLU：不像不扣分。对 head 求和：多视角投票。不是 softmax，不在 block 上归一化。然后 `topk(min(512, 块数))`，选中块整块展开（每块 4 个全要）。实际允许 token 数 = `2048 + (可见长度 % 4)`，最多再加 3 个最近 token。

---

## 5. 真正的 Gated Attention

Indexer 只出 mask。Gated Attention 用 **另一套投影**（与 indexer 不共享），结构即 Qwen3.5 Attention。

### GQA

```
q_proj: 2560 → 24×256×2     # ×2 因为一半是 gate
k_proj: 2560 →  2×256
v_proj: 2560 →  2×256
o_proj: 24×256 → 2560
```

2 个 KV head，每个被 12 个 Q 共用。Cache 按 2 头存。

### QK-RMSNorm

RoPE 前，每个 head 的 256 维各自 RMSNorm（`(1+γ)`）。**V 不 Norm**。这是 RMS，不是 GDN kernel 里的 L2 单位化。

### Partial RoPE

Q 和 K 都按 **该 token 自己的位置** 转前 64 维。V 不转。点积缩放 \(1/\sqrt{256}\)。

`cos/sin` 在 QSA 里先是全历史（给 indexer）；进注意力前切成当前这段，见下一节。

### `q_proj` 里的 output gate

`q_proj` 沿最后一维切成两半：Q `[B,T,24,256]` 和 gate `[B,T,24×256]`。

`softmax(qk⊤/√d + mask)`，mask = 因果 ∧ indexer。然后：

```
attn_out = attn_out * sigmoid(gate)
y = o_proj(attn_out)
```

先算完注意力，再决定放多少进残差。Gate 来自当前 hidden，不看 KV。

| | GDN 输出门 | 这里 |
|---|---|---|
| 从哪来 | 单独 `in_proj_z` | 塞在 `q_proj` 里 |
| 激活 | silu | sigmoid |
| 何时乘 | RMSNorm(y) 之后 | 加权 V 之后、o_proj 之前 |

然后 GR Write，同一层再 MoE。

---

## 6. `[:, -T:, :]` 和这里的 \(T\)

Indexer 需要历史上每个 block 起点的 RoPE，所以 `cos/sin` 按 **全长 position_ids**（cache 旧的 + 这一拍新的）来算，时间维是「已经见过的全部位置」。

Gated Attention 这一步只投影 **当前这段** 的 Q/K，RoPE 必须和这段对齐，不能拿全长 `cos/sin` 去广播：

```python
position_embeddings = (x[:, -hidden_states.shape[1] :, :] for x in position_embeddings)
```

**\(T\) = 这一次 `forward` 的 `hidden_states.shape[1]`**，不是 2048，也不是上下文总长。

| 场景 | 这次的 hidden | \(T\) |
|---|---|---|
| 一次跑完整 prompt | `[B, prompt长度, 2560]` | prompt 长度 |
| decode 一个新 token | `[B, 1, 2560]` | **1** |
| chunked prefill | `[B, chunk, 2560]` | chunk 长度 |

历史 K 进 cache 时已经按当时位置转过；这次只转新 Q/新 K，再和旧 KV 拼接。
