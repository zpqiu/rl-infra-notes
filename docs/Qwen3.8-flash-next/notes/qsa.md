# Qwen Sparse Attention（QSA）

备查笔记。对应 [ARCHITECTURE.md](../ARCHITECTURE.md) §5，以及关于 Indexer、RoPE、block 打分、Gated Attention、两阶段训练、质量/效率消融和 DeepSeek CSA/HCA 对照的讨论。

实现：`Qwen4ExpTextQSAIndexer` + `Qwen4ExpTextAttention`（后者继承 `Qwen3_5Attention`）。每 4 层里 1 层 QSA；进层前同样 GR Read `[B,T,2560]`，出来 `o_proj` 再 GR Write。

---

## 1. 它是什么

GDN 把历史压进固定白板（有损）。QSA 在全上下文上做一次 **稀疏但精确** 的查找。

真正的 Sparse Core Attention 不再让每个 query 看全部 \(T\) 个 token，而是大约只看 **2048 个 token**（512 block × 4，再加 tail）。选人不是逐 token top-k，而是：

1. 连续 4 个 token 收成 1 个 micro-block（K 平均）
2. block 级打分，最多留 512 块
3. 展开回 token，跑真正的注意力

但 **2048 只是 Core Attention 的 token budget，不是整个 QSA 的复杂度**。Indexer 仍需让每个 query 与压缩后的 key blocks 打分：完整 prefill 约为 \(O(T^2/r)\)，Core Attention 约为 \(O(TK)\)；这里 \(r=4\)、\(K=2048\)。单 token decode 时两部分分别约为 \(O(T/r)\) 和 \(O(K)\)。

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

这一层仍存 KV cache（2 头 × 256，随上下文涨）；Sparse Core Attention 只在约 2048 个位置上做。Indexer 另外 cache 所有 token 的 raw K（128 维），后续 query 仍要给约 \(T/4\) 个完整 blocks 打分。

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

ReLU：不像不扣分。对 head 求和：多视角投票。不是 softmax，不在 block 上归一化。然后 `topk(min(512, 块数))`，选中块整块展开（每块 4 个全要）。令可见长度 \(L=t+1\)，实际允许 token 数为

\[
4\min\!\left(512,\left\lfloor\frac{L}{4}\right\rfloor\right)+(L\bmod4).
\]

只有已经存在至少 512 个完整 blocks 时，才达到 `2048 + tail`；tail 最多 3 个 token。

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
| 激活 | checkpoint 为 sigmoid（配置可选 silu） | sigmoid |
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

---

## 7. Indexer 为什么不能只靠 LM loss 训练

Indexer 的最终输出是一个离散集合：

\[
B_i=\operatorname{TopK}(I_{i,:}).
\]

这些 indices 被展开成 bool / additive attention mask，再交给 Sparse Core Attention。对一次普通反向传播而言，mask 被当作已经确定的常量：

```text
LM loss → Sparse Core Attention 参数、backbone hidden
        ↛ TopK 选择边界
        ↛ Indexer score I
        ↛ index_qk_proj
```

LM loss 能训练“在当前选中 token 上怎样做 attention”，却不能告诉 Indexer“刚才应该换一个 block”。所以 Indexer 需要一个绕开 TopK 的连续目标：让它直接拟合主干 Attention 认为重要的位置。

Transformers 参考实现也明确体现了这个边界：Indexer forward 只执行 `scores.topk(...)` 并返回 mask，不计算训练 loss；对应测试关闭了 `test_all_params_have_gradient`，注释说明 Indexer 参数由独立 objective 训练。它是结构和推理参考，不包含 report 中的 distillation pipeline。

---

## 8. Teacher distribution 怎么构造

对 query token \(i\)，Full Attention teacher 的第 \(h\) 个 head 先产生正常的 token-level attention probability：

\[
A^{(h)}_{i,j}
=\operatorname{Softmax}_j(z^{(h)}_{i,:})_j.
\]

把所有 teacher heads 的 probability 相加，再做 L1 normalization：

\[
a_{i,j}
=\frac{\sum_h A^{(h)}_{i,j}}
       {\sum_{j'}\sum_h A^{(h)}_{i,j'}}.
\]

因为每个 head 自己已经和为 1，这等价于对 heads 求平均。它把多头的不同检索偏好压成一份 head-agnostic token relevance distribution。

Indexer 的候选单位是 4-token block，不是 token，所以 teacher 还要对齐到 block level。对从 \(p_b=br\) 开始的完整 block：

\[
\bar a_{i,b}
=\max_{j\in[p_b,p_b+r-1]} a_{i,j},
\qquad r=4.
\]

然后在所有完整、因果可见的 blocks 上重新做 L1 normalization：

\[
\hat a_{i,b}
=\frac{\bar a_{i,b}}{\sum_c\bar a_{i,c}}.
\]

这里选 max pooling 而不是 average pooling，是因为一个 block 中只要有一个非常关键的 token，整个 block 就值得保留；average 可能把这个尖峰稀释掉。Max pooling 后总和不再是 1，所以必须再次归一化。

末尾不足 4 个 token 的 tail 不参与这个 KL target：它没有 block score，而且在 Sparse Core Attention 中本来就会无条件保留。

---

## 9. Stage 1：Dense Distillation

Stage 1 仍然使用 Full Attention teacher，只训练 Indexer，backbone 冻结。

Indexer 对所有完整、因果可见 blocks 产生 raw importance scores：

\[
I_{i,:}=[I_{i,0},I_{i,1},\ldots,I_{i,B-1}].
\]

训练时在 **全部这些 blocks** 上做 softmax，而不是先 TopK：

\[
p^{\text{idx}}_{i,:}=\operatorname{Softmax}(I_{i,:}).
\]

目标是 teacher-to-student KL：

\[
L_{\mathrm{KL}}
=\frac1N\sum_i
D_{\mathrm{KL}}
\left(
\hat a_{i,:}\;\|\;p^{\text{idx}}_{i,:}
\right).
\]

去掉与 student 无关的常数后，它就是用 soft target 做 cross-entropy。对 Indexer logits 的梯度具有熟悉的形式：

\[
\frac{\partial L}{\partial I_{i,b}}
=p^{\text{idx}}_{i,b}-\hat a_{i,b}.
\]

所以 Indexer 给某块的概率过高就压低 score，低于 teacher relevance 就抬高 score。梯度直接经过连续的 \(I\) 回到：

```text
KL
 ↓
all-block softmax(I)
 ↓
ReLU(sum_h q_h · k_block)
 ↓
index_qk_proj、q_layernorm、k_layernorm
```

这条路径没有 TopK，因此可以正常求导。

Flash-Next 的 Stage 1 配置：

| 项 | 数值 |
|---|---:|
| 序列长度 | 256K |
| 训练步数 | 1,000 |
| 学习率 | \(1\times10^{-3}\) |
| 每步数据 | 8 条 256K 序列 |
| 总 token | 约 2B |
| 更新参数 | 仅 Indexer |

### 一个三块的例子

假设 teacher 聚合后的 12 个 token probability 是：

```text
block 0: [0.02, 0.03, 0.01, 0.04]
block 1: [0.01, 0.30, 0.02, 0.02]
block 2: [0.10, 0.20, 0.15, 0.10]
```

MaxPool 得到：

```text
[0.04, 0.30, 0.20]
```

再归一化：

```text
teacher block target ≈ [0.074, 0.556, 0.370]
```

如果 Indexer logits 是 `[1.0, 0.5, -0.2]`，softmax 约为：

```text
student distribution ≈ [0.524, 0.318, 0.158]
```

于是梯度 \(p-\hat a\) 约为：

```text
[+0.450, -0.238, -0.212]
```

梯度下降会明显压低 block 0 的 score，同时提高 block 1、2；这正是 LM loss 经过离散 TopK 无法直接提供的排序监督。

---

## 10. Stage 2：Sparse Joint Training

仅把蒸馏好的 Indexer 接上 sparse mask 还不够：Indexer 可以近似 Full Attention 的重要性，但 backbone 以前仍依赖完整上下文。直接切成 2048-token sparse attention 会出现性能下降，因此还需要让二者共同适应。

Stage 2 真正启用：

\[
B_i=\operatorname{TopK}_{K_B}(I_{i,:}),
\qquad
K_B=K/r=2048/4=512.
\]

选中的 blocks 展开回 token，并加上当前不完整 tail：

\[
S_i=\operatorname{Expand}(B_i)\cup\operatorname{Tail}(i).
\]

这里同时存在两条训练路径。

### 10.1 LM 路径：训练 sparse backbone

```text
next-token LM loss
  ↓
Sparse Core Attention（只看 S_i）
  ↓
core Q/K/V/O、GR、其他 backbone 参数
```

这使 backbone 学会在 Indexer 给出的有限上下文上工作。由于 \(S_i\) 来自离散 TopK，autograd 不会通过 mask 把这条 LM 梯度传回 Indexer score。

### 10.2 KL 路径：继续训练 Indexer

此时为了避免在全部 blocks 上保留 dense distillation 成本，KL 只在已经选中的 \(B_i\) 内计算。先把这些块对应的 teacher probabilities 重新归一化：

\[
\hat a^{\text{sel}}_{i,b}
=\frac{\hat a_{i,b}}
       {\sum_{c\in B_i}\hat a_{i,c}},
\qquad b\in B_i.
\]

Indexer 也只在选中 scores 上做 softmax：

\[
p^{\text{sel}}_i
=\operatorname{Softmax}(I_{i,B_i}).
\]

于是：

\[
L_{\mathrm{KL}}
=\frac1N\sum_i
D_{\mathrm{KL}}
\left(
\hat a^{\text{sel}}_i
\;\|\;
p^{\text{sel}}_i
\right).
\]

注意这里仍然没有对“TopK 选谁”本身求导。梯度只进入已经被选中的 score values；未选 blocks 在这一拍没有直接 KL 梯度。这正是 Stage 1 不能省略的另一个原因：如果进入 Stage 2 时候选集合就很差，selected-only KL 很难把一个从未入选的重要 block 救回来。

沿用上面的例子，若当前 Top-2 错选了 block 0 和 1：

```text
selected teacher  [0.04, 0.30] → renorm ≈ [0.118, 0.882]
selected student  softmax([1.0, 0.5])      ≈ [0.622, 0.378]
gradient                                  ≈ [+0.505, -0.505]
```

block 0 会被压低、block 1 会被抬高，但未选中的 block 2 本步没有梯度。Stage 1 的 all-block distillation 负责先建立较好的召回，Stage 2 主要做 sparse 条件下的校准与 backbone 共适应。

Flash-Next 的 Stage 2 配置：

| 项 | 数值 |
|---|---:|
| 所处阶段 | CPT 最后阶段 |
| 序列长度 | 256K |
| 训练步数 | 8,000 |
| 学习率 | \(2.5\times10^{-5}\) |
| 每步数据 | 96 条 256K 序列 |
| 总 token | 约 200B |
| 更新参数 | backbone + Indexer |

可以概念性地把这一阶段看成：

\[
L_{\text{stage 2}}
=L_{\text{LM}}+\lambda L_{\text{KL}}.
\]

但 report 没有给出 \(\lambda\)，也没有公开 teacher branch 是否 stop-gradient、KL 是否继续影响 Indexer 输入之前的 backbone hidden 等 autograd 细节，不能从参考 forward 自行补全这些实现选择。

---

## 11. 两阶段各自解决什么

```text
Full Attention backbone
        │ teacher distribution
        ▼
Stage 1: all-block KL，只训练 Indexer
        │ 得到有召回能力的 selector
        ▼
Stage 2: TopK sparse attention
        ├─ LM loss → backbone 适应稀疏上下文
        └─ selected-block KL → Indexer 继续校准
```

- **Stage 1 解决 selector 的冷启动**：随机 Indexer 选出的 mask 无法靠 LM loss修正。
- **Stage 2 解决 backbone 的分布变化**：会模仿 Full Attention 的 selector，不等于原 backbone 立刻能承受信息裁剪。
- **KL 解决不可导选择**：用连续 score distribution 监督 Indexer，而不是尝试穿过 TopK。
- **Joint training 解决共适应**：最终 sparse attention 的 LM-loss 曲线和 Full Attention 基线差距约为 \(10^{-4}\)。

report 提到训练使用 fused QSA kernel 联合计算 sparse attention output 和 KL，避免物化庞大的中间张量；具体 kernel、loss coefficient 和完整训练 API 没有出现在当前 Transformers 参考实现中。

---

## 12. Report 的质量、消融与效率分析

### 12.1 通用能力（Table 2）

最终 QSA run 与 Full-Attention baseline 的 8 项评测：

| Method | MMLU-Pro | SuperGPQA | MATH | GSM8K | BBH | MMMLU | EvalPlus | MultiPL-E | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Attention | 72.9 | 51.7 | 69.8 | 91.0 | 90.4 | **81.8** | 70.8 | 78.4 | 75.9 |
| **QSA** | **73.7** | **52.1** | **71.6** | **92.2** | **91.6** | 81.1 | **72.3** | **79.8** | **76.8** |

QSA 在 7/8 项上更高，平均 `75.9 → 76.8`；MMMLU 低 `0.7`。report 没给多 seed 方差，因此这些结果支持“没有系统性能力退化”，但不足以证明 sparse attention 本身必然优于 dense attention。

### 12.2 长上下文检索（Table 3）

| Method | RULER ≤128K | 128–256K | 256–512K | 512K–1M | MRCR 128K | 256K | 512K | 1M | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Full Attention | 99.84 | **99.81** | 97.65 | 90.08 | **97.14** | **94.20** | 30.66 | 20.71 | 78.76 |
| **QSA** | **99.89** | 99.62 | **98.95** | **93.00** | 95.98 | 93.00 | **40.53** | **26.44** | **80.93** |

QSA 在较短 MRCR 上略低，但在最长区间优势明显：RULER `90.08 → 93.00`，MRCR 512K `30.66 → 40.53`，1M `20.71 → 26.44`。这是分别训练后的 Full-Attention baseline 与 QSA run 的比较，不是同一权重下只切换 mask 的实验。

### 12.3 MTP QSA + index reuse（Table 4）

四步 speculative decoding 中，`w/ QSA` 配置同时把 MTP attention 换成 QSA，并在 prediction steps 间复用 top-k indices：

| Method | MT-Bench | GSM8K | MATH | HumanEval | MBPP | Avg. accepted length |
|---|---:|---:|---:|---:|---:|---:|
| Full Attention | 3.44 | 4.19 | 4.29 | 4.24 | 4.12 | 4.06 |
| QSA + reuse | 3.47 | 4.20 | 4.30 | 4.26 | 4.13 | 4.07 |

这套 **QSA + reuse 整体配置**的 accepted length 与 Full Attention 基本不变。report 没有提供“QSA 但不 reuse”的控制组，因此不能从 Table 4 单独归因 index reuse；也没单列 MTP draft latency 或不同复用步数的消融。

### 12.4 Compression ratio 与 IndexShare（Figure 5a）

35B-A3B、Stage 2 后的 RULER 消融比较 Block 2/4/8/16：

- Block 8/16 更快，但低于 Full Attention；
- **Block 4** 在 relative indexer latency `0.25` 时达到 Full-Attention 水平；
- Block 2 把 latency 提高到约 `0.5`，质量没有明显继续提升。

最终选择 `r=4`。同图的跨层 IndexShare 在 relative latency `0.5` 时仍低于 baseline；QSA 的层内压缩不依赖被三层 GDN 隔开的 attention layers 具有相似 top-k，因此更适合 hybrid architecture。图中各点没有数值表，不应把目测值当精确结果。

### 12.5 Indexer query heads 与 Stage 2（Figure 5b）

实验比较 1/2/4/16 个 Indexer Q heads：

- Stage 1 后直接启用 sparse attention，RULER 明显低于 Full Attention；
- Stage 2 joint training 后恢复到 Full-Attention 附近；
- 4 heads 已达到/略高于 baseline，16 heads 只带来很小增益。

因此最终 Indexer 使用 4 Q heads，而 Core GQA 使用 24 Q heads。Figure 5 是曲线图，没有公开精确分数表。

### 12.6 Kernel-level latency（Figure 6）

在 1M context：

| 对比 | Prefill | Decode |
|---|---:|---:|
| Indexer `r=4` vs `r=1` | `3.8×` | `4.4×` |
| 完整 QSA attention module vs dense GQA | `7.6×` | `4.9×` |

条件：prefill 测最后一个 16K chunk、BS=1；decode 为 BS=4、`next_n=4`，含 3 个额外 MTP steps。完整 QSA 数字包含 Indexer 和 Sparse Core Attention；dense baseline 是 FlashInfer paged GQA。

这些是 attention-module kernel latency，不是端到端模型吞吐。若从头处理全部 \(T\) 个 queries，Indexer prefill 是 `O(T²/r)`；Figure 6 固定只测最后一个 16K chunk，因此随图中 context length \(T\) 的工作量更接近 `O(16K · T/r)`。Sparse Core 对完整 prefill 才是约 `O(TK)`。

---

## 13. 与 DeepSeek V4 CSA / HCA 的核心区别

详细的 DeepSeek 路径见 [deepseek-v4/architecture_notes.md](../../deepseek-v4/architecture_notes.md)。三者最关键的区别是“压缩表示是否进入最终 Attention”。

| | QSA | DeepSeek CSA | DeepSeek HCA |
|---|---|---|---|
| 压缩率 | 4 | 4，带 overlap | 128，无 overlap |
| Indexer | 有 | 有 | 无 |
| Indexer 压缩 K | 只用于选 block | 只用于选 compressed position | 不适用 |
| 进入 Core 的主 KV | 原始 token K/V | 另一套主 Compressor 的 selected KV | 主 Compressor 的全部因果合法 KV |
| Top-k 后是否展开 | **展开回 4 个原始 tokens** | 不展开 | 不适用 |
| 原始局部路径 | 仅 incomplete tail 必留 | 最近 128 个未压缩 token-level KV | 最近 128 个未压缩 token-level KV |
| 长上下文末端约看 | 上限 `2048 + tail` 个原始 KV | 上限 `512 compressed + 128 token-level` | `T/128 compressed + 128 token-level` |
| 长程主 KV cache | HF 参考实现保存原始 GQA KV | 约 `T/4` compressed KV | 约 `T/128` compressed KV |

QSA 是 **block-level retrieval、token-level attention**：Indexer 的 AvgPool block K 只决定选谁，最终 GQA 使用另一套原始 token K/V。这个语义要求选中位置的 token-level K/V 可访问；HF 参考实现保存全部历史 GQA KV，但 report 没公开 production kernel 的物理 cache 分配与生命周期。

CSA 也有独立的 Indexer Compressor，但它只产生用于评分的 128 维压缩 K。Indexer 选出 block positions 后，Core Attention 用这些位置访问另一套主 Compressor 产生的 512 维 compressed KV；不是把 Indexer K 直接交给 Core。

CSA/HCA 是 **compressed-representation attention**：多个 tokens 经 learned gated pooling 形成一个真实 KV entry，最终 softmax 直接作用于这些 compressed entries，并与最近 128 个未压缩 token-level KV entries 联合归一化。CSA 每 4 tokens 产生一个 entry；除首项外，一个 entry 由前一个 4-token chunk 的 \(C^a\) 分支和当前 chunk 的 \(C^b\) 分支聚合，stride 为 4、覆盖最多 8 个原 token。HCA 每 128 tokens 产生一个非重叠 entry，并对全部因果合法、已经完成的 compressed entries 做 dense attention。

对位置 \(i\) 的 prefill query，DeepSeek 实际可见条目数为：

\[
N_{\mathrm{CSA}}(i)=
\min\!\left(512,\left\lfloor\frac{i+1}{4}\right\rfloor\right)
+\min(128,i+1),
\]

\[
N_{\mathrm{HCA}}(i)=
\left\lfloor\frac{i+1}{128}\right\rfloor
+\min(128,i+1).
\]

表中的 `640` 和 `T/128+128` 是长上下文末端或 decode query 的近似值。

因此 QSA 保留选中 block 内的 token 细节；DeepSeek 在参考实现中直接压缩长期主 KV，代价是 block 内信息在 Core Attention 前已经聚合。表中的 cache 行只比较主长期 KV：DeepSeek 还保存 SWA ring buffer、Compressor state，CSA 另有 Indexer cache，不能据此当作完整总 cache 内存对比。

---

## 14. 公开证据边界

report 尚未公开：

- Top-K token budget（如 512/1024/2048/4096）的 sweep；
- block K AvgPool、teacher MaxPool、ReLU-sum score 和 tail policy 的独立消融；
- Indexer 对 teacher mass 的 recall / coverage；
- Stage 1 步数和 Stage 2 KL coefficient \(\lambda\) 的 sweep；
- teacher branch、stop-gradient 和 fused KL 的具体实现；
- fused training kernel 的 latency / peak-memory 数字；
- 多 seed、置信区间和同一权重下 dense/sparse 的严格对照；
- 完整模型的端到端 prefill/decode 吞吐。
