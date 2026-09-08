# PLE：N-gram Embedding 与四路残差注入

备查笔记。对应 [ARCHITECTURE.md](../ARCHITECTURE.md) §7，以及关于 embedding 矩阵、hash collision、PLE 注入路径、layer placement 和 vocabulary scaling 的讨论。

实现：`Qwen4ExpTextNGramEmbedding` + `Qwen4ExpTextPLELayer`（`modular_qwen4_exp.py`）。Flash-Next 只在第 2 层使用 PLE；该层是 GDN 层。

---

## 1. 一句话

PLE 分两步：

1. 用当前位置结尾的 **bigram / trigram** hash 查询 16 个小维度 embedding head，拼成一个 2560 维短语向量。
2. 根据当前四路 Expanded Residual 各自的内容，决定把这一个短语向量分别写入每一路多少；再补一条 dilated depthwise convolution 路径。

它发生在第 2 层的 GR Read 之前：

```text
input_ids ── N-gram lookup ── PLE gate / conv ──┐
                                                ↓ add
Expanded Residual H ─────────────────────────── H'
                                                ↓
                                      GR Read → GDN → GR Write
                                                ↓
                                      GR Read → MoE → GR Write
```

PLE 不是 GDN recurrent state 内部的一部分；它先修改四路 residual，之后 GDN 和同层 MoE 都会间接看到这次写入。

---

## 2. 真实 checkpoint 规格

| 项 | 数值 |
|---|---:|
| `ple_layer_ids` | `[2]`，1-indexed |
| `ple_embed_dim` | 2560 |
| `ngram_size` | 3：bigram + trigram |
| `heads_per_ngram` | 8 |
| N-gram head 总数 | 16 |
| 每个 head 的 embedding dim | `2560 / 16 = 160` |
| `ngram_vocab_size_base` | 20,000,000 |
| `ple_conv_kernel_size` | 4 |
| PLE conv dilation | 3（等于 `ngram_size`） |
| `split_ngram_parts` | 128（checkpoint 分片数；实现默认值是 512） |

架构约束：

- `ple_layer_ids` 使用 1-indexed 层号。
- PLE 只能挂在 `linear_attention`（GDN）层。
- 有 PLE 时，每层 cache 预留 3 个 conv-state 槽：GDN、PLE conv、N-gram token context。

---

## 3. Embedding 矩阵怎么组织

### 3.1 逻辑上 16 张表

Unigram 已由普通 vocabulary embedding 表示，所以这里的循环从 2-gram 开始：

```text
bigram： 8 heads × [约 20M rows, 160 dim]
trigram：8 heads × [约 20M rows, 160 dim]
```

前 8 个 head 属于 bigram，后 8 个属于 trigram。每个 token 会从每个 head 查一行：

```text
8 × bigram vectors  → 8 × 160 = 1280 dim
8 × trigram vectors → 8 × 160 = 1280 dim
concat                         = 2560 dim
```

### 3.2 物理上是一张大表

运行时不是 16 个 `nn.Embedding`，而是把 16 张逻辑表沿 row 方向拼成一张：

```text
head 0 rows | head 1 rows | ... | head 15 rows | padding rows
```

每个 head 保存：

```text
head_vocab_sizes[j] = p_j
head_offsets[j]     = sum(p_k for k < j)
```

本 head 的局部索引会加 offset，变成大表中的全局 row：

\[
\operatorname{global\_id}_j
=\operatorname{offset}_j+(H\bmod p_j).
\]

因此不同 head 的 row 区间完全不重叠。即使两个 head 得到相同的 local id，它们也不会读取同一个物理参数。

16 个逻辑表的 row 数之和是 320,001,446，再向 128 对齐为：

```text
ngram_embedding.weight: [320,001,536, 160]
```

总参数量：

```text
320,001,536 × 160 = 51,200,245,760 ≈ 51.2B
```

BF16 权重约 95.4 GiB。checkpoint 的 `split_ngram_parts=128` 只控制权重如何拆分保存；加载时会把 shards 拼回这一张 runtime embedding。

---

## 4. 从 token tuple 得到 hash

### 4.1 当前 token 和历史 token

对位置 \(t\)：

```text
x0 = input_ids[t]       # 当前 token
x1 = input_ids[t - 1]   # 前一个 token
x2 = input_ids[t - 2]   # 前两个 token
```

于是：

```text
bigram key  = (x1, x0)
trigram key = (x2, x1, x0)
```

序列开头或 EOS 之后若历史不足，用 EOS token 补齐。`_shift_right_ignore_eos` 不允许 n-gram 跨 EOS，把两个文档或对话段拼成一个短语。

### 4.2 SplitMix64 只负责生成固定乘数

SplitMix64 是确定性的 64-bit 整数 mixer：相近输入会被打散成看起来无关的 64-bit 输出。它不是可训练模块，也不会在每个 token 上运行。

初始化时先构造 layer seed：

```text
base_seed = seed + 10007 × ple_layer_index
```

再为相对位置 0、1、2 生成三个奇数乘数：

```text
random_bits = SplitMix64(base_seed + constant × (position + 1))
m_position  = 2 × (random_bits % bound) + 1
```

`2k+1` 保证乘数是奇数；bound 保证合法 token id 与乘数的乘积不超过 signed int64 正数范围。奇数在模 \(2^{64}\) 的整数空间中可逆，不会像偶数乘法那样系统性地清掉低位信息。

Flash-Next 这一层确定性得到：

```text
m0 = 23703573157769   # 当前 token
m1 = 20109073645365   # 前一个 token
m2 =  8052911324071   # 前两个 token
```

如果以后有多个 PLE layer，`ple_layer_index` 会改变 seed，使各层使用另一组乘数。

### 4.3 乘位置常数，再 XOR

XOR（异或）逐 bit 操作：相同为 0，不同为 1。

| A | B | `A XOR B` |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

Bigram 和 trigram 的 mixed id：

\[
H_2=(x_0m_0)\oplus(x_1m_1),
\]

\[
H_3=(x_0m_0)\oplus(x_1m_1)\oplus(x_2m_2).
\]

XOR 本身满足交换律，但 token 先绑定了不同的位置乘数。序列 `(a, b)` 与 `(b, a)` 分别得到：

\[
H_2(a,b)=(bm_0)\oplus(am_1),
\]

\[
H_2(b,a)=(am_0)\oplus(bm_1).
\]

由于 \(m_0\ne m_1\)，两边的操作数不是简单交换，结果通常不同。乘数负责注入位置身份，XOR 负责把多个已经打散的 63-bit pattern 合成一个固定宽度整数。

---

## 5. 八个素数取模和碰撞

### 5.1 素数从哪里来

`heads_per_ngram=8` 决定每种 n-gram 查 8 张表。表大小不是 seed 生成或训练得到，而是从 `ngram_vocab_size_base=20,000,000` 开始依次寻找素数。

Bigram 使用前 8 个：

```text
20,000,003  20,000,023  20,000,033  20,000,047
20,000,059  20,000,063  20,000,069  20,000,077
```

Trigram 使用后 8 个：

```text
20,000,081  20,000,093  20,000,107  20,000,147
20,000,153  20,000,159  20,000,161  20,000,171
```

严格按当前 HF 参考实现，每种 n-gram 先得到 **一个** \(H_2\) 或 \(H_3\)，然后用 8 个不同素数取模；并不是先生成 8 个完全独立的 pre-hash：

\[
id_j=H\bmod p_j,\qquad j=0,\ldots,7.
\]

### 5.2 它不保证单表无碰撞

每张表只有约 2000 万行，而可能出现的 n-gram 更多，所以单个 head 内碰撞是允许且不可避免的。代码没有链表、二次探测或动态扩表；碰撞的 n-gram 直接共享该 head 的 160 维 row。

设计目标是避免一次单表碰撞扩散到全部 8 个 head。如果：

\[
H_a\bmod p_0=H_b\bmod p_0,
\]

只说明 \(H_a-H_b\) 是 \(p_0\) 的倍数，不代表它也是 \(p_1\) 的倍数。因为各表大小是不同素数，它们两两互质；两个值要在多个 head 同时碰撞，差值必须是这些素数乘积的倍数。

约 2000 万的前三个素数乘积已经大于 \(2^{63}\)。mixed id 被限制在非负 signed-int64 范围内，所以只要 \(H_a\ne H_b\)，它们不可能在同一 n-gram order 的所有 head 上同时发生取模碰撞。

### 5.3 仍然存在的碰撞边界

若两个不同 token tuple 在乘法 + XOR 阶段就得到相同 mixed id：

\[
H(x_0,x_1,x_2)=H(x'_0,x'_1,x'_2),
\]

后续对多少个素数取模都无法区分。当前实现没有证明 XOR mixer 对所有合法 bigram / trigram 完全无碰撞。

因此准确说法是：

- 单个 head 允许 hash collision。
- 多个不同素数 head 将碰撞限制成局部的 160 维共享，而不是完整 1280 维表示一起共享。
- Bigram 与 trigram 使用不同 head 区间，也不会跨 n-gram order 读取同一物理 row。

---

## 6. 完整查表与 shape

对每个 n-gram order：

```python
mixed_ids = shifted_tokens[0] * m0
mixed_ids ^= shifted_tokens[1] * m1
# trigram 再做：
mixed_ids ^= shifted_tokens[2] * m2

local_ids  = mixed_ids[..., None] % head_vocab_sizes
global_ids = local_ids + head_offsets
```

两种 order 的 id 拼起来：

```text
bigram ids          [B,T,8]
trigram ids         [B,T,8]
concat              [B,T,16]
embedding lookup    [B,T,16,160]
flatten             [B,T,2560]
```

记最终结果为：

\[
E\in\mathbb{R}^{B\times T\times2560}.
\]

Decode 时 cache 最近 `ngram_size-1=2` 个 token id，复用 `conv_states[2]`。这样新 token 到来时无需重新传入完整历史，也能构造其 bigram / trigram。

---

## 7. 如何注入四路 Expanded Residual

### 7.1 输入与位置

进入第 2 个 DecoderLayer 时：

```text
H: [B,T,4×2560] = [B,T,10240]   # Expanded Residual
E: [B,T,2560]                    # N-gram lookup 结果
```

Decoder 的顺序是：

```python
if self.ple is not None:
    H = H + self.ple(H, input_ids, cache)

h = GR_attn.Read(H)
y = GDN(h)
H = GR_attn.Write(H, y)
```

也就是说，PLE 先写 Expanded Residual，再由正常 GR Read 决定 GDN 实际读到什么。

### 7.2 N-gram embedding 产生四份 K、一份共享 V

```python
K = GroupRMSNorm(W_k(E))
V = W_v(E)
```

Shape：

```text
E                    [B,T,2560]
W_k(E)               [B,T,10240]
K reshape            [B,T,4,2560]
V                    [B,T,2560]
```

同一个 n-gram embedding 产生四份不同的 key：

\[
k_{t,1},k_{t,2},k_{t,3},k_{t,4},
\]

但四路共享同一个待写入内容：

\[
v_t=W_v e_t.
\]

### 7.3 当前四路 residual 产生 Q

```python
Q = GroupRMSNorm(H).reshape(B, T, 4, 2560)
```

每一路用自己的 residual 状态作为 query：

\[
q_{t,i}=\operatorname{RMSNorm}(H_{t,i}).
\]

Group RMSNorm 让四路分别计算 RMS，避免某一路幅度影响其他路的 gate。

### 7.4 每个 token、每一路独立算 gate

\[
s_{t,i}=\frac{\langle k_{t,i},q_{t,i}\rangle}{\sqrt{2560}}.
\]

Shape：

```text
K, Q             [B,T,4,2560]
dot product      [B,T,4,1]
```

然后做保留符号的平方根：

\[
\tilde s_{t,i}=\operatorname{sign}(s_{t,i})\sqrt{|s_{t,i}|},
\]

再做 sigmoid：

\[
g_{t,i}=\sigma(\tilde s_{t,i})\in(0,1).
\]

代码对绝对值先 `clamp_min(1e-6)`，避免平方根在 0 附近出现数值问题。Signed sqrt 保留正负方向，同时放大小绝对值、压缩大绝对值，再交给 sigmoid。

这不是跨 token Attention：没有和其他位置的 key 比较，也没有 softmax。它只是每个 token 内部，用四个独立 Q/K compatibility score 控制四路写入。

### 7.5 同一个 V，以不同强度写入四路

\[
u_{t,i}=g_{t,i}v_t.
\]

Broadcast 后：

```text
V                 [B,T,1,2560]
gate              [B,T,4,1]
U = gate × V      [B,T,4,2560]
```

例如同一短语向量可以按不同权重进入四路：

```text
branch 0 ← 0.9 × V
branch 1 ← 0.2 × V
branch 2 ← 0.7 × V
branch 3 ← 0.1 × V
```

Key 决定当前短语和某一路 residual 是否匹配；shared Value 决定真正写入的内容。

### 7.6 Dilated depthwise convolution 路径

Direct gated value 之外，还有：

```python
U_norm = GroupRMSNorm(U.flatten(-2))
C = SiLU(DilatedDepthwiseConv1d(U_norm))
```

配置：

```text
channels    = 4×2560 = 10240
kernel      = 4
dilation    = 3
groups      = 10240
```

所以每个通道独立沿时间维组合，大致读取：

```text
t, t-3, t-6, t-9
```

有效历史长度是 `(kernel-1)×dilation = 9`，decode cache 使用 `conv_states[1]`。卷积权重初始化为 0，因此初始化时这条支路没有贡献，训练后再逐渐学会利用它。

最终 PLE 返回：

\[
\operatorname{PLEOutput}=U+C
\in\mathbb{R}^{B\times T\times10240},
\]

Decoder 执行：

\[
H\leftarrow H+U+C.
\]

若有 padding，direct 和 conv 输入都会先用 `conv_mask` 清零。

---

## 8. 注入路径的直觉

对某个 token \(t\)、某条 residual branch \(i\)，可以压缩成：

```text
e_t = 当前 bigram / trigram 查到的短语记忆
k_ti = 这份短语记忆在 branch i 上的匹配 key
q_ti = branch i 当前需要什么
v_t = 真正可写入的短语内容

gate_ti = sigmoid(signed_sqrt(<q_ti, k_ti>/sqrt(d)))
H_ti += gate_ti * v_t + dilated_local_path
```

所以各组件分工是：

- **Hash lookup**：提供大容量、近乎零 FLOPs 的局部短语记忆。
- **Q/K gate**：决定当前四路 residual 各自接收多少。
- **Shared V**：承载实际写入的内容。
- **Dilated DWConv**：在已经门控的注入信号上再做局部时间组合。
- **后续 GR Read**：把写过 PLE 的四路 residual 混成 GDN 的单路输入。

---

## 9. Offload 与代码边界

`ngram_embedding.weight` 被列入 `_no_placement_params`。这张约 95.4 GiB 的 BF16 表通常不能放入单卡显存；实际推理系统可让它常驻 host memory，提前计算确定性的 lookup ids，并通过异步 prefetch 与 backbone 计算重叠。

Hugging Face 参考路径需要显式把 ids 移到 embedding weight 所在 device，查表后再把结果移回原 device。若普通 `device_map` hook 在每次 forward 前尝试把整张 CPU-offloaded 参数搬回 accelerator，会直接 OOM，所以该参数需要跳过常规 placement hook。

代码地图：

| 内容 | 位置 |
|---|---|
| 素数表大小、offset、物理 embedding | `Qwen4ExpTextNGramEmbedding.__init__` |
| SplitMix64、位置乘数 | `_splitmix64`、`_build_layer_multipliers` |
| EOS-safe shift、hash、lookup | `Qwen4ExpTextNGramEmbedding.forward` |
| Q/K/V gate、signed sqrt、conv | `Qwen4ExpTextPLELayer.forward` |
| 注入顺序 | `Qwen4ExpTextDecoderLayer.forward` |
| Cache 槽位 | `Qwen4ExpTextConfig.number_of_conv_states` 与 `DynamicCache` |

---

## 10. Technical report 的 PLE 分析

report 使用 **N-gram Embedding** 这个名称，公开实验集中在两个问题：注入哪一层，以及 N-gram table 应占多少容量。以下实验统一使用每个 active parameter 300 tokens（300 TPP）；前文的 hash、gate 和 convolution 细节主要来自参考实现。

### 10.1 注入位置（Table 7）

固定 N-gram Embedding 总参数量，比较单层和多层注入：

| 位置（1-indexed） | Loss | Avg.（9 benchmarks） |
|---|---:|---:|
| 无 N-gram | 1.585 | 45.44 |
| Layer 1 | 1.541 | 47.30 |
| Layer 2 | 1.541 | **47.94** |
| Layer 3 | 1.543 | 46.76 |
| Layer 4 | 1.544 | 46.89 |
| Layer 10 | 1.544 | 46.62 |
| Layer 15 | 1.543 | 47.37 |
| Layer 25 | 1.541 | 47.40 |
| Layer 2 + 15 | 1.541 | 47.01 |
| Layer 2 + 25 | **1.540** | 47.75 |

结论：

- 没有单一深度在所有任务上一致占优；前两层较强，中层和深层仍有竞争力。
- 固定预算分散到两层没有稳定收益；`2 + 25` 只多降 `0.001` loss，Avg. 反而低于只放 Layer 2。
- 不同位置的相对排序在 Full Attention 与 GDN 下相近，说明 placement 对 token mixer 不太敏感。
- 最终选 **Layer 2**：单层 Avg. 最好，并可让 host-memory prefetch 与 Layer 1 计算重叠。Layer 1 前没有足够计算隐藏预取延迟。

最终实现中，PLE 在 Layer 2 的 GR Read 之前写入四路 Expanded Residual；report 的 placement 表本身没有单独消融这个具体注入算子。

### 10.2 固定总模型参数预算（Table 8）

扩大 N-gram vocabulary 的同时减少 MoE experts，使总模型参数不变。Scale 相对 250K tokenizer vocabulary；括号是 N-gram 参数占比。

| N-gram scale | 参数占比 | Loss | Uncheatable PPL |
|---|---:|---:|---:|
| None | 0% | 1.202 | 5.55 |
| 5× | 10% | 1.200 | **5.54** |
| 10× | 25% | **1.197** | 5.55 |
| 30× | 50% | 1.201 | 5.59 |

10× 在 loss 上最好，但 out-of-domain PPL 基本不变，下游任务也没有相对 MoE-only baseline 的一致提升。这个固定预算实验说明，在所测 recipe 和容量点上，N-gram memory 没有表现为 MoE conditional compute 的直接等价替代；两者在扩展容量时承担不同角色。

### 10.3 固定 MoE、额外扩大 N-gram table（Table 9）

此时总参数随 N-gram vocabulary 增长：

| Scale | Loss | MMLU | MATH | GSM8K | C-Eval | CMMLU | MMMLU |
|---|---:|---:|---:|---:|---:|---:|---:|
| None | 1.585 | 62.78 | 32.52 | 59.21 | 66.91 | 68.10 | 54.06 |
| 20× | 1.553 | 64.14 | **37.38** | **65.09** | 71.75 | 72.29 | 55.94 |
| 50× | 1.541 | 64.71 | 37.32 | 64.00 | 72.12 | 72.48 | 56.64 |
| 100× | 1.534 | 64.70 | 36.98 | 63.08 | 73.75 | 72.73 | **56.65** |
| 200× | **1.526** | **64.85** | 35.34 | 62.96 | **74.94** | **73.24** | 55.82 |

- Loss 随 table 增大单调下降：`1.585 → 1.526`。
- 下游能力不单调：MATH/GSM8K 在 20× 最好，其他任务在更大 scale 饱和或波动。
- C-Eval 和 CMMLU 随 scale 持续提高，是 report 特别指出的例外。

因此扩大 N-gram table 是低 FLOPs 的额外容量轴，但不能只根据 pretraining loss 决定规模。

### 10.4 其他尝试与证据边界

report 还尝试 token normalization、不同 n-gram orders 的非均匀容量分配、按频率划分 slots，但没有一致收益，也没有给具体表格。

公开 report 没有给出：

- bigram-only / trigram-only、head 数和 hash collision 的消融；
- signed-sqrt gate、四路 residual injection 和 dilated convolution 的独立贡献；
- 直接注入 vocabulary embedding 与 Layer-2 residual injection 的对照；
- host prefetch 的 latency / bandwidth 和端到端吞吐；
- 最终 51B N-gram 参数在 125B backbone 上的单变量收益。

