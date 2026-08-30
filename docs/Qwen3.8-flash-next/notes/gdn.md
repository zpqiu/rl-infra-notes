# Gated DeltaNet（GDN）

备查笔记。对应 [ARCHITECTURE.md](../ARCHITECTURE.md) §4，以及之后关于投影、短卷积、遗忘、\(\beta\)、外积、读取、输出门、L2 / \(1/\sqrt{d}\) 的问答。

实现：`Qwen4ExpTextGatedDeltaNet` ← `Qwen3_5GatedDeltaNet`（`modeling_qwen3_5.py`）。Flash-Next 规格：QK 16 head × 128，V 48 head × 128，短卷积 kernel=4。

一条 GDN 层的前向（GR Read 给出 `[B,T,2560]` 之后）：

```
投影 QKV / z / b / a
 → QKV 短卷积（时间维、depthwise、kernel=4）
 → 每头：g 遗忘整张 S → β 沿 k 做外积改写 → q⊤S 读取
 → RMSNorm(y) * silu(z)
 → out_proj
 → GR Write →（同一层）MoE
```

还没细抠、不影响「GDN 在干什么」的：prefill 的 chunked kernel、decode 的 conv/recurrent cache、16 vs 48 head 的 `repeat_interleave`。

---

## 1. 它是什么

Softmax 注意力存全量 KV，代价随长度 \(T\) 涨。GDN 不存 KV，维护一块固定大小的矩阵 \(S\)（每头约 `128×128`），历史压进去。新 token 只更新这块白板再读一次。

一层 GDN 的 state 大约：`48 heads × 128 × 128 ≈ 78 万个数`。Flash-Next 里 3/4 的层是 GDN（廉价记过去），1/4 是 QSA（精确检索）。

把 \(S\) 想成「key 空间 → value 空间」的表。每个 token 三步：

1. **遗忘**：\(S \leftarrow g\cdot S\)
2. **按误差改写**：\(\hat v=k^\top S\)，\(S \leftarrow S + k\otimes\beta(v-\hat v)\)
3. **读取**：\(y=q^\top S\)

线性注意力只会 \(S += kv^\top\)（流水账，改不掉）。DeltaNet 能按误差覆盖；再加 Gate 才能主动忘记。

---

## 2. 投影就是线性层；z / b / a 不是全层一个标量

「投影」= 无 bias 的 `nn.Linear`。四个都从 `[B,T,2560]` 出去：

| 名字 | 线性层 | 每个 token 的输出 | 含义 |
|---|---|---|---|
| QKV | `2560 → 16×128×2 + 48×128` | 一长条再 split | 进短卷积，再进白板 |
| **z** | `2560 → 48×128` | `[B,T,48,128]` | 每个 V 通道一个门，给输出 `RMSNorm(y)*silu(z)` |
| **b** | `2560 → 48` | `[B,T,48]` | 每个 V head 一个数，`β=sigmoid(b)` |
| **a** | `2560 → 48` | `[B,T,48]` | 每个 V head 一个数，参与遗忘 \(g\) |

`A_log`、`dt_bias` 才是每 head 一份、序列上共享（长度 48）。

---

## 3. QKV 短卷积（kernel=4）

在 **时间维 T** 上做 depthwise causal Conv1d，通道之间不混。

```
hidden          [B, T, 2560]
in_proj_qkv     [B, T, 10240]
transpose       [B, 10240, T]     ← 在 T 上卷积
causal conv + SiLU
transpose 回来  [B, T, 10240]
split → Q/K [B,T,2048], V [B,T,6144]
reshape → Q/K [B,T,16,128], V [B,T,48,128]
```

`groups=conv_dim`：10240 路各用自己的 4 个权。某通道 \(c\)、时刻 \(t\) 只看 \(x_{c,t-3}\ldots x_{c,t}\)。输出 shape 与输入相同。解码时 cache 最近 3 个数（`kernel-1`）再和新 token 拼。

局部 4-gram 交给卷积，白板留给更长的压缩历史。

---

## 4. 遗忘 \(g\)

发生在 delta 改写之前：整张 \(S\) 乘一个 \((0,1)\) 系数。每个 token、每个 V head 一个数，广播到整张 `128×128`。

顺序固定：**先衰减，再按误差写**。

### \(g_t\) 公式为什么绕

不是「两个强度加在一起再压到 0~1」，而是连续时间衰减的离散化：

\[
g_t=\exp(-A\cdot\Delta t_t)\in(0,1)
\]

- \(A=\exp(A_{\log})>0\)：这个 head 固有衰减率（与 token 无关）
- \(\Delta t=\mathrm{softplus}(a+\mathrm{dt\_bias})>0\)：这一拍走多大一步（随 token）
- \(A\) 或 \(\Delta t\) **越大，忘得越狠，\(g_t\) 越小**（和「遗忘强度」反着走）

代码里外层算的是对数域 `g_log = -exp(A_log)*softplus(a+dt_bias)`（恒负），kernel 里再 `.exp()`。

**乘法 vs 加法**：遗忘率 = 头的时间尺度 × 步长。慢头（\(A\) 小）不会被一个激进 token 直接清掉。初始化 \(A\sim\mathrm{Uniform}(0.01,16)\) 让各头一上来就有跨数量级的记忆长度。

`exp(A_log)`、`softplus`、最后 `exp(-AΔt)` 主要是 **保证 \(g\in(0,1)\)、递推不爆**，不是多一个遗忘因子。

---

## 5. \(\beta=\sigma(b)\) 是写入力度

不是遗忘。遗忘 \(g\) 先把整张 \(S\) 缩小；\(\beta\) 决定误差要改进去多少：

\[
\hat v=k^\top S,\qquad S\leftarrow S+k\otimes\beta(v-\hat v)
\]

\(\beta\in(0,1)\)，每 token、每 V head 一个。

| | \(g\) | \(\beta\) |
|---|---|---|
| 干什么 | 整张 \(S\) 乘一个数 | 只沿当前 \(k\) 做 rank-1 修正 |
| 0 | 这头记忆清掉 | 不写，只忘、只读 |
| 1 | 几乎不忘 | \(k\) 已单位化时，这个 key 的旧联想整段换成 \(v\) |

---

## 6. \(k\otimes\delta\) 外积

\(\otimes\) = 外积：\(k\in\mathbb{R}^{128}\) 和 \(\delta=\beta(v-\hat v)\in\mathbb{R}^{128}\) 得到 `128×128` 补丁。

```python
S = S + k.unsqueeze(-1) * delta.unsqueeze(-2)
```

\(k\) 已单位化时 \(k^\top(k\otimes\delta)=\delta\)，所以再查同一个 \(k\) 会往 \(v\) 走一步。含义：\(k\) 是地址，\(\delta\) 是修正，外积是只贴在这个地址上的 rank-1 补丁。

---

## 7. \(q\) 读取；`.sum(dim=-2)`

写前用 \(k^\top S\) 看这个地址存了什么；读用 \(q^\top S\) 按 query 方向取。同一操作，换向量。

\(S\) 最后两维 `[k_dim, v_dim]`。`k.unsqueeze(-1)` 后与 \(S\) 相乘，`sum(dim=-2)` 沿 **key 维** 加点积，剩下 value 维：

\[
(k^\top S)_j=\sum_i k_i S_{ij}
\]

等价于 `einsum('...kv,...k->...v', S, k)`。

---

## 8. `RMSNorm(y)*silu(z)`

\(y=q^\top S\) 与 \(z\) 都是 `[B,T,48,128]`。每个 head 的 128 维先 RMSNorm，再逐元素乘 `silu(z)`，然后 `out_proj`：`48×128 → 2560`。

- **RMSNorm(\(y\))**：沿 128 维拉回 O(1)。代码里 `variance = mean(x²)`（没减均值），`rsqrt(variance+eps)` 即除以 RMS。`self.weight` 是可学的 \(\gamma\)（这个 gated 版本初始化为 1）。RMS 用的是 \(\sqrt{\mathrm{mean}(x^2)}\)，不是 \(\sqrt{\sum x^2}\)（那是 L2 范数，差 \(\sqrt{d}\)）。
- **\(\times\mathrm{silu}(z)\)**：\(z\) 不进卷积、不进白板，从当前 hidden 直接投。输出门：读到了不等于这拍要用。先 Norm 再乘门。

\(g,\beta\) 管白板上怎么忘/写；\(z\) 管读完要不要用。

---

## 9. `out_proj` 之后

GDN 模块到 `out_proj` 结束，返回 \(y\in\mathbb{R}^{2560}\)。Decoder 里立刻 GR Write：

`H ← H + concat(g_write[i] · y)`，形状回到 `[B,T,4×2560]`。

然后同一层：GR Read → MoE → GR Write，再进下一层（或最后 mixer）。

---

## 10. Q/K 的 L2 norm 和 \(1/\sqrt{d}\)

在 delta kernel 里、进循环之前：

1. 每个 Q/K 沿 128 维 **L2 单位化**（除以 \(\sqrt{\sum x_j^2}\)，长度为 1，没有 \(\gamma\)）。
2. **只给 Q** 再乘 \(1/\sqrt{128}\)。K 保持单位长。

L2 保证 \(k^\top k=1\)，\(\beta=1\) 时覆盖几何干净。\(1/\sqrt{d}\) 只缩放读取 \(y=\hat q^\top S/\sqrt{d}\)，避免随维数漂；不要给 \(k\) 也除，否则更新会弱掉 \(d\) 倍。
