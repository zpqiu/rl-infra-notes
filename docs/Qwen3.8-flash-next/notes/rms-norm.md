# RMSNorm 与 Group RMSNorm

备查笔记。对应 [ARCHITECTURE.md](../ARCHITECTURE.md) §3.3（GR Read 里的 Group RMSNorm），以及后续两问：

1. Group RMSNorm 具体怎么做？
2. 普通 RMSNorm 是什么原理？

实现：`Qwen4ExpTextRMSNorm`（`modular_qwen4_exp.py`），继承 `Qwen3_5RMSNorm`。

---

## Q1. 普通 RMSNorm 是什么原理？

普通 RMSNorm 做一件事：用向量自己的均方根把尺度拉回到大约 1，**不减均值**。

### 公式

对最后一个维度 \(x \in \mathbb{R}^{d}\)（一个 token 的 hidden）：

\[
\mathrm{RMS}(x)=\sqrt{\frac{1}{d}\sum_{j=1}^{d} x_j^{2}+\varepsilon},\qquad
\mathrm{RMSNorm}(x)=\frac{x}{\mathrm{RMS}(x)}\odot \gamma
\]

\(\gamma \in \mathbb{R}^{d}\) 是可学习的逐通道缩放。没有偏置 \(\beta\)，也没有减均值。

Qwen3.5 / Flash-Next 把 \(\gamma\) 写成 \(1+\gamma'\)，\(\gamma'\) 初始化为 0，起步就是恒等：

```python
def forward(self, x):
    output = self._norm(x.float())
    output = output * (1.0 + self.weight.float())
    return output.type_as(x)
```

Llama 常见写法是 \(\gamma\) 直接初始化为 1。两种等价，只是参数化不同。

### 和 LayerNorm 的差别

LayerNorm：

\[
\hat x=\frac{x-\mu}{\sigma},\quad
\mu=\frac{1}{d}\sum x_j,\quad
\sigma=\sqrt{\frac{1}{d}\sum(x_j-\mu)^{2}+\varepsilon}
\]

再乘 \(\gamma\)、加 \(\beta\)。

| | LayerNorm | RMSNorm |
|---|---|---|
| 减均值 | 是 | **否** |
| 除以什么 | 标准差 | **均方根**（含均值信息） |
| 仿射 | \(\gamma,\beta\) | 通常只有 \(\gamma\) |
| 计算 | 要两次归约（均值 + 方差） | 一次归约（平方均值） |

RMS 和标准差的关系：\(\mathrm{RMS}^{2}(x)=\mathrm{Var}(x)+\mu^{2}\)。不减均值时，整体平移也会被缩放掉一部分，但方向（各维相对大小）基本保留。

直观上：LayerNorm 既校正「平移」也校正「尺度」；RMSNorm 只校正尺度，把 \(\|x\|_2\) 拉到大约 \(\sqrt{d}\)。

### 为什么 LLM 普遍用它

1. **更便宜**：少一次减均值，融合 kernel 更简单，是现在 decoder 里的默认选择。
2. **对残差更友好**：残差希望保留方向；很多经验表明，深层 Transformer 真正需要压的是激活爆炸，不是均值漂移。
3. **数值稳**：除的是 \(\sqrt{\mathbb{E}[x^{2}]+\varepsilon}\)，不会出现方差接近 0 时除以极小 \(\sigma\) 那么敏感。

Pre-LN 里它通常放在子层入口：`x + f(RMSNorm(x))`，让 Attention/FFN 看到的向量尺度稳定，残差仍走原始 \(x\)。

---

## Q2. Group RMSNorm 具体是怎么做的？

Group RMSNorm 只改 **RMS 统计量的计算范围**：把拼在一起的 4 路残差当成 4 组，**每组单独算 RMS**，仿射参数仍是整条 `4d` 上的一个向量。

GR 里是这样构造的：

```python
self.hc_norm = Qwen4ExpTextRMSNorm(
    hc_hidden_size,                 # 4 * hidden_size
    group_size=self.hidden_size,    # 2560
    eps=config.rms_norm_eps,
)
```

Flash-Next 上就是：`dim = 4×2560 = 10240`，`group_size = 2560`，因此 **4 组**。

### 一步步在做什么

输入 `H` 形状 `[B, T, 4d]`，4 路已经在最后一维拼好：`[h₁ | h₂ | h₃ | h₄]`。

**1. 切组（只发生在 `_norm` 里）**

```python
def _norm(self, x: torch.Tensor) -> torch.Tensor:
    if self.group_size is not None:
        x = x.reshape(*x.shape[:-1], -1, self.group_size)
    out = x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
    return out.flatten(-2) if self.group_size is not None else out
```

`reshape` 把最后一维从 `4d` 变成 `(4, d)`：

```
[B, T, 10240]  →  [B, T, 4, 2560]
```

**2. 每组自己算 RMS，不减均值**

`mean(-1)` 只沿 `2560` 这一维，得到 `[B, T, 4, 1]` 共 4 个 RMS。第 `i` 路：

\[
\mathrm{RMS}(h_i)=\sqrt{\frac{1}{d}\sum_{j=1}^{d} h_{i,j}^2+\varepsilon},\qquad
\hat h_i=\frac{h_i}{\mathrm{RMS}(h_i)}
\]

四路的尺度互不影响：某一路过冲只会缩放它自己，不会把另外三路一起压下去。

然后 `flatten(-2)` 拼回 `[B, T, 10240]`。

**3. 仿射是整条 `4d` 上的 `(1+γ)`，不分组**

`forward` 继承自 Qwen3.5，没有改：先 `_norm(x.float())`，再乘 `(1.0 + weight)`。

`weight` 长度是 **10240**，初始化为 **0**，所以起步是恒等 `×1`。`γ` 按通道学，**不在 4 路之间共享**。先在 float32 里算完再 cast 回去。

整体就是：

\[
\mathrm{GroupRMSNorm}(H)_i = \hat h_i \odot (1+\gamma_i),\quad i=1,\ldots,4
\]

### 和普通 RMSNorm / GroupNorm 的差别

| | 统计范围 | 减均值 | 仿射 |
|---|---|---|---|
| 普通 RMSNorm（`group_size=None`） | 整个最后一维 `4d` | 否 | `(1+γ)`，长度 `4d` |
| **这里的 Group RMSNorm** | 每组 `d=2560` | 否 | 同样 `(1+γ)`，长度 `4d` |
| GroupNorm | 每组 | 是（减均值再除标准差） | 通常 per-channel `γ,β` |

所以它更像「按残差支路切开的 RMSNorm」，不是标准 GroupNorm。

同文件里 Q/K 的 `q_norm` / `k_norm`、以及 `group_size=None` 时，走的就是普通 RMSNorm：沿整个 `head_dim` 或 `hidden` 算一个 RMS。

### 为什么 GR 要分组

Read 门随后会做 `mean_over_4(g_read ⊙ H_norm)`。如果 4 路拼在一起做一次 RMS，能量大的那一路会抬高分母，把其他路压小，读门还没工作残差就已经偏了。按支路归一化之后，4 路在相近尺度上，element-wise 读门才是在「选路」，而不是在抢救量纲。

---

## 在 Flash-Next 里的两种用法

- **普通 RMSNorm**（`group_size=None`）：Q/K head、以及没有分组时，沿整个最后一维算 **一个** RMS。
- **Group RMSNorm**（GR 的 `hc_norm`）：把 `4d` 切成 4 组，**每组各自**算一个 RMS，仿射仍是整条 `4d` 的 \(1+\gamma\)。原理相同，只是统计范围从「整个向量」变成「每一路残差」。
