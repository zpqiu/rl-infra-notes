# Gated Residual（GR）：结构、mHC 对照与报告实验

备查笔记。对应 [ARCHITECTURE.md](../ARCHITECTURE.md) §3，以及 technical report §2.2（Residual）和 §3.3（Stability Stress Test）。

实现：`Qwen4ExpTextGatedResidual`。Qwen3.8-Flash-Next 使用 4 路 Expanded Residual；每个 decoder layer 的 token mixer（GDN / QSA）和 MoE 各有一套独立 GR。

本文重点回答四类问题：

1. GR 的读写公式是什么；
2. GR 与 HC / mHC 的差异是什么；
3. report 中的 static、dynamic、no-GR 分别指什么；
4. report 用哪些实验支持最终设计。

---

## 1. 最终 GR 结构

设一个 token 在某个子层前的 4 路 residual 为

\[
R=[R_1,\ldots,R_{n_r}]\in\mathbb R^{n_r\times d},
\qquad n_r=4.
\]

### 1.1 每条 branch 独立 RMSNorm

\[
\bar R_i=\operatorname{RMSNorm}(R_i;\gamma_i),
\qquad i=1,\ldots,n_r.
\]

每条 branch 独立计算 RMS，并有自己的逐通道有效 gain。这里的 \(\gamma_i\) 表示乘到归一化结果上的有效 gain；实现采用零中心参数化，checkpoint 参数为 \(\gamma_i^{\mathrm{param}}\)，实际 gain 是 \(1+\gamma_i^{\mathrm{param}}\)。某一路幅度过大不会改变其他路的归一化尺度。

### 1.2 Element-wise dynamic read

把所有 normalized branches 拼成 `n_r d` 维向量，经低秩 bottleneck 产生逐 branch、逐 channel 的门：

\[
G=
\operatorname{unvec}
\sigma\!\left(
W_u\operatorname{SiLU}\!\left(
\frac{1}{n_r}W_d\operatorname{vec}(\bar R)
\right)
\right)
\in(0,1)^{n_r\times d}.
\]

Flash-Next 的 bottleneck rank 为

\[
r=d/8=320.
\]

子层实际看到的单路输入为

\[
x=\frac1{n_r}\sum_{i=1}^{n_r}G_i\odot\bar R_i.
\]

因此不同 channel 可以从不同 branch 读取；这不是每条 branch 共用一个 scalar 的普通 HC read。两个 `/ n_r` 的缩放原因见 [gr-read-divide-by-4.md](gr-read-divide-by-4.md)。

### 1.3 Branch-wise dynamic write

令子层输出为

\[
y=F(x).
\]

GR 为每条 branch 预测一个写入 scalar：

\[
s=2\sigma\!\left(
\frac1{n_r}W_w\operatorname{vec}(\bar R)
\right)
\in(0,2)^{n_r}.
\]

更新为

\[
R_i'=R_i+s_i y.
\]

Read 是 `[n_r,d]` 的 element-wise gate，Write 仍是 `[n_r]` 的 branch-wise scalar。原始、未经 Norm 的 \(R_i\) 走 identity bypass。

### 1.4 在 decoder layer 中的位置

```text
# Attention / token mixer 半层
x = GR_attn.Read(R)
y = GDN(x) 或 QSA(x)
R = GR_attn.Write(R, y)

# MoE 半层
x = GR_mlp.Read(R)
y = MoE(x)
R = GR_mlp.Write(R, y)
```

GR Read 已经包含 Group RMSNorm，所以 block 外不再额外放传统 Pre-Norm。

---

## 2. HC / mHC 与 GR 的统一写法

### 2.1 HC 的三个 operator

Hyper-Connections（HC）可写成

\[
x^{(l)}=H_{\mathrm{mix}}^\top R^{(l)},
\]

\[
y^{(l)}=F^{(l)}(\operatorname{Norm}(x^{(l)})),
\]

\[
R^{(l+1)}=
H_{\mathrm{res}}R^{(l)}+
H_{\mathrm{combine}}y^{(l)\top}.
\]

三者分别是：

| Operator | 形状 | 作用 |
|---|---:|---|
| \(H_{\mathrm{mix}}\) | `[n_r]` | 把多路 residual 读成一路 |
| \(H_{\mathrm{combine}}\) | `[n_r]` | 把子层输出写回多路 |
| \(H_{\mathrm{res}}\) | `[n_r,n_r]` | 直接混合旧 residual branches |

### 2.2 mHC 中的 `m`

mHC 是 **Manifold-Constrained Hyper-Connections**。`m` 指 manifold-constrained。

mHC 主要把 \(H_{\mathrm{res}}\) 约束在非负双随机矩阵集合中：

\[
H_{\mathrm{res}}\mathbf 1=\mathbf 1,
\qquad
H_{\mathrm{res}}^\top\mathbf 1=\mathbf 1.
\]

DeepSeek V4 用 Sinkhorn-Knopp 迭代得到这个 `[4,4]` 矩阵；代码中的 `comb` 就是 \(H_{\mathrm{res}}\)。该约束限制多层 residual mixing 连乘时的放大。

### 2.3 Static 不等于 frozen

report 用“静态项 + residual 预测的动态项”描述 operator。把维度显式写出，可以记为：

\[
z_*=W_*\operatorname{vec}(\bar R),
\qquad
W_*\in\mathbb R^{m_*\times n_rd},
\]

\[
H_*=H_*^s+
\operatorname{reshape}_*\!\left(\lambda_*\odot\phi(z_*)\right),
\]

其中 \(*\in\{\mathrm{mix},\mathrm{combine},\mathrm{res}\}\)，并且

\[
m_{\mathrm{mix}}=n_r,
\qquad
m_{\mathrm{combine}}=n_r,
\qquad
m_{\mathrm{res}}=n_r^2.
\]

- \(H_*^s\)：**可学习但 input-independent** 的静态参数；
- \(W_*\)、\(\lambda_*\)：动态分支的可学习参数；
- \(W_*\operatorname{vec}(\bar R)\)：从当前 token 的全部 residual branches 预测出的数据依赖项；
- `reshape`：将 \(n_r^2\) 个 residual-mix 值还原成 `[n_r,n_r]`。

因此：

- **static mHC**：令 \(\lambda_*=0\) 并在该设置中保持为零，关闭动态分支；静态项 \(H_*^s\) 仍可训练，但不同 token 使用同一组 operator；
- **dynamic mHC**：operator 随当前 token 的 residual state 改变。

DeepSeek V4 每个 token 动态产生：

```text
pre   [4]     = H_mix
post  [4]     = H_combine
comb  [4,4]   = H_res，经过 Sinkhorn
```

### 2.4 GR 相当于固定 \(H_{\mathrm{res}}=I\)

GR 的 residual 更新没有 branch-to-branch bypass mixing：

\[
R'=R+s\,y^\top.
\]

放进 HC 统一公式，就是

\[
H_{\mathrm{res}}=I.
\]

四路仍会通过“联合 Read → 子层计算 → Write 回四路”间接交换信息，但旧 residual 本身不经过 `[4,4]` mix。

---

## 3. GR 与 dynamic mHC 的核心区别

| 项 | Qwen3.8 GR | Dynamic mHC / DeepSeek V4 |
|---|---|---|
| Branch 数 | 4 | 4 |
| Read | 每 branch、每 channel 一个 gate `[4,d]` | 每 branch 一个 scalar `[4]` |
| Write | 每 branch 一个 scalar `[4]` | 每 branch 一个 scalar `[4]` |
| 旧 residual 路径 | Identity，\(H_{\mathrm{res}}=I\) | 动态 `[4,4]` mixing |
| 稳定性处理 | 不需要约束 residual mix | Sinkhorn 投影为双随机矩阵 |
| Normalization | 每条 branch 独立 RMSNorm，融合进 Read | HC read 后再做普通 Pre-Norm |
| 表达能力放在哪里 | 逐通道读取 | 分支间 residual mixing |
| Residual-side 额外工作 | Element-wise Read + scalar Write；identity bypass，无 \(H_{\mathrm{res}}\) mixing | Scalar Read/Write + 完整 residual mixing；另有小型 `[4,4]` Sinkhorn |

一句话：

> mHC 用较粗的 scalar read/write 配合强 `[4,4]` residual mixing；GR 去掉 residual mixing，把容量集中到 element-wise read。

---

## 4. report 中不同 baseline 分别是什么

| 名称 | Residual 路数 | 是否可学习 | 是否随 token 动态 |
|---|---:|---:|---:|
| Pre-norm / no-GR | 1 | 普通模型参数 | 没有 GR operator |
| Widening-only / 简化 AltUp | 4 | 静态 read 等参数 | 否 |
| mHC static | 4 | 是 | 否 |
| mHC dynamic | 4 | 是 | 是 |
| GR | 4 | 是 | 是；Read 细化到 channel |

Figure 7 信息路径分析里的 **no-GR reference 是单路 pre-norm residual**，不是“四路但去掉 gate”。

更接近“四路、无动态 gate”的是 widening-only 或 mHC static：

- widening-only 使用静态 read，并按深度 round-robin 把输出写入一条 branch；
- mHC static 保留 4 路 HC operator，但 operator 不依赖当前 token。

---

## 5. Widening-only：四路本身是否有用

report 先用简化 AltUp 检查“只加宽 residual”能带来多少收益：

- 25B-A3B MoE；
- 训练 400B tokens；
- 4 路 residual；
- 静态 scalar read；
- block output 按层号 round-robin 写入一条 branch。

仅加宽 residual 就让训练 loss 降低约

\[
\Delta L\approx-0.01.
\]

所以 GR 的收益不全来自动态 gate；多路 residual 本身已经增加了跨层信息容量。

---

## 6. Residual read/write 消融（Table 5）

设置：

- 25B-A3B MoE；
- 训练 560B tokens；
- 所有 widened variants 都使用 4 branches；
- 相同 benchmark suite 和 evaluation pipeline。

| Residual | Loss | MMLU | MMLU-Pro | SuperGPQA | MATH | GSM8K | BBH | MMMLU | EvalPlus | MultiPL-E | Avg. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| Pre-norm | 1.617 | 64.29 | 38.40 | 21.78 | 53.92 | 77.41 | 64.73 | 51.26 | 49.25 | 37.15 | 50.91 |
| mHC static | 1.596 | 64.62 | 43.69 | 22.20 | 55.08 | 78.05 | 65.42 | 52.78 | 49.59 | 40.94 | 52.49 |
| mHC dynamic | 1.594 | 66.11 | 45.84 | **24.20** | 59.54 | **78.51** | 66.01 | **56.61** | **52.16** | 41.30 | 54.47 |
| **GR** | **1.590** | **66.69** | **46.02** | 23.80 | **61.18** | 78.20 | **66.54** | 56.19 | 51.36 | **42.00** | **54.66** |

### 6.1 Static widening 贡献主要 loss gain

从 Pre-norm 到 mHC static：

\[
1.617\rightarrow1.596,
\qquad \Delta L=-0.021.
\]

从 static 到 dynamic：

\[
1.596\rightarrow1.594,
\qquad \Delta L=-0.002.
\]

但 benchmark 的变化相反：

- Pre-norm → static：Avg. `+1.58`；
- static → dynamic：Avg. `+1.98`。

动态 read/write 对 loss 的影响很小，却明显影响下游能力。只看 pretraining loss 会低估 data dependence 的价值。

### 6.2 report 给出的五条设计观察

1. **Bounded positive gate**：sigmoid 在 loss 和训练稳定性上优于 tanh。
2. **Data dependence**：动态 read/write 的 benchmark 增益明显大于 loss 所显示的差异。
3. **Read granularity 更重要**：Read 从 per-branch scalar 细化到 per-branch-and-channel 有收益；Write 细化到 channel 几乎无收益。
4. **使用全部 branches**：从所有 branches 预测 operator 优于只看最后一路或先 pooling；每条 branch 独立 RMSNorm 还有额外收益。
5. **\(H_{\mathrm{res}}\) 收益很小**：Read/Write 足够强后，增加 `[4,4]` residual mixing 没有显著改善。

GR 与 dynamic mHC 在该规模下整体接近。报告选择 GR 不只是因为表中小幅更低的 loss / 更高的 Avg.，还因为去掉 \(H_{\mathrm{res}}\) 后内存流量更低、稳定性约束更简单。

report 没有公开每个组件的完整 factorial table，无法从现有数字严格拆出 element-wise read、Group RMSNorm 和移除 \(H_{\mathrm{res}}\) 各自的独立贡献。

---

## 7. 与 Attention Residual 的比较（Table 6）

Attention Residual（AttnRes）用 softmax attention 读取更早子层的输出：

- Full AttnRes：读取所有历史 sublayer outputs；
- Block AttnRes：每 \(S\) 个 sublayers 汇总成一份表示后再读取。

28-layer 模型共有 56 个 sublayers：

| Residual design | Loss | Loss + GatedNorm |
|---|---:|---:|
| Pre-norm residual | 1.789 | 1.787 |
| Block AttnRes，\(S=4\) | 1.773 | 1.768 |
| Block AttnRes，\(S=2\) | 1.770 | 1.766 |
| Full AttnRes | 1.762 | 1.758 |
| GR，\(n_r=4\) | — | 1.762 |

结论：

- Full AttnRes 不加 GatedNorm 时和 GR 都是 `1.762`；
- Full AttnRes 再加 GatedNorm 可到 `1.758`；
- 汇总历史 sublayers 会损失效果，且 \(S=4\) 比 \(S=2\) 更差；
- 在 48-layer 设置中，Block AttnRes \(S=4\) 为 `1.711`，GR 为 `1.707`。

GR 不是唯一能增强跨层读取的方案，但它只维护固定 4 路 residual，不需要对越来越多的历史 sublayer outputs 做 attention。

---

## 8. 四条 branches 实际学到了什么（Figure 7）

### 8.1 Figure 7 的路径统计与适用范围

在 Figure 7 的 20-layer probe 中，report 将 branch \(c\) 在 reader block \(v\) 前写成历史 block outputs 的累加：

\[
R_c^{(v)}=R_c^{(0)}+\sum_{u<v}s_c^{(u)}y^{(u)}.
\]

固定这次 forward 实际取到的 gate 和 RMS 分母后，writer \(u\) 对 reader \(v\) 的向量项为

\[
a_{u\to v}
=
\frac1{n_r}
\sum_{c=1}^{n_r}
G_c^{(v)}\odot\gamma_c\odot
\frac{s_c^{(u)}y^{(u)}}{\operatorname{rms}(R_c^{(v)})}.
\]

report 再比较这些 writer 向量项的 normalized magnitude：

\[
\pi_{uv}
=
\frac{\lVert a_{u\to v}\rVert}
{\sum_{u'<v}\lVert a_{u'\to v}\rVert}.
\]

按定义，每个 reader 的这些 normalized magnitudes 之和应为 1；实现的数值误差小于

\[
3\times10^{-8}.
\]

\(\pi_{uv}\) 是 **writer 向量范数的归一化统计量**，不是严格的 causal attribution：它不反映不同 writer 向量之间的抵消或同向增强，上式的分母也不包含初始 embedding。该累加式描述的是 Figure 7 的 probe；若直接分析包含 Layer-2 PLE 注入的最终 production 架构，还必须把 PLE 对四路 residual 的额外写入项计入。

### 8.2 实验设置

- 20-layer MoE；
- GR 与一个其他条件相同的单路 no-GR reference；
- 相同 recipe、数据、优化器和训练 step；
- 在相同 tokens 上 probe；
- 报告还检查了 5 个 GR checkpoints。

比较量为

\[
\Delta_{uv}=\pi_{uv}^{\mathrm{GR}}-\pi_{uv}^{\mathrm{reference}}.
\]

### 8.3 一条 branch 负责长程，其他 branches 偏局部

Figure 7 展示 780 个 ordered writer-reader pairs 中满足

\[
\Delta_{uv}\ge0.05
\]

且至少跨一个 layer 的 21 条路径。

跨 5 个 GR checkpoints，report 都观察到：

- 一条 long-range branch，正文所称的 typical skip 约 `10.9` layers；
- 其余三条 branch，正文所称的 typical skip 约 `3.4–3.9` layers。

report 没有进一步定义这里 `typical` 的聚合统计量；Figure 7 图注针对展示的 checkpoint 另给出 local branches 的 median skip `1.2–3.5`。具体是哪条 branch 不固定，因为四路在初始化时可交换。

### 8.4 代表性路径

**Layer 0 GDN → Layer 15 Attention**：

\[
\pi: 0.020\rightarrow0.138.
\]

这条路径在 layer 10–19 的 reader 中持续保持 `0.072–0.138`，没有明显随深度衰减。

**Layer 10 GDN → Layer 11 Attention**：

\[
\Delta_{uv}=0.117.
\]

说明 GR 不只强化长程路径，也会强化重要的相邻层连接。

**Layer 0 MLP 的双时间尺度**：

- 到 layer 2 的局部路径：`0.008 → 0.058`；
- 到 layer 15 的长程路径：`0.139 → 0.192`。

同一个 writer output 可以经不同 branches 同时具有局部和长期两种保留强度，单 residual stream 很难表达这种差异。

### 8.5 Softmax attention 是主要长程 reader

读取这些增强路径最多的 sublayers 主要是每四层一次的 softmax-attention layers。报告据此认为 global attention 是整合显式长程 residual 信息的重要枢纽，而 GDN 负责压缩式历史记忆。

### 8.6 GR 改变的是分布，不是平均传播距离

将所有路径按 skip 分组，GR 相对 reference：

- 相邻路径（skip 1）合计增加 `0.96`；
- 中程路径（skip 2–12）合计减少 `3.21`；
- 长程路径（skip > 12）合计增加 `0.91`。

但加权平均 skip 几乎不变：

- GR：`3.97`；
- reference：`3.91`。

所以 GR 不是让所有信息平均传播得更远，而是选择并放大少数关键的局部和超长路径。

---

## 9. Inference efficiency 实验

### 9.1 Top-2 branch 稀疏化没有采用

训练后的 write 往往由两条 branches 主导，因此 report 尝试每层只访问 gate 最大的两条 branches。

结果：

- pretraining loss 几乎不变；
- pretraining benchmarks 几乎不变；
- post-training 后质量明显下降；
- 从训练开始稀疏、中途切换、按层改变 sparsity 都没有解决。

report 这一段在 sparse read / sparse write 的措辞上不完全一致；可以确认的是，最终没有采用 top-2 branch access。它是“pretraining 指标看起来安全，但 post-training 暴露退化”的例子。

### 9.2 FP8 residual state

GR、GDN 和 gated attention 的有界 gate 使 residual 值域较窄。将 4 路 residual state 从 BF16 改为 FP8：

- residual state 的搬运字节减半；
- 质量几乎不变。

### 9.3 Fused Read / Write

- Group RMSNorm 融进 GR Read；
- Read 和 Write 各自融合为单 kernel；
- widened residual 每个 block 每个方向只遍历一次。

report 没有单独给出 GR 模块的 latency 数值。

---

## 10. 稳定性压力测试（§3.3）

### 10.1 整体 recipe 对照

设置：

- 28-layer、25B-A3B MoE；
- 固定在最优学习率的 2× / 4×，不走正常 decay；
- gradient clipping threshold `0.5`。

比较：

1. Qwen3.5 structure + AdamW；
2. Qwen3.5 structure + Muon；
3. Muon + GR。

观察：

- 在 2× 学习率下，两条 Muon runs 都未越过 clipping threshold，明显比 AdamW baseline 稳定；
- 在 4× 学习率下，AdamW baseline 的 loss spike rate 达 `183 / 10k steps`；
- 同一高压设置下，带 GR 的配置记录到 0 个 loss spikes；
- GR 同时降低 gradient-norm spike 的频率/幅度和 activation outlier。

这组三路对照可以分两步读：AdamW baseline 与 Qwen3.5 + Muon 隔离 optimizer 变化；Qwen3.5 + Muon 与 Muon + GR 保持 Muon 不变，支持“加入完整 GR 组件会改善稳定性”。但后一步仍把 widening、动态 Read/Write 和 GatedNorm 作为整体加入，不能拆出 gate 本身的独立贡献；为此 report 还做了下面的单变量实验。

### 10.2 单独隔离 GatedNorm

固定：

- 28-layer 模型；
- AdamW；
- 其他结构和数据顺序；
- 3× 最优学习率。

只开关 GatedNorm。开启 gate 后：

\[
\text{spike rate}: 32.0\rightarrow3.2
\quad\text{per 10k steps},
\]

\[
\text{clip-threshold crossings}:256\rightarrow20.
\]

在最高学习率下开启 gate 后，activation outlier 甚至低于最低学习率的 ungated baseline。

report 的机制解释是：高学习率需要 residual update rescaling；没有显式 gate 时，网络通过增大 activation outlier 间接调整，因而更脆弱。乘法 gate 直接提供重缩放，使训练更稳定。

---

## 11. Production learning-rate 下的验证（Figure 13）

比较前 276B tokens：

1. Qwen3.5 + Muon；
2. Qwen3.5 + GR + Muon；
3. 完整 Qwen3.8-Flash-Next recipe。

三者共享数据顺序、学习率 schedule 和 optimizer。

### 11.1 Loss

在 276B tokens：

- 加 GR：loss 降低 `0.026`；
- 完整 Flash-Next 再降低 `0.032`；
- 相对 Qwen3.5 + Muon 总计降低 `0.058`。

额外的 `0.032` 不能归因于同一个 GR 改动，因为 report 在完整 recipe 中明确还包含进一步 refined GR 和 N-gram embedding layer。

### 11.2 Gradient norm

| 配置 | Median | p99.9 |
|---|---:|---:|
| Qwen3.5 + Muon | 0.097 | 0.298 |
| + GR | 0.053 | 0.071 |
| Full Flash-Next | 0.043 | 0.066 |

- 无 GR / GatedNorm residual gate 的 Muon baseline 是唯一触发 clipping 的运行；
- gated runs 在 1000-step window 内的 gradient-norm standard deviation 低 `4.3–4.7×`；
- GR 在所有被 probe 的深度上都明显降低 residual activation maximum。

report 推测，将最终 residual read 和 LM head 前的 final normalization 融成 gated read，是完整 Flash-Next 与普通 `Muon + GR` 之间差异的主要来源之一。

---

## 12. 证据边界

公开 report 支持以下结论：

- 4 路 widening 本身有效；
- data-dependent read/write 对 benchmark 的价值大于 loss 所显示的差异；
- read 细化到 channel 比 write 细化更重要；
- 去掉 \(H_{\mathrm{res}}\) 在所测规模下没有明显质量代价，并减少 residual memory traffic；
- 在 Figure 7 的 20-layer probe 及所检查的 5 个 checkpoints 中，GR 形成一条长程 branch 和三条偏局部 branches；
- top-2 branch 稀疏化会在 post-training 后退化；
- gate 对高学习率下的 spike、gradient norm 和 activation outlier 有明确改善。

公开材料仍没有给出：

- element-wise read、Group RMSNorm、移除 \(H_{\mathrm{res}}\) 的完整 factorial 数值；
- GR 独立模块的 latency / bandwidth 表；
- top-2 branch 稀疏化的具体 post-training 分数；
- 全尺寸模型上 GR 与 dynamic mHC 的独立对照；
- 每个 branch 的完整 gate 分布和不同任务间的一致性。
