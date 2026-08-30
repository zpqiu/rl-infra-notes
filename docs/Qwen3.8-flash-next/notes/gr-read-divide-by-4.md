# GR Read 为什么除以 4

备查笔记。对应 [ARCHITECTURE.md](../ARCHITECTURE.md) §3.3：

`u = SiLU( W_down(H_norm) / 4 )`

以及问答：这里为什么要除以 4？

实现：`Qwen4ExpTextGatedResidual.forward`（`modular_qwen4_exp.py`）。`4` 就是 `hc_count`（残差支路数）。写门同样除以 `hc_count`。

---

## 一句话

Group RMSNorm 之后每一路已经是 O(1)，4 路拼成 `4d` 再做线性层，预激活会按支路数变大。除以 4 是按 **平均** 而不是求和来缩放，让后面的 SiLU / sigmoid 起步落在非饱和区：读门大约均匀混合，写门大约恒等写回。

细节（初始化方差、std 估算）可先跳过，需要时再看下面。

---

## 代码

```python
input_mix_weight = F.silu(self.input_mix_weight_down(hyper_input_normed) / self.hc_count)
input_mix_weight = torch.sigmoid(self.input_mix_weight_up(input_mix_weight))
# ...
injection_weights = 2 * torch.sigmoid(self.block_inject_weight(hyper_input_normed) / self.hc_count)
```

两处都是 `/ hc_count`，意图同一套。

---

## 原因（短版）

1. **fan-in 大了 4 倍**
   `H_norm` 是 `[B, T, 4d]`。每路 RMS ≈ 1，拼起来线性层要加 4 倍那么多项。权重按 HF 默认 `N(0, 0.02^2)` 初始化，**不按 fan-in 缩放**，预激活会偏大。`/ 4` 把它压回接近单路的尺度。

2. **护的是门，不是 W_down 本身**
   `/4` 在 SiLU **之前**。预激活小，低秩瓶颈不会一上来就很大，后面的 `sigmoid(W_up(·))` 才不会饱和成 0/1。初始化时 sigmoid(0)≈1/2，4 路近似均匀混合。
   写门 `2 * sigmoid(0) = 1`，起步按恒等写回。

3. **和 GR 的「平均读写」对齐**
   Read 端是 `mean_over_4(g ⊙ H)`。线性预激活也按 `/n` 平均，加宽支路时残差尺度不会一层层漂。这是 mean 缩放（`/n`），不是方差保持的 `/sqrt(n)`。

无 bias 时，线性部分 `W(H)/4` 和 `W(H/4)` 相同；SiLU 非线性，所以必须先除再激活，不能把 `/4` 挪到 SiLU 后面。

---

## 原因（需要时再看的数字）

`W_down`：`4d → 320`，无 bias，W ~ N(0, 0.02^2)。单个输出维的标准差大约：

`0.02 * sqrt(4d) ≈ 0.02 * sqrt(10240) ≈ 2.0`

多出来的 sqrt(4) 来自 4 路拼接。`/ 4` 之后标准差大约 `0.5`。

换成 8 路仍 `/ hc_count` 的话，尺度继续按平均压，不会随加宽线性爆炸。
