# Qwen3.8-Flash-Next 模型结构

本文对照官方架构图，以及 Hugging Face `transformers` 里的实现（`src/transformers/models/qwen4_exp/`，源文件是 `modular_qwen4_exp.py`）讲解模型结构。

官方实现的类名前缀是 `Qwen4Exp*`：Qwen3.8-Flash-Next 是 **Qwen4 架构的提前开源预览**，角色等同于当年的 Qwen3-Next 之于 Qwen3.5。

> 注意：`Qwen4ExpTextConfig` 里的默认值（`hidden=2048`、`40` 层等）只是代码 stub。下文规格以 **Qwen3.8-Flash-Next 真实 checkpoint** 为准。

---

## 1. Overview

### 1.1 定位

Qwen3.8-Flash-Next 是一个多模态 MoE 模型，在 Qwen3.5 的 **GDN + Gated Attention + MoE** 骨架上，沿四个方向改架构：

| 方向 | 改了什么 |
|---|---|
| Attention | 原来的 Gated Attention 换成 **QSA**（Qwen Sparse Attention），仍和 GDN 3:1 混排 |
| Residual | 单流 Pre-LN residual 换成 **Gated Residual（GR）**：4 路并行残差 + 动态读写门 |
| Embedding | 额外的 **N-gram Embedding**（代码里叫 PLE），用查表扩容量、几乎不加 FLOPs |
| Optimization | **Muon + AdamW** 分工，并按新架构重拟合 scaling law（训练侧，本文不展开） |

主体 **125B** 参数，每 token 激活约 **6B**，外挂 **51B** N-gram embedding 和 **4B** MTP。相对 Qwen3.7-Plus，官方称训练成本大约只有 1/9。

### 1.2 官方架构图

![Qwen3.8-Flash-Next Architecture](https://qianwen-res.oss-accelerate.aliyuncs.com/Qwen3.8-Flash-Next/architecture.png)

图从左到右、从上到下可以这样读：

1. **Input Tokens → Vocabulary Embedding**：普通 token embedding。
2. **Hybrid Block × L/4**：主干重复 12 次。每个 Hybrid Block 含 **3 个 GDN Layer + 1 个 QSA Layer**（图里把 QSA 画在块顶上，实现顺序是先 3 个 GDN 再 1 个 QSA）。
3. 每个 Layer 的骨架相同：**GR Read →（GDN 或 QSA）→ GR Write → GR Read → MoE → GR Write**，残差流是加宽后的 **Expanded Residual**（4 路）。
4. **N-gram Embedding** 只接到 **Layer 2** 的某个 GDN Layer 上。
5. 主干结束后：**GR Read → Prediction Head**，旁边还有 **MTP Modules**（训练用的多 token 预测头）。

### 1.3 真实 checkpoint 规格

| 项 | 数值 |
|---|---|
| 类型 | Causal LM + Vision Encoder |
| 主体参数 | 125B，每 token 激活 **6B** |
| N-gram embedding | 额外 **51B**（可 CPU offload） |
| MTP | 额外 **4B**（1 层，multi-step 训练） |
| Hidden | 2560 |
| Token / LM vocab | 248320（padded） |
| 层数 | **48** |
| 排布 | `12 × (3 × (GDN → MoE) → 1 × (QSA → MoE))` |
| 原生上下文 | 262,144；YaRN 可到 1M |
| Gated Residual | 4 路，bottleneck rank 320 |
| MoE | 512 experts，每 token 10 routed + 1 shared，FFN dim 640 |

**Gated DeltaNet**

| 项 | 数值 |
|---|---|
| QK heads | 16 |
| V heads | 48 |
| Head dim | 128 |
| 短卷积 kernel | 4 |

**Qwen Sparse Attention**

| 项 | 数值 |
|---|---|
| Q heads | 24 |
| KV heads | 2（GQA） |
| Head dim | 256 |
| RoPE 维 | 64（`partial_rotary_factor=0.25`） |
| Indexer | MQA：4 Q heads + 1 共享 K head，dim 128 |
| Budget | 512 blocks ≈ 2048 tokens |
| 压缩比 | 4（连续 4 个 token 合成 1 个 micro-block） |

### 1.4 端到端数据流（实现）

```
input_ids
    │
    ├─ embed_tokens → inputs_embeds [B, T, 2560]
    │       ▲
    │       └─ 若有图像/视频：Vision Encoder 的 token scatter 进 placeholder
    │
    ├─ hidden = inputs_embeds.repeat(..., hc_count=4)   # Expanded Residual [B, T, 4×2560]
    │
    ├─ 48 × DecoderLayer
    │       ├─ (仅 layer 2) hidden += PLE(n-gram lookup)
    │       ├─ GR Read → GDN 或 QSA → GR Write
    │       └─ GR Read → MoE → GR Write
    │
    ├─ hyper_connection_mixer：4 路 → 1 路 [B, T, 2560]
    │
    └─ lm_head → logits
```

对应代码：`Qwen4ExpTextModel.forward` 里 `hidden_states.repeat(..., hc_count)`，循环 `Qwen4ExpTextDecoderLayer`，最后 `hyper_connection_mixer`。

---

## 2. Hybrid Block：层怎么排

48 层按每 4 层一组，重复 12 次：

```
Layer  0, 1, 2   GDN + MoE     ← 第 2 层（1-indexed）挂 N-gram / PLE
Layer  3         QSA + MoE
Layer  4, 5, 6   GDN + MoE
Layer  7         QSA + MoE
...
Layer 44,45,46   GDN + MoE
Layer 47         QSA + MoE
```

配置里 `layer_types` 的生成规则是：`(i + 1) % 4 == 0` 为 `qwen_sparse_attention`，否则 `linear_attention`。checkpoint 若写成 `full_attention`，加载时会被改写成 `qwen_sparse_attention`——QSA 就是挂在原来的 full attention 层上。

每一层的 Python 骨架（`Qwen4ExpTextDecoderLayer`）几乎一样，只换 token mixer：

```text
if PLE:
    H ← H + PLE(H, input_ids)

# Attention 半层
h, H, g_write ← GR_attn.Read(H)          # h: [B,T,d],  H: [B,T,4d]
y ← GDN(h)  或  QSA(h)
H ← H + g_write ⊙ y                      # GR Write，写回 4 路

# FFN 半层
h, H, g_write ← GR_mlp.Read(H)
y ← MoE(h)
H ← H + g_write ⊙ y
```

没有传统的 `input_layernorm` / `post_attention_layernorm`。Norm 做在 4 路 Expanded Residual 上，由 GR 内部完成。Attention 和 MoE **各有一套独立的 GR**。

---

## 3. Gated Residual（图中的 GR Read / GR Write / Expanded Residual）

论文名 **Gated Residual (GR)**，代码类名 `Qwen4ExpTextGatedResidual`（注释里也称 hyper-connection）。

### 3.1 在解决什么问题

普通 Transformer 残差是单条流：`x ← x + f(LN(x))`。GR 做两件事：

1. 把残差加宽成 **4 条并行支路**（Hyper-Connection 思路），让不同层可以走不同的信息通道。
2. 用 **数据依赖的门** 控制从哪条路读、往哪条路写（GatedNorm 思路），深层训练更稳，推理开销很小。

### 3.2 Expanded Residual

进第一层之前：

```python
hidden_states = inputs_embeds.repeat(1, 1, hc_count)  # [B, T, 4d]
```

之后整网都在 `[B, T, 4×2560]` 上跑，直到最后的 mixer 再压回 `[B, T, 2560]`。

### 3.3 GR Read（图中橙色）

对 4 路做 **Group RMSNorm**（每组 `d=2560` 独立方差，公式是 Qwen3.5 风格的 `(1 + γ)`）。

然后一个低秩 mixer（rank **320**）：

```
u = SiLU( W_down(H_norm) / 4 )     # 4d → 320
g_read = σ( W_up(u) )              # 320 → 4d，再 reshape 成 [4, d]
h = mean_over_4( g_read ⊙ H_norm ) # 加权平均 → [B, T, d]
```

`h` 才是 GDN / QSA / MoE 看到的输入。读门是 **element-wise** 的，每条支路、每个通道都可以单独开关。

### 3.4 GR Write（图中蓝色）

写门是 **每条支路一个标量**（每个 token、每条路一个数）：

```
g_write = 2 · σ( W_inject(H_norm) / 4 )    # [B, T, 4]
H ← H + concat_4( g_write[i] · y )         # y 是子层输出 [B, T, d]
```

残差加的是 **未经 Read 的原始 `H`**，不是 norm 之后的值。`2·sigmoid` 让写回幅度以 1 为中心，可大于或小于恒等映射。

### 3.5 最后的 mixer

`Qwen4ExpTextModel.hyper_connection_mixer` 也是一个 GR，但 `use_combine=False`：**只 Read、不 Write**，把 4 路合成一条 hidden，交给 LM head。

---

## 4. GDN Layer（图中 Gated DeltaNet）

每组 Hybrid Block 里有 **3 个** GDN Layer。实现类 `Qwen4ExpTextGatedDeltaNet`，基本原样继承 Qwen3.5 的 Gated DeltaNet。

### 4.1 角色

线性注意力 / 递推状态：把历史压进 **固定大小** 的 recurrent state，对长度近似线性，不存全量 KV。3/4 的层用它「廉价地记住过去」，剩下 1/4 用 QSA 做精确检索。

### 4.2 计算步骤

1. **投影**：`in_proj_qkv / z / b / a` 分别出 QKV、输出门 `z`、delta 的 `β` 和衰减相关的 `a`。
2. **短卷积**：kernel=4 的 depthwise causal conv 作用在 QKV 上（解码时维护 conv state）。
3. **Gated Delta Rule**：  
   `β = sigmoid(b)`，`g = −exp(A_log) · softplus(a + dt_bias)`。  
   V heads 是 QK 的 3 倍（48 vs 16），Q/K 做 `repeat_interleave`。  
   Prefill 走 chunked kernel，decode 走 recurrent kernel；Q/K 在 kernel 里 L2 normalize。
4. **输出门**：`RMSNorm(y) * silu(z)`，再 `out_proj` 回到 `d`。

输出形状 `[B, T, 2560]`，再交给 GR Write 写回 4 路。

### 4.3 和 Qwen3.5 的差异

几乎没有算法差异。Qwen4-Exp 只把 gated RMSNorm 的激活改成可配置（`output_gate_type`，默认仍是 `silu`）。

---

## 5. QSA Layer（图中 Qwen Sparse Attn）

每组 Hybrid Block 里有 **1 个** QSA Layer，替换 Qwen3.5 里的 full Gated Attention。

### 5.1 角色

GDN 的 state 是有损压缩。QSA 负责在全上下文上做一次 **稀疏、精确** 的检索。和「逐 token 选 top-k」不同，QSA 先把序列收成 **micro-block**，在 block 级打分，再把选中的 block 展开成 token 做注意力。长序列上，这一层的计算/访存从 `O(T²)` 降到大约 `O(T · 2048)`。

### 5.2 两段式结构

```
hidden
  ├─ Indexer（轻量 MQA）→ 稀疏 token mask
  └─ Gated Attention（GQA + QK-Norm + partial RoPE + output gate）
         attention_mask = causal_mask ∧ indexer_mask
```

对应 `Qwen4ExpTextQSAIndexer` + `Qwen4ExpTextAttention`（后者继承 `Qwen3_5Attention`）。

### 5.3 Indexer：怎么选 token

- 从 hidden 投影出 4 个 Q head + 1 个 K head（dim 128），各自 RMSNorm。
- Q 用 **当前** token 的 RoPE；K 先按 `compress_ratio=4` 做平均，得到 block key，再用该 block **起始位置** 的 RoPE。
- 只在 causal 可见、且凑满完整 block 的 token 上打分：

  ```
  score(block) = sum_heads ReLU(q · k_block) / √d
  ```

- `topk` 选出最多 **512** 个 block，展开成最多 **2048** 个 token。
- 末尾凑不满一个 block 的 **tail token 全部保留**（保证最近上下文不被丢掉）。
- 得到一张 0/1 mask，叠到 causal mask 上。未选中的位置直接不参与 attention。

实现里 indexer 的 Python 路径是双重 for-loop（eager 参考实现）。训练/推理引擎（SGLang、vLLM、TokenSpeed）会换成向量化/融合 kernel。

### 5.4 真正的 Attention

选完 token 之后，就是 Qwen3.5 的 Gated Attention：

- GQA：24 Q / 2 KV，`head_dim=256`
- Q/K RMSNorm
- **Partial RoPE**：只旋转前 64 维，后 192 维不转。约束：RoPE 维必须 ≤ indexer head dim（128）
- `q_proj` 同时产出 Q 和 output gate，`attn_out * sigmoid(gate)` 后再 `o_proj`

这一层 **仍然存 KV cache**（只是计算时每个 query 只 attend 到约 2048 个位置）。Indexer 自己也要 cache 所有 token 的 raw K，用来给后续 query 打分。

---

## 6. MoE（图中每层都有的红色 MoE）

每个 GDN Layer 和每个 QSA Layer 后面都跟一个 MoE，没有 dense FFN 层。类：`Qwen4ExpTextSparseMoeBlock`，结构沿用 Qwen3-Next。

```
h [B,T,d]
  ├─ Router：softmax → top-10，再对这 10 个概率归一化（norm_topk_prob=True）
  ├─ 10 个 routed expert：SwiGLU，intermediate=640
  └─ 1 个 shared expert：同样 SwiGLU，再乘 sigmoid 标量门
y = routed_sum + gated_shared
```

- Expert 数 **512**，权重是 3D 张量 `gate_up_proj [E, 2I, d]`、`down_proj [E, d, I]`
- 每 token 激活：10 routed + 1 shared
- Router 是线性 `d → 512`，没有 load-balancing 以外的额外结构（aux loss 系数 `router_aux_loss_coef=0.001`）

粗算激活量：`10 × 2 × 2560 × 640` 量级加上 shared 和注意力，和官方「6B activated」一致。容量主要来自 512 个 expert 的稀疏组合，而不是把单层 FFN 做宽。

---

## 7. N-gram Embedding（图中 Layer 2 only）

论文名 **N-gram Embedding**，代码名 **PLE（Per-Layer Embedding）**。图上明确画了：只在 Layer 2 注入，接到一个 GDN Layer 上。

约束（`validate_architecture`）：

- `ple_layer_ids` 是 **1-indexed**
- PLE **只能挂在 `linear_attention`（GDN）层**
- Flash-Next 实际是 `ple_layer_ids = [2]`

### 7.1 在解决什么问题

MoE 用「条件计算」扩容量，但 expert 必须常驻加速器。Embedding 查表几乎零 FLOPs，而且整张表可以放在 **host 内存**，lookup 下标提前算好、异步 prefetch。这是另一条扩参轴：Flash-Next 用一张约 **51B / ~90GiB** 的表，专门记局部短语（bigram / trigram）。

### 7.2 Hash 查表（`Qwen4ExpTextNGramEmbedding`）

- `ngram_size=3` → bigram + trigram
- 每种 n-gram **8 个独立 hash head**，共 16 head
- 每个 head 一张约 **2000 万** 的词表，大小取互不相同的素数（避免各 head 碰撞对齐）
- hash：`token_id × 层相关奇数乘数`，再 XOR 各位置，再 `mod vocab`
- 跨 EOS **不混上下文**（shift 时用 EOS 填充）
- 16 个头的向量 concat 成 `ple_embed_dim`（默认 = hidden 2560）

解码时要把前 `ngram_size-1` 个 token 放进 cache（实现上复用 conv_state 槽位 2）。

### 7.3 注入 GDN 层（`Qwen4ExpTextPLELayer`）

查完表之后不是直接加到 token embedding 上，而是 **按 4 路残差做一次门控写入**：

```
emb = NGramLookup(input_ids)
K = RMSNorm(W_k(emb))     # 4 路各一份 key
V = W_v(emb)              # 共享 value
Q = RMSNorm(H)            # 当前 4 路 hidden 当 query
s = ⟨K, Q⟩ / √d
gate = sign(s) · √|s|
H ← H + σ(gate) · V
H ← H + DilatedDWConv( RMSNorm(σ(gate)·V) )   # dilation=3, kernel=4
```

膨胀卷积给局部词法一点「平滑」，kernel 按 dilation 拉长，cache 里另占 conv_state 槽位 1。

### 7.4 和 device_map

`ngram_embedding.weight` 被标成 `_no_placement_params`。大多数机器上必须跳过 `device_map`，否则 accelerate 会在 forward 时把 CPU offload 的参数搬回 GPU，直接 OOM。官方推理栈用 host 常驻 + 异步 prefetch。

---

## 8. MTP 与 Prediction Head

图右侧：主干结束后一条 **GR Read → Prediction Head**，旁边还有 **MTP Modules**。

### 8.1 Prediction Head

推理路径：

```
H_4way → hyper_connection_mixer（只 Read）→ [B, T, 2560] → lm_head → vocab logits
```

`tie_word_embeddings=False`，embedding 和 LM head 不共享。

### 8.2 MTP Modules

官方规格：**1 层 MTP，约 4B，multi-step 训练**。用来在训练时同时预测多个未来 token，加强规划/长程信号。

**当前 `transformers` 的 `qwen4_exp` 实现不加载 MTP**：

```python
_keys_to_ignore_on_load_unexpected = [r"^mtp.*"]
```

权重文件里有 `mtp.*`，推理时丢掉。训练框架（Megatron / 内部栈）才会用到图里那一块。

---

## 9. Vision Encoder（图上没画，权重里有）

Flash-Next 是 **Causal LM + Vision Encoder**。文本主干图只画了 LLM；视觉部分几乎原样继承 Qwen3.5-MoE ViT（`Qwen4ExpVisionModel`）。

| 项 | 默认 / 行为 |
|---|---|
| Patch | 3D conv，`temporal=2`，空间 `16×16` |
| Merge | `spatial_merge=2`（2×2 patch 合成 1 个视觉 token） |
| ViT 深度 | 27（config 默认） |
| 位置 | 可插值的 2D pos embed + Vision RoPE |
| 接到 LLM | `PatchMerger` 投到文本 hidden，scatter 进 image/video placeholder |

文本侧位置编码是 **interleaved 3D MRoPE**（T/H/W），YaRN 到 1M 时 `mrope_section` 类似 `[11, 11, 10]`，`partial_rotary_factor=0.25`。

PLE 用的是原始 `input_ids`（视觉 placeholder 对 n-gram 来说就是特殊 token）。若用户只传 `inputs_embeds`、不传 `input_ids`，实现会尝试把 embedding 反查回 id（`reverse_embedding`），对不上会报错。

---

## 10. Cache 与长上下文

有 PLE 时 `number_of_conv_states = 3`：

| Cache 槽 | 谁用 |
|---|---|
| conv_states[0] + recurrent_states | GDN 短卷积 + delta 状态 |
| conv_states[1] | PLE 膨胀卷积 |
| conv_states[2] | n-gram 需要的前几个 token id |
| KV + indexer raw K | 仅 QSA 层 |

QSA indexer 需要对 **全部历史位置** 做 RoPE，所以 cache 上还挂了完整 `position_ids`。生成时 `allow_is_causal_skip=False`，必须物化 4D mask，再叠 indexer mask。

原生 262K。超过这个长度用 **静态 YaRN**（`factor` 按目标长度设，例如 1M 用 4.0）。静态 YaRN 对短文本可能有副作用，官方建议只在真正需要超长上下文时改 `rope_parameters`。

---

## 11. 和 Qwen3-Next / Qwen3.5 的对照

| | Qwen3-Next / 3.5 | Qwen3.8-Flash-Next（Qwen4-Exp） |
|---|---|---|
| 混合注意力 | 3 GDN + 1 **Gated Attention** | 3 GDN + 1 **QSA**（block 级稀疏） |
| 残差 | 单流 Pre-LN | **4 流 Gated Residual** |
| Embedding | 只有 vocab embedding | + Layer 2 的 **n-gram 查表** |
| MLP | MoE（部分层可能 dense） | **每层都是 MoE**，expert 更窄（640）、激活更多（10+1） |
| 长上下文代价 | full attention 层仍是满 KV 计算 | QSA 每 query 最多看 ~2048 token |
| 视觉 | Qwen3.5 ViT | 基本相同 |

一句话：

- **GDN** 廉价压缩历史
- **QSA** 在压缩索引上精确取回
- **4 路 GR** 负责深层信息分流和训练稳定
- **N-gram 表** 用查表换局部短语容量

主体 125B 里每 token 只跑约 6B，再外挂一张几乎零计算的 51B 短语记忆。

---

## 12. 代码地图

实现目录：`../transformers/src/transformers/models/qwen4_exp/`

| 模块 | 类 | 文件 |
|---|---|---|
| 配置 | `Qwen4ExpConfig` / `TextConfig` / `VisionConfig` | `modular_qwen4_exp.py` |
| GR | `Qwen4ExpTextGatedResidual` | 同上 |
| GDN | `Qwen4ExpTextGatedDeltaNet` ← `Qwen3_5GatedDeltaNet` | 同上 + `qwen3_5/modeling_qwen3_5.py` |
| QSA indexer | `Qwen4ExpTextQSAIndexer` | `modular_qwen4_exp.py` |
| QSA attention | `Qwen4ExpTextAttention` ← `Qwen3_5Attention` | 同上 |
| MoE | `Qwen4ExpTextSparseMoeBlock` ← `Qwen3NextSparseMoeBlock` | 同上 + `qwen3_next/` |
| N-gram / PLE | `Qwen4ExpTextNGramEmbedding`, `Qwen4ExpTextPLELayer` | `modular_qwen4_exp.py` |
| Decoder 层 | `Qwen4ExpTextDecoderLayer` | 同上 |
| 文本主干 | `Qwen4ExpTextModel` | 同上 |
| 多模态 | `Qwen4ExpModel`, `Qwen4ExpForConditionalGeneration` | 同上 |
| 视觉 | `Qwen4ExpVisionModel` ← `Qwen3_5MoeVisionModel` | `qwen3_5_moe/` |

生成后的 `modeling_qwen4_exp.py` 不要手改，改 `modular_qwen4_exp.py`。
