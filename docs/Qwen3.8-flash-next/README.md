# Qwen3.8-Flash-Next 结构笔记

对照官方架构图和 Hugging Face `transformers` 的 `qwen4_exp` 实现（`modular_qwen4_exp.py`）整理的模型结构笔记。Qwen3.8-Flash-Next 是 Qwen4 架构的提前开源预览。

当前覆盖到 **N-gram / PLE**；MoE、MTP、Vision 仍以总览文档为主，细节笔记待补。

## 文档

| 文档 | 内容 |
|------|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 总览：规格、Hybrid Block、GR / GDN / QSA / MoE / PLE / Vision |
| [notes/rms-norm.md](notes/rms-norm.md) | 普通 RMSNorm 与 Group RMSNorm |
| [notes/gr.md](notes/gr.md) | GR：结构、HC / mHC 对照、消融、信息路径、效率与稳定性实验 |
| [notes/gr-read-divide-by-4.md](notes/gr-read-divide-by-4.md) | GR Read 为什么除以 `hc_count` |
| [notes/gdn.md](notes/gdn.md) | Gated DeltaNet：递推机制、实现细节、hybrid 消融与 FlashQLA |
| [notes/qsa.md](notes/qsa.md) | QSA：Indexer、两阶段 KL、质量/效率消融及 CSA/HCA 对照 |
| [notes/ple.md](notes/ple.md) | PLE：N-gram 表与注入实现、layer placement 和 vocabulary scaling |

实现对照：`transformers/src/transformers/models/qwen4_exp/`（以及继承的 `qwen3_5` / `qwen3_next`）。
