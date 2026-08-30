# Qwen3.8-Flash-Next 结构笔记

对照官方架构图和 Hugging Face `transformers` 的 `qwen4_exp` 实现（`modular_qwen4_exp.py`）整理的模型结构笔记。Qwen3.8-Flash-Next 是 Qwen4 架构的提前开源预览。

当前覆盖到 **QSA**；MoE、N-gram / PLE、MTP、Vision 仍以总览文档为主，细节笔记待补。

## 文档

| 文档 | 内容 |
|------|------|
| [ARCHITECTURE.md](ARCHITECTURE.md) | 总览：规格、Hybrid Block、GR / GDN / QSA / MoE / PLE / Vision |
| [notes/rms-norm.md](notes/rms-norm.md) | 普通 RMSNorm 与 Group RMSNorm |
| [notes/gr-read-divide-by-4.md](notes/gr-read-divide-by-4.md) | GR Read 为什么除以 `hc_count` |
| [notes/gdn.md](notes/gdn.md) | Gated DeltaNet：投影、短卷积、遗忘、\(\beta\)、外积、读取、输出门 |
| [notes/qsa.md](notes/qsa.md) | Qwen Sparse Attention：Indexer、RoPE、block 打分、Gated Attention |

实现对照：`transformers/src/transformers/models/qwen4_exp/`（以及继承的 `qwen3_5` / `qwen3_next`）。
