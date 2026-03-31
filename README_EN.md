# RL Infra Notes

English | [中文](README.md)

> Deep source-code walkthroughs of LLM RL training infrastructure — async RL scheduling, weight synchronization, FP8 mixed-precision, MoE routing precision, and more.

**Source-level** deep-dive notes on LLM reinforcement learning training infrastructure. Goes beyond "what it is" to focus on "why it's designed this way" and "what the code actually does."

## Why This Repo?

There are more and more open-source RL frameworks, but most documentation only tells you how to use the API. When you need to understand:

- How exactly are rollout and training scheduled in async RL training?
- What happens to the inference engine during weight synchronization? Abort or drain?
- Which operators does FP8 training actually quantize? What's the scale format?
- What problems arise with bf16 topk in MoE Routers?

The answers are only in the source code. This repo is a structured record of the "reading source code" process, with code locations, comparison tables, and architecture diagrams.

## Notes

### Async RL Training

Comparative analysis of three frameworks' design choices in async RL training, covering 4 core dimensions from the [HuggingFace Async RL Survey](https://huggingface.co/blog/async-rl-training-landscape): Rollout Buffer, Weight Synchronization, Staleness Management, and Partial Rollout.

| Note | Framework | Highlights |
|------|-----------|------------|
| [SLIME Async RL Walkthrough](docs-en/async-rl/slime-async-rl-walkthrough.md) | [THUDM/slime](https://github.com/THUDM/slime) | Double-buffer scheduling, TIS + OPSM staleness correction, abort + recycle mechanism |
| [veRL Async RL Walkthrough](docs-en/async-rl/verl-async-rl-walkthrough.md) | [volcengine/verl](https://github.com/volcengine/verl) | Bounded queue + backpressure, NCCL bucketed broadcast, MIS multi-version IS, prefix continuation |
| [NeMo-RL Async RL Walkthrough](docs-en/async-rl/nemo-rl-async-rl-walkthrough.md) | [NVIDIA/NeMo-RL](https://github.com/NVIDIA/NeMo-RL) | Replay Buffer + target weight matching, in-flight weight update, TIS / ICE-POP / seq-mask-TIS |
### FP8 Mixed-Precision Training & Inference

Detailed analysis of quantization scope, scale formats, and communication precision in FP8 training and inference.

| Note | Framework | Highlights |
|------|-----------|------------|
| [Megatron Overview](docs-en/fp8/megatron-overview.md) | Megatron-LM / Bridge / TE | Component relationships, FP8 Blockwise quantization scope |
| [fp8_param_gather Deep Dive](docs-en/fp8/fp8-param-gather.md) | Megatron-LM | FP8 all-gather communication optimization, parameter update flow comparison |
| [FP8 Blockwise Scale Analysis](docs-en/fp8/fp8_blockwise_scale_analysis.md) | vLLM | DeepGEMM UE8M0 vs FP32 scale, kernel dispatch priority |
| [MoE Router Dtype Analysis](docs-en/fp8/megatron_moe_router_dtype_analysis.md) | Megatron-LM + vLLM | End-to-end router dtype tracing (training vs inference), bf16 topk precision risks |

## Frameworks Studied

| Framework | Focus |
|-----------|-------|
| [NVIDIA NeMo-RL](https://github.com/NVIDIA/NeMo-RL) | RL training pipeline, async GRPO |
| [veRL](https://github.com/volcengine/verl) | Async RL, weight sync |
| [SLIME](https://github.com/THUDM/slime) | Async RL, TIS/OPSM |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | Distributed training, FP8, MoE |
| [Megatron-Bridge](https://github.com/NVIDIA/Megatron-Bridge) | HF↔Megatron conversion |
| [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) | FP8 kernels |
| [vLLM](https://github.com/vllm-project/vllm) | Inference, FP8, MoE routing |

## Contributing

Issues for discussion or supplementary analysis are welcome. If you find that code references in the notes are outdated (frameworks update quickly), PRs to fix them are also appreciated.

## License

[MIT](LICENSE)
