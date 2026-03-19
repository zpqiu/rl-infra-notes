# RL Infra Notes

> Deep source-code walkthroughs of LLM RL training infrastructure — async RL scheduling, weight synchronization, FP8 mixed-precision, MoE routing precision, and more.

LLM 强化学习训练基础设施的**源码级**深度分析笔记。不止于"是什么"，更关注"为什么这样设计"和"代码里实际怎么做的"。

## Why This Repo?

开源 RL 框架越来越多，但大部分文档只告诉你 API 怎么用。当你需要理解：

- Async RL 训练中 rollout 和 training 到底怎么调度的？
- 权重同步时推理引擎发生了什么？abort 还是 drain？
- FP8 训练到底量化了哪些算子？scale 格式是什么？
- MoE Router 在 bf16 下 topk 会出什么问题？

答案只在源码里。这个 repo 就是把"读源码"的过程结构化记录下来，附带代码位置、对比表和架构图。

## Notes

### Async RL Training

对比分析 SLIME 和 veRL 两个框架在异步 RL 训练中的设计选择，覆盖 [HuggingFace Async RL Survey](https://huggingface.co/blog/async-rl-training-landscape) 的 4 个核心维度：Rollout Buffer、权重同步、Staleness 管理、Partial Rollout。

| Note | Framework | Highlights |
|------|-----------|------------|
| [SLIME Async RL Walkthrough](docs/async-rl/slime-async-rl-walkthrough.md) | [THUDM/slime](https://github.com/THUDM/slime) | Double-buffer 调度、TIS + OPSM staleness 修正、abort + recycle 机制 |
| [veRL Async RL Walkthrough](docs/async-rl/verl-async-rl-walkthrough.md) | [volcengine/verl](https://github.com/volcengine/verl) | Bounded queue + backpressure、NCCL bucketed broadcast、MIS 多版本 IS、prefix continuation |

### FP8 Mixed-Precision Training & Inference

FP8 训练和推理中的量化范围、scale 格式、通信精度等细节分析。

| Note | Framework | Highlights |
|------|-----------|------------|
| [Megatron Overview](docs/fp8/megatron-overview.md) | Megatron-LM / Bridge / TE | 组件关系、FP8 Blockwise 量化范围 |
| [fp8_param_gather 详解](docs/fp8/fp8-param-gather.md) | Megatron-LM | FP8 all-gather 通信优化、参数更新流程对比 |
| [FP8 Blockwise Scale 分析](docs/fp8/fp8_blockwise_scale_analysis.md) | vLLM | DeepGEMM UE8M0 vs FP32 scale、kernel dispatch 优先级 |
| [MoE Router Dtype 分析](docs/fp8/megatron_moe_router_dtype_analysis.md) | Megatron-LM + vLLM | Router 全链路 dtype 追踪（训练 vs 推理）、bf16 topk 精度风险 |

## Frameworks Studied

| Framework | Focus |
|-----------|-------|
| [NVIDIA NeMo RL](https://github.com/NVIDIA-NeMo/RL) | RL training pipeline |
| [veRL](https://github.com/volcengine/verl) | Async RL, weight sync |
| [SLIME](https://github.com/THUDM/slime) | Async RL, TIS/OPSM |
| [Megatron-LM](https://github.com/NVIDIA/Megatron-LM) | Distributed training, FP8, MoE |
| [Megatron-Bridge](https://github.com/NVIDIA/Megatron-Bridge) | HF↔Megatron conversion |
| [TransformerEngine](https://github.com/NVIDIA/TransformerEngine) | FP8 kernels |
| [vLLM](https://github.com/vllm-project/vllm) | Inference, FP8, MoE routing |

## Contributing

欢迎提 Issue 讨论或补充分析。如果你发现笔记中的代码引用已过时（框架更新很快），也欢迎 PR 修正。

## License

[MIT](LICENSE)
