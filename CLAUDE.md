# Megatron 项目概览

本目录包含三个 NVIDIA Megatron 相关库，用于大规模 Transformer 模型训练。

## 组件关系

```
Megatron-Bridge (高层API)
    │
    ├── Megatron-LM/Core (分布式训练基础设施)
    ├── TransformerEngine (FP8加速内核)
    └── HuggingFace Transformers (模型互转)
```

## 各库定位

| 目录 | 定位 |
|------|------|
| `Megatron-LM/` | 核心库，提供分布式并行策略(TP/PP/DP/CP/EP)和模型组件 |
| `TransformerEngine/` | GPU优化内核，提供FP8精度训练加速 |
| `Megatron-Bridge/` | 高层封装，提供HF↔Megatron转换、训练recipes、PEFT |

## 使用场景

- **快速训练主流模型** → Megatron-Bridge（有30+模型的预配置recipes）
- **自定义模型/并行策略** → Megatron-LM/Core
- **HuggingFace模型互转** → Megatron-Bridge

## 注意事项

- Megatron-Bridge 内部 vendor 了 Megatron-LM（在 `3rdparty/` 目录）
- 三个库都需要 Python 3.10+
- FP8 训练需要 Hopper/Ada/Blackwell GPU

## FP8 训练细节

### Blockwise Recipe 量化范围

**量化的操作**（仅 GEMM/矩阵乘法）：
- `Linear`, `LayerNormLinear`, `LayerNormMLP`, `GroupedLinear`
- 量化对象：权重、激活、梯度

**不量化的操作**：
- Attention (DPA/MHA) - 显式禁止
- LayerNorm / RMSNorm
- Softmax

参考代码：`TransformerEngine/transformer_engine/common/recipe/__init__.py:364-365`

```python
assert (
    not self.fp8_dpa and not self.fp8_mha
), "FP8 attention is not supported for Float8BlockScaling."
```
