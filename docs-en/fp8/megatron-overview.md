# Megatron Project Overview

This directory contains three NVIDIA Megatron-related libraries for large-scale Transformer model training.

## Component Relationships

```
Megatron-Bridge (High-level API)
    │
    ├── Megatron-LM/Core (Distributed training infrastructure)
    ├── TransformerEngine (FP8 acceleration kernels)
    └── HuggingFace Transformers (Model interconversion)
```

## Library Roles

| Directory | Role |
|-----------|------|
| `Megatron-LM/` | Core library providing distributed parallelism strategies (TP/PP/DP/CP/EP) and model components |
| `TransformerEngine/` | GPU-optimized kernels providing FP8 precision training acceleration |
| `Megatron-Bridge/` | High-level wrapper providing HF↔Megatron conversion, training recipes, and PEFT |

## Use Cases

- **Quickly training mainstream models** → Megatron-Bridge (has pre-configured recipes for 30+ models)
- **Custom models/parallelism strategies** → Megatron-LM/Core
- **HuggingFace model interconversion** → Megatron-Bridge

## Notes

- Megatron-Bridge internally vendors Megatron-LM (in the `3rdparty/` directory)
- All three libraries require Python 3.10+
- FP8 training requires Hopper/Ada/Blackwell GPUs

## FP8 Training Details

### Blockwise Recipe Quantization Scope

**Quantized operations** (GEMM/matrix multiply only):
- `Linear`, `LayerNormLinear`, `LayerNormMLP`, `GroupedLinear`
- Quantization targets: weights, activations, gradients

**Non-quantized operations**:
- Attention (DPA/MHA) - explicitly prohibited
- LayerNorm / RMSNorm
- Softmax

Reference code: `TransformerEngine/transformer_engine/common/recipe/__init__.py:364-365`

```python
assert (
    not self.fp8_dpa and not self.fp8_mha
), "FP8 attention is not supported for Float8BlockScaling."
```
