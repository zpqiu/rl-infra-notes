# fp8_param_gather Parameter Deep Dive

## Parameter Definition

```
--fp8-param-gather: Keep the compute param in fp8 (do not use any other intermediate
                    dtype) and perform the param all-gather in fp8.
```

Source: `Megatron-LM/megatron/training/arguments.py:1541-1543`

## Prerequisites

Can only be used with one of the following:
- `--use-distributed-optimizer`
- `--use-torch-fsdp2`
- `--use-megatron-fsdp`
- Inference mode

Source: `Megatron-LM/megatron/training/arguments.py:737-739`

## Enabled vs Disabled Comparison

| Aspect | fp8_param_gather=False (default) | fp8_param_gather=True |
|--------|----------------------------------|----------------------|
| Parameter storage format | BF16/FP16 | FP8 |
| All-gather communication precision | BF16 (16-bit) | FP8 (8-bit) |
| Communication volume | Baseline | Reduced by 50% |
| Quantization timing | Dynamic quantization on every forward | One-time quantization after optimizer step |
| Memory usage | Requires BF16 parameter copy | No extra copy needed |

## Parameter Update Flow

**fp8_param_gather = False (default):**
```
FP32 main_param (optimizer update)
        ↓ copy
BF16/FP16 model_param
        ↓ all-gather (BF16 precision)
Full parameters (for next forward)
        ↓ dynamic quantization
FP8 for computation
```

**fp8_param_gather = True:**
```
FP32 main_param (optimizer update)
        ↓ quantize_param_shard (quantization)
FP8 model_param (stored directly as FP8)
        ↓ all-gather (FP8 precision)
Full parameters (for next forward)
        ↓ no further quantization needed
Directly used for computation
```

## Impact on Compute Precision

**No direct impact on Forward/Backward compute precision.**

Reasons:
1. FP8 compute precision is controlled by `--fp8-format`, unaffected by `fp8_param_gather`
2. Regardless of this flag, TransformerEngine's GEMM computations execute at FP8 precision
3. Optimizer states (main_param) always remain in FP32, which is the core guarantee of mixed-precision training

## Impact on Parameter Updates

`fp8_param_gather` affects the conversion path from FP32 main_param to model_param:

- **Disabled**: FP32 → BF16 → store → all-gather → dynamic quantization to FP8 for computation
- **Enabled**: FP32 → directly quantize to FP8 → store → all-gather → directly used for computation

Related code: `Megatron-LM/megatron/core/optimizer/distrib_optimizer.py:2430-2432`

```python
quantize_param_shard(
    *self._get_fp8_params_and_shard_fp32_from_fp8(), self.data_parallel_group
)
```

## Recommended Configuration

For MXFP8 training, the official recommendation is to enable both:
- `--fp8-param-gather`
- `--reuse-grad-buf-for-mxfp8-param-ag`

Otherwise you will see a warning:
```
mxfp8 without using reuse_grad_buf_for_mxfp8_param_ag and fp8_param_gather
will use significant amount additional GPU memory.
```

Source: `Megatron-LM/megatron/core/optimizer/optimizer_config.py:369-372`

## Caveats

- FSDP2 + TE 2.0.0 currently does not support FP8 param gather and will automatically fall back to BF16
- Source: `Megatron-LM/megatron/training/arguments.py:707-711`

---

*Last updated: 2026-02-03*
