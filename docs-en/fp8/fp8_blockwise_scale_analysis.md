# vLLM FP8 Blockwise Scale Format Analysis

## Conclusion

In vLLM, FP8 blockwise scales use **FP32 format** by default. Only on the DeepGEMM path (Hopper/Blackwell GPUs), FP32 scales are converted to **UE8M0 (power-of-2) format** by default.

## Kernel Dispatch Priority

Dispatch order for `W8A8BlockFp8LinearOp.apply()`:

| Priority | Condition | Kernel | Scale Format |
|----------|-----------|--------|-------------|
| 1 | DeepGEMM available + bf16 output + N/K aligned | **DeepGEMM** | UE8M0 (default) or FP32 |
| 2 | CUTLASS block FP8 supported (SM90+) | **CUTLASS** | FP32, `use_ue8m0=False` |
| 3 | ROCm AITer available | **AITer** | — |
| 4 | None of the above | **Triton** | FP32, `use_ue8m0=False` |

## DeepGEMM Uses UE8M0 by Default

When both environment variables `VLLM_USE_DEEP_GEMM` (default 1) and `VLLM_USE_DEEP_GEMM_E8M0` (default 1) are True, the DeepGEMM path converts FP32 scales to power-of-2 format:

```python
# vllm/utils/deep_gemm.py
def _ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))
```

UE8M0 can be disabled via `VLLM_USE_DEEP_GEMM_E8M0=0`, in which case DeepGEMM is still used but scales remain in FP32.

## Relationship Between quantization_config.scale_fmt and DeepGEMM

`quantization_config.scale_fmt` is only read during model config loading (`vllm/transformers_utils/config.py:624-644`), and **only responds to the `"ue8m0"` value**:

```python
scale_fmt = quantization_config.get("scale_fmt", None)
if scale_fmt in ("ue8m0",):        # only matches "ue8m0"
    if not envs.is_set("VLLM_USE_DEEP_GEMM_E8M0"):
        os.environ["VLLM_USE_DEEP_GEMM_E8M0"] = "1"
```

**`scale_fmt="fp32"` does not affect DeepGEMM behavior**. Reasons:

1. `"fp32" in ("ue8m0",)` → False, the config loading code does nothing
2. At runtime, all DeepGEMM paths check whether to use UE8M0 via `is_deep_gemm_e8m0_used()`
3. That function only reads the `VLLM_USE_DEEP_GEMM_E8M0` environment variable (default True)
4. `scale_fmt` is not passed to any runtime kernel dispatch logic

| `scale_fmt` Value | Config Loading Action | `VLLM_USE_DEEP_GEMM_E8M0` | Actual DeepGEMM Behavior |
|---|---|---|---|
| `"ue8m0"` | Proactively sets env=1 | 1 | UE8M0 |
| `"fp32"` | **No-op** | 1 (default) | **Still UE8M0** |
| `None` | No-op | 1 (default) | Still UE8M0 |

To make DeepGEMM use FP32 scales, you must explicitly set `VLLM_USE_DEEP_GEMM_E8M0=0`.

## Key Code Paths

| Verification Point | File | Line |
|--------------------|------|------|
| Environment variable defaults | `vllm/envs.py` | :157-159 |
| E8M0 enablement check | `vllm/utils/deep_gemm.py` | :59-80 (`is_deep_gemm_e8m0_used()`) |
| ceil_to_ue8m0 conversion | `vllm/utils/deep_gemm.py` | :313 |
| Weight requantization to UE8M0 | `vllm/model_executor/layers/quantization/utils/fp8_utils.py` | :873-930 |
| apply dispatch to DeepGEMM | `vllm/model_executor/layers/quantization/utils/fp8_utils.py` | :252-255 |
| blockscale op dispatch (CUTLASS/Triton) | `vllm/model_executor/layers/quantization/utils/fp8_utils.py` | :375-409 |
| CUTLASS block FP8 support detection | `vllm/model_executor/layers/quantization/utils/w8a8_utils.py` | :54-61 |
| scale_fmt reading (config loading only) | `vllm/transformers_utils/config.py` | :624-644 |
