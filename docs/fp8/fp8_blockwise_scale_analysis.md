# vLLM FP8 Blockwise Scale 格式分析

## 结论

vLLM 中 FP8 blockwise scale 默认使用 **FP32 格式**。仅在 DeepGEMM 路径下（Hopper/Blackwell GPU），会默认将 FP32 scale 转换为 **UE8M0（power-of-2）格式**。

## Kernel Dispatch 优先级

`W8A8BlockFp8LinearOp.apply()` 的 dispatch 顺序：

| 优先级 | 条件 | Kernel | Scale 格式 |
|--------|------|--------|-----------|
| 1 | DeepGEMM 可用 + bf16 输出 + N/K 对齐 | **DeepGEMM** | UE8M0（默认）或 FP32 |
| 2 | CUTLASS block FP8 支持（SM90+） | **CUTLASS** | FP32, `use_ue8m0=False` |
| 3 | ROCm AITer 可用 | **AITer** | — |
| 4 | 以上都不满足 | **Triton** | FP32, `use_ue8m0=False` |

## DeepGEMM 默认使用 UE8M0

环境变量 `VLLM_USE_DEEP_GEMM`（默认1）和 `VLLM_USE_DEEP_GEMM_E8M0`（默认1）同时为 True 时，DeepGEMM 路径会将 FP32 scale 转换为 power-of-2 格式：

```python
# vllm/utils/deep_gemm.py
def _ceil_to_ue8m0(x: torch.Tensor):
    return torch.pow(2.0, torch.ceil(torch.log2(x.abs())))
```

可通过 `VLLM_USE_DEEP_GEMM_E8M0=0` 禁用 UE8M0，此时 DeepGEMM 仍使用但 scale 保持 FP32。

## quantization_config.scale_fmt 与 DeepGEMM 的关系

`quantization_config.scale_fmt` 仅在模型 config 加载时被读取（`vllm/transformers_utils/config.py:624-644`），且**只对 `"ue8m0"` 值有响应**：

```python
scale_fmt = quantization_config.get("scale_fmt", None)
if scale_fmt in ("ue8m0",):        # 只匹配 "ue8m0"
    if not envs.is_set("VLLM_USE_DEEP_GEMM_E8M0"):
        os.environ["VLLM_USE_DEEP_GEMM_E8M0"] = "1"
```

**`scale_fmt="fp32"` 不会影响 DeepGEMM 行为**。原因：

1. `"fp32" in ("ue8m0",)` → False，config 加载代码什么都不做
2. 运行时所有 DeepGEMM 路径通过 `is_deep_gemm_e8m0_used()` 判断是否用 UE8M0
3. 该函数只读 `VLLM_USE_DEEP_GEMM_E8M0` 环境变量（默认 True）
4. `scale_fmt` 不会传递到任何运行时 kernel dispatch 逻辑

| `scale_fmt` 值 | config 加载动作 | `VLLM_USE_DEEP_GEMM_E8M0` | DeepGEMM 实际行为 |
|---|---|---|---|
| `"ue8m0"` | 主动设 env=1 | 1 | UE8M0 |
| `"fp32"` | **无操作** | 1（默认） | **仍然 UE8M0** |
| `None` | 无操作 | 1（默认） | 仍然 UE8M0 |

要让 DeepGEMM 使用 FP32 scale，必须显式设置 `VLLM_USE_DEEP_GEMM_E8M0=0`。

## 关键代码路径

| 验证点 | 文件 | 行号 |
|--------|------|------|
| 环境变量默认值 | `vllm/envs.py` | :157-159 |
| E8M0 启用判断 | `vllm/utils/deep_gemm.py` | :59-80 (`is_deep_gemm_e8m0_used()`) |
| ceil_to_ue8m0 转换 | `vllm/utils/deep_gemm.py` | :313 |
| 权重 requant 到 UE8M0 | `vllm/model_executor/layers/quantization/utils/fp8_utils.py` | :873-930 |
| apply dispatch 到 DeepGEMM | `vllm/model_executor/layers/quantization/utils/fp8_utils.py` | :252-255 |
| blockscale op dispatch（CUTLASS/Triton） | `vllm/model_executor/layers/quantization/utils/fp8_utils.py` | :375-409 |
| CUTLASS block FP8 支持检测 | `vllm/model_executor/layers/quantization/utils/w8a8_utils.py` | :54-61 |
| scale_fmt 读取（仅 config 加载） | `vllm/transformers_utils/config.py` | :624-644 |
