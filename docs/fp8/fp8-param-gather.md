# fp8_param_gather 参数详解

## 参数定义

```
--fp8-param-gather: Keep the compute param in fp8 (do not use any other intermediate
                    dtype) and perform the param all-gather in fp8.
```

来源: `Megatron-LM/megatron/training/arguments.py:1541-1543`

## 使用前提条件

只能与以下之一配合使用：
- `--use-distributed-optimizer`
- `--use-torch-fsdp2`
- `--use-megatron-fsdp`
- 推理模式

来源: `Megatron-LM/megatron/training/arguments.py:737-739`

## 开启 vs 关闭对比

| 方面 | fp8_param_gather=False (默认) | fp8_param_gather=True |
|------|------------------------------|----------------------|
| 参数存储格式 | BF16/FP16 | FP8 |
| All-gather 通信精度 | BF16 (16-bit) | FP8 (8-bit) |
| 通信量 | 基准 | 减少 50% |
| 量化时机 | 每次 forward 动态量化 | 优化器步骤后一次量化 |
| 显存占用 | 需要 BF16 参数副本 | 无需额外副本 |

## 参数更新流程

**fp8_param_gather = False（默认）：**
```
FP32 main_param (优化器更新)
        ↓ copy
BF16/FP16 model_param
        ↓ all-gather（BF16精度）
完整参数（用于下一次forward）
        ↓ 动态量化
FP8 用于计算
```

**fp8_param_gather = True：**
```
FP32 main_param (优化器更新)
        ↓ quantize_param_shard（量化）
FP8 model_param（直接存储为FP8）
        ↓ all-gather（FP8精度）
完整参数（用于下一次forward）
        ↓ 无需再量化
直接用于计算
```

## 对计算精度的影响

**对 Forward/Backward 计算精度没有直接影响。**

原因：
1. FP8 计算精度由 `--fp8-format` 控制，不受 `fp8_param_gather` 影响
2. 无论该 flag 开启与否，TransformerEngine 的 GEMM 计算都在 FP8 精度下执行
3. 优化器状态（main_param）始终保持 FP32，这是混合精度训练的核心保证

## 对参数更新的影响

`fp8_param_gather` 影响的是从 FP32 main_param 到 model_param 的转换路径：

- **关闭时**: FP32 → BF16 → 存储 → all-gather → 动态量化到 FP8 用于计算
- **开启时**: FP32 → 直接量化到 FP8 → 存储 → all-gather → 直接用于计算

相关代码: `Megatron-LM/megatron/core/optimizer/distrib_optimizer.py:2430-2432`

```python
quantize_param_shard(
    *self._get_fp8_params_and_shard_fp32_from_fp8(), self.data_parallel_group
)
```

## 推荐配置

对于 MXFP8 训练，官方建议同时开启：
- `--fp8-param-gather`
- `--reuse-grad-buf-for-mxfp8-param-ag`

否则会有警告：
```
mxfp8 without using reuse_grad_buf_for_mxfp8_param_ag and fp8_param_gather
will use significant amount additional GPU memory.
```

来源: `Megatron-LM/megatron/core/optimizer/optimizer_config.py:369-372`

## 注意事项

- FSDP2 + TE 2.0.0 目前不支持 FP8 param gather，会自动回退到 BF16
- 来源: `Megatron-LM/megatron/training/arguments.py:707-711`

---

*最后更新: 2026-02-03*
