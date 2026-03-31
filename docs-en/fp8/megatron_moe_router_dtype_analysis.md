# MoE Router Dtype Analysis (Qwen3 MoE) — Training & Inference

> **Analysis based on the following versions:**
>
> | Library | Version / Tag | Commit | Date |
> |---|---|---|---|
> | vLLM | v0.12.0 | `4fd9d6a8` | 2025-12-02 |
> | Megatron-LM (Core) | core_v0.15.0rc7+688 | `b47c376f` | 2026-01-27 |
> | Megatron-Bridge | v0.2.0rc6+477 | `56200ade` | 2026-01-30 |

---

# Part I — Megatron Training Side

## 1. Key Files

| Component | File Path | Key Lines |
|------|----------|--------|
| Router Base Class | `Megatron-LM/megatron/core/transformer/moe/router.py` | L28-101 |
| TopKRouter | `Megatron-LM/megatron/core/transformer/moe/router.py` | L500-562 |
| Router GEMM (autograd) | `Megatron-LM/megatron/core/transformer/moe/moe_utils.py` | L1099-1199 |
| topk + softmax/sigmoid | `Megatron-LM/megatron/core/transformer/moe/moe_utils.py` | L561-679 |
| unpermute (probs weighting) | `Megatron-LM/megatron/core/transformer/moe/moe_utils.py` | L333-410 |
| MoE Layer Call Flow | `Megatron-LM/megatron/core/transformer/moe/moe_layer.py` | L357-409 |
| AlltoAll Dispatcher | `Megatron-LM/megatron/core/transformer/moe/token_dispatcher.py` | L592-849 |
| GroupedMLP Expert | `Megatron-LM/megatron/core/transformer/moe/experts.py` | L65-278 |
| TransformerConfig | `Megatron-LM/megatron/core/transformer/transformer_config.py` | `moe_router_dtype` |
| Qwen3 MoE Config | `Megatron-Bridge/src/megatron/bridge/models/qwen/qwen_provider.py` | L362-396 |
| Qwen3Next Config | `Megatron-Bridge/src/megatron/bridge/models/qwen/qwen_provider.py` | L433-454 |
| Weight Mapping | `Megatron-Bridge/src/megatron/bridge/models/qwen/qwen3_moe_bridge.py` | L86 |

## 2. Router Weight Initialization

```python
# router.py:53-54 — Always created in fp32
self.weight = torch.nn.Parameter(
    torch.empty((self.config.num_moe_experts, self.config.hidden_size), dtype=torch.float32)
)

# router.py:67-77 — reset_parameters: Convert to params_dtype after initialization
def reset_parameters(self):
    if self.config.perform_initialization:
        self.config.init_method(self.weight)          # Execute init_method in fp32
    self.weight.data = self.weight.data.to(dtype=self.config.params_dtype)  # Convert to bf16
```

- Qwen3 MoE: `params_dtype=bf16`, `add_bias_linear=False` (no bias)
- Weights are ultimately stored as **bf16**, unaffected by FP8 quantization (FP8 only applies to expert FFN)

## 3. Router Gating (dtype Decision Logic)

```python
# router.py:79-101
def gating(self, input: torch.Tensor):
    router_dtype = input.dtype                          # Default: follows input dtype
    if self.config.moe_router_dtype == 'fp32':
        router_dtype = torch.float32                    # Explicit override
    elif self.config.moe_router_dtype == 'fp64':
        router_dtype = torch.float64
    logits = router_gating_linear(input, self.weight, self.bias, router_dtype)
    return logits
```

`router_dtype` priority: `moe_router_dtype` config > `input.dtype`

## 4. Router GEMM (RouterGatingLinearFunction)

```python
# moe_utils.py:1105-1142
class RouterGatingLinearFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, inp, weight, bias, router_dtype):
        ctx.router_dtype = router_dtype
        ctx.input_dtype = inp.dtype          # Save original dtype for backward
        ctx.weight_dtype = weight.dtype

        if te_general_gemm is not None and router_dtype != torch.float64:
            output = te_general_gemm(weight, inp, router_dtype, layout="TN", bias=bias)
        elif bias is None:
            output = torch.mm(inp.to(router_dtype), weight.to(router_dtype).t())
        else:
            output = torch.addmm(bias.to(router_dtype), inp.to(router_dtype), weight.to(router_dtype).t())

        return output    # dtype = router_dtype, not converted back to params_dtype
```

Key points:
- Input and weights are **explicitly cast to `router_dtype`** before GEMM
- `torch.mm` output dtype = dtype of both operands = `router_dtype`
- **Forward output is not converted back** — logits are returned directly in `router_dtype`

Gradients are **converted back to original dtype** in backward:
```python
# moe_utils.py:1164-1175
grad_input = grad_input[0].to(ctx.input_dtype)    # Convert back to input's dtype
grad_weight = grad_weight[0].to(ctx.weight_dtype)  # Convert back to weight's dtype
```

## 5. TopK + Softmax

```python
# moe_utils.py:662-668
# Qwen3 MoE: score_function="softmax", use_pre_softmax=False
if score_function == "softmax":
    if use_pre_softmax:
        scores = torch.softmax(logits, dim=-1, dtype=torch.float32).type_as(logits)
        probs, top_indices = compute_topk(scores, topk, ...)
    else:  # ← Qwen3 MoE takes this branch
        scores, top_indices = compute_topk(logits, topk, ...)           # topk executed on logits' original dtype
        probs = torch.softmax(scores, dim=-1, dtype=torch.float32)      # softmax forced to fp32
                      .type_as(logits)                                   # convert back to logits' dtype
```

The effect of `.type_as(logits)` depends on the dtype of logits:
- `moe_router_dtype=None` → logits=bf16 → probs **truncated back to bf16**
- `moe_router_dtype='fp32'` → logits=fp32 → probs **remain fp32**

## 6. Probs Weighting Inside Experts (GroupedMLP)

```python
# experts.py:112-115 — activation_func_with_probs
def activation_func_with_probs(x, probs):
    dtype = x.dtype                            # fc1_output's dtype = bf16
    res = self.activation_func(x) * probs      # bf16 * probs_dtype → broadcast promotion
    return res.to(dtype)                        # convert back to bf16

# experts.py:256-258 — Called between fc1 and fc2
intermediate_parallel = self.activation_func_with_probs(
    fc1_output, permuted_probs.unsqueeze(-1)   # probs weighting happens here
)
fc2_output = gg.ops.gmm(intermediate_parallel, w2, ...)  # input is already bf16
```

## 7. Token Combine (AlltoAll Dispatcher)

```python
# token_dispatcher.py:779-783 — combine_preprocess (only executed when TP>1)
hidden_states = reduce_scatter_to_sequence_parallel_region(
    hidden_states.to(self.probs.dtype),       # convert to probs' dtype for reduction
    group=self.tp_group,
).to(hidden_states.dtype)                      # convert back to hidden's dtype

# token_dispatcher.py:833-840 — combine_postprocess
output = unpermute(
    permutated_local_input_tokens,
    self.reversed_local_input_permutation_mapping,
    restore_shape=self.hidden_shape_before_permute,
    routing_map=self.routing_map,
    # Note: **no probs argument passed here**, no weighting is done
    # AlltoAll's probs weighting is done inside experts (see Section 6)
)
```

## 8. Complete Dtype Comparison for Both Paths

Qwen3 MoE config: `params_dtype=bf16`, `pre_softmax=False`, `score_function="softmax"`, dispatcher=AlltoAll, experts=GroupedMLP, `topk=8`

| Stage | Code Location | `moe_router_dtype=None` (default) | `moe_router_dtype='fp32'` |
|------|----------|------|------|
| weight storage | `router.py:73` | bf16 | bf16 |
| router_dtype decision | `router.py:95-99` | `input.dtype` → bf16 | `torch.float32` → fp32 |
| Router GEMM | `moe_utils.py:1134-1135` | bf16 × bf16 → **bf16** | bf16→fp32 × bf16→fp32 → **fp32** |
| logits | | **bf16** | **fp32** |
| topk selection | `moe_utils.py:667` | topk on **bf16** | topk on **fp32** |
| softmax computation | `moe_utils.py:668` | `softmax(dtype=fp32)` → fp32 | `softmax(dtype=fp32)` → fp32 |
| `.type_as(logits)` | `moe_utils.py:668` | fp32 → **bf16** (truncation) | fp32 → **fp32** (lossless) |
| probs output | | **bf16** | **fp32** |
| permute + AlltoAll | `token_dispatcher.py:637-671` | probs remain **bf16** | probs remain **fp32** |
| Expert weighting | `experts.py:114` | `act(x_bf16) * probs_bf16` → **bf16** | `act(x_bf16) * probs_fp32` → **fp32** |
| `.to(dtype)` convert back | `experts.py:115` | → **bf16** | fp32 → **bf16** |
| fc2 GEMM input | `experts.py:259` | **bf16** | **bf16** |
| reduce_scatter precision | `token_dispatcher.py:779-783` | `hidden.to(bf16)` → **bf16** reduction | `hidden.to(fp32)` → **fp32** reduction |
| reduce_scatter convert back | `token_dispatcher.py:779-783` | → **bf16** | fp32 → **bf16** |
| unpermute (combine) | `token_dispatcher.py:833-840` | no probs weighting, **bf16** | no probs weighting, **bf16** |
| final output | | **bf16** | **bf16** |

## 9. Core Differences Summary

1. **TopK precision**: The default path performs topk on bf16 (~3.3 mantissa bits). For 128/512 experts, close logit values may lead to unstable expert selection; the fp32 path (~7.2 mantissa bits) is more reliable.

2. **Implicit truncation from `.type_as(logits)`**: The same line of code `softmax(..., dtype=fp32).type_as(logits)` truncates the fp32 softmax result back to bf16 in the default path, losing the precision advantage of softmax.

3. **Expert weighting precision**: In the fp32 path, the `activation(x) * probs` multiplication executes in fp32 (broadcast promotion). Although the final `.to(dtype)` converts back to bf16, the multiplication itself has higher precision.

4. **Communication reduction precision**: The fp32 path accumulates in fp32 during reduce_scatter before converting back to bf16, avoiding bf16 precision loss during multi-GPU reduction.

5. **Final output is consistent**: Both paths produce bf16 final output; the differences only affect the precision of intermediate routing computations.

## 10. Qwen3 Variant Configuration Comparison

```python
# qwen_provider.py:362-396
class Qwen3MoEModelProvider:
    num_moe_experts: int = 128
    moe_router_topk: int = 8
    # moe_router_dtype not set → None → defaults to bf16 path

# qwen_provider.py:433-454
class Qwen3NextModelProvider(Qwen3MoEModelProvider):
    num_moe_experts: int = 512
    moe_router_topk: int = 10
    moe_router_dtype: str = "fp32"    # Explicit fp32, 512 experts need higher precision
```

The Qwen3Next 512-expert variant explicitly enables `moe_router_dtype='fp32'`, indicating that the NVIDIA team considers bf16 routing precision insufficient for a large number of experts.

---

# Part II — vLLM Inference Side

## 11. Key Files

| Component | File Path | Key Lines |
|------|----------|--------|
| Qwen3 MoE Block | `vllm/model_executor/models/qwen3_moe.py` | L121-210 |
| ReplicatedLinear (gate) | `vllm/model_executor/layers/linear.py` | L296-367 |
| LinearBase (params_dtype) | `vllm/model_executor/layers/linear.py` | L243-279 |
| UnquantizedLinearMethod | `vllm/model_executor/layers/linear.py` | L196-240 |
| Gate GEMM Actual Call | `vllm/model_executor/layers/utils.py` | L99-105 |
| FusedMoE.select_experts | `vllm/model_executor/layers/fused_moe/layer.py` | L1519-1636 |
| fused_topk (Python) | `vllm/model_executor/layers/fused_moe/fused_moe.py` | L1101-1130 |
| topk_softmax (C++ entry) | `vllm/_custom_ops.py` | L1977-1986 |
| topk_softmax (CUDA kernel) | `csrc/moe/topk_softmax_kernels.cu` | L675-705 |
| FusedMoE.forward_impl | `vllm/model_executor/layers/fused_moe/layer.py` | L1889-1985 |

## 12. Router Module Initialization

```python
# qwen3_moe.py:178-184
self.gate = ReplicatedLinear(
    config.hidden_size,      # e.g. 2560
    config.num_experts,      # e.g. 128
    bias=False,
    quant_config=quant_config,  # ← passes the model's quant_config!
    prefix=f"{prefix}.gate",
)
```

**No explicit `params_dtype` specified**, falls through to `LinearBase.__init__` default logic:

```python
# linear.py:275-276
if params_dtype is None:
    params_dtype = torch.get_default_dtype()  # vllm sets this based on --dtype, typically bf16/fp16
```

**Key difference from other MoE models**: Qwen3 MoE **passes `quant_config` to the gate**, whereas most models explicitly pass `quant_config=None`:

| Model | gate's `quant_config` | gate's `params_dtype` |
|------|----------------------|----------------------|
| **Qwen3 MoE** | **`quant_config` (follows model)** | default (bf16) |
| Qwen2 MoE | `None` | default |
| DeepSeek-V2 | `None` | default |
| Mixtral | `None` | explicitly specified |
| MiniMax-Text-01 | `None` | **`torch.float32`** |
| Nemotron-H | `None` | **`torch.float32`** |
| ERNIE 4.5 MoE | `None` | **`torch.float32`** |

## 13. Gate GEMM Computation

**Forward path**:

```python
# qwen3_moe.py:198
router_logits, _ = self.gate(hidden_states)   # hidden_states: [num_tokens, hidden_dim]
```

For non-quantized scenarios, the final call is:

```python
# linear.py:240 → utils.py:99-105
def default_unquantized_gemm(layer, x, weight, bias):
    return torch.nn.functional.linear(x, weight, bias)
    # x: bf16, weight: bf16 → output: bf16
```

**No dtype cast logic whatsoever** — unlike Megatron's `RouterGatingLinearFunction`, vLLM uses a plain `F.linear`: input dtype in, same dtype out.

## 14. TopK + Softmax

Router logits enter `FusedMoE.select_experts` (`layer.py:1609-1616`), Qwen3 MoE takes the `fused_topk` path:

```python
# layer.py:1610-1616 — no grouped_topk, no bias correction
topk_weights, topk_ids, token_expert_indices = fused_topk(
    hidden_states=hidden_states,
    gating_output=router_logits,   # bf16 (from gate GEMM)
    topk=self.top_k,              # 8
    renormalize=self.renormalize,  # config.norm_topk_prob
)
```

Inside `fused_topk` (`fused_moe.py:1101-1130`):

```python
# Output topk_weights is always allocated as FP32
topk_weights = torch.empty(M, topk, dtype=torch.float32, ...)
topk_ids = torch.empty(M, topk, dtype=torch.int32, ...)

# Calls CUDA kernel — gating_output can be float/half/bf16
ops.topk_softmax(topk_weights, topk_ids, token_expert_indices,
                 gating_output, renormalize)
```

## 15. topk_softmax CUDA Kernel Internals

`csrc/moe/topk_softmax_kernels.cu:675-705` dispatches by input dtype:

```cpp
if (gating_output.scalar_type() == at::ScalarType::Float) {
    dispatch_topk_softmax_launch<float>(...);
} else if (gating_output.scalar_type() == at::ScalarType::Half) {
    dispatch_topk_softmax_launch<__half>(...);
} else if (gating_output.scalar_type() == at::ScalarType::BFloat16) {
    dispatch_topk_softmax_launch<__nv_bfloat16>(...);
}
```

Inside the kernel, **all softmax/topk computations are first converted to FP32**:

```cpp
// topk_softmax_kernels.cu:55-63
template <typename T>
__device__ __forceinline__ float toFloat(T value) {
    if constexpr (std::is_same_v<T, float>)          return value;
    else if constexpr (std::is_same_v<T, __nv_bfloat16>) return __bfloat162float(value);
    else if constexpr (std::is_same_v<T, __half>)        return __half2float(value);
}

// topk_softmax_kernels.cu:91,107,122 — immediately convert to float after reading gating_output
const float val = toFloat(input[idx]);
```

The output `topk_weights` is FP32 (`torch.float32`).

## 16. Routing Method

```python
# qwen3_moe.py:175
routing_method_type=RoutingMethodType.Renormalize
```

Semantics: First select experts via TopK, then apply softmax normalization on the selected weights. This is done in one step inside the `topk_softmax` CUDA kernel.

## 17. vLLM Inference Side Complete Dtype Flow

```
hidden_states (BF16, from previous layer)
    │
    ▼
[Gate GEMM]  F.linear(hidden_states, gate_weight)
    │  weight: bf16 (non-quantized) / possibly fp8 (if quant_config didn't skip gate)
    │  computation: bf16 × bf16 → bf16
    ▼
router_logits (BF16)              ← precision bottleneck
    │
    ▼
[topk_softmax CUDA kernel]
    │  internal: toFloat() convert to fp32 → softmax fp32 → topk fp32
    ▼
topk_weights (FP32), topk_ids (INT32)
    │
    ▼
[FusedMoE Expert GEMM]  Triton kernel, FP32 accumulation
    ▼
final_hidden_states (BF16)
```

## 18. Configurations Affecting Router Dtype

| Config Item | Scope | Effect |
|--------|--------|------|
| `--dtype` (vllm launch parameter) | `torch.get_default_dtype()` | Determines gate weight and GEMM dtype |
| `--quantization` | quant_config | If FP8 and gate is not excluded, gate **may be quantized** |
| quant config's `ignored_layers` | excluded by prefix | Can explicitly exclude gate layer from quantization |
| `config.norm_topk_prob` | renormalize | Controls whether topk_softmax performs normalization |

---

# Part III — Training vs Inference Comparison

## 19. Full Comparison Table

Qwen3 MoE (128 experts, topk=8, `params_dtype=bf16`), comparing Megatron training (default `moe_router_dtype=None`) and vLLM inference (default config).

| Stage | Megatron Training (default) | Megatron Training (`moe_router_dtype='fp32'`) | vLLM Inference (default) |
|------|------|------|------|
| **Gate Weight Storage** | bf16 | bf16 | bf16 |
| **Gate Quantization Protection** | N/A (FP8 does not affect router weights) | N/A | **None** (`quant_config` passed through, FP8 may quantize gate) |
| **Router GEMM Method** | `RouterGatingLinearFunction` (custom autograd) | same | `F.linear` (plain PyTorch) |
| **Pre-GEMM dtype cast** | explicit `.to(router_dtype)` | explicit `.to(fp32)` | **no cast** |
| **GEMM Computation dtype** | **bf16** | **fp32** | **bf16** |
| **logits dtype** | **bf16** | **fp32** | **bf16** |
| **TopK Selection dtype** | **bf16** | **fp32** | **bf16** (converted to fp32 inside kernel before selection) |
| **Softmax Computation** | `softmax(dtype=fp32)` → fp32 | `softmax(dtype=fp32)` → fp32 | fp32 inside CUDA kernel |
| **Softmax → probs Conversion** | `.type_as(logits)` → **bf16 truncation** | `.type_as(logits)` → **fp32 lossless** | directly outputs **fp32** |
| **topk_weights Final dtype** | **bf16** | **fp32** | **fp32** |
| **Expert Weighting Location** | inside expert (post-fc1 `act(x) * probs`) | same | inside FusedMoE Triton kernel |
| **Expert Weighting Precision** | bf16 × bf16 → **bf16** | bf16 × fp32 → **fp32** → bf16 | fp32 weights enter kernel |
| **Router dtype Config** | `TransformerConfig.moe_router_dtype` | same | **no equivalent config** |
| **Final Output** | bf16 | bf16 | bf16 |

## 20. Key Differences Analysis

### 20.1 Gate GEMM Precision Control

- **Megatron** has a dedicated `RouterGatingLinearFunction` that **explicitly casts** input and weights to `router_dtype` before GEMM, providing the ability to promote from bf16 to fp32.
- **vLLM** uses a plain `F.linear` with **no dtype cast logic whatsoever**. Gate GEMM always executes in the original dtype of weights/input.

### 20.2 Gate Protection Under Quantization

- **Megatron**'s router is an independent module (`Router` class). FP8 quantization only applies to the `GroupedMLP` expert FFN, so router weights are **inherently unaffected**.
- **vLLM**'s Qwen3 MoE **passes `quant_config` through to the gate** (`qwen3_moe.py:182`). Under FP8 quantization scenarios, the gate layer may be quantized (depending on whether the specific quant method handles `ReplicatedLinear`). Most other models (Qwen2-MoE, DeepSeek-V2, Mixtral, etc.) explicitly pass `quant_config=None` to avoid this issue.

### 20.3 TopK Selection Precision

- **Megatron default**: TopK is executed on bf16 logits. For 128 experts, the logit differences may be smoothed out by bf16's low mantissa precision (~3.3 bits).
- **vLLM**: Although logits are also bf16, the CUDA kernel internally **converts to fp32 before performing topk**, so the topk selection precision is higher than Megatron's default path.
- **Megatron `fp32` path**: Logits are already fp32, topk is executed on fp32, achieving the highest precision.

### 20.4 Softmax → Probs Truncation

- **Megatron default**: `softmax(dtype=fp32).type_as(logits)` **truncates the fp32 result back to bf16**, causing probs precision loss.
- **Megatron fp32**: In `.type_as(logits)`, logits are fp32, so probs remain fp32.
- **vLLM**: `topk_weights` output is always fp32, so **there is no truncation issue**. In this regard, vLLM is better than Megatron's default path.

### 20.5 Configuration Flexibility

- **Megatron** provides the `moe_router_dtype` config — a single parameter that controls the precision of the entire routing path.
- **vLLM** has no equivalent unified config. To improve gate precision, manual modifications to the model code are required (e.g., adding `params_dtype=torch.float32` and `quant_config=None`, and casting input in forward).

## 21. Summary

```
                     Gate GEMM        TopK Selection    Softmax→Probs     Config Capability
                     ─────────        ──────────────    ─────────────     ─────────────────
Megatron (default)    bf16             bf16              fp32→bf16 trunc   has moe_router_dtype
Megatron (fp32)       fp32 ✓           fp32 ✓            fp32 lossless ✓   ↑
vLLM (default)        bf16             bf16→fp32(kernel) fp32 lossless ✓   no unified config
```

vLLM's topk_softmax kernel performs fp32 promotion internally, making softmax and topk selection precision better than Megatron's default path. However, the Gate GEMM itself still executes in bf16 with no configurable option, and the gate may be FP8-quantized. **Overall routing precision protection is weaker than Megatron's `moe_router_dtype='fp32'` path**.
