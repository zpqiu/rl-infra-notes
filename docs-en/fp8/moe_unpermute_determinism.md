# MoE Unpermute Non-Determinism Analysis

The issue was discovered during on-policy GRPO experiments in NeMo-RL, where the importance sampling ratio was not equal to 1 — meaning MCore produced inconsistent results across two forward passes on the same input. Original issue: [NVIDIA-NeMo/RL#2255](https://github.com/NVIDIA-NeMo/RL/issues/2255). Root cause was traced to `moe_permute_fusion=False`.

This note provides a root-cause analysis of the `scatter_add_`-induced non-determinism in MoE forward pass when `moe_permute_fusion=False`, and explains how the TE fused kernel avoids this problem via a gather-reduce pattern.

> **Analysis based on the following versions:**
>
> | Library | Commit |
> |---|---|
> | Megatron-LM | `17a67b9a9` |
> | TransformerEngine | `6638fefb` (v0.1+1653) |

---

## 1. Problem Statement

With `moe_permute_fusion=False`, the MoE model's forward pass produces non-deterministic results. The root cause is the `scatter_add_` operation in `unpermute()`: under top-K routing, multiple expert outputs need to be accumulated back to the same token position, and the CUDA implementation of `scatter_add_` does not guarantee the order of concurrent floating-point additions to the same address.

## 2. MoE Forward Data Flow

```
Original tokens [num_tokens, hidden]
        │
        ▼  permute()     ── group by expert: token → sorted by expert assignment
Permuted tokens [num_permuted_tokens, hidden]
        │
        ▼  Expert FFN    ── each expert processes its token chunk
Expert output [num_permuted_tokens, hidden]
        │
        ▼  unpermute()   ── scatter results back to original token positions and accumulate
Output tokens [num_tokens, hidden]
```

Key point: with top-K routing, the same token is routed to K experts. `permute()` **duplicates** the token K times to different positions, and `unpermute()` must **sum** K expert outputs back to the same position.

## 3. Non-Deterministic Code Path (Megatron-LM PyTorch)

### 3.1 `permute()` — Deterministic, No Issue

`Megatron-LM/megatron/core/transformer/moe/moe_utils.py:382-428`

```python
# Sort by routing_map, grouping tokens by expert
sorted_indices = routing_map.argsort(dim=-1, descending=True, stable=True)
# Gather tokens to new positions via index_select
permuted_input = tokens.index_select(0, sorted_indices)   # L426
```

`index_select` is a one-to-one gather — each output position reads from exactly one input position. **Deterministic.**

### 3.2 `unpermute()` — Root Cause of Non-Determinism

`Megatron-LM/megatron/core/transformer/moe/moe_utils.py:431-530`

```python
def unpermute(permuted_tokens, sorted_indices, restore_shape, probs=None,
              routing_map=None, fused=False, drop_and_pad=False, pad_offsets=None):
    if fused:
        return fused_unpermute(...)   # → TE path, see Section 4

    _, hidden = restore_shape

    if probs is not None:
        # Apply router probs to expert outputs (weighted)
        permuted_tokens = permuted_tokens * permuted_probs.unsqueeze(-1)  # L510

    # Create zero-filled output
    output_tokens = torch.zeros(restore_shape, dtype=permuted_tokens.dtype,
                                device=permuted_tokens.device)      # L513

    if torch.are_deterministic_algorithms_enabled():
        # Deterministic path: index_add_
        output_tokens.index_add_(0, sorted_indices, permuted_tokens)  # L524
    else:
        # Default path: scatter_add_ ← NON-DETERMINISTIC!
        output_tokens.scatter_add_(
            0,
            sorted_indices.unsqueeze(1).expand(-1, hidden),           # L527-528
            permuted_tokens,
        )
    return output_tokens.to(dtype=input_dtype)
```

### 3.3 Why `scatter_add_` Is Non-Deterministic

`sorted_indices` **contains duplicate values** — under top-K routing, the same token is processed by multiple experts. For example, token 5 is selected by expert 0 and expert 3:

```
permuted_tokens[42] = expert_0 output for token_5
permuted_tokens[91] = expert_3 output for token_5
sorted_indices[42] = 5
sorted_indices[91] = 5

scatter_add_ needs: output[5] += permuted_tokens[42] + permuted_tokens[91]
```

The CUDA implementation of `scatter_add_` launches one thread per **source row**, and multiple threads concurrently perform floating-point additions to `output[5]`. Since floating-point addition is not associative:

```
Run 1:  0.0 + a + b = x    (thread 42 executes first)
Run 2:  0.0 + b + a = x'   (thread 91 executes first)
x ≠ x'                     (FP rounding error ~1e-7 for FP32)
```

CUDA thread scheduling order is non-deterministic across runs → results are non-deterministic.

### 3.4 Behavior Under Three Configurations

| Configuration | Code Path | Deterministic? |
|---|---|---|
| `moe_permute_fusion=True` | TE `fused_unpermute()` | Yes |
| `moe_permute_fusion=False` + `torch.use_deterministic_algorithms(True)` | `index_add_()` | Yes |
| `moe_permute_fusion=False` (default) | **`scatter_add_()`** | **No** |

## 4. TE Fused Kernel: Deterministic Gather-Reduce Design

### 4.1 Call Chain

```
Megatron moe_utils.py: fused_unpermute
  └── megatron.core.extensions.transformer_engine: moe_unpermute        # alias
      └── transformer_engine.pytorch.permutation: moe_unpermute         # autograd Function
          └── _moe_unpermute_mask_map.forward                           # L339
              └── triton_permutation.unpermute_with_mask_map             # PyTorch wrapper
                  └── _unpermute_kernel                                  # Triton kernel
```

### 4.2 `row_id_map` Data Structure

TE does not use Megatron's `sorted_indices`. Instead, it builds a dedicated `row_id_map` tensor.

`TE/transformer_engine/pytorch/triton/permutation.py:24-118` — `make_row_id_map()`

Input `routing_map` shape `[num_tokens, num_experts]`, output `row_id_map` shape `[num_tokens, num_experts * 2 + 1]`.

Example from TE source comments (5 tokens, 3 experts):

```
routing_map:
  [[1, 1, 0],    # token 0 → expert 0, 1
   [1, 0, 1],    # token 1 → expert 0, 2
   [0, 0, 1],    # token 2 → expert 2
   [1, 1, 0],    # token 3 → expert 0, 1
   [0, 0, 0]]    # token 4 → not routed
```

After the 3-pass Triton kernel:

```
row_id_map (final):
  token 0: [3, 0, _, 1, 0, _, 2]   → routed to 2 experts, permuted rows 3 and 0, expert idx 1 and 0
  token 1: [5, 1, _, 2, 0, _, 2]   → routed to 2 experts, permuted rows 5 and 1
  token 2: [6, _, _, 2, _, _, 1]   → routed to 1 expert
  token 3: [4, 2, _, 1, 0, _, 2]   → routed to 2 experts
  token 4: [_, _, _, _, _, _, 0]   → not routed

Layout: [dst_row_0..dst_row_{K-1}, expert_idx_0..expert_idx_{K-1}, n_routed]
         ├── first num_experts cols ┤├── middle num_experts cols ──────┤├─ 1 ─┤
```

Key design: **each token knows which experts it was routed to and the corresponding permuted row indices**.

### 4.3 `_unpermute_kernel` — Gather-Reduce Core

`TE/transformer_engine/common/triton/permutation.py:314-414`

```python
@triton.jit
def _unpermute_kernel(...):
    # Grid: (num_tokens, cdiv(hidden_size, BLOCK_SIZE))
    # One program per token ← KEY!
    pid_t = tl.program_id(0)    # current output token
    pid_h = tl.program_id(1)    # hidden dim block tile

    accumulator = tl.zeros((BLOCK_SIZE,), dtype=compute_type)  # FP32 local accumulator

    # Read from row_id_map: how many experts was this token routed to?
    n_routed = tl.load(row_id_map_ptr + pid_t * ... + num_experts * 2 * ...)

    for idx in tl.range(n_routed):
        # Read from row_id_map: permuted row index for the idx-th expert
        src_row = tl.load(row_id_map_ptr + pid_t * ... + idx * ...).to(tl.int64)

        # Load expert output
        inp = tl.load(input_ptr + src_row * stride_input_token + ...)
        inp = inp.to(compute_type)          # upcast to FP32

        if WITH_MERGING_PROBS:
            # Read expert idx from row_id_map, look up router prob
            expert_idx = tl.load(row_id_map_ptr + ... + (num_experts + idx) * ...)
            merging_prob = tl.load(merging_probs_ptr + pid_t * ... + expert_idx * ...)
            inp *= merging_prob

        accumulator += inp                  # sequential accumulation, no contention

    # Single write to output
    tl.store(output_ptr + pid_t * stride_output_token + ..., accumulator.to(data_type))
```

### 4.4 Scatter vs Gather: Inversion of the Parallelization Axis

| | PyTorch `scatter_add_` | TE `_unpermute_kernel` |
|---|---|---|
| **Grid dimension** | One thread per **source row** (permuted token) | One program per **destination row** (original token) |
| **Accumulation** | Multiple threads concurrently write to the same output address | Single program sequentially reads all contributing sources |
| **Pattern** | Scatter (write-side conflict) | **Gather-reduce** (read-side, no conflict) |
| **Deterministic** | No — thread scheduling order is non-deterministic | **Yes** — single owner, sequential loop |
| **Precision** | Depends on input dtype | Forced FP32 accumulation (`compute_type = tl.float32`) |

Core insight: **inversion of the parallelization axis**. `scatter_add_` parallelizes from the source perspective (one thread per source row), causing multiple threads to race on the same output. The TE kernel parallelizes from the destination perspective (one program per output token), gathering all contributions and accumulating sequentially. Single owner = no contention = deterministic.

### 4.5 Permute Kernel Comparison

`TE/transformer_engine/common/triton/permutation.py:196-293` — `_permute_kernel`

Permute goes in the opposite direction: distributing from original tokens to permuted positions. TE also uses per-token programs:

```python
pid_t = tl.program_id(0)     # current original token
src_row = pid_t.to(tl.int64)
inp = tl.load(input_ptr + src_row * ...)

n_routed = tl.load(row_id_map_ptr + pid_t * ... + num_experts * 2 * ...)
for idx in tl.range(n_routed):
    dst_row = tl.load(row_id_map_ptr + pid_t * ... + idx * ...).to(tl.int64)
    tl.store(output_ptr + dst_row * ..., inp)    # writes to different destination rows, no conflict
```

Permute is a scatter-write (one-to-many), but each destination row is written by only one source (guaranteed by routing), so it is inherently deterministic. Unpermute is many-to-one (multiple expert outputs → one token), which is where the accumulation conflict arises.

## 5. `row_id_map` Construction: 3-Pass Triton Pipeline

`TE/transformer_engine/common/triton/permutation.py:82-192`

Converts the sparse `routing_map [num_tokens, num_experts]` into the dense `row_id_map [num_tokens, num_experts * 2 + 1]` in three steps:

| Pass | Kernel | Purpose | Grid |
|---|---|---|---|
| 1 | `_row_id_map_pass_1_kernel` | Per-expert column block-wise cumsum | `(num_experts, cdiv(num_tokens, 1024))` |
| 2 | `_row_id_map_pass_2_kernel` | Cross-block prefix sum → global row index; set unrouted positions to -1 | same |
| 3 | `_row_id_map_pass_3_kernel` | Per-token: compact sparse expert columns into dense [dst_rows | expert_idxs | n_routed] | `(num_tokens,)` |

Pass 3 uses a bitonic argsort (`_argsort`) internally to move valid entries to the front and invalid -1s to the back.

## 6. Backward Path

`_moe_unpermute_mask_map` backward (`TE/transformer_engine/pytorch/permutation.py:396-510`):

- **Without probs**: The backward of unpermute is simply permute (scatter ↔ gather are inverse operations), directly calls `permute_with_mask_map`
- **With probs**: Requires additional `probs_grad` computation, calls `unpermute_with_mask_map_bwd_with_merging_probs` (dedicated backward kernel)

Both paths use TE's per-token Triton kernels, maintaining determinism.
