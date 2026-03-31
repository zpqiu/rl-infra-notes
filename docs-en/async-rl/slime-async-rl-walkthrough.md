# SLIME Async RL Training: Code Walk-Through

> **Codebase**: [THUDM/slime](https://github.com/THUDM/slime) @ commit `f71f7103`
>
> **Reference Blog**: [Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries](https://huggingface.co/blog/async-rl-training-landscape) (HuggingFace, 2026-03-10)
>
> **Dimensions of Focus**: 4 out of the Blog's 7 dimensions:
> - Dimension 2: Rollout Buffer Design
> - Dimension 3: Weight Synchronization Protocol
> - Dimension 4: Staleness Management
> - Dimension 5: Partial Rollout Handling

## SLIME's Position in the Blog's Framework

| Aspect | SLIME's Choice | Aggressiveness |
|------|-------------|---------|
| **Rollout Buffer** | Double-buffer (depth=1) | Conservative — staleness at most 1 step |
| **Weight Sync** | NCCL broadcast + bucketing, pause→flush→sync→continue | Moderate — has bucketing optimization but requires pausing inference |
| **Staleness Management** | Dual mechanism: double-buffer (depth=1) + TIS/OPSM; records version but does not do per-sample rejection by default | Moderate-conservative — first uses structure to limit lag to 1 step, then uses loss correction for residual off-policy |
| **Partial Rollout** | Abort + recycle + off-policy token masking | Moderate — better than discarding, but not as good as per-forward-pass switching |

## File Reading Order

Open these files in execution flow order to walk through the entire process:

1. `train_async.py` — Skeleton (30 lines to see the big picture)
2. `slime/ray/rollout.py:RolloutManager.generate()` — Scheduling layer
3. `slime/rollout/sglang_rollout.py:generate_rollout_async()` — Generation + abort + recycle
4. `slime/rollout/data_source.py:RolloutDataSourceWithBuffer` — Buffer recycling mechanism
5. `slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py` — Full weight sync flow
6. `slime/backends/megatron_utils/actor.py:train_actor()` — Training data preparation
7. `slime/backends/megatron_utils/loss.py:policy_loss_function()` — TIS + OPSM correction

---

## Step 0: Startup — Allocate GPUs, Create Components

**Entry point**: `train_async.py:10` → `train(args)`

**File path**: `slime/ray/placement_group.py:79-119`

```
create_placement_groups(args)
  ├── actor GPU pool:   [GPU 0 .. actor_num_gpus-1]
  ├── critic GPU pool:  [immediately after actor]  (optional)
  └── rollout GPU pool: [last rollout_num_gpus GPUs]
```

In non-colocate mode (`train_async.py:11` enforces `assert not args.colocate`), training and inference run on **physically isolated GPU pools**. This is the physical foundation of the entire async architecture.

Then two core components are created:

**RolloutManager** (`placement_group.py:181-201` → `slime/ray/rollout.py:349-391`):
- A **Ray remote actor** (no GPU, pure scheduling)
- Internally launches SGLang inference engine fleet + router
- Holds `data_source` (data source) and `rollout_engine_lock` (weight sync lock)

**RayTrainGroup** (`placement_group.py:132-178` → `slime/ray/actor_group.py:10-99`):
- One `MegatronTrainRayActor` per GPU
- Rank 0 discovers master_addr/port, remaining actors join
- `async_init()` loads model, optimizer, checkpoint

**Initial weight sync** (`train_async.py:25-26`):
```python
if not args.critic_train_only:
    actor_model.update_weights()  # Ensure SGLang engines get the training model's initial weights
```
This is the first and only "blocking" full weight push.

---

## Step 1: Main Loop — Double-Buffer Pipeline Scheduling

**File**: `train_async.py:32-74`

This is the **scheduling heart** of SLIME async, only ~40 lines, worth reading line by line:

```python
# [A] Warm-up: initiate the first rollout
rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)

for rollout_id in range(args.start_rollout_id, args.num_rollout):
    # [B] Retrieve current rollout data (blocking wait)
    rollout_data_curr_ref = ray.get(rollout_data_next_future)

    # [C] Immediately initiate next rollout (parallel with training)
    if rollout_id + 1 < args.num_rollout:
        rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

    # [D] Train with current data (blocking wait for training to finish)
    ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

    # [E] Weight sync (triggered at intervals)
    if (rollout_id + 1) % args.update_weights_interval == 0:
        # ⚠️ Key: wait for the next rollout to complete first, preventing mid-rollout weight updates
        rollout_data_curr_ref = ray.get(rollout_data_next_future)
        rollout_data_next_future = None
        actor_model.update_weights()
```

> **[Dimension 2: Rollout Buffer]** This is a **double-buffer** with depth exactly 1. The timeline looks like:
> ```
> Time →
> Rollout:  [===Gen 0===]  [===Gen 1===]  [===Gen 2===]
> Train:                   [==Train 0==]  [==Train 1==]
>                                    ↑ Overlap region ↑
> ```
> Gen N+1 and Train N execute in parallel. But at most 1 rollout is in flight at any given time.

> **[Dimension 3: Weight Sync Trigger Point]** Note at `[E]`: before triggering weight sync, **the in-flight rollout must be drained first** (`ray.get(rollout_data_next_future)`). This ensures the inference engine is idle during weight updates, avoiding the complexity of mid-generation weight switching. The Blog classifies this as **"Per Training Step/Batch (blocking)"** interruption granularity.

---

## Step 2: Rollout Phase — From Prompt to Training Data

**Call chain**:
```
rollout_manager.generate.remote()
  → RolloutManager.generate()                  # rollout.py:460
    → generate_rollout()                        # sglang_rollout.py:578
      → generate_rollout_async()                # sglang_rollout.py:366
```

### 2a. Data Sampling

`sglang_rollout.py:399-403`:
```python
while len(data) < target_data_size:
    while state.remaining_batch_size < target_data_size:
        samples = data_source(args.over_sampling_batch_size)  # Fetch prompts from data source
        state.submit_generate_tasks(samples)                   # Submit async generation tasks
```

> **[Dimension 5: Partial Rollout]** `data_source` is `RolloutDataSourceWithBuffer.get_samples()` (`data_source.py:175-187`). It **prioritizes fetching abort-recycled partial samples from the buffer**, and only fetches new prompts from the dataset when the buffer is insufficient. This is the "consumer side" of recycle.

### 2b. Concurrent Generation

`sglang_rollout.py:111-123` — `submit_generate_tasks`:
```python
def submit_generate_tasks(self, samples):
    for group in samples:
        self.pendings.add(asyncio.create_task(
            generate_and_rm_group(args, group, sampling_params, evaluation=False)
        ))
```

Each group (G completions for the same prompt) runs as an asyncio task, sent concurrently to the SGLang router.

**Single sample generation** (`sglang_rollout.py:127-220`):
```python
async def generate(args, sample, sampling_params):
    payload = {"sampling_params": ..., "return_logprob": True}
    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True   # For MoE routing replay
    output = await post(url, payload)

    # Collect metadata
    sample.tokens += new_response_tokens
    sample.rollout_log_probs += new_response_log_probs  # ← Key: record logprob at generation time
    sample.update_from_meta_info(args, output["meta_info"])  # ← Record weight_version
```

> **[Dimension 4: Staleness]** Two key pieces of information are collected here:
> - `rollout_log_probs`: token-level log prob from policy π_old at generation time, used later by TIS to compute IS ratio
> - `weight_versions`: records the inference engine's weight version when generating this sample
>
> But the two serve different purposes:
> - `rollout_log_probs` actually participates in TIS/OPSM computation in the training loss
> - `weight_versions` in the default main path is primarily used for tracking/observation, and does not do per-sample hard drop based on version gap

### 2c. Oversampling + Dynamic Filtering + Abort

`sglang_rollout.py:405-438`:
```python
# Collect completed groups
done, state.pendings = await asyncio.wait(state.pendings, return_when=FIRST_COMPLETED)
for task in done:
    group = task.result()
    # Dynamic filtering (e.g., discard groups with zero reward variance)
    if not call_dynamic_filter(dynamic_filter, args, group).keep:
        state.remaining_batch_size -= 1
        continue
    data.append(group)

# Collected enough → abort remaining in-flight requests
aborted_samples = await abort(args, rollout_id)
```

> **[Dimension 5: Partial Rollout — Abort Flow]**
> `abort()` (`sglang_rollout.py:322-363`) does three things:
> 1. Sets `state.aborted = True` (causes subsequent tasks to return early)
> 2. Sends `abort_request` to all SGLang workers (`{"abort_all": True}`)
> 3. Waits for all pending tasks to complete, collects partially generated samples, marks `start_rollout_id`

### 2d. Recycle Recovery

`sglang_rollout.py:597-598`:
```python
output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
data_source.add_samples(aborted_samples)  # Put back into buffer, fetched first on next generation
```

> **[Dimension 5: Off-policy token masking]**
> When a partial sample is retrieved and continues generation (`sglang_rollout.py:229-231`):
> ```python
> if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
>     sample.loss_mask = [0] * sample.response_length  # Tokens generated by old weights → loss mask = 0
> ```
> Newly generated tokens append `[1]`. During training, old tokens do not participate in loss computation.

### 2e. Convert to Training Data

`rollout.py:460-473`:
```python
def generate(self, rollout_id):
    data, metrics = self._get_rollout_data(rollout_id)
    data = self._convert_samples_to_train_data(data)        # Sample → dict
    return self._split_train_data_by_dp(data, dp_size)       # Split by DP rank
```

`_convert_samples_to_train_data()` (`rollout.py:664-727`) converts the `Sample` list into a dict:
```python
train_data = {
    "tokens": [...],
    "response_lengths": [...],
    "rewards": [...],
    "loss_masks": [...],              # ← Contains off-policy mask from partial rollout
    "rollout_log_probs": [...],       # ← Input for TIS
    "rollout_routed_experts": [...],  # ← Input for MoE routing replay
}
```

Finally split by DP rank, each piece placed into Ray object store via `ray.put()`.

---

## Step 3: Weight Sync — Pause → Gather → Broadcast → Continue

**Trigger**: `train_async.py:64-69`, when `(rollout_id + 1) % update_weights_interval == 0`

**Call chain**:
```
actor_model.update_weights()
  → RayTrainGroup.update_weights()                      # actor_group.py:119
    → MegatronTrainRayActor.update_weights()             # actor.py:534
      → UpdateWeightFromDistributed.update_weights()     # update_weight_from_distributed.py:82
```

### 3a. Pause + Flush

`update_weight_from_distributed.py:88-99`:
```python
if dist.get_rank() == 0:
    ray.get([engine.pause_generation.remote() for engine in self.rollout_engines])
    ray.get([engine.flush_cache.remote() for engine in self.rollout_engines])
dist.barrier(group=get_gloo_group())
```
Rank 0 tells all inference engines to stop accepting new requests and flush the KV cache.

### 3b. Non-Expert Parameters: TP Gather → HF Conversion → Bucketed Broadcast

`update_weight_from_distributed.py:106-115`:
```python
for name, param in named_params_and_buffers(args, model):
    if ".experts." in name:
        continue
    buffer_size = self._update_weight_from_distributed(name, param, ...)
```

Processing for each parameter (`update_weight_from_distributed.py:142-164`):
```
param (TP-sharded)
  → all_gather_param()         # TP all-gather to restore full tensor (common.py:15)
  → convert_to_hf()           # Megatron format → HuggingFace format
  → accumulate into bucket
  → exceeds update_weight_buffer_size → _update_bucket_weights_from_distributed()
```

Bucket broadcast (`update_weight_from_distributed.py:228-249`):
```python
def _update_bucket_weights_from_distributed(self, converted_named_tensors):
    while not ray.get(self.rollout_engine_lock.acquire.remote()):
        time.sleep(0.1)   # Acquire lock to prevent concurrent NCCL deadlock

    # First send metadata via Ray (name, dtype, shape), then send tensor data via NCCL broadcast
    refs = update_weights_from_distributed(group_name, group, weight_version, engines, tensors)
    ray.get(refs)

    ray.get(self.rollout_engine_lock.release.remote())
```

`update_weights_from_distributed()` (`update_weight_from_distributed.py:310-337`):
```python
# Metadata via Ray (names, dtypes, shapes, weight_version)
refs = [engine.update_weights_from_distributed.remote(...) for engine in rollout_engines]
# Tensor data via NCCL broadcast (rank 0 → all engines)
for _, param in converted_named_tensors:
    handles.append(dist.broadcast(param.data, 0, group=group, async_op=True))
for handle in handles:
    handle.wait()
```

### 3c. Expert Parameters: Additional EP Gather

`update_weight_from_distributed.py:118-128`:
```python
for name, param in named_params_and_buffers(args, model):
    if ".experts." not in name:
        continue
    buffer_size = self._update_expert_weight_from_distributed(name, param, ...)
```

Expert parameters require an additional cross-EP-rank all-gather (`_update_expert_bucket_weights_from_distributed`, line 190-226), because each EP rank only holds a subset of experts. After gathering, they go through the same HF conversion + NCCL broadcast.

### 3d. Continue + Version Increment

`update_weight_from_distributed.py:86, 131-140`:
```python
self.weight_version += 1  # Incremented at entry

# ... after all parameter transfers are complete ...
if dist.get_rank() == 0:
    ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
dist.barrier(group=get_gloo_group())
```

### 3e. Colocated Path (Supplementary)

When some inference engines are on the same machine as training, the `UpdateWeightFromTensor` path is used (`update_weight_from_tensor.py`):
```
HF params → FlattenedTensorBucket packing → Gloo gather_object (CPU) → Ray IPC → SGLang engine
```
The difference from the distributed path is using CPU Gloo + IPC instead of GPU NCCL, avoiding cross-machine communication. Both paths can **coexist** (some engines colocated, some distributed).

---

## Step 4: Training — Forward → Advantage → Loss(TIS/OPSM) → Backward

**Call chain**:
```
actor_model.async_train()
  → RayTrainGroup.async_train()           # actor_group.py:111
    → MegatronTrainRayActor.train()        # actor.py:355
      → train_actor()                      # actor.py:398
```

### 4a. Data Preparation + Ref Model Forward

`actor.py:398-445`:
```python
def train_actor(self, rollout_id, rollout_data):
    data_iterator, num_microbatches = get_data_iterator(args, model, rollout_data)

    # If ref model exists → switch weights → compute ref_log_probs
    if "ref" in self.weights_backuper.backup_tags:
        self._switch_model("ref")
        rollout_data.update(self.compute_log_prob(..., store_prefix="ref_"))

    # Switch back to actor → compute current policy's log_probs (if not using rollout_logprobs)
    self._switch_model("actor")
    if not args.use_rollout_logprobs or args.get_mismatch_metrics:
        rollout_data.update(self.compute_log_prob(..., store_prefix=""))
```

> **[Dimension 4: Staleness]** There are two sources for old log probs here:
> - `use_rollout_logprobs=True`: directly use the `rollout_log_probs` recorded by the inference engine (more accurately reflects the policy at generation time, but may be off-policy)
> - `use_rollout_logprobs=False`: re-forward with the current actor weights to compute (on-policy but incurs an extra forward pass)
>
> The prerequisite for TIS correction is having `rollout_log_probs`, so `use_tis` requires `use_rollout_logprobs` or at least `get_mismatch_metrics`.

### 4b. Advantage Computation

`actor.py:460`:
```python
compute_advantages_and_returns(self.args, rollout_data)
```

`loss.py:400-561` — depending on `advantage_estimator`:
- **grpo/gspo**: `rewards × ones_like(kl)` — reward broadcast to each token
- **ppo**: GAE with value baseline
- **reinforce_plus_plus**: per-token discounted return

Advantage normalization across DP group (`loss.py:504-557`).

### 4c. Policy Loss — Where TIS and OPSM Take Effect

**File**: `loss.py:613-831` — `policy_loss_function()`, this is the **core battlefield** for staleness correction.

```python
# ① Get old_log_probs (source depends on use_rollout_logprobs)
old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]

# ② Current policy forward → compute log_probs and entropy
log_probs = get_log_probs_and_entropy(logits, ...)

# ③ Standard PPO clipped loss
pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high)
```

> **[Dimension 4: OPSM — Off-Policy Sequence Masking]** (`loss.py:682-689`):
> ```python
> if args.use_opsm:
>     opsm_mask, opsm_clipfrac = compute_opsm_mask(
>         full_log_probs, full_old_log_probs, advantages, loss_masks
>     )
>     pg_loss = pg_loss * opsm_mask  # Sequence-level KL > δ and advantage < 0 → zeroed out
> ```
> Intuition: off-policy negative samples (advantage<0 + high KL) are unreliable — don't let them push down the policy probability.
>
> `compute_opsm_mask()` (`ppo_utils.py:54-92`):
> ```python
> seq_kl = ((old_logprob - log_prob) * loss_mask).sum() / loss_mask.sum()
> mask = ((advantage < 0) & (seq_kl > delta)).float()
> return 1 - mask  # 0 = masked out, 1 = kept
> ```

> **[Dimension 4: TIS — Truncated Importance Sampling]** (`loss.py:712-740`):
> ```python
> if args.use_tis:
>     tis_func = vanilla_tis_function  # or custom
>     pg_loss, modified_masks, tis_metrics = tis_func(
>         pg_loss=pg_loss,
>         train_log_probs=batch["log_probs"],        # π_θ (current)
>         rollout_log_probs=batch["rollout_log_probs"], # π_old (at generation time)
>     )
> ```
> `vanilla_tis_function()` (`loss.py:563-584`):
> ```python
> tis = exp(π_θ - π_rollout)                        # IS ratio
> tis_weights = clamp(tis, tis_clip_low, tis_clip)  # Truncation
> pg_loss = pg_loss * tis_weights                    # Weighted correction
> ```
> There is also `icepop_function` (line 587): directly zeroes out values beyond the truncation range. This is still an **IS/RS variant at the loss level**, not per-sample version rejection based on `weight_version`.

**Execution order**: OPSM zeroing first → then TIS weighting. The two are decoupled and can be toggled independently.

> **More precisely, SLIME's default main path uses two staleness strategies from the blog simultaneously**:
> - **Strategy 2: Depth Bounding**. `train_async.py` has only one `rollout_data_next_future`, forming a one-step-ahead double-buffer; and before `update_weights()`, it first calls `ray.get(future)` to drain the in-flight rollout, thus structurally limiting the policy lag to at most 1 rollout step.
> - **Strategy 3: IS-weighted loss correction**. That is TIS + OPSM here.
>
> **What is not implemented by default is Strategy 1: Per-sample version rejection**. The code records `weight_versions`, but the default training path does not discard samples based on version gap.

### 4d. Backward + Optimizer Step

`actor.py:474-482`:
```python
with timer("actor_train"):
    train(rollout_id, model, optimizer, opt_param_scheduler, data_iterator, num_microbatches)
```
Standard Megatron pipeline-parallel train step. The loss, after TIS/OPSM correction, is backpropagated.

### 4e. Weight Backup

`actor.py:492`:
```python
self.weights_backuper.backup("actor")  # CPU backup, used for ref model switching and weight sync
```

---

## Step 5: Back to Step 1

The main loop returns to `train_async.py:33`, where `ray.get(rollout_data_next_future)` retrieves the next batch of rollout data that was **generated in parallel** during Step 4's training, and begins a new round of training.

---

## Full Flow + Dimension Annotation Diagram

```
train_async.py                                  Blog Dimensions
─────────────────────────────────────────────────────────
[Startup]
  create_placement_groups()                     ← Physical isolation (disaggregated)
  create_rollout_manager() → SGLang engines
  create_training_models() → Megatron actors
  actor_model.update_weights()                  ← Initial weight sync [Dim 3]

[Main Loop]
  future = rollout_manager.generate(0)          ← Warm-up
  for rollout_id in range(...):
    data = ray.get(future)                      ← [Dim 2] double-buffer fetch data
    future = generate(rollout_id+1)             ← [Dim 2] next batch generated in parallel
    │
    │ ┌─ generate_rollout_async() ──────────────────────────────
    │ │  data_source.get_samples()              ← [Dim 5] prioritize partial samples from buffer
    │ │  submit_generate_tasks() → asyncio
    │ │  │  generate() → SGLang
    │ │  │    record rollout_log_probs           ← [Dim 4] raw material for staleness correction
    │ │  │    record weight_versions             ← [Dim 4] version tracking
    │ │  │    record rollout_routed_experts       ← MoE routing replay
    │ │  │    partial rollout: mask_offpolicy     ← [Dim 5] old tokens loss_mask=0
    │ │  wait(FIRST_COMPLETED) + dynamic filtering
    │ │  abort() → recycle partial samples        ← [Dim 5] abort + recycle
    │ │  data_source.add_samples(aborted)         ← [Dim 5] put back into buffer
    │ │  convert_samples_to_train_data()
    │ │  split_by_dp()
    │ └──────────────────────────────────────────────────────────
    │
    actor_model.async_train(data)
    │ ┌─ train_actor() ─────────────────────────────────────────
    │ │  ref forward → ref_log_probs
    │ │  actor forward → log_probs               ← [Dim 4] on-policy logprob
    │ │  compute_advantages_and_returns()
    │ │  policy_loss_function():
    │ │    compute_policy_loss() → PPO clipped
    │ │    OPSM: mask(adv<0 & kl>δ)              ← [Dim 4] sequence-level off-policy mask
    │ │    TIS: loss × clamp(π_θ/π_old)          ← [Dim 4] token-level IS weighting
    │ │  backward + optimizer step
    │ └──────────────────────────────────────────────────────────
    │
    if (rollout_id+1) % interval == 0:
      ray.get(future)                            ← [Dim 3] drain in-flight rollout
      future = None
      actor_model.update_weights()
      │ ┌─ UpdateWeightFromDistributed ─────────────────────────
      │ │  pause_generation()                    ← [Dim 3] pause inference
      │ │  flush_cache()                         ← [Dim 3] flush KV cache
      │ │  for param (non-expert):
      │ │    all_gather_param() [TP]             ← [Dim 3] TP gather
      │ │    convert_to_hf()
      │ │    accumulate → bucket
      │ │    if bucket full:
      │ │      lock → NCCL broadcast → unlock    ← [Dim 3] bucketed transfer
      │ │  for param (expert):
      │ │    EP all-gather → HF → broadcast      ← [Dim 3] MoE expert handling
      │ │  weight_version++                      ← [Dim 4] version number increment
      │ │  continue_generation()                 ← [Dim 3] resume inference
      │ └──────────────────────────────────────────────────────
```

---

## Key Configuration Parameters Quick Reference

| Parameter | Purpose | Corresponding Dimension |
|------|------|---------|
| `--update_weights_interval` | Sync weights every N rollouts | Dimension 3 |
| `--update_weight_buffer_size` | Bucket size for bucketed broadcast | Dimension 3 |
| `--use_rollout_logprobs` | Use inference engine's logprob as old policy | Dimension 4 |
| `--use_tis` | Enable TIS correction | Dimension 4 |
| `--tis_clip` / `--tis_clip_low` | TIS truncation range | Dimension 4 |
| `--custom_tis_function_path` | Custom TIS function (e.g., icepop) | Dimension 4 |
| `--use_opsm` | Enable OPSM masking | Dimension 4 |
| `--opsm_delta` | OPSM KL threshold δ | Dimension 4 |
| `--partial_rollout` | Enable abort + recycle | Dimension 5 |
| `--mask_offpolicy_in_partial_rollout` | Old tokens in recycled samples don't participate in loss | Dimension 5 |
| `--get_mismatch_metrics` | Record train/rollout logprob discrepancy | Dimension 4 |
| `--use_rollout_routing_replay` | MoE routing consistency (record + replay) | Blog §5.4 |
