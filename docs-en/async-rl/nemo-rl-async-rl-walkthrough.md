# NeMo-RL Async RL Training: Code Walk-Through

> **Codebase**: [NVIDIA/NeMo-RL](https://github.com/NVIDIA/NeMo-RL) @ commit `94fa37d9` (latest release: v0.5.0)
>
> **Reference Blog**: [Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries](https://huggingface.co/blog/async-rl-training-landscape) (HuggingFace, 2026-03-10)
>
> **Dimensions of Interest**: 4 out of the Blog's 7 dimensions:
> - Dimension 2: Rollout Buffer Design
> - Dimension 3: Weight Synchronization Protocol
> - Dimension 4: Staleness Management
> - Dimension 5: Partial Rollout Handling

## NeMo-RL's Position in the Blog's Framework

| Dimension | NeMo-RL's Choice | Aggressiveness |
|------|----------------|---------|
| **Rollout Buffer** | Replay Buffer (depth=max_trajectory_age_steps, default 1) | Flexible — equivalent to double-buffer at depth 1, adjustable up to 8 |
| **Weight Sync** | NCCL collective broadcast (non-colocate) / ZMQ IPC (colocate), supports in-flight weight update | Aggressive — can update weights while inference is in progress |
| **Staleness Management** | Hybrid: version-aware filtering (target match + age window) + required IS correction | Moderately conservative — first filters out samples that shouldn't enter the current step, then applies IS to correct residual off-policy |
| **Partial Rollout** | No abort/recycle — all samples complete independently before entering buffer | Conservative — no partial rollout |

## File Reading Order

Open these files in execution flow order to walk through the entire pipeline:

1. `RL/nemo_rl/algorithms/grpo.py:2368` — `async_grpo_train()` entry point
2. `RL/nemo_rl/algorithms/async_utils.py:36` — `ReplayBuffer` Replay Buffer
3. `RL/nemo_rl/algorithms/async_utils.py:239` — `AsyncTrajectoryCollector` background generation scheduling
4. `RL/nemo_rl/experience/rollouts.py:862` — `run_async_multi_turn_rollout()` async rollout
5. `RL/nemo_rl/models/generation/vllm/vllm_worker_async.py:674` — vLLM async generation
6. `RL/nemo_rl/algorithms/grpo.py:1104` — `refit_policy_generation()` weight synchronization
7. `RL/nemo_rl/algorithms/loss/loss_functions.py:367` — Importance Sampling correction

---

## Step 0: Startup — Validation & Component Creation

**Entry point**: `grpo.py:2368` → `async_grpo_train()`

### 0a. Pre-validation

`grpo.py:2401-2416`:
```python
# Must use vLLM async engine
assert _should_use_async_rollouts(master_config)
# Must enable IS correction (off-policy convergence guarantee)
assert master_config["loss_fn"]["use_importance_sampling_correction"] is True
# Colocated inference is forbidden (training and inference must be physically separated)
assert not colocated_inference
```

> **[Dimension 3: Weight Sync]** Async mode mandates **disaggregated deployment** (training and inference on separate GPUs), consistent with SLIME's design. Weight synchronization uses NCCL collective broadcast (non-colocate path).

### 0b. Creating the ReplayBuffer

`grpo.py:2490-2501`:
```python
# Buffer size = num_prompts_per_step × max_trajectory_age_steps × 2 (slack)
optimal_buffer_size = num_prompts_per_step * max_trajectory_age_steps * late_arrival_slack
replay_buffer = ReplayBuffer.remote(max_size=optimal_buffer_size)
```

> **[Dimension 2: Rollout Buffer]** This is a **per-prompt group granularity** Replay Buffer. Each entry is 1 prompt × `num_generations_per_prompt` completions. Buffer depth is controlled by `max_trajectory_age_steps`.

### 0c. Creating the AsyncTrajectoryCollector

`grpo.py:2522-2537`:
```python
trajectory_collector = AsyncTrajectoryCollector.remote(
    policy_generation=policy_generation,
    tokenizer=tokenizer,
    task_to_env=task_to_env,
    master_config=master_config,
    replay_buffer=replay_buffer,
    start_step=step,
)
# Start background generation thread
trajectory_collector.start_collection.remote(dataloader)
trajectory_collector.set_weight_version.remote(weight_version)
```

The Collector is a **Ray remote actor** that internally starts a **daemon background thread** that continuously fetches batches from the dataloader, initiates inference, and pushes results into the ReplayBuffer.

### 0d. Initial Weight Sync (Refit)

`grpo.py:2546-2568`:
```python
if NEED_REFIT and POLICY_GENERATION_STALE:
    refit_policy_generation(policy, policy_generation, colocated_inference)
    POLICY_GENERATION_STALE = False
```

Ensures the vLLM inference engine has the training model's initial weights.

### 0e. Buffer Warm-up

`grpo.py:2607-2624`:
```python
while True:
    buffer_size_current = ray.get(replay_buffer.size.remote())
    if buffer_size_current >= min_trajectories_needed:
        break
    time.sleep(1.0)
```

Training must wait until the Buffer has at least `num_prompts_per_step` trajectories before starting. During this time, the background Collector continues filling the Buffer.

---

## Step 1: Main Loop — Async Pipeline Driven by Replay Buffer

**File**: `grpo.py:2626-2887`

Unlike SLIME's double-buffer main loop (~40 lines), NeMo-RL's asynchrony is decoupled via **ReplayBuffer + background Collector**, and the main loop itself is a standard train loop:

```python
while step < max_num_steps:
    # [A] Sample from ReplayBuffer (may stall)
    sample_result = ray.get(replay_buffer.sample.remote(
        num_prompt_groups=num_prompts_per_step,
        current_weight_version=weight_version,
        max_age_steps=max_trajectory_age_steps,
    ))
    if sample_result is None:
        time.sleep(0.5)
        continue  # Buffer insufficient → wait for background Collector to fill

    # [B] Training (logprob → advantage → policy loss → backward)
    fprop_logprobs = policy.get_logprobs(train_data)
    reference_logprobs = policy.get_reference_policy_logprobs(train_data)
    advantages = adv_estimator.compute_advantage(...)
    train_results = policy.train(train_data, loss_fn)

    # [C] Weight synchronization
    ray.get(trajectory_collector.prepare_for_refit.remote())
    refit_policy_generation(policy, policy_generation, colocated_inference)
    weight_version += 1
    trajectory_collector.set_weight_version.remote(weight_version)
    trajectory_collector.resume_after_refit.remote()
```

> **[Dimension 2: Rollout Buffer]** Comparison with SLIME:
> ```
> SLIME (double-buffer):
>   Main loop schedules rollouts directly, at most 1 in-flight rollout
>   Timeline: [Gen 0] [Gen 1] [Gen 2]
>                    [Train 0] [Train 1]
>
> NeMo-RL (replay buffer):
>   Background Collector generates continuously, main loop only fetches data from Buffer to train
>   Timeline: [Gen 0][Gen 1][Gen 2][Gen 3][Gen 4]...  ← continuous background
>                         [Train 0]  [Train 1]...    ← on-demand fetch
>   Up to num_prompts_per_step × max_trajectory_age_steps in-flight
> ```
> NeMo-RL's Buffer is deeper, and Generation and Training are **fully decoupled**.

---

## Step 2: Background Rollout — AsyncTrajectoryCollector

**File**: `async_utils.py:239-754`

### 2a. Background Collection Loop

`async_utils.py:392-448` — `_collection_loop()`:
```python
def _collection_loop(self):
    for batch in self.dataloader:
        if not self.running: break
        self._manual_pause_cleared.wait()   # Manual pause check
        self._refit_pause_cleared.wait()    # Weight sync pause check
        # Generation limit check (all target weights already generated → pause waiting for weight update)
        if self._should_pause_for_generation_limits():
            self._generation_limit_cleared.wait()
        self._process_batch(batch)
```

Three layers of Event gating:
- `_manual_pause_cleared`: Manual pause (used during validation)
- `_refit_pause_cleared`: Pause new generation during weight synchronization
- `_generation_limit_cleared`: All available target weights already have data → wait for new weight version

### 2b. Weight Version & Target Weight Mechanism

`async_utils.py:294-321` — `_calculate_target_weights()`:
```python
def _calculate_target_weights(self, generation_weight_version):
    """
    Example:
      generation_weight_version = 10
      max_trajectory_age_steps = 4
    Returns: [11, 12, 13, 14]
    i.e., data generated with v10 weights can serve training steps 11, 12, 13, 14
    """
    return [generation_weight_version + i for i in range(1, max_age + 1)]
```

> **[Dimension 4: Staleness]** This is NeMo-RL's unique **target weight** design:
> - Each trajectory records not only "which weight version **generated** it" (`trajectory_version`), but also labels "which training step it is **intended to serve**" (`target_weight_version`)
> - Buffer sampling only selects data where `target_weight_version == current_weight_version`
> - This ensures each training step receives data that was "designed to serve that step", avoiding random staleness
>
> This goes beyond mere "version tracking" — it is **genuine version-aware filtering that participates in sampling/filtering**: which old samples can enter the current training step is not left for the loss to handle as a fallback, but is explicitly constrained on the buffer side.

`async_utils.py:323-342` — `_get_next_target_for_generation()`:
```python
def _get_next_target_for_generation(self, generation_weight_version):
    target_weights = self._calculate_target_weights(generation_weight_version)
    last_generated = ray.get(
        self.replay_buffer.get_last_target_weight_already_generated.remote()
    )
    for target_weight in target_weights:
        if target_weight > last_generated and target_weight not in self._generating_targets:
            self._generating_targets.add(target_weight)  # Reserve
            return target_weight
    return None  # All targets are already being generated or have been generated
```

### 2c. Concurrent Generation — One Worker Thread per Prompt

`async_utils.py:451-508` — `_process_batch()`:
```python
def _process_batch(self, batch):
    target_weight = self._get_next_target_for_generation(generation_weight_version)
    if target_weight is None: return  # No generation needed

    for prompt_idx in range(num_prompts):
        self._inflight_sema.acquire()  # Concurrency cap control
        worker = threading.Thread(
            target=self._run_prompt_group_worker,
            args=(repeated_batch, generation_weight_version, target_weight, prompt_idx),
        )
        self._inflight_threads.add(worker)
        worker.start()
```

Concurrency control:
- `_inflight_sema`: Semaphore, upper bound is `num_prompts_per_step × max_trajectory_age_steps`
- Each worker thread independently runs a complete rollout for one prompt group

> **Batch boundaries are not synchronization barriers**: `_process_batch()` only spawns threads, it does not join — after returning, `_collection_loop` immediately fetches the next batch. Prompt groups across batches can be concurrently in-flight. The actual flow control mechanisms are the semaphore (prompt granularity), generation limit (target weight granularity), and refit pause (during weight synchronization), not batch completion.

### 2d. Generation of a Single Prompt Group

`async_utils.py:637-754` — `_run_prompt_group_worker()`:
```python
def _run_prompt_group_worker(self, repeated_batch, generation_weight_version, target_weight_version, prompt_idx):
    # Run complete rollout (may be multi-turn conversation)
    final_batch, rollout_metrics = run_async_multi_turn_rollout(
        policy_generation=self.policy_generation,
        input_batch=repeated_batch,
        tokenizer=self.tokenizer,
        task_to_env=self.task_to_env,
        max_seq_len=...,
        max_rollout_turns=...,
    )

    # Package result and push into ReplayBuffer
    trajectory_group = {
        "batch": final_batch.to("cpu"),
        "rollout_metrics": rollout_metrics,
        "timestamp": time.time(),
    }
    # Exponential backoff retry until Buffer accepts
    while self.running:
        status = ray.get(self.replay_buffer.push_with_wait_signal.remote(
            trajectory_group, generation_weight_version, target_weight_version,
        ))
        if status == "success": break
        elif status == "full": time.sleep(min(backoff_delay, 0.5))
```

> **[Dimension 5: Partial Rollout]** NeMo-RL **does not support** partial rollout/abort/recycle. All completions for a prompt group run to completion independently before entering the Buffer. This is simpler than SLIME but sacrifices some GPU utilization — long-tail samples may slow down overall throughput.

---

## Step 3: ReplayBuffer — Version-Aware Sampling

**File**: `async_utils.py:36-236`

### 3a. Data Structure

```python
class ReplayBuffer:
    trajectories = []           # List[dict], each is a prompt group
    trajectory_versions = []    # Weight version at generation time
    target_weight_versions = [] # Target training step
    last_target_weight_already_generated = -1  # Maximum target already generated
```

### 3b. Sampling Logic

`async_utils.py:102-223` — `sample()`:
```python
def sample(self, num_prompt_groups, current_weight_version, max_age_steps):
    # 1. Compute valid version window
    min_valid_version = max(0, current_weight_version - max_age_steps)

    # 2. Filter valid data
    valid_indices = [i for i, v in enumerate(self.trajectory_versions)
                     if min_valid_version <= v <= current_weight_version]

    # 3. Select only those with matching target version
    intended_indices = [i for i in valid_indices
                        if self.target_weight_versions[i] == current_weight_version]

    # 4. Insufficient quantity → return None (stall training)
    if len(intended_indices) < num_prompt_groups:
        return None

    # 5. Select + remove from buffer
    selected = intended_indices[:num_prompt_groups]
    avg_trajectory_age = current_weight_version - mean(trajectory_versions[selected])
    # ... remove and return
```

> **[Dimension 2 + Dimension 4]** Key design choices in the sampling strategy:
> - **Target matching**: Only selects data where `target_weight_version == current_weight_version`, ensuring each training step's data was "prepared for it"
> - **Age window**: Data older than `max_age_steps` is considered stale, triggering a ValueError
> - **Stall semantics**: If data is insufficient, training stalls — preferring to wait rather than making do — conservative but stable
> - **avg_trajectory_age** is returned as a metric, reflecting the degree of off-policy
>
> If we strictly consider only **staleness management** itself, without mixing in general concurrency/capacity control, NeMo-RL's default async path mainly uses two types of mechanisms:
> - **Strategy 1: Per-sample version rejection / filtering**: Target matching where `target_weight_version == current_weight_version`, plus the `trajectory_version` age window, determines which samples can enter the current step. This is **sample filtering**.
> - **Strategy 3: IS-weighted loss correction**: TIS / ICE-POP / seq-mask-TIS covered later
>
> `ReplayBuffer` size, `_inflight_sema`, and the background collector's flow control are better understood under the **Rollout Buffer / Orchestration** dimension; they indirectly affect the stale backlog but are not the rules that directly determine whether a sample is admitted to the current training step.

---

## Step 4: Weight Synchronization — Two Modes

**Trigger**: `grpo.py:2857-2883`, **triggered immediately after each training step** (unlike SLIME's `update_weights_interval` spacing)

### 4a. Prepare for Refit — Pause / Wait

`async_utils.py:529-576` — `prepare_for_refit()`:

```python
def prepare_for_refit(self):
    self._refit_pause_cleared.clear()  # Pause new generation

    if is_async_engine and in_flight_weight_updates:
        # Mode A: In-Flight — don't wait, ongoing generation continues with old weights
        print("Skipping wait for pending generations")
    else:
        # Mode B: Blocking — wait for all pending threads to complete
        self.wait_for_pending_generations()
```

> **[Dimension 3: Weight Sync]** Comparison of the two modes:
>
> | | **Blocking Mode** | **In-Flight Mode** |
> |---|---|---|
> | Config | `in_flight_weight_updates: false` | `in_flight_weight_updates: true` |
> | Waiting | Waits for all pending generation to complete | Only pauses new generation, doesn't wait for in-progress |
> | KV cache | No invalidation needed | Optional recompute (AREAL-style) |
> | Staleness | Low (no mid-generation weight change) | High (some tokens generated with old weights) |
> | Use case | `max_trajectory_age_steps=1` | Only benefits performance when `max_trajectory_age_steps>1` |
>
> If `max_trajectory_age_steps > 1` but in-flight is not enabled, the code prints a warning (`grpo.py:2409-2416`).

### 4b. Refit — NCCL Collective Broadcast

`grpo.py:1104-1198` — `refit_policy_generation()`:

Non-colocate path (the only path for async mode):
```python
# Training side broadcasts weights
futures_train = policy.broadcast_weights_for_collective(kv_scales=kv_scales)
# Inference side receives
futures_inference = policy_generation.update_weights_from_collective()
# Wait for completion
ray.get(futures_train)
results = ray.get(futures_inference)
```

> **[Dimension 3]** Comparison with SLIME:
> - SLIME: TP all-gather → HF conversion → bucketed NCCL broadcast, requires pause → flush KV cache → lock
> - NeMo-RL: Direct NCCL collective broadcast (implemented via vLLM `collective_rpc`), more concise
> - NeMo-RL does not require HF format conversion (vLLM directly receives training format)

vLLM-side reception (`vllm_generation.py:833-852`):
```python
def update_weights_from_collective(self):
    method_name = "update_weights_from_collective_async"  # async engine path
    futures = self.worker_group.run_all_workers_single_data(method_name, ...)
    return futures
```

Ultimately calls vLLM's `collective_rpc("update_weights_from_collective")` (`vllm_worker_async.py:1073-1085`).

### 4c. Resume — Restore Generation + Optional KV Cache Invalidation

`async_utils.py:578-601` — `resume_after_refit()`:
```python
def resume_after_refit(self):
    # AREAL-style: invalidate KV cache → new generation uses new-weight KV
    if in_flight_weight_updates and recompute_kv_cache_after_weight_updates:
        self.policy_generation.invalidate_kv_cache()

    self._refit_pause_cleared.set()  # Resume new generation
```

> **[Dimension 3]** KV cache invalidation strategy choices:
> - **recompute = true (AREAL-style)**: After in-flight generation completes, subsequent generation recomputes KV cache with new weights. More accurate but has overhead.
> - **recompute = false (Magistral-style)**: Keeps old KV cache and continues using it. Faster but the prefix portion's KV is based on old weights.

### 4d. Version Increment & Notification

`grpo.py:2880-2883`:
```python
weight_version += 1
trajectory_collector.set_weight_version.remote(weight_version)
trajectory_collector.resume_after_refit.remote()
```

`async_utils.py:344-353` — After the Collector receives a new version, it wakes up generation that was paused due to generation limit:
```python
def set_weight_version(self, version):
    self.current_weight_version = version
    if not self._generation_limit_cleared.is_set():
        self._generation_limit_cleared.set()  # Wake up the paused collection loop
```

---

## Step 5: Rollout Layer — vLLM Async Engine

**Call chain**:
```
_run_prompt_group_worker()
  → run_async_multi_turn_rollout()          # rollouts.py:862
    → asyncio.gather(*sample_tasks)         # Concurrently run all samples
      → run_sample_multi_turn_rollout()
        → generate_responses_async()
          → VllmGeneration.generate_async()  # vllm_generation.py:710
            → VllmAsyncGenerationWorker.generate_async()  # vllm_worker_async.py:674
              → AsyncLLM.generate()          # vLLM V1 async engine
```

### 5a. Async Rollout Orchestration

`rollouts.py:891-1053` — `_async_rollout_implementation()`:
```python
async def _async_rollout_implementation():
    # Each sample independently runs the complete multi-turn conversation
    sample_tasks = [
        run_single_sample_with_error_handling(i, sample_state)
        for i, sample_state in enumerate(sample_initial_states)
    ]
    # All samples execute concurrently
    sample_results = await asyncio.gather(*sample_tasks, return_exceptions=False)
```

### 5b. vLLM Async Generation

`vllm_worker_async.py:674-707`:
```python
async def generate_async(self, data, greedy=False):
    """Async generation for a single sample, using vLLM V1 AsyncLLM"""
    assert batch_size == 1  # Only processes one sample at a time
    vllm_request_generator = self.llm.generate(
        prompt=prompt, sampling_params=..., request_id=str(uuid.uuid4()),
    )
    async for req_output in vllm_request_generator:
        final_request_output = req_output
    # Extract token ids + logprobs
```

> **Key difference**: NeMo-RL uses vLLM V1's `AsyncLLM` (`vllm_worker_async.py:168`), while SLIME uses SGLang. vLLM V1's `collective_rpc` allows broadcasting weights while inference is in progress (the foundation for in-flight weight updates).

### 5c. Generation Logprobs

`vllm_worker_async.py:838-854`:
```python
if hasattr(generation_details, "logprobs") and generation_details.logprobs:
    for idx, logprob_dict_per_token in enumerate(generation_details.logprobs):
        # Extract log probability for each token
```

These generation logprobs are stored in the `generation_logprobs` field of `message_log`, and are later used during training as the behavior policy pi_old.

---

## Step 6: Training — Logprob → Advantage → Loss(IS) → Backward

**File**: `grpo.py:2786-2856`

### 6a. Logprob Computation

`grpo.py:2788-2802`:
```python
# Current policy forward → logprobs
fprop_logprobs = policy.get_logprobs(train_data)["logprobs"]
# Reference policy forward → reference logprobs
reference_logprobs = policy.get_reference_policy_logprobs(train_data)["reference_logprobs"]
train_data["prev_logprobs"] = fprop_logprobs           # pi_theta (current policy)
train_data["reference_policy_logprobs"] = reference_logprobs
```

> **[Dimension 4: Staleness]** Comparison with SLIME's two sources of old log probs:
> - SLIME `use_rollout_logprobs=True`: Uses `rollout_log_probs` recorded by the inference engine
> - SLIME `use_rollout_logprobs=False`: Re-forwards with current actor weights
> - NeMo-RL: Always re-forwards with the **current policy** to compute `prev_logprobs`, while retaining `generation_logprobs` (recorded during inference) for IS correction

### 6b. Sequence Logprob Error Masking

`grpo.py:2804-2814`:
```python
max_seq_mult_prob_error, num_masked_seqs, masked_correct_pct = \
    compute_and_apply_seq_logprob_error_masking(
        train_data=train_data,
        rewards=rewards,
        seq_logprob_error_threshold=master_config["grpo"]["seq_logprob_error_threshold"],
    )
```

When the sequence-level difference between `generation_logprobs` and `prev_logprobs` exceeds the threshold, that sequence is masked. This provides additional protection against staleness.

### 6c. Advantage Computation

`grpo.py:2817-2832`:
```python
train_data["advantages"] = adv_estimator.compute_advantage(
    prompt_ids=prompt_ids_for_adv,
    rewards=rewards,
    mask=mask,
    repeated_batch=repeated_batch,
    logprobs_policy=train_data["prev_logprobs"],
    logprobs_reference=train_data.get("reference_policy_logprobs"),
)
```

Supports GRPO, GDPO, Reinforce++, and other advantage estimators.

### 6d. Policy Loss — Importance Sampling Correction

**File**: `loss_functions.py:367-492` — `ClippedPGLossFn.__call__()`

This is the **core battlefield** for staleness correction:

```python
# 1. Standard PPO clipped ratio
log_ratios = curr_logprobs - prev_logprobs       # log(pi_theta / pi_theta_old)
ratios = log_ratios.exp()
ratios_clamped = ratios.clamp(1 - eps_min, 1 + eps_max)
clip_loss = max(-advantages * ratios, -advantages * ratios_clamped)

# 2. Importance Sampling correction
actor_importance_weights = exp(prev_logprobs - generation_logprobs)  # pi_theta_old / pi_gen
#                                                                       ^ current policy / generation-time policy

# 3. Truncated IS (three variants)
if tis_type == "tis":
    weights = clamp(weights, max=tis_ratio)           # Truncate upper bound
elif tis_type == "icepop":
    weights = where(in_bounds, weights, 0)             # Out of bounds → zero
elif tis_type == "seq-mask-tis":
    seq_geomean = exp(mean(log(weights)))              # Sequence-level geometric mean
    seq_mask = (seq_geomean >= min) & (seq_geomean <= max)
    weights = weights * seq_mask                       # Sequence-level gating, retaining token-level weights

# 4. Final loss
loss = masked_mean(importance_weights * clip_loss, mask)
```

> **[Dimension 4: Staleness Management]** Comparison of three IS strategies (`loss_functions.py:389-468`):
>
> | Strategy | Granularity | Behavior | Typical Parameters |
> |------|---------|------|---------|
> | **TIS** | Token-level | clamp(IS weight, max=T) | T=5.0 |
> | **ICE-POP** | Token-level | IS weight not in [min,max] → zero | [0.5, 5.0] |
> | **seq-mask-TIS** | Sequence-level gating + Token-level correction | Geometric mean IS ratio not in [min,max] → zero entire sequence | [0.999, 1.002] |
>
> Comparison with SLIME:
> - SLIME uses **TIS + OPSM** two-layer correction (sequence-level mask first, then token-level weighting)
> - **However, NeMo-RL's overall staleness management is not just IS at the loss layer**: The default async path, before entering the loss, has already applied system-level constraints through the replay buffer's target matching, age window, and in-flight/buffer depth
> - The loss layer then provides **TIS / ICE-POP / seq-mask-TIS** as three selectable IS variants
> - NeMo-RL's `seq-mask-tis` is similar to SLIME's OPSM (sequence-level gating), but based on IS ratio rather than KL + advantage

### 6e. Additional Support: Sequence-level IS (GSPO)

`loss_functions.py:372-380`:
```python
if self.sequence_level_importance_ratios:
    # GSPO: Sequence-level IS weights
    seq_lp_diff = ((prev_logprobs - generation_logprobs) * mask).sum(dim=-1)
    actor_importance_weights = exp(seq_lp_diff)
```

When `sequence_level_importance_ratios=True`, IS weights are computed at the sequence level (the approach from the GSPO paper).

---

## Step 7: Back to Step 1

Training step completes → weight synchronization → `weight_version++` → background Collector is woken up → generates new trajectories with new weights and pushes them into the Buffer → main loop samples next step's data from the Buffer.

---

## Full Pipeline + Dimension Annotation Diagram

```
grpo.py:async_grpo_train()                      Blog Dimensions
─────────────────────────────────────────────────────────
[Startup]
  Validation: async_engine=true, IS=true, non-colocate  ← Physical isolation (disaggregated)
  ReplayBuffer.remote(max_size=...)              ← [Dim 2] Buffer creation
  AsyncTrajectoryCollector.remote(...)
  start_collection(dataloader)                   ← Background generation thread starts
  refit_policy_generation()                      ← Initial weight sync [Dim 3]
  wait for buffer >= min_trajectories             ← Buffer warm-up

[Main Loop]
  while step < max_num_steps:
    ┌─ replay_buffer.sample() ──────────────────────────────
    │  filter: target_weight_version == current   ← [Dim 2+4] Version-aware sampling
    │  insufficient → stall wait                  ← [Dim 2] Training pace controlled by buffer
    └───────────────────────────────────────────────────────

    ┌─ Training ─────────────────────────────────────────────
    │  policy.get_logprobs() → prev_logprobs      ← [Dim 4] On-policy logprob
    │  policy.get_reference_policy_logprobs()
    │  seq_logprob_error_masking()                ← [Dim 4] Additional staleness protection
    │  adv_estimator.compute_advantage()
    │  ClippedPGLossFn():
    │    PPO clipped ratio
    │    IS weights = exp(prev_lp - gen_lp)       ← [Dim 4] Behavior/generation ratio
    │    TIS / ICE-POP / seq-mask-TIS truncation  ← [Dim 4] IS variants
    │    loss = IS_weights × clip_loss
    │  policy.train() → backward + optimizer step
    └───────────────────────────────────────────────────────

    ┌─ Weight Sync ──────────────────────────────────────────
    │  trajectory_collector.prepare_for_refit()
    │  │  Pause new generation
    │  │  [IF in_flight_weight_updates]
    │  │    Skip waiting — in-progress generation continues   ← [Dim 3] In-flight mode
    │  │  [ELSE]
    │  │    Wait for all pending threads to complete           ← [Dim 3] Blocking mode
    │  │
    │  refit_policy_generation()
    │  │  policy.broadcast_weights_for_collective() ← [Dim 3] NCCL broadcast
    │  │  policy_generation.update_weights_from_collective()
    │  │   → vLLM collective_rpc("update_weights_from_collective")
    │  │
    │  weight_version += 1                         ← [Dim 4] Version number increment
    │  trajectory_collector.set_weight_version()
    │  trajectory_collector.resume_after_refit()
    │  │  [IF recompute_kv_cache]
    │  │    invalidate_kv_cache()                  ← [Dim 3] AREAL-style
    └───────────────────────────────────────────────────────

[Background — AsyncTrajectoryCollector (independent thread running continuously)]
  _collection_loop():
    for batch in dataloader:
      wait manual_pause / refit_pause / generation_limit
      _process_batch(batch):
        target_weight = _get_next_target()         ← [Dim 4] Target training step
        for prompt_idx in range(num_prompts):
          _inflight_sema.acquire()                 ← Concurrency control
          Thread → _run_prompt_group_worker():
            run_async_multi_turn_rollout()
              asyncio.gather(*sample_tasks)        ← Concurrent generation
                VllmAsyncGenerationWorker.generate_async()
                  AsyncLLM.generate()              ← vLLM V1 async
            replay_buffer.push_with_wait_signal(
              trajectory, weight_version, target_weight_version
            )                                      ← [Dim 2] Push to buffer with version metadata
```

---

## Key Configuration Parameter Quick Reference

| Parameter | Purpose | Corresponding Dimension |
|------|------|---------|
| `grpo.async_grpo.enabled` | Enable async GRPO | — |
| `grpo.async_grpo.max_trajectory_age_steps` | Buffer depth / maximum trajectory age | Dimension 2, 4 |
| `grpo.async_grpo.in_flight_weight_updates` | Don't wait for in-progress generation during weight updates | Dimension 3 |
| `grpo.async_grpo.recompute_kv_cache_after_weight_updates` | Invalidate KV cache after in-flight (AREAL-style) | Dimension 3 |
| `policy.generation.vllm_cfg.async_engine` | Use vLLM V1 AsyncLLM | — |
| `policy.generation.colocated.enabled` | Must be false (async requires disaggregated) | Dimension 3 |
| `loss_fn.use_importance_sampling_correction` | Enable IS correction (mandatory for async) | Dimension 4 |
| `loss_fn.truncated_importance_sampling_ratio` | TIS/ICE-POP truncation upper bound | Dimension 4 |
| `loss_fn.truncated_importance_sampling_ratio_min` | ICE-POP/seq-mask-TIS truncation lower bound | Dimension 4 |
| `loss_fn.truncated_importance_sampling_type` | IS strategy: `tis` / `icepop` / `seq-mask-tis` | Dimension 4 |
| `loss_fn.sequence_level_importance_ratios` | Sequence-level IS (GSPO) | Dimension 4 |
| `grpo.seq_logprob_error_threshold` | Mask sequence when logprob difference exceeds threshold | Dimension 4 |

## Example Configurations

**Llama 3.1 8B — 2 nodes 8 GPUs, async 1-off** (`grpo-llama3.1-8b-instruct-2n8g-async-1off.yaml`):
```yaml
grpo:
  async_grpo:
    enabled: true
    max_trajectory_age_steps: 1         # At most 1 step of staleness
    in_flight_weight_updates: true
loss_fn:
  use_importance_sampling_correction: true
policy:
  generation:
    colocated:
      enabled: false
    vllm_cfg:
      async_engine: true
      gpu_memory_utilization: 0.8
```

**Qwen3 30B-A3B MoE — 24 nodes, async 8-off** (`grpo-qwen3-30ba3b-24n8g-async-8off.yaml`):
```yaml
grpo:
  async_grpo:
    enabled: true
    max_trajectory_age_steps: 8         # Allow 8 steps of staleness
    in_flight_weight_updates: true
loss_fn:
  use_importance_sampling_correction: true
policy:
  megatron_cfg:
    expert_model_parallel_size: 8       # MoE EP=8
  generation:
    colocated:
      enabled: false
    vllm_cfg:
      async_engine: true
      tensor_parallel_size: 2
```

## NeMo-RL vs SLIME Async Architecture Comparison

| Aspect | SLIME | NeMo-RL |
|------|-------|---------|
| **Scheduling Model** | Main loop schedules directly, double-buffer | Background Collector + ReplayBuffer, fully decoupled |
| **Buffer Depth** | Fixed 1 (double-buffer) | Configurable 1~N (`max_trajectory_age_steps`) |
| **Concurrency Granularity** | At most 1 rollout in-flight at a time | `num_prompts_per_step × max_age` concurrent |
| **Inference Engine** | SGLang | vLLM V1 (AsyncLLM) |
| **Weight Sync** | pause → flush → bucketed NCCL broadcast | NCCL collective broadcast via vLLM `collective_rpc` |
| **In-flight Updates** | Not supported (must drain in-flight) | Supported (`in_flight_weight_updates: true`) |
| **Partial Rollout** | Abort + recycle + off-policy mask | Not supported |
| **Staleness Correction** | depth=1 + TIS + OPSM | version-aware filtering + IS correction |
| **Version Tracking** | `weight_versions` per sample | `trajectory_version` + `target_weight_version` per prompt group |
| **KV Cache Strategy** | flush_cache before sync | Optional invalidate (AREAL vs Magistral) |
