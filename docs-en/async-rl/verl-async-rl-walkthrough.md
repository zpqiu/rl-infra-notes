# verl Fully Async RL Training: Code Walk-Through

> **Codebase**: [volcengine/verl](https://github.com/volcengine/verl) (Meituan fully_async_policy module)
>
> **Reference Blog**: [Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries](https://huggingface.co/blog/async-rl-training-landscape) (HuggingFace, 2026-03-10)
>
> **Dimensions of Interest**: 4 out of the 7 dimensions from the Blog:
> - Dimension 2: Rollout Buffer Design
> - Dimension 3: Weight Synchronization Protocol
> - Dimension 4: Staleness Management
> - Dimension 5: Partial Rollout Handling

## Positioning of verl in the Blog Framework

| Dimension | verl's Choice | Aggressiveness |
|------|-------------|---------|
| **Rollout Buffer** | Bounded async queue (depth controlled by staleness_threshold) | Aggressive — true multi-batch in-flight |
| **Weight Sync** | NCCL broadcast + bucketing, abort→sleep→sync→wake→resume | Moderate — abort in-flight rather than drain |
| **Staleness Management** | Depth bounding + rollout correction (TIS + rejection sampling), optional MIS; with version tracking but no hard discard based on version gap | Moderately rich — depth throttling + IS weighting + rejection sampling + multi-version |
| **Partial Rollout** | Abort + application-level prefix continuation | Moderate — same category as SLIME/SkyRL, but with cross-version tracking |

### Key Differences from SLIME

| Dimension | SLIME | verl |
|------|-------|------|
| Buffer depth | Double-buffer (depth=1) | Bounded queue (depth=N, controlled by staleness_threshold) |
| Scheduling model | Main loop blocking with ray.get | Rollouter/Trainer as two independent Ray actors, decoupled via MessageQueue |
| Data granularity | Rollout by entire batch | Per-sample streaming generation |
| Weight sync trigger | Training side drains in-flight before sync | Training side proactively triggers every N steps, aborts in-flight |
| Partial Rollout | abort + recycle buffer + off-policy token masking | abort + prefix continuation + cross-version global_steps tracking |
| Staleness correction | TIS + OPSM (inside loss function) | rollout_corr_helper (standalone module: TIS + rejection sampling + off-policy metrics) |
| MIS | None | Yes — save_model_to_cpu / restore_model_from_cpu supports multi-version IS |

## Recommended File Reading Order

Open these files in execution-flow order to walk through the entire pipeline:

1. `verl/experimental/fully_async_policy/fully_async_main.py` — Entry point: how the three Ray actors are assembled
2. `verl/experimental/fully_async_policy/message_queue.py` — Complete bounded buffer implementation
3. `verl/experimental/fully_async_policy/fully_async_rollouter.py` — Producer: streaming generation + backpressure
4. `verl/experimental/fully_async_policy/fully_async_trainer.py` — Consumer: training loop + weight sync triggering
5. `verl/checkpoint_engine/base.py:308-445` — CheckpointEngineManager: 8-step weight sync flow
6. `verl/checkpoint_engine/nccl_checkpoint_engine.py` — NCCL bucketed broadcast implementation
7. `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py` — ZMQ + CUDA IPC transfer (colocated path)
8. `verl/experimental/fully_async_policy/agent_loop/agent_loop.py` — partial rollout: prefix continuation
9. `verl/trainer/ppo/rollout_corr_helper.py` — TIS + rejection sampling math implementation
10. `verl/experimental/fully_async_policy/fully_async_trainer.py:473-493` — MIS: multi-version old_log_prob computation

---

## Step 0: Startup — Assembling Three Ray Actors

**Entry point**: `fully_async_main.py:34` → `FullyAsyncTaskRunner`

**Architecture**:
```
FullyAsyncTaskRunner (orchestrator, 1 CPU)
  │
  ├── FullyAsyncRollouter (Ray actor, 10 CPU)
  │     └── rollout worker group → vLLM/SGLang engines
  │
  ├── MessageQueue (Ray actor, 2 CPU)
  │     └── deque(maxlen=max_queue_size)
  │
  └── FullyAsyncTrainer (Ray actor, 10 CPU)
        └── actor worker group → FSDP/Megatron training
```

**Assembly flow** (`fully_async_main.py:50-116`):

```python
# 1. Create Trainer and Rollouter (initialized in parallel)
trainer = FullyAsyncTrainer.remote(...)
rollouter = FullyAsyncRollouter.remote(...)

# 2. Synchronize total_train_steps
total_train_steps = ray.get(rollouter.get_total_train_steps.remote())
ray.get(trainer.set_total_train_steps.remote(total_train_steps))

# 3. Create MessageQueue and inject into both sides
message_queue = MessageQueue.remote(config, max_queue_size)
message_queue_client = MessageQueueClient(message_queue)
ray.get(rollouter.set_message_queue_client.remote(message_queue_client))
ray.get(trainer.set_message_queue_client.remote(message_queue_client))

# 4. Trainer holds a reference to Rollouter (used to trigger staleness reset after weight sync)
ray.get(trainer.set_rollouter.remote(rollouter))

# 5. Initial weight sync + optional val_before_train
ray.get(trainer._fit_update_weights.remote())
```

**Starting training** (`fully_async_main.py:158-170`):
```python
# Rollouter and Trainer start in parallel, each running independently
rollouter_future = rollouter.fit.remote()
trainer_future = trainer.fit.remote()
# ray.wait monitors; handles completion or failure of either
```

> **Difference from SLIME**: SLIME's main loop is a `for` loop that uses `ray.get` to blockingly coordinate alternating rollout and train phases. verl's Rollouter and Trainer are fully independent Ray actors that communicate asynchronously via MessageQueue, requiring no central scheduler.

---

## Step 1: MessageQueue — Bounded Buffer

**File**: `message_queue.py`

### 1a. Data Structure

`message_queue.py:27-53`:
```python
@ray.remote(num_cpus=2, max_concurrency=20)
class MessageQueue:
    def __init__(self, config, max_queue_size=1000):
        self.queue = deque(maxlen=self.max_queue_size)    # bounded queue
        self._lock = asyncio.Lock()
        self._consumer_condition = asyncio.Condition(self._lock)  # consumer wait notification
```

### 1b. Producer Side: put_sample

`message_queue.py:55-83`:
```python
async def put_sample(self, sample):
    async with self._lock:
        if len(self.queue) >= self.max_queue_size:
            self.queue.popleft()       # queue full → discard oldest sample, don't block producer
            self.dropped_samples += 1
        self.queue.append(sample)
        self._consumer_condition.notify_all()  # wake up waiting consumer
```

### 1c. Consumer Side: get_sample

`message_queue.py:85-103`:
```python
async def get_sample(self):
    async with self._lock:
        while len(self.queue) == 0 and self.running:
            await self._consumer_condition.wait()  # queue empty → block and wait
        data = self.queue.popleft()
        return data, len(self.queue)
```

> **[Dimension 2: Rollout Buffer]** This is an **unbounded-producer, blocking-consumer** design: the producer never blocks (drops old data when full), while the consumer waits when the queue is empty. This is fundamentally different from SLIME's double-buffer (depth=1) — verl allows rollout data from multiple batches to accumulate in the queue.

### 1d. Queue Depth Calculation

`fully_async_rollouter.py:198-212`:
```python
self.max_required_samples = (
    required_samples                          # ppo_mini_batch_size × require_batches
    × (staleness_threshold + 1)               # allowed staleness depth levels
    × trigger_parameter_sync_step             # how many train steps per weight sync
)
self.max_queue_size = self.max_required_samples
```

**Example**: `ppo_mini_batch_size=64, require_batches=2, staleness_threshold=1, trigger_parameter_sync_step=2`
→ `max_queue_size = 64 × 2 × 2 × 2 = 512`

> **Meaning**: Queue depth directly encodes the staleness tolerance. `staleness_threshold=0` degenerates to SLIME-like behavior (generate one batch, wait for training to consume it before continuing).

---

## Step 2: Rollouter — Streaming Generation + Backpressure

**File**: `fully_async_rollouter.py`

### 2a. Overall Structure: Two Coroutines

`fully_async_rollouter.py:519-574` — `_streaming_generation_main()`:
```python
# Launch two concurrent coroutines
feed_task = safe_create_task(self._feed_samples())        # fetch samples from dataloader into pending_queue
processor_task = safe_create_task(self._processor_worker())  # fetch from pending_queue, submit async generation
await asyncio.wait([feed_task, processor_task], return_when=FIRST_COMPLETED)
```

```
DataLoader ──→ pending_queue ──→ processor_worker ──→ vLLM/SGLang ──→ MessageQueue
  (_feed_samples)     asyncio.Queue(128)    (_processor_worker)              (Ray actor)
```

### 2b. Feed: Per-Sample Feeding

`fully_async_rollouter.py:400-431`:
```python
async def _feed_samples(self):
    for epoch, batch_dict in continuous_iterator:
        full_batch = prepare_single_generation_data(batch_dict, self.config)
        rollout_sample = RolloutSample(full_batch=full_batch, sample_id=..., epoch=epoch)
        await self.pending_queue.put(rollout_sample)  # asyncio.Queue, applies backpressure when full
```

> **Difference from SLIME**: SLIME submits by batch (submitting `over_sampling_batch_size` prompts at once), while verl submits per sample (`gen_batch_size=1`), providing finer granularity.

### 2c. Processor: Concurrent Generation + Backpressure

`fully_async_rollouter.py:433-498`:
```python
async def _processor_worker(self):
    while True:
        # ① Check whether generation should be paused
        if self.paused or await self._should_pause_generation():
            # Wait for all active_tasks to complete
            while self.active_tasks:
                done_tasks, self.active_tasks = await asyncio.wait(self.active_tasks, ...)
            # Suspend, wait for reset_staleness() to wake up
            while self.paused:
                await self.condition.wait()

        # ② Fetch sample
        rollout_sample = await self.pending_queue.get()
        self.staleness_samples += 1

        # ③ Enforce concurrency limit
        while len(self.active_tasks) >= self.max_concurrent_samples:
            done_tasks, self.active_tasks = await asyncio.wait(self.active_tasks, ...)

        # ④ Submit async generation
        task = safe_create_task(self._process_single_sample_streaming(rollout_sample))
```

### 2d. Complete Flow for a Single Sample

`fully_async_rollouter.py:500-517`:
```python
async def _process_single_sample_streaming(self, rollout_sample):
    # Call vLLM/SGLang for generation
    ret = await self.async_rollout_manager.generate_sequences_single(rollout_sample.full_batch)
    rollout_sample.full_batch = ret
    # Push into MessageQueue
    success = await self.message_queue_client.put_sample(ray.cloudpickle.dumps(rollout_sample))
```

### 2e. Backpressure: Two Pause Conditions

`fully_async_rollouter.py:643-665`:
```python
async def _should_pause_generation(self):
    queue_stats = self.message_queue_client.get_statistics_sync()
    # Condition 1: MessageQueue is full
    if queue_stats["queue_size"] >= self.max_queue_size:
        return True
    # Condition 2: samples generated since last weight sync exceed limit
    if self.staleness_samples >= self.max_required_samples:
        return True
    return False
```

> **[Dimension 2: Backpressure]** Two layers of backpressure: physical queue full + staleness count logical limit. The purpose of the latter: even if the queue isn't full (trainer is consuming), if the rollouter has generated too many samples without a weight sync, it should pause to avoid excessive off-policy drift.

### 2f. Staleness Counting and Reset

`fully_async_rollouter.py:237-261`:
```python
async def reset_staleness(self):
    async with self.lock:
        self.paused = False
        self.condition.notify_all()                                 # wake up the paused processor
        # Reset staleness, but not to zero — in-flight and queued samples are still from old version
        self.staleness_samples = len(self.active_tasks) + await self.message_queue_client.get_queue_size()
```

> **Detail**: After reset, staleness is not 0 but rather the count of "already generated but not yet consumed by training" samples. This is a conservative but accurate choice.

---

## Step 3: Trainer — Consuming + Training

**File**: `fully_async_trainer.py`

### 3a. Main Loop

`fully_async_trainer.py:389-419`:
```python
async def fit(self):
    while True:
        try:
            await self.fit_step()
        except TrainingStopException:
            break
```

### 3b. Single Training Step Flow

`fully_async_trainer.py:421-461`:
```python
async def fit_step(self):
    batch = await self._fit_generate(None)      # fetch data from queue
    batch = self._fit_compute_reward(batch)      # compute reward
    batch = self._fit_compute_log_prob(batch)    # current policy forward → log_prob
    batch = self._fit_compute_ref_log_prob(batch) # ref policy forward → ref_log_prob
    batch = self._fit_compute_critic(batch)      # critic forward (optional)
    batch = self._fit_compute_advantage(batch)   # advantage computation
    batch = self._fit_update_critic(batch)       # critic backward (optional)
    batch = self._fit_update_actor(batch)        # actor backward
    self._fit_update_local_step()                # version counter
    await self._fit_update_weights()             # weight sync (conditionally triggered)
```

### 3c. Fetching Data from Queue

`fully_async_trainer.py:227-284`:
```python
async def _get_samples_from_queue(self):
    queue_samples = []
    while len(queue_samples) < self.required_samples:
        sample, queue_len = self.message_queue_client.get_sample_sync()
        if sample is None:
            break
        queue_samples.append(sample)

    queue_samples = [ray.cloudpickle.loads(x) for x in queue_samples]
    batch = assemble_batch_from_rollout_samples(queue_samples, self.tokenizer, self.config)
    return 0, batch
```

> Each fetch retrieves `required_samples` items (= `ppo_mini_batch_size × require_batches`) to assemble a training batch.

---

## Step 4: Weight Synchronization — Abort → Sleep → Sync → Wake → Resume

### 4a. Trigger Timing

`fully_async_trainer.py:495-527`:
```python
def _fit_update_local_step(self):
    if self.local_trigger_step < self.trigger_parameter_sync_step:
        self.local_trigger_step += 1       # hasn't reached trigger threshold, increment
    else:
        self.current_param_version += 1    # threshold reached → increment version number
        self.local_trigger_step = 1        # reset

async def _fit_update_weights(self):
    if self.local_trigger_step != 1:       # only execute when version number just incremented
        return
    await self.checkpoint_manager.update_weights(global_steps=self.current_param_version)
    # Notify Rollouter to reset staleness
    ray.get(self.rollouter.reset_staleness.remote())
```

> **[Dimension 3: Trigger Frequency]** Controlled by `trigger_parameter_sync_step`. A value of 2 means weights are synced every 2 train steps.

### 4b. CheckpointEngineManager: 8-Step Flow

**File**: `checkpoint_engine/base.py:404-445`

```python
async def update_weights(self, global_steps=None):
    # Step 1: abort all in-flight generation requests
    await asyncio.gather(*[r.abort_all_requests() for r in self.replicas])

    # Step 2: build temporary RayWorkerGroup
    workers = []
    for replica in self.replicas:
        workers.extend(replica.workers)
    rollout = RayWorkerGroup(worker_handles=workers, ...)

    # Step 3: sleep replicas — free KV cache to make GPU memory available for weight transfer
    await self.sleep_replicas()

    # Step 4: establish NCCL communication group (trainer rank 0 + all rollout workers)
    self.build_process_group(rollout)

    # Step 5: actual weight transfer — trainer send + rollout receive
    ray.get(trainer.update_weights(...) + rollout.update_weights(...))

    # Step 6: destroy communication group
    ray.get(trainer.execute_checkpoint_engine(["finalize"] * ...) + ...)

    # Step 7: wake up replicas — restore KV cache
    await self.wake_up_replicas()

    # Step 8: resume generation — engine accepts requests again
    await asyncio.gather(*[r.resume_generation() for r in self.replicas])
```

> **[Dimension 3: Interruption Semantics]** Step 1 is a **hard abort**, not a drain. vLLM calls `pause_generation(wait_for_inflight_requests=False)`, SGLang calls `pause_generation(mode="abort")`. Both immediately terminate in-flight requests.
>
> **[Dimension 3: Memory Management]** Step 3 sleep (free KV cache) → Step 7 wake (restore KV cache) is a memory borrowing trick: syncing weights needs extra GPU memory for buffers, and sleeping frees up KV cache space to accommodate them.

### 4c. NCCL Bucketed Broadcast (Disaggregated Path)

**File**: `checkpoint_engine/nccl_checkpoint_engine.py:97-275`

Topology: trainer rank 0 (master) → all rollout workers (slaves)

```
Trainer rank 0                      Rollout workers 1..N
     │                                    │
     │── ZMQ PUB: bucket metadata ───────>│  (name, shape, dtype, offset)
     │                                    │
     │── NCCL broadcast: bucket data ────>│  (actual tensor bytes)
     │                                    │
     │   (swap send_buf/recv_buf,         │
     │    pipeline next bucket)           │
```

`nccl_checkpoint_engine.py:223-275` — `send_weights()`:
```python
for name, weight in weights:
    # Accumulate into bucket
    if offset + weight.nbytes > self.bucket_size:
        # bucket full → wait for previous broadcast to complete
        await broadcast_op.wait_for_complete()
        # initiate new broadcast (ZMQ sends metadata + NCCL broadcasts data)
        broadcast_op = BroadcastOperation(rank=0, bucket=send_buf, metadata=bucket_meta, ...)
        # double buffer swap, pipeline transfer
        send_buf, recv_buf = recv_buf, send_buf
```

> **[Dimension 3: Transfer Optimization]** Double buffer + async broadcast achieves pipelining of transfer and packing. One buffer is being NCCL broadcast while the other is simultaneously being filled by the CPU with the next batch of weights.

### 4d. ZMQ + CUDA IPC (Colocated Path)

**File**: `workers/rollout/vllm_rollout/bucketed_weight_transfer.py:73-192`

When trainer and rollout are on the same machine, CUDA IPC is used instead of NCCL:

```python
class BucketedWeightSender:
    async def async_send_weights(self, weights):
        for name, weight in weights:
            if offset + weight.nbytes > self.bucket_size:
                self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})  # ZMQ sends metadata
                self.socket.recv()  # wait for receiver acknowledgment
            # Write directly to CUDA IPC buffer (zero-copy)
            self.buffer[offset:offset+weight.nbytes].copy_(weight.view(-1).view(torch.uint8))
```

> ZMQ handles metadata coordination, while actual tensors are transferred via CUDA IPC shared memory, avoiding CPU-GPU copies.

### 4e. vLLM `pause_generation` Semantics

**File**: `vllm/v1/engine/async_llm.py:563-607`

verl calls `pause_generation(wait_for_inflight_requests=False, clear_cache=True)`:

1. **`self._paused = True`** — New requests are blocked at the `generate()` entry by `asyncio.Condition.wait_for()`
2. **`abort(all request_ids)`** — Immediately abort all in-flight requests
3. **`wait_for_requests_to_drain()`** — Wait for engine_core to actually clear (abort is asynchronous)
4. **`reset_prefix_cache()` + `reset_mm_cache()`** — Clear caches (old weight KV cache is inconsistent with new weights)

`resume_generation()` simply does `self._paused = False` + `notify_all()`.

### 4f. SGLang `pause_generation(mode="abort")` Semantics

**File**: `sglang/python/sglang/srt/managers/tokenizer_manager.py:1373-1386`

```python
async def pause_generation(self, obj):
    self.is_pause = True                    # block new requests
    while True:
        self.abort_request(abort_all=True)  # repeatedly send abort
        if not await self.model_update_lock.is_locked():
            break                           # no requests being processed → drain complete
        await asyncio.sleep(1.0)            # poll every 1 second
```

Scheduler side (`scheduler.py:2989-3086`):
- **waiting_queue**: directly pop, no extra overhead
- **running_batch**: set `to_finish = FINISH_ABORT()`, **still runs one decode forward** before actually freeing KV cache

> **SGLang vs vLLM**: Semantics are roughly aligned (block new requests + abort in-flight + wait for drain), but SGLang's drain uses 1-second polling on `model_update_lock`, which is less precise than vLLM's event-driven `_requests_drained`. Also, SGLang's abort mode **does not go through the scheduler's `pause_generation()`**, so the `_engine_paused` flag is not set.

---

## Step 5: Staleness Management — Aligning with the Blog First, Then Examining verl Code

The HuggingFace blog divides staleness management into 3 orthogonal strategies. Mapping these to verl's fully async main path, a more accurate correspondence is:

| Blog Strategy | Supported by verl? | Corresponding Code |
|-----------|---------------|------------------|
| **Strategy 1: Per-sample version rejection** | **No exact implementation** | Samples do carry version information (`global_steps` / `min/max_global_steps`), and the trainer also tracks stale trajectories; but there is no main-path logic that hard-discards samples when `current_version - sample_version > K` before training |
| **Strategy 2: Depth Bounding** | **Yes, and it is the core mechanism** | `staleness_threshold` + `max_required_samples` + bounded `MessageQueue` + rollouter pause/backpressure, limiting the maximum number of stale samples that can accumulate at the system depth level |
| **Strategy 3: IS-weighted loss correction** | **Yes** | TIS and rejection sampling in `rollout_corr_helper.py`, plus MIS when `trigger_parameter_sync_step > 1` |

> **Key Clarification**: Previously in this section we mainly expanded on Strategy 3, but verl is not "only IS correction." It simultaneously clearly implements **Strategy 2 (depth throttling)**. Conversely, although it records sample version numbers, it **does not implement the kind of hard discard by version gap threshold described in the blog's Strategy 1**.

### 5a. Strategy 2 — Depth Bounding (System-Level Depth Throttling)

**File**: `experimental/fully_async_policy/fully_async_rollouter.py`

`staleness_threshold` is not "filtering by sample version at training time," but rather directly participates in computing the maximum sample depth allowed in the system:

```python
self.max_required_samples = int(
    self.required_samples
    * (self.staleness_threshold + 1)
    * self.config.async_training.trigger_parameter_sync_step
)
self.max_queue_size = self.max_required_samples
```

The Rollouter continuously checks two pause conditions during streaming generation:

```python
if queue_size >= self.max_queue_size:
    return True

if self.staleness_samples >= self.max_required_samples:
    return True
```

After each parameter sync, the rollouter resets `staleness_samples` to "current in-flight task count + unconsumed samples in queue":

```python
self.staleness_samples = len(self.active_tasks) + await self.message_queue_client.get_queue_size()
```

> **[Dimension 4: Strategy 2]** This is exactly the **Depth Bounding** from the blog: instead of checking each sample's version gap at the trainer entry, it limits pipeline depth at the architectural level, thereby bounding worst-case staleness.
>
> **Addendum**: When the `MessageQueue` is full, it `popleft()` to discard the oldest sample, but this is a physical protection against queue overflow, not a semantically meaningful hard rejection based on "version gap exceeding threshold."

### 5b. Strategy 3 — Rollout Correction Helper

**File**: `trainer/ppo/rollout_corr_helper.py`

This is a standalone off-policy correction module providing three core functions:

| Function | Purpose |
|------|------|
| `compute_rollout_correction_weights()` | Compute truncated IS weights |
| `compute_rollout_rejection_mask()` | Compute rejection sampling mask |
| `compute_rollout_correction_and_rejection_mask()` | Unified interface: IS + rejection |

The training main loop explicitly calls this logic:

```python
batch, is_metrics = compute_rollout_correction_and_add_to_batch(batch, rollout_corr_config)
```

### 5c. TIS — Truncated Importance Sampling

`rollout_corr_helper.py:481-540`:
```python
def compute_rollout_correction_weights(log_ratio, response_mask, rollout_is="token", rollout_is_threshold=2.0):
    # log_ratio = log(π_train / π_rollout)

    if rollout_is == "token":
        rollout_is_weights = torch.exp(torch.clamp(log_ratio, -20, 20))
    elif rollout_is == "sequence":
        log_ratio_sum = masked_sum(log_ratio, response_mask, axis=-1).unsqueeze(-1)
        rollout_is_weights = torch.exp(torch.clamp(log_ratio_sum, -20, 20))

    rollout_is_weights = torch.clamp(rollout_is_weights, max=rollout_is_threshold)
```

> **Comparison with SLIME TIS**: The mechanism is the same (exp ratio → clamp), but verl additionally supports sequence-level aggregation and batch normalization. SLIME only has token-level.

### 5d. Rejection Sampling

`rollout_corr_helper.py:605+`:

verl supports multiple rejection strategies (via `rollout_rs` configuration):
- `token_k*`: token-level divergence exceeds limit
- `seq_sum_k*` / `seq_mean_k*` / `seq_max_k*`: sequence-level divergence exceeds limit
- Threshold format: `lower_upper` or single-sided upper bound

Rejected tokens / sequences are excluded from the loss by modifying `response_mask`.

> **Note**: The rejection sampling here is **part of Strategy 3**. It filters based on drift in `π_train / π_rollout`, not the blog's Strategy 1 approach of "hard discarding based on how many versions behind a sample is."
>
> **Comparison with SLIME**: SLIME uses OPSM (only mask when advantage<0 and KL>δ), a strategy that "only masks harmful off-policy tokens." verl's rejection sampling is more general but does not consider advantage direction.

### 5e. bypass_mode — Skip old_log_prob Recomputation

`rollout_corr_helper.py:1039-1074`:

When `rollout_correction.bypass_mode=True`:
```python
def apply_bypass_mode(batch, rollout_corr_config, policy_loss_config):
    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
    policy_loss_config["loss_mode"] = "bypass_mode"
```

> **Meaning**: Saves one forward pass, but old_log_prob may drift from the training policy. Suitable for scenarios with low staleness.

### 5f. MIS — Multi-version Importance Sampling

**File**: `fully_async_trainer.py:473-493`

When `trigger_parameter_sync_step > 1`, there are multiple train steps within one weight version, and each step's batch comes from the same rollout policy version; but the actor has continued updating. MIS ensures the old policy log-prob is still computed using the weights of the "corresponding rollout version":

```python
def _compute_old_log_prob(self, batch):
    if self.local_trigger_step == 1:
        self.actor_rollout_wg.save_model_to_cpu(1)
        old_log_prob = super()._compute_old_log_prob(batch)
    else:
        self.actor_rollout_wg.save_model_to_cpu(self.local_trigger_step)
        self.actor_rollout_wg.restore_model_from_cpu(1)
        old_log_prob = super()._compute_old_log_prob(batch)
        self.actor_rollout_wg.restore_model_from_cpu(self.local_trigger_step)
        self.actor_rollout_wg.clear_cpu_model(self.local_trigger_step)
    return old_log_prob
```

> **[Dimension 4: MIS]** This is a distinctive feature of verl. It doesn't fall under any of the blog's three listed strategies, but can be seen as an extension of Strategy 3 for the "multiple train steps / multiple old policy versions" scenario: CPU-cached multi-version weights ensure the IS ratio's baseline version is correct.

### 5g. What Is the Status of Strategy 1 in verl?

verl is not completely without "per-sample version recording." In fact:

- The rollout server stamps output with `extra_fields["global_steps"]`
- Partial rollout further records `min_global_steps` / `max_global_steps`
- The trainer tracks metrics like `trajectory_param_versions`, `stale_trajectory_processed`, `partial_ratio`

However, these version fields are currently mainly used for:

1. Monitoring how many stale / partial trajectories are mixed into a batch
2. Providing observability for partial rollout about "how many parameter versions were crossed"
3. Letting the trainer count the number of stale trajectories

**They are not used in the fully async main path to implement**:

```python
if current_param_version - sample_version > K:
    drop(sample)
```

This kind of per-sample version rejection.

> **Conclusion**: If strictly following the blog's taxonomy, verl fully async's Staleness Management should be labeled as:
> - **Strategy 2: yes**
> - **Strategy 3: yes**
> - **Strategy 1: no exact hard-drop implementation in the main path**

---

## Step 6: Partial Rollout — Abort + Prefix Continuation

**File**: `experimental/fully_async_policy/agent_loop/agent_loop.py:40-123`

### 6a. Core Mechanism

`FullyAsyncLLMServerManager` inherits from `AsyncLLMServerManager` and overrides `generate()`:

```python
class FullyAsyncLLMServerManager(AsyncLLMServerManager):
    async def generate(self, request_id, *, prompt_ids, sampling_params, ...):
        final_output = TokenOutput(token_ids=[], log_probs=[], num_preempted=0)
        min_global_steps, max_global_steps = None, None

        while True:
            # ① Call parent generate, prompt = original prompt + already generated tokens
            output = await super().generate(
                prompt_ids=prompt_ids + final_output.token_ids,
                sampling_params=sampling_params,
            )

            # ② Append this round's output to the accumulated buffer
            final_output.token_ids.extend(output.token_ids)
            final_output.log_probs.extend(output.log_probs)

            # ③ Record cross-version information
            global_steps = output.extra_fields.get("global_steps", None)
            if min_global_steps is None:
                min_global_steps = global_steps
            max_global_steps = global_steps

            # ④ Deduct remaining max_tokens
            sampling_params[limit_key] = original_max_tokens - len(final_output.token_ids)

            # ⑤ If not interrupted by abort, or partial_rollout is disabled, exit loop
            if output.stop_reason not in ("aborted", "abort") or not self.config.async_training.partial_rollout:
                break

        final_output.extra_fields["min_global_steps"] = min_global_steps
        final_output.extra_fields["max_global_steps"] = max_global_steps
        return final_output
```

### 6b. Timing Diagram

```
Rollouter                              Engine                         Trainer
   │                                     │                              │
   │── generate(prompt) ────────────────>│ decode t1,t2,t3...          │
   │                                     │                              │
   │                                     │       weight sync triggered ─┤
   │                                     │<── abort_all_requests ──────│
   │<── output(tokens=[t1,t2,t3],        │                              │
   │         stop_reason="abort") ───────│                              │
   │                                     │<── update_weights ──────────│
   │                                     │<── resume_generation ───────│
   │                                     │                              │
   │  stop_reason=="abort" &&            │                              │
   │  partial_rollout=True               │                              │
   │  → while loop continues             │                              │
   │                                     │                              │
   │── generate(prompt+[t1,t2,t3]) ─────>│ continue decode with        │
   │         max_tokens -= 3             │ new weights t4,t5,...       │
   │                                     │                              │
   │<── output(tokens=[t4,t5,...],       │                              │
   │         stop_reason="stop") ────────│                              │
   │                                     │                              │
   │  stop_reason=="stop" → break        │                              │
   │  return [t1,t2,t3,t4,t5,...]       │                              │
   │  min_global_steps=V1                │                              │
   │  max_global_steps=V2                │                              │
```

### 6c. Comparison with SLIME Partial Rollout

| Dimension | SLIME | verl |
|------|-------|------|
| Partial tokens after abort | Placed in recycle buffer, retrieved next time to continue generation | Retry within the same `generate()` call via loop |
| Handling of old tokens | `loss_mask = [0] * old_len` (off-policy token masking) | No token-level masking, but records `min/max_global_steps` for trainer to assess |
| Recovery method | Retrieved in next `data_source.get_samples()` call | Resubmitted directly in the current `while True` loop |
| Prefix cache utilization | Possible (SGLang radix cache) | Not possible (`clear_cache=True` clears the prefix cache) |
| Observability | Rollouter knows abort occurred, manages explicitly | Transparent to Rollouter upper layer — abort/resume handled internally in `FullyAsyncLLMServerManager` |

> **[Dimension 5]** Both are essentially **abort + prefix retry**. The survey's classification ("explicit save/resume" vs "abort + retry with prefix") shows no difference at the code level. verl's distinguishing features are: (1) transparent to the upper-layer caller — the `generate()` internal while loop automatically retries; (2) records `min/max_global_steps` so the trainer knows how many policy versions this rollout spans.

---

## Full Pipeline + Dimension Annotation Diagram

```
fully_async_main.py                                Blog Dimensions
──────────────────────────────────────────────────────────
[Startup]
  FullyAsyncTrainer.remote()                       ← training actor
  FullyAsyncRollouter.remote()                     ← inference actor
  MessageQueue.remote(max_queue_size)              ← [Dim 2] bounded buffer
  trainer.set_rollouter(rollouter)                 ← reference injection
  trainer._fit_update_weights()                    ← initial weight sync [Dim 3]

[Parallel Execution]
  rollouter.fit() ─────────────────────────────────── independent Ray actor
  │ ┌─ _streaming_generation_main() ──────────────────────────────
  │ │  _feed_samples():
  │ │    for epoch, batch_dict in dataloader:
  │ │      pending_queue.put(RolloutSample)
  │ │
  │ │  _processor_worker():
  │ │    while True:
  │ │      _should_pause_generation()?              ← [Dim 2] backpressure
  │ │        queue_size >= max_queue_size
  │ │        staleness_samples >= max_required
  │ │      → paused: await condition.wait()
  │ │
  │ │      rollout_sample = pending_queue.get()
  │ │      staleness_samples += 1
  │ │      │
  │ │      └─ _process_single_sample_streaming() ─────────────────
  │ │           FullyAsyncLLMServerManager.generate()
  │ │           │  while True:                      ← [Dim 5] partial rollout loop
  │ │           │    output = super().generate(
  │ │           │      prompt + accumulated_tokens)
  │ │           │    accumulate tokens, log_probs
  │ │           │    track min/max_global_steps      ← [Dim 4] cross-version tracking
  │ │           │    if stop_reason != "abort":
  │ │           │      break
  │ │           │
  │ │           message_queue.put_sample(result)     ← [Dim 2] push to queue
  │ └──────────────────────────────────────────────────────────────

  trainer.fit() ───────────────────────────────────── independent Ray actor
  │ while True:
  │   fit_step():
  │   │ ┌─ _fit_generate() ──────────────────────────────────────
  │   │ │  _get_samples_from_queue()                ← [Dim 2] fetch N items from queue
  │   │ │  assemble_batch_from_rollout_samples()
  │   │ └────────────────────────────────────────────────────────
  │   │
  │   │ _fit_compute_reward()
  │   │ _fit_compute_log_prob()                     ← [Dim 4] current policy forward
  │   │ _fit_compute_ref_log_prob()
  │   │ _fit_compute_advantage()
  │   │ _fit_update_critic()
  │   │
  │   │ ┌─ _fit_update_actor() ──────────────────────────────────
  │   │ │  (internally uses rollout_corr_helper)
  │   │ │  compute_rollout_correction_weights()     ← [Dim 4] TIS
  │   │ │  compute_rollout_rejection_mask()         ← [Dim 4] rejection sampling
  │   │ │  _compute_old_log_prob()                  ← [Dim 4] MIS (multi-version)
  │   │ │    save_model_to_cpu() / restore()
  │   │ │  backward + optimizer step
  │   │ └────────────────────────────────────────────────────────
  │   │
  │   │ _fit_update_local_step()
  │   │   local_trigger_step++
  │   │   if >= trigger_parameter_sync_step:
  │   │     current_param_version++
  │   │
  │   │ ┌─ _fit_update_weights() (conditionally triggered) ────────
  │   │ │  CheckpointEngineManager.update_weights():
  │   │ │    1. abort_all_requests()                ← [Dim 3][Dim 5] hard abort
  │   │ │    2. build temporary worker group
  │   │ │    3. sleep_replicas()                    ← [Dim 3] free KV cache
  │   │ │    4. build_process_group()               ← [Dim 3] establish NCCL group
  │   │ │    5. NCCL bucketed broadcast             ← [Dim 3] transfer weights
  │   │ │    6. finalize()                          ← [Dim 3] clean up comm group
  │   │ │    7. wake_up_replicas()                  ← [Dim 3] restore KV cache
  │   │ │    8. resume_generation()                 ← [Dim 3][Dim 5] resume generation
  │   │ │
  │   │ │  rollouter.reset_staleness()              ← [Dim 2] reset backpressure
  │   │ └────────────────────────────────────────────────────────
  │   │
  │   │ _fit_validate()
  │   │ _fit_save_checkpoint()
```

---

## Key Configuration Parameters Quick Reference

| Parameter | Purpose | Corresponding Dimension |
|------|------|---------|
| `async_training.staleness_threshold` | Allowed staleness depth levels, affects queue depth | Dimension 2 |
| `async_training.require_batches` | How many mini-batches to fetch per training step | Dimension 2 |
| `async_training.trigger_parameter_sync_step` | Sync weights every N train steps | Dimension 3 |
| `async_training.partial_rollout` | Enable abort + prefix continuation | Dimension 5 |
| `algorithm.rollout_correction.bypass_mode` | Skip old_log_prob recomputation | Dimension 4 |
| `algorithm.rollout_correction.rollout_is` | IS aggregation level: "token" / "sequence" | Dimension 4 |
| `algorithm.rollout_correction.rollout_is_threshold` | TIS truncation upper bound (default 2.0) | Dimension 4 |
| `algorithm.rollout_correction.rollout_rs` | Rejection sampling strategy | Dimension 4 |
| `actor_rollout_ref.rollout.checkpoint_engine.backend` | Weight transfer backend: "nccl" / "nixl" / "naive" | Dimension 3 |
| `actor_rollout_ref.rollout.checkpoint_engine.bucket_size_mb` | Bucketed broadcast bucket size | Dimension 3 |

---

## Survey Description Corrections

Based on code analysis, several descriptions of verl in the survey need correction:

| What the Survey Says | Actual Code |
|---|---|
| "Soft pause with resume-oriented handling" | **Hard abort** (`wait_for_inflight_requests=False`), does not wait for natural completion |
| "Explicit save/resume" (partial rollout) | **Abort + prefix retry**, same category as SLIME/SkyRL. Tokens are saved at the application layer in `final_output`, not at the engine layer |
| "NCCL + bucketing" (implying pure NCCL) | NCCL is the disaggregated path; the colocated path uses ZMQ + CUDA IPC. Both paths can coexist |
| "Clipped TIS, optional OPSM" | Actually TIS + rejection sampling (based on KL divergence, not OPSM). verl's `rollout_corr_helper` does not contain OPSM |
