# verl Fully Async RL Training: Code Walk-Through

> **Codebase**: [volcengine/verl](https://github.com/volcengine/verl) (Meituan fully_async_policy module)
>
> **参考 Blog**: [Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries](https://huggingface.co/blog/async-rl-training-landscape) (HuggingFace, 2026-03-10)
>
> **关注维度**: Blog 7 个维度中的 4 个:
> - 维度 2: Rollout Buffer 设计
> - 维度 3: 权重同步协议
> - 维度 4: Staleness 管理
> - 维度 5: Partial Rollout 处理

## verl 在 Blog 框架中的定位

| 维度 | verl 的选择 | 激进程度 |
|------|-------------|---------|
| **Rollout Buffer** | Bounded async queue (深度由 staleness_threshold 控制) | 激进 — 真正的多 batch in-flight |
| **权重同步** | NCCL broadcast + bucketing, abort→sleep→sync→wake→resume | 中等 — abort in-flight 而非 drain |
| **Staleness 管理** | Rollout correction (TIS + rejection sampling)，可选 MIS | 中等偏丰富 — IS 加权 + 拒绝采样 + 多版本 |
| **Partial Rollout** | Abort + application-level prefix continuation | 中等 — 与 SLIME/SkyRL 同类，但有跨版本追踪 |

### 与 SLIME 的关键差异

| 维度 | SLIME | verl |
|------|-------|------|
| Buffer 深度 | Double-buffer (深度=1) | Bounded queue (深度=N，由 staleness_threshold 控制) |
| 调度模型 | 主循环 ray.get 阻塞式 | Rollouter/Trainer 两个独立 Ray actor，通过 MessageQueue 解耦 |
| 数据粒度 | 按 batch 整体 rollout | 逐 sample streaming 生成 |
| 权重同步触发 | 训练侧 drain in-flight 后同步 | 训练侧每 N 步主动触发，abort in-flight |
| Partial Rollout | abort + recycle buffer + off-policy token masking | abort + prefix continuation + 跨版本 global_steps 追踪 |
| Staleness 修正 | TIS + OPSM (在 loss 函数内) | rollout_corr_helper (独立模块：TIS + rejection sampling + off-policy metrics) |
| MIS | 无 | 有 — save_model_to_cpu / restore_model_from_cpu 支持多版本 IS |

## 文件阅读顺序

按执行流顺序打开这些文件即可走完全流程：

1. `verl/experimental/fully_async_policy/fully_async_main.py` — 入口：三个 Ray actor 如何组装
2. `verl/experimental/fully_async_policy/message_queue.py` — bounded buffer 完整实现
3. `verl/experimental/fully_async_policy/fully_async_rollouter.py` — producer：streaming 生成 + backpressure
4. `verl/experimental/fully_async_policy/fully_async_trainer.py` — consumer：training loop + weight sync 触发
5. `verl/checkpoint_engine/base.py:308-445` — CheckpointEngineManager：weight sync 8 步流程
6. `verl/checkpoint_engine/nccl_checkpoint_engine.py` — NCCL bucketed broadcast 实现
7. `verl/workers/rollout/vllm_rollout/bucketed_weight_transfer.py` — ZMQ + CUDA IPC 传输（colocated 路径）
8. `verl/experimental/fully_async_policy/agent_loop/agent_loop.py` — partial rollout: prefix continuation
9. `verl/trainer/ppo/rollout_corr_helper.py` — TIS + rejection sampling 数学实现
10. `verl/experimental/fully_async_policy/fully_async_trainer.py:473-493` — MIS: 多版本 old_log_prob 计算

---

## 第 0 步：启动 — 三个 Ray Actor 组装

**入口**: `fully_async_main.py:34` → `FullyAsyncTaskRunner`

**架构**:
```
FullyAsyncTaskRunner (调度者，1 CPU)
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

**组装流程** (`fully_async_main.py:50-116`):

```python
# 1. 创建 Trainer 和 Rollouter（并行初始化）
trainer = FullyAsyncTrainer.remote(...)
rollouter = FullyAsyncRollouter.remote(...)

# 2. 同步 total_train_steps
total_train_steps = ray.get(rollouter.get_total_train_steps.remote())
ray.get(trainer.set_total_train_steps.remote(total_train_steps))

# 3. 创建 MessageQueue，注入双方
message_queue = MessageQueue.remote(config, max_queue_size)
message_queue_client = MessageQueueClient(message_queue)
ray.get(rollouter.set_message_queue_client.remote(message_queue_client))
ray.get(trainer.set_message_queue_client.remote(message_queue_client))

# 4. Trainer 持有 Rollouter 引用（用于触发 weight sync 后的 staleness reset）
ray.get(trainer.set_rollouter.remote(rollouter))

# 5. 初始权重同步 + 可选 val_before_train
ray.get(trainer._fit_update_weights.remote())
```

**启动训练** (`fully_async_main.py:158-170`):
```python
# Rollouter 和 Trainer 并行启动，各自独立运行
rollouter_future = rollouter.fit.remote()
trainer_future = trainer.fit.remote()
# ray.wait 监控，任一完成或失败则处理
```

> **与 SLIME 的区别**: SLIME 的主循环是一个 `for` 循环，用 `ray.get` 阻塞协调 rollout 和 train 的交替。verl 的 Rollouter 和 Trainer 是完全独立运行的 Ray actor，通过 MessageQueue 异步通信，不需要中央调度。

---

## 第 1 步：MessageQueue — Bounded Buffer

**文件**: `message_queue.py`

### 1a. 数据结构

`message_queue.py:27-53`:
```python
@ray.remote(num_cpus=2, max_concurrency=20)
class MessageQueue:
    def __init__(self, config, max_queue_size=1000):
        self.queue = deque(maxlen=self.max_queue_size)    # 有界队列
        self._lock = asyncio.Lock()
        self._consumer_condition = asyncio.Condition(self._lock)  # consumer 等待通知
```

### 1b. Producer 端：put_sample

`message_queue.py:55-83`:
```python
async def put_sample(self, sample):
    async with self._lock:
        if len(self.queue) >= self.max_queue_size:
            self.queue.popleft()       # 队列满 → 丢弃最老 sample，不阻塞 producer
            self.dropped_samples += 1
        self.queue.append(sample)
        self._consumer_condition.notify_all()  # 唤醒等待的 consumer
```

### 1c. Consumer 端：get_sample

`message_queue.py:85-103`:
```python
async def get_sample(self):
    async with self._lock:
        while len(self.queue) == 0 and self.running:
            await self._consumer_condition.wait()  # 队列空 → 阻塞等待
        data = self.queue.popleft()
        return data, len(self.queue)
```

> **[维度 2: Rollout Buffer]** 这是一个 **unbounded-producer, blocking-consumer** 的设计：producer 永不阻塞（满了就丢老数据），consumer 在队列空时等待。与 SLIME 的 double-buffer（深度=1）本质不同 —— verl 允许多个 batch 的 rollout 数据在 queue 中积累。

### 1d. Queue 深度计算

`fully_async_rollouter.py:198-212`:
```python
self.max_required_samples = (
    required_samples                          # ppo_mini_batch_size × require_batches
    × (staleness_threshold + 1)               # 允许的 staleness 层数
    × trigger_parameter_sync_step             # 多少个 train step 触发一次 weight sync
)
self.max_queue_size = self.max_required_samples
```

**示例**: `ppo_mini_batch_size=64, require_batches=2, staleness_threshold=1, trigger_parameter_sync_step=2`
→ `max_queue_size = 64 × 2 × 2 × 2 = 512`

> **含义**: queue 深度直接编码了 staleness 容忍度。`staleness_threshold=0` 退化为类似 SLIME 的行为（生成一批，等训练消费完才能继续）。

---

## 第 2 步：Rollouter — Streaming 生成 + Backpressure

**文件**: `fully_async_rollouter.py`

### 2a. 整体结构：两个协程

`fully_async_rollouter.py:519-574` — `_streaming_generation_main()`:
```python
# 启动两个并发协程
feed_task = safe_create_task(self._feed_samples())        # 从 dataloader 取样放入 pending_queue
processor_task = safe_create_task(self._processor_worker())  # 从 pending_queue 取样，提交异步生成
await asyncio.wait([feed_task, processor_task], return_when=FIRST_COMPLETED)
```

```
DataLoader ──→ pending_queue ──→ processor_worker ──→ vLLM/SGLang ──→ MessageQueue
  (_feed_samples)     asyncio.Queue(128)    (_processor_worker)              (Ray actor)
```

### 2b. Feed: 逐条喂样

`fully_async_rollouter.py:400-431`:
```python
async def _feed_samples(self):
    for epoch, batch_dict in continuous_iterator:
        full_batch = prepare_single_generation_data(batch_dict, self.config)
        rollout_sample = RolloutSample(full_batch=full_batch, sample_id=..., epoch=epoch)
        await self.pending_queue.put(rollout_sample)  # asyncio.Queue, 满了会背压
```

> **与 SLIME 的区别**: SLIME 按 batch 提交（一次提交 `over_sampling_batch_size` 个 prompt），verl 逐条提交（`gen_batch_size=1`），粒度更细。

### 2c. Processor: 并发生成 + Backpressure

`fully_async_rollouter.py:433-498`:
```python
async def _processor_worker(self):
    while True:
        # ① 检查是否应该暂停
        if self.paused or await self._should_pause_generation():
            # 等待所有 active_tasks 完成
            while self.active_tasks:
                done_tasks, self.active_tasks = await asyncio.wait(self.active_tasks, ...)
            # 挂起，等 reset_staleness() 唤醒
            while self.paused:
                await self.condition.wait()

        # ② 取样
        rollout_sample = await self.pending_queue.get()
        self.staleness_samples += 1

        # ③ 控制并发上限
        while len(self.active_tasks) >= self.max_concurrent_samples:
            done_tasks, self.active_tasks = await asyncio.wait(self.active_tasks, ...)

        # ④ 提交异步生成
        task = safe_create_task(self._process_single_sample_streaming(rollout_sample))
```

### 2d. 单条样本的完整流程

`fully_async_rollouter.py:500-517`:
```python
async def _process_single_sample_streaming(self, rollout_sample):
    # 调用 vLLM/SGLang 生成
    ret = await self.async_rollout_manager.generate_sequences_single(rollout_sample.full_batch)
    rollout_sample.full_batch = ret
    # 推入 MessageQueue
    success = await self.message_queue_client.put_sample(ray.cloudpickle.dumps(rollout_sample))
```

### 2e. Backpressure: 两个暂停条件

`fully_async_rollouter.py:643-665`:
```python
async def _should_pause_generation(self):
    queue_stats = self.message_queue_client.get_statistics_sync()
    # 条件 1: MessageQueue 满
    if queue_stats["queue_size"] >= self.max_queue_size:
        return True
    # 条件 2: 自上次 weight sync 以来生成的样本数超限
    if self.staleness_samples >= self.max_required_samples:
        return True
    return False
```

> **[维度 2: Backpressure]** 两层背压：queue 物理满 + staleness 计数逻辑满。后者的作用是：即使 queue 还没满（trainer 在消费），但如果 rollouter 已经跑了太多样本没有 sync 权重，也要暂停，避免 off-policy 偏移过大。

### 2f. Staleness 计数与重置

`fully_async_rollouter.py:237-261`:
```python
async def reset_staleness(self):
    async with self.lock:
        self.paused = False
        self.condition.notify_all()                                 # 唤醒被暂停的 processor
        # 重置 staleness，但不清零 — in-flight 和 queue 中的样本仍是旧版本
        self.staleness_samples = len(self.active_tasks) + await self.message_queue_client.get_queue_size()
```

> **细节**: reset 后 staleness 不是 0，而是当前残留的"已生成但未被训练消费"的样本数。这是一个保守但准确的选择。

---

## 第 3 步：Trainer — 消费 + 训练

**文件**: `fully_async_trainer.py`

### 3a. 主循环

`fully_async_trainer.py:389-419`:
```python
async def fit(self):
    while True:
        try:
            await self.fit_step()
        except TrainingStopException:
            break
```

### 3b. 单步训练流程

`fully_async_trainer.py:421-461`:
```python
async def fit_step(self):
    batch = await self._fit_generate(None)      # 从 queue 取数据
    batch = self._fit_compute_reward(batch)      # 计算 reward
    batch = self._fit_compute_log_prob(batch)    # 当前策略 forward → log_prob
    batch = self._fit_compute_ref_log_prob(batch) # ref 策略 forward → ref_log_prob
    batch = self._fit_compute_critic(batch)      # critic forward (可选)
    batch = self._fit_compute_advantage(batch)   # advantage 计算
    batch = self._fit_update_critic(batch)       # critic backward (可选)
    batch = self._fit_update_actor(batch)        # actor backward
    self._fit_update_local_step()                # 版本计数
    await self._fit_update_weights()             # 权重同步（条件触发）
```

### 3c. 从 Queue 取数据

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

> 每次取 `required_samples` 条（= `ppo_mini_batch_size × require_batches`），凑成一个训练 batch。

---

## 第 4 步：权重同步 — Abort → Sleep → Sync → Wake → Resume

### 4a. 触发时机

`fully_async_trainer.py:495-527`:
```python
def _fit_update_local_step(self):
    if self.local_trigger_step < self.trigger_parameter_sync_step:
        self.local_trigger_step += 1       # 还没到触发阈值，递增
    else:
        self.current_param_version += 1    # 达到阈值 → 版本号 +1
        self.local_trigger_step = 1        # 重置

async def _fit_update_weights(self):
    if self.local_trigger_step != 1:       # 只在版本号刚递增时执行
        return
    await self.checkpoint_manager.update_weights(global_steps=self.current_param_version)
    # 通知 Rollouter 重置 staleness
    ray.get(self.rollouter.reset_staleness.remote())
```

> **[维度 3: 触发频率]** 由 `trigger_parameter_sync_step` 控制。值为 2 意味着每 2 个 train step 同步一次权重。

### 4b. CheckpointEngineManager: 8 步流程

**文件**: `checkpoint_engine/base.py:404-445`

```python
async def update_weights(self, global_steps=None):
    # Step 1: abort 所有 in-flight 生成请求
    await asyncio.gather(*[r.abort_all_requests() for r in self.replicas])

    # Step 2: 构建临时 RayWorkerGroup
    workers = []
    for replica in self.replicas:
        workers.extend(replica.workers)
    rollout = RayWorkerGroup(worker_handles=workers, ...)

    # Step 3: sleep replicas — 释放 KV cache 腾出显存给权重传输
    await self.sleep_replicas()

    # Step 4: 建立 NCCL 通信组（trainer rank 0 + 所有 rollout workers）
    self.build_process_group(rollout)

    # Step 5: 实际权重传输 — trainer send + rollout receive
    ray.get(trainer.update_weights(...) + rollout.update_weights(...))

    # Step 6: 销毁通信组
    ray.get(trainer.execute_checkpoint_engine(["finalize"] * ...) + ...)

    # Step 7: wake up replicas — 恢复 KV cache
    await self.wake_up_replicas()

    # Step 8: resume generation — 引擎重新接受请求
    await asyncio.gather(*[r.resume_generation() for r in self.replicas])
```

> **[维度 3: 中断语义]** Step 1 是 **hard abort**，不是 drain。vLLM 调的是 `pause_generation(wait_for_inflight_requests=False)`，SGLang 调的是 `pause_generation(mode="abort")`。两者都是立即终止 in-flight 请求。
>
> **[维度 3: 显存管理]** Step 3 sleep（释放 KV cache）→ Step 7 wake（恢复 KV cache）是一个显存借用技巧：sync 权重需要额外显存放 buffer，sleep 腾出 KV cache 的空间来放。

### 4c. NCCL Bucketed Broadcast（disaggregated 路径）

**文件**: `checkpoint_engine/nccl_checkpoint_engine.py:97-275`

拓扑：trainer rank 0 (master) → 所有 rollout workers (slaves)

```
Trainer rank 0                      Rollout workers 1..N
     │                                    │
     │── ZMQ PUB: bucket metadata ───────>│  (name, shape, dtype, offset)
     │                                    │
     │── NCCL broadcast: bucket data ────>│  (实际 tensor 字节)
     │                                    │
     │   (swap send_buf/recv_buf,         │
     │    pipeline 下一个 bucket)          │
```

`nccl_checkpoint_engine.py:223-275` — `send_weights()`:
```python
for name, weight in weights:
    # 累积到 bucket
    if offset + weight.nbytes > self.bucket_size:
        # bucket 满 → 等上一个 broadcast 完成
        await broadcast_op.wait_for_complete()
        # 发起新 broadcast（ZMQ 发 metadata + NCCL broadcast 发数据）
        broadcast_op = BroadcastOperation(rank=0, bucket=send_buf, metadata=bucket_meta, ...)
        # 双 buffer 交换，pipeline 传输
        send_buf, recv_buf = recv_buf, send_buf
```

> **[维度 3: 传输优化]** 双 buffer + async broadcast 实现了传输和打包的 pipeline。一个 buffer 在 NCCL broadcast，另一个同时在被 CPU 填充下一批权重。

### 4d. ZMQ + CUDA IPC（colocated 路径）

**文件**: `workers/rollout/vllm_rollout/bucketed_weight_transfer.py:73-192`

当 trainer 和 rollout 在同一机器上时，走 CUDA IPC 而非 NCCL：

```python
class BucketedWeightSender:
    async def async_send_weights(self, weights):
        for name, weight in weights:
            if offset + weight.nbytes > self.bucket_size:
                self.socket.send_pyobj({"bucket_meta": bucket_meta, "is_last": False})  # ZMQ 发 metadata
                self.socket.recv()  # 等 receiver 确认
            # 直接写入 CUDA IPC buffer（零拷贝）
            self.buffer[offset:offset+weight.nbytes].copy_(weight.view(-1).view(torch.uint8))
```

> ZMQ 负责元数据协调，实际 tensor 通过 CUDA IPC shared memory 传输，避免了 CPU-GPU 拷贝。

### 4e. vLLM `pause_generation` 语义

**文件**: `vllm/v1/engine/async_llm.py:563-607`

verl 调用 `pause_generation(wait_for_inflight_requests=False, clear_cache=True)`:

1. **`self._paused = True`** — 新请求在 `generate()` 入口被 `asyncio.Condition.wait_for()` 阻塞
2. **`abort(all request_ids)`** — 立即 abort 所有 in-flight 请求
3. **`wait_for_requests_to_drain()`** — 等 engine_core 真正清空（abort 是异步的）
4. **`reset_prefix_cache()` + `reset_mm_cache()`** — 清除缓存（旧权重的 KV cache 与新权重不一致）

`resume_generation()` 只做 `self._paused = False` + `notify_all()`。

### 4f. SGLang `pause_generation(mode="abort")` 语义

**文件**: `sglang/python/sglang/srt/managers/tokenizer_manager.py:1373-1386`

```python
async def pause_generation(self, obj):
    self.is_pause = True                    # 阻塞新请求
    while True:
        self.abort_request(abort_all=True)  # 反复发 abort
        if not await self.model_update_lock.is_locked():
            break                           # 无请求在处理中 → drain 完成
        await asyncio.sleep(1.0)            # 1 秒轮询
```

Scheduler 侧 (`scheduler.py:2989-3086`):
- **waiting_queue**: 直接 pop，无额外开销
- **running_batch**: 设 `to_finish = FINISH_ABORT()`，**仍跑一次 decode forward** 才真正释放 KV cache

> **SGLang vs vLLM**: 语义大致对齐（阻塞新请求 + abort in-flight + 等 drain），但 SGLang 的 drain 是 1 秒轮询 `model_update_lock`，不如 vLLM 的 event-driven `_requests_drained` 精确。且 SGLang 的 abort mode **不经过 scheduler 的 `pause_generation()`**，`_engine_paused` 标记不被设置。

---

## 第 5 步：Staleness 管理 — Rollout Correction

### 5a. Rollout Correction Helper

**文件**: `trainer/ppo/rollout_corr_helper.py`

这是一个独立的 off-policy 修正模块，提供三个核心函数：

| 函数 | 作用 |
|------|------|
| `compute_rollout_correction_weights()` (L481) | 计算 truncated IS weights |
| `compute_rollout_rejection_mask()` (L605) | 计算 rejection sampling mask |
| `compute_rollout_correction_and_rejection_mask()` (L716) | 统一接口：IS + rejection |

### 5b. TIS — Truncated Importance Sampling

`rollout_corr_helper.py:481-540`:
```python
def compute_rollout_correction_weights(log_ratio, response_mask, rollout_is="token", rollout_is_threshold=2.0):
    # log_ratio = log(π_train / π_rollout)

    if rollout_is == "token":
        # Per-token IS: exp(log_ratio)，逐 token 独立
        rollout_is_weights = torch.exp(torch.clamp(log_ratio, -20, 20))

    elif rollout_is == "sequence":
        # Sequence-level IS: exp(sum(log_ratio))，整条序列一个权重
        log_ratio_sum = masked_sum(log_ratio, response_mask, axis=-1).unsqueeze(-1)
        rollout_is_weights = torch.exp(torch.clamp(log_ratio_sum, -20, 20))

    # 截断：clamp 到 [0, threshold]
    rollout_is_weights = torch.clamp(rollout_is_weights, max=rollout_is_threshold)
```

> **与 SLIME TIS 对比**: 机制相同（exp ratio → clamp），但 verl 额外支持 sequence-level aggregation 和 batch normalization。SLIME 只有 token-level。

### 5c. Rejection Sampling

`rollout_corr_helper.py:605+`:

verl 支持多种 rejection 策略（通过 `rollout_rs` 配置）：
- `token_k*`: token-level KL 超限 → mask 整条序列
- `seq_sum_k*` / `seq_mean_k*` / `seq_max_k*`: sequence-level KL 超限 → mask
- 阈值格式: `lower_upper`，指定 IS ratio 的上下界

被 reject 的序列其 `response_mask` 被置 0，不参与 loss 计算。

> **与 SLIME 对比**: SLIME 用 OPSM（advantage<0 且 KL>δ 才 mask），是一种"只 mask 有害的 off-policy 样本"的策略。verl 的 rejection sampling 更通用——纯按 KL divergence 过滤，不看 advantage 方向。

### 5d. bypass_mode — 跳过 old_log_prob 重算

`rollout_corr_helper.py:1039-1074`:

当 `rollout_correction.bypass_mode=True` 时：
```python
def apply_bypass_mode(batch, rollout_corr_config, policy_loss_config):
    # 不用当前策略重算 old_log_prob
    # 直接用 rollout 时记录的 rollout_log_probs 作为 old_log_prob
    batch.batch["old_log_probs"] = batch.batch["rollout_log_probs"]
    policy_loss_config["loss_mode"] = "bypass_mode"
```

> **含义**: 省一次 forward pass，但 old_log_prob 和训练策略可能有偏移。适合 staleness 不大的场景。

### 5e. MIS — Multi-version Importance Sampling

**文件**: `fully_async_trainer.py:473-493`

当 `trigger_parameter_sync_step > 1` 时，一个 weight version 内会有多个 train step，每步的 batch 是在同一版本的 rollout policy 下生成的。但随着 train step 推进，actor 权重在变化。MIS 确保每步都能正确计算 IS ratio：

```python
def _compute_old_log_prob(self, batch):
    if self.local_trigger_step == 1:
        # 第一步：保存当前权重到 CPU（作为 version 1）
        self.actor_rollout_wg.save_model_to_cpu(1)
        old_log_prob = super()._compute_old_log_prob(batch)
    else:
        # 后续步骤：
        self.actor_rollout_wg.save_model_to_cpu(self.local_trigger_step)  # 保存当前版本
        self.actor_rollout_wg.restore_model_from_cpu(1)                   # 恢复 version 1
        old_log_prob = super()._compute_old_log_prob(batch)                # 用 version 1 算 old_log_prob
        self.actor_rollout_wg.restore_model_from_cpu(self.local_trigger_step)  # 恢复当前版本
        self.actor_rollout_wg.clear_cpu_model(self.local_trigger_step)
    return old_log_prob
```

> **[维度 4: MIS]** 这是 verl 独有的机制。SLIME 没有 MIS —— 它每个 rollout 只训练一步就同步权重。verl 允许多步训练（`trigger_parameter_sync_step > 1`），用 CPU 缓存多版本权重来保持 IS ratio 的正确性。代价是 CPU 内存 + GPU↔CPU 拷贝。

---

## 第 6 步：Partial Rollout — Abort + Prefix Continuation

**文件**: `experimental/fully_async_policy/agent_loop/agent_loop.py:40-123`

### 6a. 核心机制

`FullyAsyncLLMServerManager` 继承 `AsyncLLMServerManager`，重写 `generate()`：

```python
class FullyAsyncLLMServerManager(AsyncLLMServerManager):
    async def generate(self, request_id, *, prompt_ids, sampling_params, ...):
        final_output = TokenOutput(token_ids=[], log_probs=[], num_preempted=0)
        min_global_steps, max_global_steps = None, None

        while True:
            # ① 调父类 generate，prompt = 原始 prompt + 已生成的 token
            output = await super().generate(
                prompt_ids=prompt_ids + final_output.token_ids,
                sampling_params=sampling_params,
            )

            # ② 追加本轮输出到累积 buffer
            final_output.token_ids.extend(output.token_ids)
            final_output.log_probs.extend(output.log_probs)

            # ③ 记录跨版本信息
            global_steps = output.extra_fields.get("global_steps", None)
            if min_global_steps is None:
                min_global_steps = global_steps
            max_global_steps = global_steps

            # ④ 扣减剩余 max_tokens
            sampling_params[limit_key] = original_max_tokens - len(final_output.token_ids)

            # ⑤ 如果不是被 abort 打断的，或者没开 partial_rollout，退出循环
            if output.stop_reason not in ("aborted", "abort") or not self.config.async_training.partial_rollout:
                break

        final_output.extra_fields["min_global_steps"] = min_global_steps
        final_output.extra_fields["max_global_steps"] = max_global_steps
        return final_output
```

### 6b. 时序

```
Rollouter                              Engine                         Trainer
   │                                     │                              │
   │── generate(prompt) ────────────────>│ decode t1,t2,t3...          │
   │                                     │                              │
   │                                     │         weight sync 触发 ────┤
   │                                     │<── abort_all_requests ──────│
   │<── output(tokens=[t1,t2,t3],        │                              │
   │         stop_reason="abort") ───────│                              │
   │                                     │<── update_weights ──────────│
   │                                     │<── resume_generation ───────│
   │                                     │                              │
   │  stop_reason=="abort" &&            │                              │
   │  partial_rollout=True               │                              │
   │  → while 循环继续                    │                              │
   │                                     │                              │
   │── generate(prompt+[t1,t2,t3]) ─────>│ 用新权重继续 decode          │
   │         max_tokens -= 3             │ t4,t5,...                    │
   │                                     │                              │
   │<── output(tokens=[t4,t5,...],       │                              │
   │         stop_reason="stop") ────────│                              │
   │                                     │                              │
   │  stop_reason=="stop" → break        │                              │
   │  返回 [t1,t2,t3,t4,t5,...]         │                              │
   │  min_global_steps=V1                │                              │
   │  max_global_steps=V2                │                              │
```

### 6c. 与 SLIME Partial Rollout 的对比

| 维度 | SLIME | verl |
|------|-------|------|
| abort 后的 partial tokens | 放入 recycle buffer，下次取出继续生成 | 在同一 `generate()` 调用内循环重试 |
| 旧 token 的处理 | `loss_mask = [0] * old_len`（off-policy token masking） | 不做 token-level masking，但记录 `min/max_global_steps` 供 trainer 判断 |
| 恢复方式 | 下一次 `data_source.get_samples()` 取出 | 当前 `while True` 循环直接重提交 |
| prefix cache 利用 | 可以（SGLang radix cache） | 不可以（`clear_cache=True` 清掉了 prefix cache） |
| 可感知性 | Rollouter 知道 abort 发生，显式管理 | 对 Rollouter 上层透明 —— abort/resume 在 `FullyAsyncLLMServerManager` 内部处理 |

> **[维度 5]** 两者本质上都是 **abort + prefix retry**。Survey 的分类（"explicit save/resume" vs "abort + retry with prefix"）在代码层面没有区别。verl 的特色是：(1) 对上层调用者透明——`generate()` 内部 while 循环自动重试；(2) 记录 `min/max_global_steps` 让 trainer 知道这个 rollout 跨了多少 policy version。

---

## 全流程 + 维度标注图

```
fully_async_main.py                                Blog 维度
──────────────────────────────────────────────────────────
[启动]
  FullyAsyncTrainer.remote()                       ← 训练 actor
  FullyAsyncRollouter.remote()                     ← 推理 actor
  MessageQueue.remote(max_queue_size)              ← [维度2] bounded buffer
  trainer.set_rollouter(rollouter)                 ← 引用注入
  trainer._fit_update_weights()                    ← 初始权重同步 [维度3]

[并行运行]
  rollouter.fit() ─────────────────────────────────── 独立 Ray actor
  │ ┌─ _streaming_generation_main() ──────────────────────────────
  │ │  _feed_samples():
  │ │    for epoch, batch_dict in dataloader:
  │ │      pending_queue.put(RolloutSample)
  │ │
  │ │  _processor_worker():
  │ │    while True:
  │ │      _should_pause_generation()?              ← [维度2] backpressure
  │ │        queue_size >= max_queue_size
  │ │        staleness_samples >= max_required
  │ │      → paused: await condition.wait()
  │ │
  │ │      rollout_sample = pending_queue.get()
  │ │      staleness_samples += 1
  │ │      │
  │ │      └─ _process_single_sample_streaming() ─────────────────
  │ │           FullyAsyncLLMServerManager.generate()
  │ │           │  while True:                      ← [维度5] partial rollout 循环
  │ │           │    output = super().generate(
  │ │           │      prompt + accumulated_tokens)
  │ │           │    accumulate tokens, log_probs
  │ │           │    track min/max_global_steps      ← [维度4] 跨版本追踪
  │ │           │    if stop_reason != "abort":
  │ │           │      break
  │ │           │
  │ │           message_queue.put_sample(result)     ← [维度2] 推入 queue
  │ └──────────────────────────────────────────────────────────────

  trainer.fit() ───────────────────────────────────── 独立 Ray actor
  │ while True:
  │   fit_step():
  │   │ ┌─ _fit_generate() ──────────────────────────────────────
  │   │ │  _get_samples_from_queue()                ← [维度2] 从 queue 取 N 条
  │   │ │  assemble_batch_from_rollout_samples()
  │   │ └────────────────────────────────────────────────────────
  │   │
  │   │ _fit_compute_reward()
  │   │ _fit_compute_log_prob()                     ← [维度4] 当前策略 forward
  │   │ _fit_compute_ref_log_prob()
  │   │ _fit_compute_advantage()
  │   │ _fit_update_critic()
  │   │
  │   │ ┌─ _fit_update_actor() ──────────────────────────────────
  │   │ │  (内部使用 rollout_corr_helper)
  │   │ │  compute_rollout_correction_weights()     ← [维度4] TIS
  │   │ │  compute_rollout_rejection_mask()         ← [维度4] rejection sampling
  │   │ │  _compute_old_log_prob()                  ← [维度4] MIS (多版本)
  │   │ │    save_model_to_cpu() / restore()
  │   │ │  backward + optimizer step
  │   │ └────────────────────────────────────────────────────────
  │   │
  │   │ _fit_update_local_step()
  │   │   local_trigger_step++
  │   │   if >= trigger_parameter_sync_step:
  │   │     current_param_version++
  │   │
  │   │ ┌─ _fit_update_weights() (条件触发) ─────────────────────
  │   │ │  CheckpointEngineManager.update_weights():
  │   │ │    1. abort_all_requests()                ← [维度3][维度5] hard abort
  │   │ │    2. 构建临时 worker group
  │   │ │    3. sleep_replicas()                    ← [维度3] 释放 KV cache
  │   │ │    4. build_process_group()               ← [维度3] 建立 NCCL 组
  │   │ │    5. NCCL bucketed broadcast             ← [维度3] 传输权重
  │   │ │    6. finalize()                          ← [维度3] 清理通信组
  │   │ │    7. wake_up_replicas()                  ← [维度3] 恢复 KV cache
  │   │ │    8. resume_generation()                 ← [维度3][维度5] 恢复生成
  │   │ │
  │   │ │  rollouter.reset_staleness()              ← [维度2] 重置 backpressure
  │   │ └────────────────────────────────────────────────────────
  │   │
  │   │ _fit_validate()
  │   │ _fit_save_checkpoint()
```

---

## 关键配置参数速查

| 参数 | 作用 | 对应维度 |
|------|------|---------|
| `async_training.staleness_threshold` | 允许的 staleness 层数，影响 queue 深度 | 维度 2 |
| `async_training.require_batches` | 每次训练取多少个 mini-batch | 维度 2 |
| `async_training.trigger_parameter_sync_step` | 每 N 个 train step 同步一次权重 | 维度 3 |
| `async_training.partial_rollout` | 启用 abort + prefix continuation | 维度 5 |
| `algorithm.rollout_correction.bypass_mode` | 跳过 old_log_prob 重算 | 维度 4 |
| `algorithm.rollout_correction.rollout_is` | IS 聚合级别: "token" / "sequence" | 维度 4 |
| `algorithm.rollout_correction.rollout_is_threshold` | TIS 截断上限 (默认 2.0) | 维度 4 |
| `algorithm.rollout_correction.rollout_rs` | rejection sampling 策略 | 维度 4 |
| `actor_rollout_ref.rollout.checkpoint_engine.backend` | 权重传输后端: "nccl" / "nixl" / "naive" | 维度 3 |
| `actor_rollout_ref.rollout.checkpoint_engine.bucket_size_mb` | bucketed broadcast 桶大小 | 维度 3 |

---

## Survey 描述修正

基于代码分析，survey 对 verl 的几处描述需要修正：

| Survey 说的 | 实际代码 |
|---|---|
| "Soft pause with resume-oriented handling" | **Hard abort** (`wait_for_inflight_requests=False`)，不等自然结束 |
| "Explicit save/resume" (partial rollout) | **Abort + prefix retry**，与 SLIME/SkyRL 同类。token 保存在应用层 `final_output`，不在引擎层 |
| "NCCL + bucketing" (暗示纯 NCCL) | NCCL 是 disaggregated 路径；colocated 路径是 ZMQ + CUDA IPC。两路径可共存 |
| "Clipped TIS, optional OPSM" | 实际是 TIS + rejection sampling（基于 KL divergence，非 OPSM）。verl 的 `rollout_corr_helper` 不包含 OPSM |
