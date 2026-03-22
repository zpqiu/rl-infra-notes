# NeMo-RL Async RL Training: Code Walk-Through

> **Codebase**: [NVIDIA/NeMo-RL](https://github.com/NVIDIA/NeMo-RL) @ commit `94fa37d9`（latest release: v0.5.0）
>
> **参考 Blog**: [Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries](https://huggingface.co/blog/async-rl-training-landscape) (HuggingFace, 2026-03-10)
>
> **关注维度**: Blog 7 个维度中的 4 个:
> - 维度 2: Rollout Buffer 设计
> - 维度 3: 权重同步协议
> - 维度 4: Staleness 管理
> - 维度 5: Partial Rollout 处理

## NeMo-RL 在 Blog 框架中的定位

| 维度 | NeMo-RL 的选择 | 激进程度 |
|------|----------------|---------|
| **Rollout Buffer** | Replay Buffer（深度=max_trajectory_age_steps，默认 1） | 灵活 — 1 步时等价 double-buffer，可调至 8 步 |
| **权重同步** | NCCL collective broadcast（非 colocate）/ ZMQ IPC（colocate），支持 in-flight weight update | 激进 — 可在推理进行中更新权重 |
| **Staleness 管理** | Hybrid：version-aware filtering（target match + age window）+ required IS correction | 中等偏保守 — 先筛掉不该进入当前步的样本，再用 IS 修正残余 off-policy |
| **Partial Rollout** | 无 abort/recycle — 所有 sample 独立完成后入 buffer | 保守 — 不存在 partial rollout |

## 文件阅读顺序

按执行流顺序打开这些文件即可走完全流程：

1. `RL/nemo_rl/algorithms/grpo.py:2368` — `async_grpo_train()` 入口
2. `RL/nemo_rl/algorithms/async_utils.py:36` — `ReplayBuffer` Replay Buffer
3. `RL/nemo_rl/algorithms/async_utils.py:239` — `AsyncTrajectoryCollector` 后台生成调度
4. `RL/nemo_rl/experience/rollouts.py:862` — `run_async_multi_turn_rollout()` 异步 rollout
5. `RL/nemo_rl/models/generation/vllm/vllm_worker_async.py:674` — vLLM 异步生成
6. `RL/nemo_rl/algorithms/grpo.py:1104` — `refit_policy_generation()` 权重同步
7. `RL/nemo_rl/algorithms/loss/loss_functions.py:367` — Importance Sampling 修正

---

## 第 0 步：启动 — 校验、创建组件

**入口**: `grpo.py:2368` → `async_grpo_train()`

### 0a. 前置校验

`grpo.py:2401-2416`:
```python
# 必须使用 vLLM async engine
assert _should_use_async_rollouts(master_config)
# 必须开启 IS 修正（off-policy 收敛保证）
assert master_config["loss_fn"]["use_importance_sampling_correction"] is True
# 禁止 colocated inference（训练和推理必须物理隔离）
assert not colocated_inference
```

> **[维度 3: 权重同步]** Async 模式强制要求 **disaggregated 部署**（训练和推理在不同 GPU 上），与 SLIME 的设计一致。权重同步通过 NCCL collective broadcast（非 colocate 路径）。

### 0b. 创建 ReplayBuffer

`grpo.py:2490-2501`:
```python
# Buffer 大小 = num_prompts_per_step × max_trajectory_age_steps × 2(slack)
optimal_buffer_size = num_prompts_per_step * max_trajectory_age_steps * late_arrival_slack
replay_buffer = ReplayBuffer.remote(max_size=optimal_buffer_size)
```

> **[维度 2: Rollout Buffer]** 这是一个 **per-prompt group 粒度**的 Replay Buffer。每个 entry 是 1 个 prompt × `num_generations_per_prompt` 个 completion。Buffer 深度由 `max_trajectory_age_steps` 控制。

### 0c. 创建 AsyncTrajectoryCollector

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
# 启动后台生成线程
trajectory_collector.start_collection.remote(dataloader)
trajectory_collector.set_weight_version.remote(weight_version)
```

Collector 是一个 **Ray remote actor**，内部启动一个 **daemon 后台线程**持续从 dataloader 取 batch、发起推理、将结果推入 ReplayBuffer。

### 0d. 初始权重同步（Refit）

`grpo.py:2546-2568`:
```python
if NEED_REFIT and POLICY_GENERATION_STALE:
    refit_policy_generation(policy, policy_generation, colocated_inference)
    POLICY_GENERATION_STALE = False
```

确保 vLLM 推理引擎拥有训练模型的初始权重。

### 0e. Buffer 预热

`grpo.py:2607-2624`:
```python
while True:
    buffer_size_current = ray.get(replay_buffer.size.remote())
    if buffer_size_current >= min_trajectories_needed:
        break
    time.sleep(1.0)
```

训练必须等到 Buffer 中有至少 `num_prompts_per_step` 条 trajectory 才开始。在此期间后台 Collector 持续填充 Buffer。

---

## 第 1 步：主循环 — Replay Buffer 驱动的异步流水线

**文件**: `grpo.py:2626-2887`

与 SLIME 的 double-buffer 主循环（~40 行）不同，NeMo-RL 的异步由 **ReplayBuffer + 后台 Collector** 解耦实现，主循环本身是一个标准的 train loop：

```python
while step < max_num_steps:
    # [A] 从 ReplayBuffer 采样（可能 stall）
    sample_result = ray.get(replay_buffer.sample.remote(
        num_prompt_groups=num_prompts_per_step,
        current_weight_version=weight_version,
        max_age_steps=max_trajectory_age_steps,
    ))
    if sample_result is None:
        time.sleep(0.5)
        continue  # Buffer 不够 → 等待后台 Collector 填充

    # [B] 训练（logprob → advantage → policy loss → backward）
    fprop_logprobs = policy.get_logprobs(train_data)
    reference_logprobs = policy.get_reference_policy_logprobs(train_data)
    advantages = adv_estimator.compute_advantage(...)
    train_results = policy.train(train_data, loss_fn)

    # [C] 权重同步
    ray.get(trajectory_collector.prepare_for_refit.remote())
    refit_policy_generation(policy, policy_generation, colocated_inference)
    weight_version += 1
    trajectory_collector.set_weight_version.remote(weight_version)
    trajectory_collector.resume_after_refit.remote()
```

> **[维度 2: Rollout Buffer]** 与 SLIME 的对比：
> ```
> SLIME (double-buffer):
>   主循环亲自调度 rollout，最多 1 个 in-flight rollout
>   时间线: [Gen 0] [Gen 1] [Gen 2]
>                    [Train 0] [Train 1]
>
> NeMo-RL (replay buffer):
>   后台 Collector 持续生成，主循环只管从 Buffer 取数据训练
>   时间线: [Gen 0][Gen 1][Gen 2][Gen 3][Gen 4]...  ← 后台持续
>                         [Train 0]  [Train 1]...    ← 按需取
>   最多 num_prompts_per_step × max_trajectory_age_steps 个 in-flight
> ```
> NeMo-RL 的 Buffer 更深，Generation 和 Training **完全解耦**。

---

## 第 2 步：后台 Rollout — AsyncTrajectoryCollector

**文件**: `async_utils.py:239-754`

### 2a. 后台收集循环

`async_utils.py:392-448` — `_collection_loop()`:
```python
def _collection_loop(self):
    for batch in self.dataloader:
        if not self.running: break
        self._manual_pause_cleared.wait()   # 手动暂停检查
        self._refit_pause_cleared.wait()    # 权重同步暂停检查
        # 生成限制检查（所有 target weights 已生成 → 暂停等权重更新）
        if self._should_pause_for_generation_limits():
            self._generation_limit_cleared.wait()
        self._process_batch(batch)
```

三层 Event 门控：
- `_manual_pause_cleared`: 手动暂停（validation 时用）
- `_refit_pause_cleared`: 权重同步期间暂停新生成
- `_generation_limit_cleared`: 所有可用 target weight 都已有数据 → 等新的权重版本

### 2b. Weight Version 与 Target Weight 机制

`async_utils.py:294-321` — `_calculate_target_weights()`:
```python
def _calculate_target_weights(self, generation_weight_version):
    """
    Example:
      generation_weight_version = 10
      max_trajectory_age_steps = 4
    Returns: [11, 12, 13, 14]
    即：用 v10 权重生成的数据可以服务于训练步 11, 12, 13, 14
    """
    return [generation_weight_version + i for i in range(1, max_age + 1)]
```

> **[维度 4: Staleness]** 这是 NeMo-RL 独特的 **target weight** 设计：
> - 每条 trajectory 不只记录"用哪个权重版本**生成**"(`trajectory_version`)，还标注"**目标服务**哪个训练步"(`target_weight_version`)
> - Buffer 采样时只取 `target_weight_version == current_weight_version` 的数据
> - 这确保每个训练步拿到的数据是"被设计服务于该步"的，避免随机 staleness
>
> 这已经不只是"version tracking"，而是**真正参与采样/过滤的 version-aware filtering**：哪些旧样本能进入当前训练步，不是交给 loss 再兜底，而是在 buffer 侧先被显式约束。

`async_utils.py:323-342` — `_get_next_target_for_generation()`:
```python
def _get_next_target_for_generation(self, generation_weight_version):
    target_weights = self._calculate_target_weights(generation_weight_version)
    last_generated = ray.get(
        self.replay_buffer.get_last_target_weight_already_generated.remote()
    )
    for target_weight in target_weights:
        if target_weight > last_generated and target_weight not in self._generating_targets:
            self._generating_targets.add(target_weight)  # 预留
            return target_weight
    return None  # 所有 target 都已在生成/已生成
```

### 2c. 并发生成 — 每个 prompt 一个 worker thread

`async_utils.py:451-508` — `_process_batch()`:
```python
def _process_batch(self, batch):
    target_weight = self._get_next_target_for_generation(generation_weight_version)
    if target_weight is None: return  # 无需生成

    for prompt_idx in range(num_prompts):
        self._inflight_sema.acquire()  # 控制并发上限
        worker = threading.Thread(
            target=self._run_prompt_group_worker,
            args=(repeated_batch, generation_weight_version, target_weight, prompt_idx),
        )
        self._inflight_threads.add(worker)
        worker.start()
```

并发控制：
- `_inflight_sema`: Semaphore，上限为 `num_prompts_per_step × max_trajectory_age_steps`
- 每个 worker thread 独立运行一个 prompt group 的完整 rollout

> **Batch 边界不是同步屏障**：`_process_batch()` 只 spawn 线程，不 join — 返回后 `_collection_loop` 立即取下一个 batch。跨 batch 的 prompt group 可以并发 in-flight。真正的流控是 semaphore（prompt 粒度）、generation limit（target weight 粒度）和 refit pause（权重同步期间），而非 batch 完成。

### 2d. 单个 prompt group 的生成

`async_utils.py:637-754` — `_run_prompt_group_worker()`:
```python
def _run_prompt_group_worker(self, repeated_batch, generation_weight_version, target_weight_version, prompt_idx):
    # 运行完整 rollout（可能是多轮对话）
    final_batch, rollout_metrics = run_async_multi_turn_rollout(
        policy_generation=self.policy_generation,
        input_batch=repeated_batch,
        tokenizer=self.tokenizer,
        task_to_env=self.task_to_env,
        max_seq_len=...,
        max_rollout_turns=...,
    )

    # 结果打包推入 ReplayBuffer
    trajectory_group = {
        "batch": final_batch.to("cpu"),
        "rollout_metrics": rollout_metrics,
        "timestamp": time.time(),
    }
    # 指数退避重试，直到 Buffer 接受
    while self.running:
        status = ray.get(self.replay_buffer.push_with_wait_signal.remote(
            trajectory_group, generation_weight_version, target_weight_version,
        ))
        if status == "success": break
        elif status == "full": time.sleep(min(backoff_delay, 0.5))
```

> **[维度 5: Partial Rollout]** NeMo-RL **不支持** partial rollout/abort/recycle。每个 prompt group 的所有 completion 独立运行完成后才入 Buffer。这比 SLIME 更简单但牺牲了部分 GPU 利用率——长尾 sample 可能拖慢整体 throughput。

---

## 第 3 步：ReplayBuffer — 带版本感知的采样

**文件**: `async_utils.py:36-236`

### 3a. 数据结构

```python
class ReplayBuffer:
    trajectories = []           # List[dict]，每个是一个 prompt group
    trajectory_versions = []    # 生成时的权重版本
    target_weight_versions = [] # 目标训练步
    last_target_weight_already_generated = -1  # 最大已生成 target
```

### 3b. 采样逻辑

`async_utils.py:102-223` — `sample()`:
```python
def sample(self, num_prompt_groups, current_weight_version, max_age_steps):
    # ① 计算有效版本窗口
    min_valid_version = max(0, current_weight_version - max_age_steps)

    # ② 过滤有效数据
    valid_indices = [i for i, v in enumerate(self.trajectory_versions)
                     if min_valid_version <= v <= current_weight_version]

    # ③ 只选目标版本匹配的
    intended_indices = [i for i in valid_indices
                        if self.target_weight_versions[i] == current_weight_version]

    # ④ 数量不足 → 返回 None（stall 训练）
    if len(intended_indices) < num_prompt_groups:
        return None

    # ⑤ 选取 + 从 buffer 中移除
    selected = intended_indices[:num_prompt_groups]
    avg_trajectory_age = current_weight_version - mean(trajectory_versions[selected])
    # ... 移除并返回
```

> **[维度 2 + 维度 4]** 采样策略的关键设计：
> - **Target matching**：只取 `target_weight_version == current_weight_version` 的数据，保证每个训练步的数据都是"为它准备的"
> - **Age window**：超过 `max_age_steps` 的数据视为过时，触发 ValueError
> - **Stall semantics**：数据不够就 stall 训练，宁可等也不凑合 — 保守但稳定
> - **avg_trajectory_age** 作为 metric 返回，反映 off-policy 程度
>
> 如果严格只看 **staleness management** 本身，而不把一般性的并发/容量控制混进来，NeMo-RL 默认 async 路径主要用了两类机制：
> - **Strategy 1: Per-sample version rejection / filtering**：`target_weight_version == current_weight_version` 的 target matching，加上 `trajectory_version` 的 age window，决定哪些样本能进当前步。这是 **sample filtering**。
> - **Strategy 3: IS-weighted loss correction**：后面的 TIS / ICE-POP / seq-mask-TIS
>
> `ReplayBuffer` 大小、`_inflight_sema`、以及后台 collector 的流控，更适合放在 **Rollout Buffer / Orchestration** 维度理解；它们会间接影响 stale backlog，但不是直接决定样本是否被当前训练步接纳的规则。

---

## 第 4 步：权重同步 — 两种模式

**触发**: `grpo.py:2857-2883`，**每个训练步结束后立即触发**（与 SLIME 的 `update_weights_interval` 间隔不同）

### 4a. Prepare for Refit — 暂停 / 等待

`async_utils.py:529-576` — `prepare_for_refit()`:

```python
def prepare_for_refit(self):
    self._refit_pause_cleared.clear()  # 暂停新生成

    if is_async_engine and in_flight_weight_updates:
        # 模式 A: In-Flight — 不等待，正在进行的生成继续用旧权重
        print("Skipping wait for pending generations")
    else:
        # 模式 B: Blocking — 等所有 pending 线程完成
        self.wait_for_pending_generations()
```

> **[维度 3: 权重同步]** 两种模式的对比：
>
> | | **Blocking 模式** | **In-Flight 模式** |
> |---|---|---|
> | 配置 | `in_flight_weight_updates: false` | `in_flight_weight_updates: true` |
> | 等待 | 等所有 pending 生成完成 | 只暂停新生成，不等待进行中的 |
> | KV cache | 不需要失效 | 可选 recompute（AREAL-style） |
> | Staleness | 低（无 mid-generation 权重变化） | 高（部分 token 用旧权重生成） |
> | 适用 | `max_trajectory_age_steps=1` | `max_trajectory_age_steps>1` 时才有性能收益 |
>
> 如果 `max_trajectory_age_steps > 1` 但没开 in-flight，代码会打印 warning（`grpo.py:2409-2416`）。

### 4b. Refit — NCCL Collective Broadcast

`grpo.py:1104-1198` — `refit_policy_generation()`:

非 colocate 路径（async 模式的唯一路径）：
```python
# 训练侧 broadcast 权重
futures_train = policy.broadcast_weights_for_collective(kv_scales=kv_scales)
# 推理侧接收
futures_inference = policy_generation.update_weights_from_collective()
# 等待完成
ray.get(futures_train)
results = ray.get(futures_inference)
```

> **[维度 3]** 与 SLIME 的对比：
> - SLIME：TP all-gather → HF 转换 → bucketed NCCL broadcast，需要 pause → flush KV cache → lock
> - NeMo-RL：直接 NCCL collective broadcast（由 vLLM `collective_rpc` 实现），更简洁
> - NeMo-RL 不需要 HF 格式转换（vLLM 直接接收训练格式）

vLLM 侧的接收（`vllm_generation.py:833-852`）：
```python
def update_weights_from_collective(self):
    method_name = "update_weights_from_collective_async"  # async engine 路径
    futures = self.worker_group.run_all_workers_single_data(method_name, ...)
    return futures
```

最终调用 vLLM 的 `collective_rpc("update_weights_from_collective")`（`vllm_worker_async.py:1073-1085`）。

### 4c. Resume — 恢复生成 + 可选 KV cache 失效

`async_utils.py:578-601` — `resume_after_refit()`:
```python
def resume_after_refit(self):
    # AREAL-style: 失效 KV cache → 新生成用新权重 KV
    if in_flight_weight_updates and recompute_kv_cache_after_weight_updates:
        self.policy_generation.invalidate_kv_cache()

    self._refit_pause_cleared.set()  # 恢复新生成
```

> **[维度 3]** KV cache 失效策略的选择：
> - **recompute = true (AREAL-style)**：In-flight 生成完成后，后续生成会用新权重重算 KV cache。更准确但有开销。
> - **recompute = false (Magistral-style)**：保留旧 KV cache 继续用。更快但 prefix 部分的 KV 基于旧权重。

### 4d. Version 递增与通知

`grpo.py:2880-2883`:
```python
weight_version += 1
trajectory_collector.set_weight_version.remote(weight_version)
trajectory_collector.resume_after_refit.remote()
```

`async_utils.py:344-353` — Collector 收到新版本后唤醒因 generation limit 暂停的生成：
```python
def set_weight_version(self, version):
    self.current_weight_version = version
    if not self._generation_limit_cleared.is_set():
        self._generation_limit_cleared.set()  # 唤醒暂停的 collection loop
```

---

## 第 5 步：Rollout 层 — vLLM Async Engine

**调用链**:
```
_run_prompt_group_worker()
  → run_async_multi_turn_rollout()          # rollouts.py:862
    → asyncio.gather(*sample_tasks)         # 并发运行所有 sample
      → run_sample_multi_turn_rollout()
        → generate_responses_async()
          → VllmGeneration.generate_async()  # vllm_generation.py:710
            → VllmAsyncGenerationWorker.generate_async()  # vllm_worker_async.py:674
              → AsyncLLM.generate()          # vLLM V1 async engine
```

### 5a. 异步 Rollout 编排

`rollouts.py:891-1053` — `_async_rollout_implementation()`:
```python
async def _async_rollout_implementation():
    # 每个 sample 独立运行完整的多轮对话
    sample_tasks = [
        run_single_sample_with_error_handling(i, sample_state)
        for i, sample_state in enumerate(sample_initial_states)
    ]
    # 所有 sample 并发执行
    sample_results = await asyncio.gather(*sample_tasks, return_exceptions=False)
```

### 5b. vLLM Async 生成

`vllm_worker_async.py:674-707`:
```python
async def generate_async(self, data, greedy=False):
    """单个 sample 的异步生成，使用 vLLM V1 AsyncLLM"""
    assert batch_size == 1  # 每次只处理一个 sample
    vllm_request_generator = self.llm.generate(
        prompt=prompt, sampling_params=..., request_id=str(uuid.uuid4()),
    )
    async for req_output in vllm_request_generator:
        final_request_output = req_output
    # 提取 token ids + logprobs
```

> **关键区别**：NeMo-RL 使用 vLLM V1 的 `AsyncLLM`（`vllm_worker_async.py:168`），而 SLIME 使用 SGLang。vLLM V1 的 `collective_rpc` 允许在推理进行中直接广播权重（in-flight weight updates 的基础）。

### 5c. Generation Logprobs

`vllm_worker_async.py:838-854`:
```python
if hasattr(generation_details, "logprobs") and generation_details.logprobs:
    for idx, logprob_dict_per_token in enumerate(generation_details.logprobs):
        # 提取每个 token 的 log probability
```

这些 generation logprobs 存入 `message_log` 中的 `generation_logprobs` 字段，后续训练时作为 behavior policy π_old 使用。

---

## 第 6 步：训练 — Logprob → Advantage → Loss(IS) → Backward

**文件**: `grpo.py:2786-2856`

### 6a. Logprob 计算

`grpo.py:2788-2802`:
```python
# 当前策略 forward → logprobs
fprop_logprobs = policy.get_logprobs(train_data)["logprobs"]
# Reference policy forward → reference logprobs
reference_logprobs = policy.get_reference_policy_logprobs(train_data)["reference_logprobs"]
train_data["prev_logprobs"] = fprop_logprobs           # π_θ (当前策略)
train_data["reference_policy_logprobs"] = reference_logprobs
```

> **[维度 4: Staleness]** 对比 SLIME 的两种 old log probs 来源：
> - SLIME `use_rollout_logprobs=True`: 用推理引擎记录的 `rollout_log_probs`
> - SLIME `use_rollout_logprobs=False`: 用当前 actor 权重重新 forward
> - NeMo-RL: 总是用**当前策略**重新 forward 算 `prev_logprobs`，同时保留 `generation_logprobs`（推理时记录的）用于 IS 修正

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

当 `generation_logprobs` 和 `prev_logprobs` 的序列级差异超过阈值时，mask 掉该序列。这是对 staleness 的额外保护。

### 6c. Advantage 计算

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

支持 GRPO、GDPO、Reinforce++ 等 advantage estimator。

### 6d. Policy Loss — Importance Sampling 修正

**文件**: `loss_functions.py:367-492` — `ClippedPGLossFn.__call__()`

这是 staleness 修正的**核心战场**：

```python
# ① 标准 PPO clipped ratio
log_ratios = curr_logprobs - prev_logprobs       # log(π_θ / π_θ_old)
ratios = log_ratios.exp()
ratios_clamped = ratios.clamp(1 - eps_min, 1 + eps_max)
clip_loss = max(-advantages * ratios, -advantages * ratios_clamped)

# ② Importance Sampling 修正
actor_importance_weights = exp(prev_logprobs - generation_logprobs)  # π_θ_old / π_gen
#                                                                       ↑ 当前策略 / 生成时策略

# ③ Truncated IS（三种变体）
if tis_type == "tis":
    weights = clamp(weights, max=tis_ratio)           # 截断上界
elif tis_type == "icepop":
    weights = where(in_bounds, weights, 0)             # 越界→置零
elif tis_type == "seq-mask-tis":
    seq_geomean = exp(mean(log(weights)))              # 序列级几何均值
    seq_mask = (seq_geomean >= min) & (seq_geomean <= max)
    weights = weights * seq_mask                       # 序列级门控，保留 token 级权重

# ④ 最终 loss
loss = masked_mean(importance_weights * clip_loss, mask)
```

> **[维度 4: Staleness 管理]** 三种 IS 策略对比（`loss_functions.py:389-468`）：
>
> | 策略 | 作用粒度 | 行为 | 典型参数 |
> |------|---------|------|---------|
> | **TIS** | Token 级 | clamp(IS weight, max=T) | T=5.0 |
> | **ICE-POP** | Token 级 | IS weight ∉ [min,max] → 置零 | [0.5, 5.0] |
> | **seq-mask-TIS** | 序列级门控 + Token 级修正 | 几何均值 IS ratio ∉ [min,max] → 整条序列置零 | [0.999, 1.002] |
>
> 与 SLIME 的对比：
> - SLIME 用 **TIS + OPSM** 两层修正（先序列级 mask，再 token 级加权）
> - **但 NeMo-RL 的整体 staleness 管理并不只是 loss 层 IS**：默认 async 路径在进入 loss 前，已经通过 replay buffer 的 target matching、age window 和 in-flight/buffer 深度做过系统层约束
> - loss 层这一段则提供 **TIS / ICE-POP / seq-mask-TIS** 三种 IS 变体可选
> - NeMo-RL 的 `seq-mask-tis` 类似 SLIME 的 OPSM（序列级门控），但基于 IS ratio 而非 KL + advantage

### 6e. 额外支持：Sequence-level IS（GSPO）

`loss_functions.py:372-380`:
```python
if self.sequence_level_importance_ratios:
    # GSPO: 序列级 IS 权重
    seq_lp_diff = ((prev_logprobs - generation_logprobs) * mask).sum(dim=-1)
    actor_importance_weights = exp(seq_lp_diff)
```

当 `sequence_level_importance_ratios=True` 时，IS 权重在序列级计算（GSPO 论文的做法）。

---

## 第 7 步：回到第 1 步

训练步完成 → 权重同步 → `weight_version++` → 后台 Collector 被唤醒 → 用新权重生成新的 trajectory 并推入 Buffer → 主循环从 Buffer 采样下一步数据。

---

## 全流程 + 维度标注图

```
grpo.py:async_grpo_train()                      Blog 维度
─────────────────────────────────────────────────────────
[启动]
  校验: async_engine=true, IS=true, 非colocate   ← 物理隔离(disaggregated)
  ReplayBuffer.remote(max_size=...)              ← [维度2] buffer 创建
  AsyncTrajectoryCollector.remote(...)
  start_collection(dataloader)                   ← 后台生成线程启动
  refit_policy_generation()                      ← 初始权重同步 [维度3]
  wait for buffer >= min_trajectories             ← buffer 预热

[主循环]
  while step < max_num_steps:
    ┌─ replay_buffer.sample() ──────────────────────────────
    │  filter: target_weight_version == current   ← [维度2+4] 版本感知采样
    │  数量不足 → stall 等待                       ← [维度2] 训练节奏由 buffer 控制
    └───────────────────────────────────────────────────────

    ┌─ 训练 ────────────────────────────────────────────────
    │  policy.get_logprobs() → prev_logprobs      ← [维度4] on-policy logprob
    │  policy.get_reference_policy_logprobs()
    │  seq_logprob_error_masking()                ← [维度4] 额外 staleness 保护
    │  adv_estimator.compute_advantage()
    │  ClippedPGLossFn():
    │    PPO clipped ratio
    │    IS weights = exp(prev_lp - gen_lp)       ← [维度4] behavior/generation ratio
    │    TIS / ICE-POP / seq-mask-TIS 截断        ← [维度4] IS 变体
    │    loss = IS_weights × clip_loss
    │  policy.train() → backward + optimizer step
    └───────────────────────────────────────────────────────

    ┌─ 权重同步 ────────────────────────────────────────────
    │  trajectory_collector.prepare_for_refit()
    │  │  暂停新生成
    │  │  [IF in_flight_weight_updates]
    │  │    跳过等待 — 进行中的生成继续              ← [维度3] in-flight 模式
    │  │  [ELSE]
    │  │    等待所有 pending 线程完成                ← [维度3] blocking 模式
    │  │
    │  refit_policy_generation()
    │  │  policy.broadcast_weights_for_collective() ← [维度3] NCCL broadcast
    │  │  policy_generation.update_weights_from_collective()
    │  │   → vLLM collective_rpc("update_weights_from_collective")
    │  │
    │  weight_version += 1                         ← [维度4] 版本号递增
    │  trajectory_collector.set_weight_version()
    │  trajectory_collector.resume_after_refit()
    │  │  [IF recompute_kv_cache]
    │  │    invalidate_kv_cache()                  ← [维度3] AREAL-style
    └───────────────────────────────────────────────────────

[后台 — AsyncTrajectoryCollector（独立线程持续运行）]
  _collection_loop():
    for batch in dataloader:
      wait manual_pause / refit_pause / generation_limit
      _process_batch(batch):
        target_weight = _get_next_target()         ← [维度4] 目标训练步
        for prompt_idx in range(num_prompts):
          _inflight_sema.acquire()                 ← 并发控制
          Thread → _run_prompt_group_worker():
            run_async_multi_turn_rollout()
              asyncio.gather(*sample_tasks)        ← 并发生成
                VllmAsyncGenerationWorker.generate_async()
                  AsyncLLM.generate()              ← vLLM V1 async
            replay_buffer.push_with_wait_signal(
              trajectory, weight_version, target_weight_version
            )                                      ← [维度2] 带版本元数据入 buffer
```

---

## 关键配置参数速查

| 参数 | 作用 | 对应维度 |
|------|------|---------|
| `grpo.async_grpo.enabled` | 启用 async GRPO | — |
| `grpo.async_grpo.max_trajectory_age_steps` | Buffer 深度 / trajectory 最大年龄 | 维度 2, 4 |
| `grpo.async_grpo.in_flight_weight_updates` | 权重更新时不等待进行中的生成 | 维度 3 |
| `grpo.async_grpo.recompute_kv_cache_after_weight_updates` | In-flight 后失效 KV cache（AREAL-style） | 维度 3 |
| `policy.generation.vllm_cfg.async_engine` | 使用 vLLM V1 AsyncLLM | — |
| `policy.generation.colocated.enabled` | 必须为 false（async 要求 disaggregated） | 维度 3 |
| `loss_fn.use_importance_sampling_correction` | 启用 IS 修正（async 必须开启） | 维度 4 |
| `loss_fn.truncated_importance_sampling_ratio` | TIS/ICE-POP 截断上界 | 维度 4 |
| `loss_fn.truncated_importance_sampling_ratio_min` | ICE-POP/seq-mask-TIS 截断下界 | 维度 4 |
| `loss_fn.truncated_importance_sampling_type` | IS 策略: `tis` / `icepop` / `seq-mask-tis` | 维度 4 |
| `loss_fn.sequence_level_importance_ratios` | 序列级 IS（GSPO） | 维度 4 |
| `grpo.seq_logprob_error_threshold` | Logprob 差异超阈值 → mask 序列 | 维度 4 |

## 示例配置

**Llama 3.1 8B — 2 节点 8 GPU，async 1-off**（`grpo-llama3.1-8b-instruct-2n8g-async-1off.yaml`）:
```yaml
grpo:
  async_grpo:
    enabled: true
    max_trajectory_age_steps: 1         # 最多 1 步 staleness
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

**Qwen3 30B-A3B MoE — 24 节点，async 8-off**（`grpo-qwen3-30ba3b-24n8g-async-8off.yaml`）:
```yaml
grpo:
  async_grpo:
    enabled: true
    max_trajectory_age_steps: 8         # 允许 8 步 staleness
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

## NeMo-RL vs SLIME Async 架构对比

| 方面 | SLIME | NeMo-RL |
|------|-------|---------|
| **调度模型** | 主循环亲自调度，double-buffer | 后台 Collector + ReplayBuffer，完全解耦 |
| **Buffer 深度** | 固定 1（double-buffer） | 可配置 1~N（`max_trajectory_age_steps`） |
| **并发粒度** | 同一时刻最多 1 个 rollout in-flight | `num_prompts_per_step × max_age` 个并发 |
| **推理引擎** | SGLang | vLLM V1 (AsyncLLM) |
| **权重同步** | pause→flush→bucketed NCCL broadcast | NCCL collective broadcast via vLLM `collective_rpc` |
| **In-flight 更新** | 不支持（必须 drain in-flight） | 支持（`in_flight_weight_updates: true`） |
| **Partial Rollout** | Abort + recycle + off-policy mask | 不支持 |
| **Staleness 修正** | depth=1 + TIS + OPSM | version-aware filtering + IS correction |
| **版本追踪** | `weight_versions` per sample | `trajectory_version` + `target_weight_version` per prompt group |
| **KV cache 策略** | flush_cache before sync | 可选 invalidate（AREAL vs Magistral） |
