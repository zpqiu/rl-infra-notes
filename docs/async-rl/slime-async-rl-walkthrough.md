# SLIME Async RL Training: Code Walk-Through

> **Codebase**: [THUDM/slime](https://github.com/THUDM/slime) @ commit `f71f7103`
>
> **参考 Blog**: [Keep the Tokens Flowing: Lessons from 16 Open-Source RL Libraries](https://huggingface.co/blog/async-rl-training-landscape) (HuggingFace, 2026-03-10)
>
> **关注维度**: Blog 7 个维度中的 4 个:
> - 维度 2: Rollout Buffer 设计
> - 维度 3: 权重同步协议
> - 维度 4: Staleness 管理
> - 维度 5: Partial Rollout 处理

## SLIME 在 Blog 框架中的定位

| 维度 | SLIME 的选择 | 激进程度 |
|------|-------------|---------|
| **Rollout Buffer** | Double-buffer (深度=1) | 保守 — staleness 最多 1 步 |
| **权重同步** | NCCL broadcast + bucketing, pause→flush→sync→continue | 中等 — 有 bucketing 优化但需要暂停推理 |
| **Staleness 管理** | 双重机制：double-buffer(depth=1) + TIS/OPSM；记录 version 但默认不做 per-sample rejection | 中等偏保守 — 先用结构把 lag 压到 1 步，再用 loss 修正残余 off-policy |
| **Partial Rollout** | Abort + recycle + off-policy token masking | 中等 — 比丢弃好，但不如 per-forward-pass 切换 |

## 文件阅读顺序

按执行流顺序打开这些文件即可走完全流程：

1. `train_async.py` — 骨架（30 行看完全局）
2. `slime/ray/rollout.py:RolloutManager.generate()` — 调度层
3. `slime/rollout/sglang_rollout.py:generate_rollout_async()` — 生成 + abort + recycle
4. `slime/rollout/data_source.py:RolloutDataSourceWithBuffer` — buffer 回收机制
5. `slime/backends/megatron_utils/update_weight/update_weight_from_distributed.py` — 权重同步全流程
6. `slime/backends/megatron_utils/actor.py:train_actor()` — 训练数据准备
7. `slime/backends/megatron_utils/loss.py:policy_loss_function()` — TIS + OPSM 修正

---

## 第 0 步：启动 — 分配 GPU、创建组件

**入口**: `train_async.py:10` → `train(args)`

**文件路径**: `slime/ray/placement_group.py:79-119`

```
create_placement_groups(args)
  ├── actor GPU pool:   [GPU 0 .. actor_num_gpus-1]
  ├── critic GPU pool:  [紧随 actor 之后]  (可选)
  └── rollout GPU pool: [最后 rollout_num_gpus 个]
```

非 colocate 模式下（`train_async.py:11` 强制要求 `assert not args.colocate`），训练和推理在**物理隔离的 GPU 池**上。这是整个 async 架构的物理基础。

接着创建两个核心组件：

**RolloutManager** (`placement_group.py:181-201` → `slime/ray/rollout.py:349-391`):
- 一个 **Ray remote actor**（无 GPU，纯调度）
- 内部启动 SGLang 推理引擎群 + router
- 持有 `data_source`（数据源）和 `rollout_engine_lock`（权重同步锁）

**RayTrainGroup** (`placement_group.py:132-178` → `slime/ray/actor_group.py:10-99`):
- 每个 GPU 上一个 `MegatronTrainRayActor`
- rank 0 发现 master_addr/port，其余 actor 加入
- `async_init()` 加载模型、optimizer、checkpoint

**初始权重同步** (`train_async.py:25-26`):
```python
if not args.critic_train_only:
    actor_model.update_weights()  # 确保 SGLang 引擎拿到训练模型的初始权重
```
这是第一次也是唯一一次"阻塞式"的全量权重推送。

---

## 第 1 步：主循环 — Double-Buffer 流水线调度

**文件**: `train_async.py:32-74`

这是 SLIME async 的**调度心脏**，只有 ~40 行，值得逐行看：

```python
# [A] 预热：发起第一次 rollout
rollout_data_next_future = rollout_manager.generate.remote(args.start_rollout_id)

for rollout_id in range(args.start_rollout_id, args.num_rollout):
    # [B] 取回当前 rollout 数据（阻塞等待）
    rollout_data_curr_ref = ray.get(rollout_data_next_future)

    # [C] 立即发起下一次 rollout（与训练并行）
    if rollout_id + 1 < args.num_rollout:
        rollout_data_next_future = rollout_manager.generate.remote(rollout_id + 1)

    # [D] 用当前数据训练（阻塞等待训练完成）
    ray.get(actor_model.async_train(rollout_id, rollout_data_curr_ref))

    # [E] 权重同步（按间隔触发）
    if (rollout_id + 1) % args.update_weights_interval == 0:
        # ⚠️ 关键：先等下一次 rollout 完成，防止 rollout 中途更新权重
        rollout_data_curr_ref = ray.get(rollout_data_next_future)
        rollout_data_next_future = None
        actor_model.update_weights()
```

> **[维度 2: Rollout Buffer]** 这是一个 **double-buffer**，深度恰好为 1。时间线如下：
> ```
> 时间 →
> Rollout:  [===Gen 0===]  [===Gen 1===]  [===Gen 2===]
> Train:                   [==Train 0==]  [==Train 1==]
>                                    ↑ 重叠区域 ↑
> ```
> Gen N+1 和 Train N 并行执行。但同一时刻最多只有 1 个 rollout 在 flight。

> **[维度 3: 权重同步触发点]** 注意 `[E]` 处：触发权重同步前，**必须先 drain in-flight rollout**（`ray.get(rollout_data_next_future)`）。这保证权重更新时推理引擎是空闲的，避免 mid-generation 权重切换的复杂性。Blog 分类为 **"Per Training Step/Batch (blocking)"** 的中断粒度。

---

## 第 2 步：Rollout 阶段 — 从 prompt 到训练数据

**调用链**:
```
rollout_manager.generate.remote()
  → RolloutManager.generate()                  # rollout.py:460
    → generate_rollout()                        # sglang_rollout.py:578
      → generate_rollout_async()                # sglang_rollout.py:366
```

### 2a. 数据采样

`sglang_rollout.py:399-403`:
```python
while len(data) < target_data_size:
    while state.remaining_batch_size < target_data_size:
        samples = data_source(args.over_sampling_batch_size)  # 从数据源取 prompt
        state.submit_generate_tasks(samples)                   # 提交异步生成任务
```

> **[维度 5: Partial Rollout]** `data_source` 就是 `RolloutDataSourceWithBuffer.get_samples()` (`data_source.py:175-187`)。它**优先从 buffer 中取被 abort 回收的 partial samples**，不够再从 dataset 取新 prompt。这就是 recycle 的"消费端"。

### 2b. 并发生成

`sglang_rollout.py:111-123` — `submit_generate_tasks`:
```python
def submit_generate_tasks(self, samples):
    for group in samples:
        self.pendings.add(asyncio.create_task(
            generate_and_rm_group(args, group, sampling_params, evaluation=False)
        ))
```

每个 group（同一 prompt 的 G 个 completion）作为一个 asyncio task，并发发送到 SGLang router。

**单个 sample 的生成** (`sglang_rollout.py:127-220`):
```python
async def generate(args, sample, sampling_params):
    payload = {"sampling_params": ..., "return_logprob": True}
    if args.use_rollout_routing_replay:
        payload["return_routed_experts"] = True   # 用于 MoE routing replay
    output = await post(url, payload)

    # 收集元数据
    sample.tokens += new_response_tokens
    sample.rollout_log_probs += new_response_log_probs  # ← 关键：记录生成时的 logprob
    sample.update_from_meta_info(args, output["meta_info"])  # ← 记录 weight_version
```

> **[维度 4: Staleness]** 两个关键信息在此采集：
> - `rollout_log_probs`：生成时策略 π_old 的 token-level log prob，后续 TIS 用它算 IS ratio
> - `weight_versions`：记录生成这个 sample 时推理引擎的权重版本号
>
> 但两者用途不同：
> - `rollout_log_probs` 会在训练 loss 中真正参与 TIS/OPSM 计算
> - `weight_versions` 在默认主路径里主要用于追踪/观测，不会按 version gap 做 per-sample hard drop

### 2c. 过采样 + 动态过滤 + Abort

`sglang_rollout.py:405-438`:
```python
# 收集完成的 group
done, state.pendings = await asyncio.wait(state.pendings, return_when=FIRST_COMPLETED)
for task in done:
    group = task.result()
    # 动态过滤（如 reward 方差为 0 的 group 丢弃）
    if not call_dynamic_filter(dynamic_filter, args, group).keep:
        state.remaining_batch_size -= 1
        continue
    data.append(group)

# 收集够了 → abort 剩余 in-flight 请求
aborted_samples = await abort(args, rollout_id)
```

> **[维度 5: Partial Rollout — Abort 流程]**
> `abort()` (`sglang_rollout.py:322-363`) 做三件事：
> 1. 设 `state.aborted = True`（让后续 task 提前返回）
> 2. 向所有 SGLang worker 发 `abort_request`（`{"abort_all": True}`）
> 3. 等所有 pending task 完成，收集部分生成的 sample，标记 `start_rollout_id`

### 2d. Recycle 回收

`sglang_rollout.py:597-598`:
```python
output, aborted_samples = run(generate_rollout_async(args, rollout_id, data_source.get_samples))
data_source.add_samples(aborted_samples)  # 放回 buffer，下次生成时优先取出
```

> **[维度 5: Off-policy token masking]**
> 当 partial sample 被重新取出继续生成时 (`sglang_rollout.py:229-231`):
> ```python
> if args.partial_rollout and args.mask_offpolicy_in_partial_rollout and sample.response_length > 0:
>     sample.loss_mask = [0] * sample.response_length  # 旧权重生成的 token → loss mask 为 0
> ```
> 新生成的 token append `[1]`。最终训练时，旧 token 不参与 loss 计算。

### 2e. 转为训练数据

`rollout.py:460-473`:
```python
def generate(self, rollout_id):
    data, metrics = self._get_rollout_data(rollout_id)
    data = self._convert_samples_to_train_data(data)        # Sample → dict
    return self._split_train_data_by_dp(data, dp_size)       # 按 DP rank 切分
```

`_convert_samples_to_train_data()` (`rollout.py:664-727`) 把 `Sample` 列表转成 dict:
```python
train_data = {
    "tokens": [...],
    "response_lengths": [...],
    "rewards": [...],
    "loss_masks": [...],              # ← 含 partial rollout 的 off-policy mask
    "rollout_log_probs": [...],       # ← TIS 的输入
    "rollout_routed_experts": [...],  # ← MoE routing replay 的输入
}
```

最后按 DP rank 切分，每份 `ray.put()` 放入 Ray object store。

---

## 第 3 步：权重同步 — Pause → Gather → Broadcast → Continue

**触发**: `train_async.py:64-69`，当 `(rollout_id + 1) % update_weights_interval == 0`

**调用链**:
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
Rank 0 告诉所有推理引擎暂停接收新请求、清空 KV cache。

### 3b. 非 Expert 参数：TP Gather → HF 转换 → Bucketed Broadcast

`update_weight_from_distributed.py:106-115`:
```python
for name, param in named_params_and_buffers(args, model):
    if ".experts." in name:
        continue
    buffer_size = self._update_weight_from_distributed(name, param, ...)
```

每个参数的处理 (`update_weight_from_distributed.py:142-164`):
```
param (TP-sharded)
  → all_gather_param()         # TP all-gather 恢复完整 tensor (common.py:15)
  → convert_to_hf()           # Megatron 格式 → HuggingFace 格式
  → 累积到 bucket
  → 超过 update_weight_buffer_size → _update_bucket_weights_from_distributed()
```

Bucket broadcast (`update_weight_from_distributed.py:228-249`):
```python
def _update_bucket_weights_from_distributed(self, converted_named_tensors):
    while not ray.get(self.rollout_engine_lock.acquire.remote()):
        time.sleep(0.1)   # 获取锁，防止并发 NCCL 死锁

    # 先用 Ray 发送元数据（name, dtype, shape），再用 NCCL broadcast 发 tensor 数据
    refs = update_weights_from_distributed(group_name, group, weight_version, engines, tensors)
    ray.get(refs)

    ray.get(self.rollout_engine_lock.release.remote())
```

`update_weights_from_distributed()` (`update_weight_from_distributed.py:310-337`):
```python
# 元数据通过 Ray（names, dtypes, shapes, weight_version）
refs = [engine.update_weights_from_distributed.remote(...) for engine in rollout_engines]
# tensor 数据通过 NCCL broadcast（rank 0 → 所有 engine）
for _, param in converted_named_tensors:
    handles.append(dist.broadcast(param.data, 0, group=group, async_op=True))
for handle in handles:
    handle.wait()
```

### 3c. Expert 参数：额外 EP Gather

`update_weight_from_distributed.py:118-128`:
```python
for name, param in named_params_and_buffers(args, model):
    if ".experts." not in name:
        continue
    buffer_size = self._update_expert_weight_from_distributed(name, param, ...)
```

Expert 参数需要先跨 EP ranks all-gather（`_update_expert_bucket_weights_from_distributed`, line 190-226），因为每个 EP rank 只持有部分 expert。Gather 后再走同样的 HF 转换 + NCCL broadcast。

### 3d. Continue + Version 递增

`update_weight_from_distributed.py:86, 131-140`:
```python
self.weight_version += 1  # 入口处递增

# ... 所有参数传输完成后 ...
if dist.get_rank() == 0:
    ray.get([engine.continue_generation.remote() for engine in self.rollout_engines])
dist.barrier(group=get_gloo_group())
```

### 3e. Colocated 路径（补充）

当部分推理引擎与训练在同一机器时，走 `UpdateWeightFromTensor` (`update_weight_from_tensor.py`)：
```
HF params → FlattenedTensorBucket 打包 → Gloo gather_object (CPU) → Ray IPC → SGLang engine
```
与 distributed 路径的区别是用 CPU Gloo + IPC 替代 GPU NCCL，避免跨机通信。两条路径可以**同时存在**（一部分 engine colocated，一部分 distributed）。

---

## 第 4 步：训练 — Forward → Advantage → Loss(TIS/OPSM) → Backward

**调用链**:
```
actor_model.async_train()
  → RayTrainGroup.async_train()           # actor_group.py:111
    → MegatronTrainRayActor.train()        # actor.py:355
      → train_actor()                      # actor.py:398
```

### 4a. 数据准备 + Ref Model Forward

`actor.py:398-445`:
```python
def train_actor(self, rollout_id, rollout_data):
    data_iterator, num_microbatches = get_data_iterator(args, model, rollout_data)

    # 如果有 ref model → 切换权重 → 算 ref_log_probs
    if "ref" in self.weights_backuper.backup_tags:
        self._switch_model("ref")
        rollout_data.update(self.compute_log_prob(..., store_prefix="ref_"))

    # 切回 actor → 算当前策略的 log_probs（如果不用 rollout_logprobs）
    self._switch_model("actor")
    if not args.use_rollout_logprobs or args.get_mismatch_metrics:
        rollout_data.update(self.compute_log_prob(..., store_prefix=""))
```

> **[维度 4: Staleness]** 这里有两种 old log probs 的来源：
> - `use_rollout_logprobs=True`：直接用推理引擎记录的 `rollout_log_probs`（更准确反映生成时策略，但可能 off-policy）
> - `use_rollout_logprobs=False`：用当前 actor 权重重新 forward 算出（on-policy 但多了一次 forward 开销）
>
> TIS 修正的前提是有 `rollout_log_probs`，所以 `use_tis` 要求 `use_rollout_logprobs` 或至少 `get_mismatch_metrics`。

### 4b. Advantage 计算

`actor.py:460`:
```python
compute_advantages_and_returns(self.args, rollout_data)
```

`loss.py:400-561` — 根据 `advantage_estimator` 选择：
- **grpo/gspo**: `rewards × ones_like(kl)` — reward broadcast 到每个 token
- **ppo**: GAE with value baseline
- **reinforce_plus_plus**: 逐 token 折扣 return

advantage 归一化跨 DP group (`loss.py:504-557`)。

### 4c. Policy Loss — TIS 和 OPSM 的作用点

**文件**: `loss.py:613-831` — `policy_loss_function()`，这是 staleness 修正的**核心战场**。

```python
# ① 拿到 old_log_probs（根据 use_rollout_logprobs 选择来源）
old_log_probs = batch["rollout_log_probs"] if args.use_rollout_logprobs else batch["log_probs"]

# ② 当前策略 forward → 算 log_probs 和 entropy
log_probs = get_log_probs_and_entropy(logits, ...)

# ③ 标准 PPO clipped loss
pg_loss, pg_clipfrac = compute_policy_loss(ppo_kl, advantages, eps_clip, eps_clip_high)
```

> **[维度 4: OPSM — Off-Policy Sequence Masking]** (`loss.py:682-689`):
> ```python
> if args.use_opsm:
>     opsm_mask, opsm_clipfrac = compute_opsm_mask(
>         full_log_probs, full_old_log_probs, advantages, loss_masks
>     )
>     pg_loss = pg_loss * opsm_mask  # 序列级 KL > δ 且 advantage < 0 → 置零
> ```
> 直觉：off-policy 的负面 sample（advantage<0 + KL 大）不可信，不让它压低策略概率。
>
> `compute_opsm_mask()` (`ppo_utils.py:54-92`):
> ```python
> seq_kl = ((old_logprob - log_prob) * loss_mask).sum() / loss_mask.sum()
> mask = ((advantage < 0) & (seq_kl > delta)).float()
> return 1 - mask  # 0 = masked out, 1 = kept
> ```

> **[维度 4: TIS — Truncated Importance Sampling]** (`loss.py:712-740`):
> ```python
> if args.use_tis:
>     tis_func = vanilla_tis_function  # 或 custom
>     pg_loss, modified_masks, tis_metrics = tis_func(
>         pg_loss=pg_loss,
>         train_log_probs=batch["log_probs"],        # π_θ (当前)
>         rollout_log_probs=batch["rollout_log_probs"], # π_old (生成时)
>     )
> ```
> `vanilla_tis_function()` (`loss.py:563-584`):
> ```python
> tis = exp(π_θ - π_rollout)                        # IS ratio
> tis_weights = clamp(tis, tis_clip_low, tis_clip)  # 截断
> pg_loss = pg_loss * tis_weights                    # 加权修正
> ```
> 还有 `icepop_function` (line 587)：超出截断范围直接置零。这仍然是 **loss 层的 IS/RS 变体**，不是按 `weight_version` 做的 per-sample version rejection。

**执行顺序**: 先 OPSM 置零 → 再 TIS 加权。两者解耦，可独立开关。

> **更准确地说，SLIME 默认主路径同时用了 blog 里的两种 staleness 策略**：
> - **Strategy 2: Depth Bounding**。`train_async.py` 只有一个 `rollout_data_next_future`，形成 one-step-ahead 的 double-buffer；而且在 `update_weights()` 前会先 `ray.get(future)` drain 掉 in-flight rollout，因此结构上把 policy lag 限制在最多 1 个 rollout step。
> - **Strategy 3: IS-weighted loss correction**。即这里的 TIS + OPSM。
>
> **没有默认实现的是 Strategy 1: Per-sample version rejection**。代码会记录 `weight_versions`，但默认训练路径并不会基于 version gap 丢弃 sample。

### 4d. Backward + Optimizer Step

`actor.py:474-482`:
```python
with timer("actor_train"):
    train(rollout_id, model, optimizer, opt_param_scheduler, data_iterator, num_microbatches)
```
标准 Megatron pipeline-parallel train step。Loss 经过 TIS/OPSM 修正后反传。

### 4e. 权重备份

`actor.py:492`:
```python
self.weights_backuper.backup("actor")  # CPU 备份，供 ref model 切换和权重同步使用
```

---

## 第 5 步：回到第 1 步

主循环回到 `train_async.py:33`，`ray.get(rollout_data_next_future)` 取回在第 4 步训练期间**并行生成**的下一批 rollout 数据，开始新一轮训练。

---

## 全流程 + 维度标注图

```
train_async.py                                  Blog 维度
─────────────────────────────────────────────────────────
[启动]
  create_placement_groups()                     ← 物理隔离(disaggregated)
  create_rollout_manager() → SGLang engines
  create_training_models() → Megatron actors
  actor_model.update_weights()                  ← 初始权重同步 [维度3]

[主循环]
  future = rollout_manager.generate(0)          ← 预热
  for rollout_id in range(...):
    data = ray.get(future)                      ← [维度2] double-buffer 取数据
    future = generate(rollout_id+1)             ← [维度2] 下一批并行生成
    │
    │ ┌─ generate_rollout_async() ──────────────────────────────
    │ │  data_source.get_samples()              ← [维度5] 优先从 buffer 取 partial samples
    │ │  submit_generate_tasks() → asyncio
    │ │  │  generate() → SGLang
    │ │  │    记录 rollout_log_probs            ← [维度4] staleness 修正原材料
    │ │  │    记录 weight_versions              ← [维度4] 版本追踪
    │ │  │    记录 rollout_routed_experts        ← MoE routing replay
    │ │  │    partial rollout: mask_offpolicy    ← [维度5] 旧 token loss_mask=0
    │ │  wait(FIRST_COMPLETED) + 动态过滤
    │ │  abort() → 回收 partial samples         ← [维度5] abort + recycle
    │ │  data_source.add_samples(aborted)        ← [维度5] 放回 buffer
    │ │  convert_samples_to_train_data()
    │ │  split_by_dp()
    │ └──────────────────────────────────────────────────────────
    │
    actor_model.async_train(data)
    │ ┌─ train_actor() ─────────────────────────────────────────
    │ │  ref forward → ref_log_probs
    │ │  actor forward → log_probs              ← [维度4] on-policy logprob
    │ │  compute_advantages_and_returns()
    │ │  policy_loss_function():
    │ │    compute_policy_loss() → PPO clipped
    │ │    OPSM: mask(adv<0 & kl>δ)             ← [维度4] 序列级 off-policy mask
    │ │    TIS: loss × clamp(π_θ/π_old)         ← [维度4] token级 IS 加权
    │ │  backward + optimizer step
    │ └──────────────────────────────────────────────────────────
    │
    if (rollout_id+1) % interval == 0:
      ray.get(future)                           ← [维度3] drain in-flight rollout
      future = None
      actor_model.update_weights()
      │ ┌─ UpdateWeightFromDistributed ─────────────────────────
      │ │  pause_generation()                   ← [维度3] 暂停推理
      │ │  flush_cache()                        ← [维度3] 清 KV cache
      │ │  for param (non-expert):
      │ │    all_gather_param() [TP]            ← [维度3] TP gather
      │ │    convert_to_hf()
      │ │    accumulate → bucket
      │ │    if bucket full:
      │ │      lock → NCCL broadcast → unlock   ← [维度3] bucketed 传输
      │ │  for param (expert):
      │ │    EP all-gather → HF → broadcast     ← [维度3] MoE expert 处理
      │ │  weight_version++                     ← [维度4] 版本号递增
      │ │  continue_generation()                ← [维度3] 恢复推理
      │ └──────────────────────────────────────────────────────
```

---

## 关键配置参数速查

| 参数 | 作用 | 对应维度 |
|------|------|---------|
| `--update_weights_interval` | 每 N 个 rollout 同步一次权重 | 维度 3 |
| `--update_weight_buffer_size` | bucketed broadcast 的桶大小 | 维度 3 |
| `--use_rollout_logprobs` | 用推理引擎的 logprob 作 old policy | 维度 4 |
| `--use_tis` | 启用 TIS 修正 | 维度 4 |
| `--tis_clip` / `--tis_clip_low` | TIS 截断范围 | 维度 4 |
| `--custom_tis_function_path` | 自定义 TIS 函数（如 icepop） | 维度 4 |
| `--use_opsm` | 启用 OPSM 掩码 | 维度 4 |
| `--opsm_delta` | OPSM 的 KL 阈值 δ | 维度 4 |
| `--partial_rollout` | 启用 abort + recycle | 维度 5 |
| `--mask_offpolicy_in_partial_rollout` | 回收 sample 的旧 token 不参与 loss | 维度 5 |
| `--get_mismatch_metrics` | 记录 train/rollout logprob 差异 | 维度 4 |
| `--use_rollout_routing_replay` | MoE routing 一致性（记录+重放） | Blog §5.4 |
