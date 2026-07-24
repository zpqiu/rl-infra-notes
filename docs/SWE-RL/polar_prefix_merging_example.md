# prefix_merging 端到端例子:一次 2 轮 codex 会话如何变成可训练 Sample

> 配合主文档 [Polar 架构分析](polar_architecture_swe_rl.md) 第 7.4 节阅读。
> 本文用一个最小但完整的例子走通:completion 捕获 → 归链 → 拼流 →
> 评估 → Slime Sample,并讨论残余的训推不一致。
>
> 设定:底模 Qwen(ChatML 模板,`<|im_end|>` 是 end-of-turn token),
> agent 是 codex,任务是"修一个 bug":第 1 轮跑测试,第 2 轮提交修复。
> token id 是假的,括号里是含义。

---

## 背景:re-tokenization drift 是什么

多轮 agent rollout 里,每一轮都是一次独立的 HTTP completion。**朴素做法**
是:rollout 结束后拿最终的 message 列表,套 chat template,
`tokenizer.encode()` 出训练 token。这会产生漂移(drift):

1. **BPE 非规范性**:模型采样出的 token 序列 `[fish][ing]`,作为文本嵌进
   下一轮 prompt 再 encode,可能变成 `[fishing]`——文本相同,token 不同;
2. **harness 重写历史**:codex 这类 CLI 会把上一轮的 tool call 重新序列化
   (JSON 格式化、空白差异)后放进下一轮请求;
3. **chat template 胶水**:role 标记、generation prompt 的拼接方式和推理时
   未必逐 token 一致。

后果:训练用的 token 序列 ≠ 策略实际采样的序列,logprob 对不上位,
GRPO/PPO 的 importance ratio 失真(SWE-RL 配置开了 `--use-tis`,对
rollout logprob 的位置对齐尤其敏感)。

Polar 的应对原则(`src/polar/trajectory/builder/prefix_merging.py:1-36`):
**token 真值只来自推理引擎,builder 只做拼接,从不 decode→re-encode**。
下面逐步走一遍。

---

## 第 0 步:两次 LLM 调用,gateway 捕获到两条 CompletionRecord

Gateway 转发请求时由 `engine.py` 注入 `return_prompt_token_ids` /
`logprobs=True`(SGLang)或 `return_token_ids`(vLLM),响应里带回
**prompt 的 canonical token ids、采样出的 response token ids、逐 token
logprob**,原样存进 `CompletionRecord`。

**C1(第 1 轮)**——agent 发来 system + user:

```
P1 (prompt_ids, 引擎的 canonical 分词, 14 个):
[1(<|im_start|>) 10(system) 5(\n) 60(你是编码助手) 2(<|im_end|>)
 1  11(user)   5  20(修复) 21(bug) 2
 1  12(assistant) 5]                     ← 末尾是 generation prompt

R1 (response_ids, 模型逐个采样, 5 个):
[70(我先跑) 31(测试) 40(pyt) 41(est) 2(<|im_end|>)]
                     ^^^^^^^^^^^ 工具调用参数 "pytest" 被采样成两个 token
L1 (logprobs): [-0.12, -0.05, -0.30, -0.21, -0.01]
finish_reason = tool_calls(自然停止,末 token 就是 EOT)
```

**C2(第 2 轮)**——codex 把"完整历史 + 工具结果"作为新 prompt 发来。
引擎对这段**文本**重新做 canonical 分词:

```
P2 (prompt_ids, 24 个):
[ ...P1 原样的 14 个...                     ← 历史前缀,分词稳定
  70 31 45(pytest) 2                       ← R1 的文本被"重新分词":
                                              [40,41] 合并成了 [45] ⚠ drift!
  5 1 13(tool) 5 50(1) 51(failed) 2        ← 工具结果
  1 12 5 ]                                 ← 新的 generation prompt

R2 (response_ids, 3 个): [80(已修复) 81(提交patch) 2]
L2: [-0.40, -0.20, -0.01],  finish_reason = stop
```

注意 drift 已经发生了:模型当年采样的是 `[40,41]`,但这段文本嵌回 prompt
后 BPE 把它合并成 `[45]`——**朴素的"最终 messages 重新 encode"方案从这里
开始就和采样序列错位,后面每个位置全部平移,logprob 全部对不上**。

---

## 第 1 步:归链(只看 prompt 前缀,不看 response)

`_find_extendable_chain`(`prefix_merging.py:339-363`):completion 加入
"其上一个 prompt 是自己 prompt 前缀"的链,最长匹配优先。

```
C1 到达:没有链 → 开新链,tip = P1
C2 到达:P2[:14] == P1 ?  逐 token 比较 → 相等 → 加入该链,tip = P2
```

比较的两边(`P1` 和 `P2` 的前 14 个)都是**服务端对同一段历史的 canonical
分词**,所以稳定相等;`[40,41]` vs `[45]` 的差异落在第 14 个 token
**之后**,根本不参与判定。

如果此时 codex 还并行开了个 sub-agent(prompt 是另一套 system),它的
prompt 不以 P1 为前缀 → 自动开第二条链,互不污染。

---

## 第 2 步:拼流(采样段用原始 token,间隙段从 canonical 里切)

`_finalize_chain`(`prefix_merging.py:134-242`):

```
eot_id 探测:C1 是自然停止 → eot = R1 末 token = 2

stream  = P1                                    ← prompt,不训练
        + [70 31 40 41 2]                       ← R1 原始采样 token
canonical_tail = P2[14:] =
          [70 31 45 2 | 5 1 13 5 50 51 2 1 12 5]
           └────┬───┘
        R1 的重分词副本 → 找到第一个 eot(下标3),
        R1 本身以 eot 结尾 → 从下标4起切:
        这 4 个 token 被【整段丢弃】,drift 随之被扔掉 ✂
        + [5 1 13 5 50 51 2 1 12 5]             ← 间隙:工具结果+模板胶水
        + [80 81 2]                             ← R2 原始采样 token
```

最终这条 Trace(prompt 14 + response 18):

| 段 | response_ids | loss_mask | logprobs |
|---|---|---|---|
| R1 采样段 | `70 31 40 41 2` | `1 1 1 1 1` | L1 真值 |
| 间隙段 | `5 1 13 5 50 51 2 1 12 5` | `0 0 0 0 0 0 0 0 0 0` | 全 0.0(占位) |
| R2 采样段 | `80 81 2` | `1 1 1` | L2 真值 |

对照:**流里的 "pytest" 是 `[40,41]`(策略真实采样的),而不是 `[45]`
(重分词的)**——所有 loss_mask=1 的位置,token 和 logprob 逐位来自采样
时刻;唯一含重分词结果的间隙段 mask=0,不产生梯度。

两个失败分支:

- codex 在第 2 轮重写了历史(比如把 tool call 参数格式化了)→
  `P2[:14] != P1` → 链在这里 break,只有第 1 轮进训练
  (**截断,不污染**,计入 `reconstruction_stats`);
- 某个 mask=1 的位置引擎没回 logprob → 整条 logprobs 置 None,下游
  adapter 直接抛 `RolloutLogprobError`,**绝不静默**
  (`_finalize_logprobs`,`prefix_merging.py:317-330`)。

EOT 的两种情况(`_slice_interstitial`,`prefix_merging.py:266-296`):
上一轮自然停止(response 已含 EOT)→ canonical tail 里的 EOT 跳过防重复;
截断停止(`finish_reason=length`)→ 把 canonical tail 的 EOT 补进来闭合
轮次(masked)。

---

## 第 3 步:评估 + 转成 Slime Sample

evaluator(swebench_harness)在新容器里 apply patch 跑测试 →
`resolved=True` → gateway 把 `reward=1.0` 广播到这条 trace 上。然后
`src/slime_bridge/adapter.py` 纯字段平移,不再碰 token:

```python
Sample(
  tokens           = P1 + response_ids,   # 14 + 18 = 32 个 token,直接拼
  response_length  = 18,
  loss_mask        = [1,1,1,1,1, 0,0,0,0,0,0,0,0,0,0, 1,1,1],
  rollout_log_probs= [L1..., 0.0×10, L2...],   # 长度必须 == 18,否则报错
  reward           = {"score": 1.0},
  group_id         = <本 trajectory 的 index>,  # 若有 sub-agent 链,那条
)                                              #  Trace 也共享此 group_id
```

最后 GRPO 组处理:同一道 SWE-Gym 题的 16 个 session 各产出这样的 Sample,
`reward_post_process` 算 leave-one-trajectory-out advantage——比如 16 条里
6 条 reward=1,这条的 advantage ≈ `1.0 − 5/15 = 0.67`;Megatron 只在
mask=1 的 8 个位置算 policy loss,TIS 用 `rollout_log_probs` 在这些位置和
trainer 重算的 logprob 做校正——这一步能成立,正是因为第 2 步保证了逐位
对齐。

---

## 残余的训推不一致(caveat)

一个精准的追问:R2 推理时是以 P2(含 R1 的 canonical 副本 `[45]`)为条件
采样和计算 logprob 的,但训练时它的条件被替换成了合并流(含 R1 原始采样的
`[40,41]`)——这不是没解决训推不一致吗?

**是的,prefix_merging 没有完全消除训推不一致**。它消除的是一类错误
(一阶的、灾难性的),留下了另一类(二阶的、有界的):

- **推理时**:引擎算 `π_rollout(R2 | P2)`;
- **训练时**:Megatron 算 `π_train(R2 | P1 + R1_raw + 间隙)`。

条件序列**文本相同、token 不同**,R2 位置上的 logprob 之间除数值精度外
多了一层"条件 tokenization 不同"的 gap。它的边界:R1 自己的条件(P1)逐
token 精确;间隙段就是 canonical token,也精确;且**只有重分词真的改变了
token 时才有差异**——采样序列多数本来就是 canonical 的,drift 是合并边界、
数字、罕见拼接上的尾部事件;无 drift 时整条链逐 token 精确。

**为什么牺牲二阶、保住一阶**:单条打包序列里 R1 段只能放一个版本——

- 放 raw(Polar 的选择):R1 的训练位置一阶精确(loss 算在真实采样的
  token 上、logprob 逐位真值);代价是 R2 的条件二阶近似(同文本不同分词,
  模型内部表征接近);
- 放 canonical 副本:R2 条件精确了,但 R1 是在**策略从未采样过的 token**
  (`[45]`)上算 loss,其"rollout logprob"根本不存在,位置全部错位——
  这正是朴素 re-encode 方案的死法。

一阶错误污染的是被训练的 token 本身,必须为零;二阶误差只影响条件的表征,
可以容忍。

**残余 gap 的兜底**:`--use-tis`(truncated importance sampling)本来就是
为 π_rollout ≠ π_train 的数值差异(bf16 kernel、推理/训练引擎实现不同)
设计的;条件 drift 造成的 logprob 差被折进同一个 ratio、被截断约束。框架的
立场不是"训推完全一致",而是"**token 级一阶对齐 + ratio 级二阶校正**"。

**要零残余的替代方案**:

1. `per_request` builder:每轮 completion 独立成 Trace,R2 的 prompt 就是
   引擎真实见到的 P2,条件逐 token 精确。代价:共享前缀每轮重复
   (n 轮 O(n²) token,SWE 任务动辄几十轮),且失去单轨迹打包结构;
2. token-in-token-out:agent 直接传 token id,历史不走文本。但 codex 只会
   发文本——这违背 Polar"不改造第三方 agent、走标准文本 API"的前提。

**一个可做的量化改进**:`prefix_merging` 目前只统计链的 full/truncated
计数,没有统计 drift 发生率(canonical tail 被丢弃段与 raw response 逐
token 不等的比例)。在 `_slice_interstitial` 加一个计数器成本很低,可以
直接量化训练数据里条件不一致的占比。
