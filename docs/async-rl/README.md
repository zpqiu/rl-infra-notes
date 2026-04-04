# Async RL Training: 三框架源码对比

> SLIME · NeMo-RL · veRL —— 同一个问题，三种工程哲学

## 一图看懂调度差异

<p align="center">
  <a href="async-rl-comparison-slides.html">
    <img src="https://github.com/user-attachments/assets/placeholder-timeline" alt="Scheduling × Buffer × Weight Sync — Full Timeline" width="900">
  </a>
</p>

> **👆 占位图** — 请将 HTML slides 中 "调度 × Buffer × 权重同步 —— 全景时间线" 页面的截图保存为 `assets/timeline-overview.png`，然后替换上方链接。
>
> 交互版请直接打开 **[async-rl-comparison-slides.html](async-rl-comparison-slides.html)**（支持键盘翻页）。

三个框架在同一张时间线上的核心区别：

| | SLIME | NeMo-RL | veRL |
|---|---|---|---|
| **调度模型** | 主循环 for-loop | 主循环 + 后台 Collector | 独立 Rollouter/Trainer actor |
| **Buffer** | Double-buffer (depth=1) | Replay Buffer (depth=1~N) | Bounded Queue (maxlen=N) |
| **同步时机** | drain in-flight → sync | 每步 sync，推理不停 | 每 N 步 abort → sync |
| **Staleness 上界** | 1 步（结构保证） | `max_trajectory_age_steps` | `staleness_threshold` |
| **Partial Rollout** | abort + recycle | 无（sample 独立完成） | abort + prefix continuation |

```
保守 ◄──────────────────────────────────► 激进
(on-policy 优先)                    (throughput 优先)

   SLIME ·····  verl ·····  NeMo-RL
    20%          55%          70%
```

## 分析维度

每篇 walkthrough 围绕 [HuggingFace Async RL Survey](https://huggingface.co/blog/async-rl-training-landscape) 的 4 个核心维度展开：

| # | 维度 | 核心问题 |
|---|------|---------|
| 2 | **Rollout Buffer** | 生成数据暂存在哪？深度多少？满了怎么办？ |
| 3 | **权重同步** | 训练完新权重怎么送到推理引擎？推理要暂停吗？ |
| 4 | **Staleness 管理** | 用旧权重生成的数据还能用吗？怎么修正？ |
| 5 | **Partial Rollout** | 同步时正在生成的 request 怎么处理？丢掉还是续上？ |

## Walkthroughs

### [SLIME Async RL Walkthrough](slime-async-rl-walkthrough.md)

**关键词**: Double-buffer · abort + recycle · TIS + OPSM

最简洁的 async 实现——主循环 ~40 行，buffer 深度锁死为 1，用结构约束把 staleness 压到最小。适合想快速理解 async RL 全流程的读者。

- 双 buffer 调度如何让 Gen N+1 与 Train N 并行
- drain → flush → sync → continue 权重同步 4 步曲
- off-policy token 的 TIS 加权 + OPSM mask 修正

### [NeMo-RL Async RL Walkthrough](nemo-rl-async-rl-walkthrough.md)

**关键词**: Replay Buffer · in-flight weight update · target version matching

最灵活的 buffer 设计——Collector 在后台持续生成，训练侧按需取数据。独特之处在于推理引擎可以在生成过程中热更新权重。

- `ReplayBuffer` 如何用 version + age window 筛选样本
- `refit_policy_generation()` 实现不停推理的权重更新
- TIS / ICE-POP / seq-mask-TIS 三种 IS 修正对比

### [veRL Async RL Walkthrough](verl-async-rl-walkthrough.md)

**关键词**: Bounded Queue · backpressure · MIS · prefix continuation

最解耦的架构——Rollouter 和 Trainer 是完全独立的 Ray actor，通过 MessageQueue 通信。支持最丰富的 staleness 修正工具箱。

- 逐 sample streaming 入队 + 双层 backpressure 控制
- NCCL bucketed broadcast 权重同步实现
- 多版本 Importance Sampling (MIS) 的 CPU 快照机制
- abort 后的 prefix continuation 跨版本续写

## 交互式对比 Slides

**[中文版](async-rl-comparison-slides.html)** · **[English](async-rl-comparison-slides-en.html)**

HTML 单文件，浏览器打开即用，键盘 ←→ 翻页。覆盖全部 4 个维度的可视化对比，包含时间线图、架构图和代码片段。

## 推荐阅读顺序

1. 先打开 **Slides** 建立全局画面（~10 min）
2. 选一篇最感兴趣的框架 walkthrough 深入（各 ~30 min）
3. 带着问题交叉阅读其他两篇的对应章节
