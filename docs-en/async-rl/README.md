# Async RL Training: Three-Framework Source-Level Comparison

> SLIME · NeMo-RL · veRL — Same problem, three engineering philosophies

## The Key Difference, at a Glance

<p align="center">
  <a href="async-rl-comparison-slides-en.html">
    <img src="https://github.com/user-attachments/assets/placeholder-timeline" alt="Scheduling × Buffer × Weight Sync — Full Timeline" width="900">
  </a>
</p>

> **👆 Placeholder** — Save a screenshot of the "Scheduling × Buffer × Weight Sync — Full Timeline" slide as `assets/timeline-overview.png`, then update the link above.
>
> For the interactive version, open **[async-rl-comparison-slides-en.html](async-rl-comparison-slides-en.html)** directly (keyboard nav supported).

Core differences across three frameworks on a single timeline:

| | SLIME | NeMo-RL | veRL |
|---|---|---|---|
| **Scheduling** | Main-loop for-loop | Main-loop + background Collector | Independent Rollouter/Trainer actors |
| **Buffer** | Double-buffer (depth=1) | Replay Buffer (depth=1~N) | Bounded Queue (maxlen=N) |
| **Sync Trigger** | Drain in-flight → sync | Sync every step, inference uninterrupted | Every N steps: abort → sync |
| **Staleness Bound** | 1 step (structural) | `max_trajectory_age_steps` | `staleness_threshold` |
| **Partial Rollout** | Abort + recycle | None (samples complete independently) | Abort + prefix continuation |

```
Conservative ◄────────────────────────► Aggressive
(on-policy first)                 (throughput first)

     SLIME ·····  verl ·····  NeMo-RL
      20%          55%          70%
```

## Analysis Dimensions

Each walkthrough covers 4 core dimensions from the [HuggingFace Async RL Survey](https://huggingface.co/blog/async-rl-training-landscape):

| # | Dimension | Key Question |
|---|-----------|-------------|
| 2 | **Rollout Buffer** | Where is generated data staged? How deep? What happens when full? |
| 3 | **Weight Sync** | How do new weights reach the inference engine? Must inference pause? |
| 4 | **Staleness Management** | Can data from old weights still be used? How to correct? |
| 5 | **Partial Rollout** | What happens to in-flight requests during sync? Drop or resume? |

## Walkthroughs

### [SLIME Async RL Walkthrough](slime-async-rl-walkthrough.md)

**Keywords**: Double-buffer · abort + recycle · TIS + OPSM

The most concise async implementation — ~40-line main loop, buffer depth locked at 1, structurally minimizing staleness. Best starting point for understanding the full async RL flow.

- How double-buffer scheduling overlaps Gen N+1 with Train N
- The drain → flush → sync → continue weight sync sequence
- Off-policy token correction via TIS weighting + OPSM masking

### [NeMo-RL Async RL Walkthrough](nemo-rl-async-rl-walkthrough.md)

**Keywords**: Replay Buffer · in-flight weight update · target version matching

The most flexible buffer design — Collector generates continuously in the background while training pulls data on demand. Unique ability to hot-update inference weights mid-generation.

- How `ReplayBuffer` filters samples via version + age window
- `refit_policy_generation()` for update-without-stopping
- TIS / ICE-POP / seq-mask-TIS: three IS correction variants compared

### [veRL Async RL Walkthrough](verl-async-rl-walkthrough.md)

**Keywords**: Bounded Queue · backpressure · MIS · prefix continuation

The most decoupled architecture — Rollouter and Trainer are fully independent Ray actors communicating via MessageQueue. Richest staleness correction toolkit.

- Per-sample streaming enqueue + dual-layer backpressure
- NCCL bucketed broadcast weight sync implementation
- Multi-version Importance Sampling (MIS) with CPU snapshots
- Post-abort prefix continuation across weight versions

## Interactive Comparison Slides

**[English](async-rl-comparison-slides-en.html)** · **[中文版](../../docs/async-rl/async-rl-comparison-slides.html)**

Single HTML file, open in any browser, navigate with ← → keys. Visual comparisons across all 4 dimensions with timelines, architecture diagrams, and code snippets.

## Suggested Reading Order

1. Start with the **Slides** for the big picture (~10 min)
2. Pick the framework walkthrough that interests you most (~30 min each)
3. Cross-reference the corresponding sections in the other two
