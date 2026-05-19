# PARA-GRPO RL training

**PARA-GRPO** (*Parseability-Anchored and Ratio-gAted GRPO*) post-trains a tool-native LMM through two components that target the two failure modes vanilla GRPO surfaces (Format Fragility + Tool Necessity Gap):

- **Exploration Anchoring** — rewards parseable format only at the structural tokens most prone to collapse (`</think>` closure, full `<think>→<answer>` flow), without restricting tool-call content. Surfaced as the `constrained` (`think_prefix` / `answer_suffix` gates) and `anchoring` (`weight` = the outer `λ_anchor` in `R_fmt = R_base_fmt + λ_anchor · R_anchor`) blocks of the recipe YAML.
- **nFrames Gating** — randomizes the per-prompt overview frame budget `K ~ Uniform({4, 8, 16, 32, 64})` so calling `crop_video` earns measurable credit on prompts where the overview alone is insufficient. Surfaced as the `gating.nframes_gating` flag.

| File | Purpose |
|---|---|
| `_base.yaml` | Shared AReaL training config — paths, scheduler, actor / ref / sglang settings |
| `paragrpo_8b.yaml` | **Default** — full PARA-GRPO recipe (both components on) |
| `vanilla_grpo_8b.yaml` | Vanilla-GRPO baseline (anchoring off, gating off) |

## Environment

```bash
bash scripts/setup_env.sh rl          # creates .venv-rl from requirements/rl.lock
source .venv-rl/bin/activate
```

The RL venv runs AReaL on top of SGLang for rollouts and FSDP2 for the policy. `requirements/rl.lock` pins torch 2.9.1+cu126, sglang 0.5.9, transformers 4.57.1, and **`nvidia-cudnn-cu12==9.16.0.29`** — the cuDNN pin is required because PyTorch 2.9.1 has a `nn.Conv3d` perf regression on cuDNN < 9.15 that hangs SGLang's multimodal rollout for tens of minutes (see [pytorch/pytorch#168167](https://github.com/pytorch/pytorch/issues/168167)).

AReaL is vendored under `paravt/rl/areal/` with our hierarchical-agent additions already applied; see `patches/areal/` for the diff vs. upstream.

## The `paravt:` recipe block

```yaml
paravt:
  constrained:
    think_prefix: true
    answer_suffix: true
  gating:
    nframes_gating: true
  anchoring:
    weight: 0.5            # set to 0.0 to disable
  reward:
    mode: f1               # f1 | exact | llm
    format_weight: 1.0
```

`paravt/rl/config.py` translates this block into the env vars (`THINK_PREFIX`, `ANSWER_SUFFIX`, `NFRAMES_GATING`, `ANCHOR_WEIGHT`, `REWARD_MODE`, `FORMAT_WEIGHT`) the reward modules read at startup. `scripts/run_rl.sh` does this translation before invoking AReaL.

## What each YAML knob does

| Knob | What it controls | Default |
|---|---|---|
| `constrained.think_prefix` | Fix the first tokens of every response to `<think>\n` so rollouts cannot blind-call a tool or jump straight to `<answer>`. | `true` |
| `constrained.answer_suffix` | Add an `<answer>`-tag component to the format reward so closure of the answer block still earns credit when the rest of the format is imperfect. | `true` |
| `gating.nframes_gating` | Randomize the per-prompt overview budget `K ~ Uniform({4, 8, 16, 32, 64})`. Reduced budgets hide part of the evidence so `crop_video` is needed; the full-budget setting keeps direct answering possible. | `true` |
| `anchoring.weight` | Outer `λ_anchor` in `R_fmt = R_base_fmt + λ_anchor · R_anchor`. The selective-anchor reward awards `+0.4` for `</think>` closure, `+0.3` for the full `<think> → </think> → <answer>` flow, and `−0.3` if `<think>` is opened but never closed. Set to `0.0` for the vanilla baseline. | `0.5` |
| `reward.mode` | Task-accuracy backend: `f1` token-F1 (open-ended QA), `exact` exact-match, `llm` LLM-as-judge. | `f1` |
| `reward.format_weight` | Outer weight on `R_fmt` in the additive reward sum `R = R_acc + format_weight · R_fmt + R_tool`. | `1.0` |

## Reproducing the PARA-GRPO checkpoint

```bash
export PARAVT_BASE_MODEL=/path/to/sft/cold-start/checkpoint   # see paravt/sft/README.md
export PARAVT_TRAIN_DATA=/path/to/paravt_rl_diverse_4k4.parquet     # from ParaVT/ParaVT-Parquet
export PARAVT_VIDEO_ROOT=/path/to/source/videos
export PARAVT_FILEROOT=./experiments

bash scripts/run_rl.sh paravt/rl/configs/paragrpo_8b.yaml \
    trial_name=paragrpo-run0
```

Wall-clock on 8 × 80 GB+ NVIDIA GPUs is ≈ 50 h to convergence (700+ steps at ≈ 250 s / step including rollout). Checkpoints land under `${PARAVT_FILEROOT}/checkpoints/.../paragrpo-run0/`; eval them via `paravt/eval/configs/withtool.yaml` (see `paravt/eval/README.md`).

## Vanilla-GRPO baseline

```bash
bash scripts/run_rl.sh paravt/rl/configs/vanilla_grpo_8b.yaml \
    trial_name=vanilla-grpo
```

## Smaller-scale sanity test (4-GPU)

> [!WARNING]
> The default `paragrpo_8b.yaml` is sized for an 8-GPU box (`d1p1t1+d7p1t1`,
> `train_dataset.batch_size: 7`). Running it on fewer GPUs **without** the
> overrides below will crash at PPOActor init with
> `ValueError: batch size(7) must be divisible by world_size(N)` after vLLM
> has already loaded the 8 B model — about 4–5 minutes of wasted warmup per
> attempt.

For a 4-GPU box you must override the AReaL allocation to fit:

```bash
bash scripts/run_rl.sh paravt/rl/configs/paragrpo_8b.yaml \
    trial_name=smoke-4gpu \
    cluster.n_gpus_per_node=4 \
    allocation_mode=sglang:d1p1t1+d3p1t1 \
    train_dataset.batch_size=3 \
    valid_dataset.batch_size=3 \
    gconfig.n_samples=2 \
    saver.freq_steps=1 \
    total_train_epochs=1
```

`d3p1t1` (3-way actor data parallelism) requires `train_dataset.batch_size` to be a multiple of 3; if you need a different batch size, pick an `allocation_mode` whose dp degree divides it. Running this end-to-end is the recommended way to verify a fresh environment install before launching a multi-day production job.

## Adding a new reward term

Reward terms live in `paravt/rl/rewards/` (one function per file). To add a new term:

1. Drop a module under `paravt/rl/rewards/` exposing one function `compute_<name>_reward(completion, ...) -> float`.
2. Wire it into `paravt/rl/rewards/compose.py:compose_reward` next to the existing `format_reward` / `tool_reward` calls.
3. (Optional) gate it behind a new field in the `paravt:` YAML block + a new env var bridged through `paravt/rl/config.py`.

`paravt/rl/train.py` does not change — it just imports `compose_reward`.
