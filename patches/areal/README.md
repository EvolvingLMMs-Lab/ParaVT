# AReaL patches

ParaVT vendors a fork of [`inclusionAI/AReaL`](https://github.com/inclusionAI/AReaL) inline at `paravt/rl/areal/`. The vendor diverges from upstream commit `7927735` along two axes:

1. **Inline edits**, applied directly to the vendored tree and visible by diffing `paravt/rl/areal/` against upstream:
   - Hierarchical agent workflow scaffolding (subagent dispatch + per-turn rollout cache)
   - Advantage broadcasting fix for sub-trajectories
   - Qwen3-VL training entry-point with a clean env + F1 reward
   - rmpad return-value discard fix in `dist_rollout.py`
   - SGLang watchdog timeout extended to 600 s in `cli_args.py`
   - wandb init timeout + retry hardening in `launch.sh`
   - Video-path resolution fix in `cropped_video.py`

2. **Discrete `.patch` files** in this directory, listed below. The vendored tree already has them applied; downstream forks reapplying against fresh upstream should consume them in numeric order.

## Pin

| Repo | Commit | Date |
|---|---|---|
| `inclusionAI/AReaL` upstream | near commit `7927735` (Ulysses context-parallel support) | 2026-01-12 |
| Vendored snapshot in this repo | `paravt/rl/areal/` (matches upstream + the changes listed above) | 2026-01-29 |

## Patches

### `0001-make-swanlab-import-optional.patch`

**Problem.** `areal/utils/stats_logger.py` unconditionally imports and calls `swanlab`. SwanLab's pin (`rich<14`) conflicts with the rest of the AReaL/sglang/transformers stack, which requires `rich>=14`. As a result, a fresh `uv pip install -r requirements/rl.lock` either fails resolution or — if SwanLab is dropped from the lock — crashes at import time inside the trainer.

**Fix.** Guard `import swanlab` with a `try/except ImportError` and gate every call site on `_HAS_SWANLAB`. The change is functionally a no-op when SwanLab is installed, and it allows ParaVT's wandb-only stack to import the trainer cleanly.

**Upstream candidate fix.** SwanLab as an optional extra (`pip install areal[swanlab]`) rather than a hard dependency.
