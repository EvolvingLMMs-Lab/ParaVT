# `requirements/`

Per-workload `uv`-managed lock files. Each lock pins the full
transitive closure for one of the three independent venvs. They
disagree on `torch` / `sglang` / `vllm` / `flash-attn` / `liger-kernel`
versions, which is why each workload gets its own venv.

| File | Venv | Pinned headline versions |
|---|---|---|
| [`sft.lock`](sft.lock) | `.venv-sft` | torch 2.8.0, flash-attn, liger-kernel, transformers 4.57.x |
| [`rl.lock`](rl.lock)  | `.venv-rl`  | torch 2.9.1+cu126, sglang 0.5.9, transformers 4.57.1 — `setup_env.sh` post-upgrades `nvidia-cudnn-cu12` to 9.16.0.29 (see below) |
| [`eval.lock`](eval.lock) | `.venv-eval` | vllm 0.17.x, torch 2.10.0, transformers, qwen-vl-utils, datasets |

## `cuDNN ≥ 9.16` is required for the RL venv

The RL venv's torch 2.9.1+cu126 has a known
[`nn.Conv3d` performance regression on cuDNN < 9.15](https://github.com/pytorch/pytorch/issues/168167)
that hangs SGLang's multimodal rollout for tens of minutes on the first
video forward pass.

The `rl.lock` file pins `nvidia-cudnn-cu12==9.10.2.21` because the
official `torch==2.9.1+cu126` wheel strict-pins it; `uv` will not
resolve a 9.16 lock there. `sft.lock` (torch 2.8.0) and `eval.lock`
(torch 2.10.0) end up at 9.10 too because that is the version pulled
in by their respective torch transitive closure on the cu126 index. The fix is post-install: `scripts/setup_env.sh's
`build_rl_env()` does `uv pip install --reinstall-package
nvidia-cudnn-cu12 9.16.0.29` so the RL venv ends up with 9.16. SFT
and eval workloads do not exercise the regression (no `nn.Conv3d` on
their hot path), so their venvs keep the lock-pinned 9.10.

## How to use

```bash
bash scripts/setup_env.sh sft         # builds .venv-sft
bash scripts/setup_env.sh rl          # builds .venv-rl
bash scripts/setup_env.sh eval        # builds .venv-eval
bash scripts/setup_env.sh all         # all three, sequential
```

Each invocation creates the matching `.venv-*` at the repo root,
installs from the lock file via `uv pip install`, and registers the
vendored framework (lmms-engine / areal / lmms-eval) as an editable
install so import paths resolve cleanly.

To regenerate a lock from the project's headline pins (say, after
bumping `[project.optional-dependencies]` in `pyproject.toml`):

```bash
uv pip compile --extra sft  pyproject.toml -o requirements/sft.lock
uv pip compile --extra rl   pyproject.toml -o requirements/rl.lock
uv pip compile --extra eval pyproject.toml -o requirements/eval.lock
```
