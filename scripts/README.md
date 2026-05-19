# `scripts/`

Top-level entry points. Each script handles one stage of the
SFT → RL → eval lifecycle, plus a single-shot inference demo and the
environment bootstrapper.

| Script | Purpose | Pairs with |
|---|---|---|
| [`setup_env.sh`](setup_env.sh) | Build the three uv venvs (`.venv-{sft,rl,eval}`) from the matching lock file under `requirements/`. Post-upgrade cuDNN to 9.16.0.29 in the RL venv (see [`requirements/README.md`](../requirements/README.md)), install editable copies of the vendored frameworks. | [`requirements/`](../requirements) |
| [`run_sft.sh`](run_sft.sh) | SFT cold-start launcher. Reads a config under [`paravt/sft/configs/`](../paravt/sft/configs), expands env vars, hands off to `lmms-engine` via `torchrun`. | [`paravt/sft/`](../paravt/sft) |
| [`run_rl.sh`](run_rl.sh) | PARA-GRPO RL launcher. Reads a config under [`paravt/rl/configs/`](../paravt/rl/configs), translates the `paravt:` block into env vars via `paravt.rl.config`, hands off to AReaL's local launcher. | [`paravt/rl/`](../paravt/rl) |
| [`run_eval.sh`](run_eval.sh) | Eval launcher. Reads a config under [`paravt/eval/configs/`](../paravt/eval/configs), iterates the `datasets:` list one at a time (kills vLLM between iterations to avoid session degradation), forwards CLI overrides to `python -m paravt.eval.driver`. | [`paravt/eval/`](../paravt/eval) |
| [`inference.py`](inference.py) | Single-shot inference demo. Loads a checkpoint via vLLM and answers one question about one video. `--prompt_mode` selects the same seven prompt shapes the eval driver uses. | [`paravt/eval/driver.py`](../paravt/eval/driver.py) |

All four shell scripts auto-activate the matching `.venv-*` if it
exists at the repo root and source `.secrets.env` for `WANDB_API_KEY`,
`HF_TOKEN`, etc. Override env vars from the calling shell to suit
your cluster.

For per-workload reproduce instructions (recipes, smoke tests, knob
descriptions), see the workload README the script pairs with above.
