#!/usr/bin/env bash
# ParaVT RL launcher.
#
# Usage:
#     bash scripts/run_rl.sh <recipe.yaml> [hydra overrides...]
#
# Examples:
#     # default PARA-GRPO recipe
#     bash scripts/run_rl.sh paravt/rl/configs/paragrpo_8b.yaml \
#         trial_name=paragrpo-run0
#
#     # 4-GPU smoke test (override allocation_mode for fewer GPUs)
#     bash scripts/run_rl.sh paravt/rl/configs/paragrpo_8b.yaml \
#         trial_name=smoke \
#         allocation_mode=sglang:d1p1t1+d3p1t1 \
#         total_train_epochs=1
#
# Required env vars (set in .secrets.env or your shell):
#     PARAVT_FILEROOT     output directory for checkpoints + logs
#     PARAVT_BASE_MODEL   HF snapshot path or hub id
#     PARAVT_TRAIN_DATA   parquet file for training rollouts
#     PARAVT_VALID_DATA   parquet file for validation rollouts
#     PARAVT_VIDEO_ROOT   directory containing source videos
#
# Optional:
#     WANDB_API_KEY               enables wandb logging when set
#     WANDB_ENTITY                wandb entity (defaults to user default)
#     SGLANG_VLM_CACHE_SIZE_MB    raise to >=4096 for 64-frame video rollouts
#     LLM_JUDGE_API_KEY/BASE_URL/MODEL    required only if reward.mode == llm
#     PARAVT_NO_AUTO_ACTIVATE=1   skip auto-activating .venv-rl

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RECIPE="${1:-paravt/rl/configs/paragrpo_8b.yaml}"
[[ $# -gt 0 ]] && shift

if [[ ! -f "${RECIPE}" ]]; then
    echo "[run_rl] recipe not found: ${RECIPE}"
    exit 1
fi

# Activate the RL venv if it exists. Users who manage their own env can
# skip this by exporting PARAVT_NO_AUTO_ACTIVATE=1.
if [[ -z "${PARAVT_NO_AUTO_ACTIVATE:-}" ]] && [[ -f .venv-rl/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv-rl/bin/activate
fi

# Optional secrets — silently no-op if missing.
[[ -f .secrets.env ]] && source .secrets.env

# Sensible defaults: 64-frame video rollouts need ~4 GB of VLM cache.
export SGLANG_VLM_CACHE_SIZE_MB="${SGLANG_VLM_CACHE_SIZE_MB:-4096}"

# SGLang 0.5.9+ runs a defensive cuDNN-version probe at startup that prints
# a noisy warning even after build_rl_env() upgrades cuDNN to 9.16.0.29
# (the probe reads the version of the symbolically-loaded driver before the
# pip-shipped 9.16 wheel takes over). Disable the probe so the launch log
# stays clean; set SGLANG_DISABLE_CUDNN_CHECK=0 if you suspect a real
# version mismatch.
export SGLANG_DISABLE_CUDNN_CHECK="${SGLANG_DISABLE_CUDNN_CHECK:-1}"

# Apply the recipe's paravt: block to env vars before AReaL boots so
# the reward modules pick them up at module-import time. The block is
# also defined in HierarchicalAgentGRPOConfig so AReaL accepts it.
python -c "from paravt.rl.config import apply_paravt_config; print('[run_rl] PARA-GRPO env:', apply_paravt_config('${RECIPE}'))"

# Extract trial_name from overrides for log filename.
TRIAL_NAME="default-run"
for arg in "$@"; do
    [[ "${arg}" == trial_name=* ]] && TRIAL_NAME="${arg#trial_name=}"
done
LOG="${PARAVT_FILEROOT:-./experiments}/logs/${TRIAL_NAME}.log"
mkdir -p "$(dirname "${LOG}")"

# Optional wandb pre-create — survives heavy GPU loading.
if [[ -n "${WANDB_API_KEY:-}" ]]; then
    export WANDB_RUN_ID="${TRIAL_NAME}-$(date +%Y%m%d%H%M%S)"
    python -c "
import os, wandb
run = wandb.init(
    entity=os.environ.get('WANDB_ENTITY'),
    project=os.environ.get('WANDB_PROJECT', 'paravt'),
    name='${TRIAL_NAME}',
    id=os.environ['WANDB_RUN_ID'],
    resume='allow',
    settings=wandb.Settings(init_timeout=30),
)
print('wandb run pre-created:', run.url)
wandb.finish()
" || echo "[run_rl] wandb pre-create failed; continuing"
fi

echo "=== ParaVT RL Launch ==="
echo "Recipe : ${RECIPE}"
echo "Trial  : ${TRIAL_NAME}"
echo "Log    : ${LOG}"
echo "Args   : $*"
echo "========================="

# AReaL launches each worker as a subprocess of areal.launcher.local.
# The training entry point is paravt/rl/train.py.
exec python -m areal.launcher.local paravt/rl/train.py \
    --config "${RECIPE}" \
    "$@" \
    2>&1 | tee "${LOG}"
