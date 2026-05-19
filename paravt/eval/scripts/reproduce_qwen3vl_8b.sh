#!/usr/bin/env bash
# Qwen3-VL-8B-Instruct (with-tool).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

export PARAVT_EVAL_MODEL="${PARAVT_EVAL_MODEL:-Qwen/Qwen3-VL-8B-Instruct}"
export PARAVT_EVAL_OUT="${PARAVT_EVAL_OUT:-./eval-results}/qwen3-vl-8b"

PARAVT_EVAL_OUT="${PARAVT_EVAL_OUT}/mcq" \
    bash scripts/run_eval.sh paravt/eval/configs/withtool.yaml \
        --prompt_mode agentic_general --video_channel image_url --no_mm_kwargs

PARAVT_EVAL_OUT="${PARAVT_EVAL_OUT}/charades" \
    bash scripts/run_eval.sh paravt/eval/configs/withtool.yaml \
        --datasets charades \
        --prompt_mode agentic_general --video_channel video_url --no_mm_kwargs
