#!/usr/bin/env bash
# VideoZoomer-7B (agentic_videozoomer, native <video_zoom> tool).
#
# NOTE: this row requires VIDEOZOOMER_AGENTIC_SYSTEM to be filled in
# inside paravt/eval/driver.py (currently a placeholder).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

export PARAVT_EVAL_MODEL="${PARAVT_EVAL_MODEL:-zsgvivo/videozoomer}"
export PARAVT_EVAL_OUT="${PARAVT_EVAL_OUT:-./eval-results}/videozoomer-7b"

PARAVT_EVAL_OUT="${PARAVT_EVAL_OUT}/mcq" \
    bash scripts/run_eval.sh paravt/eval/configs/withtool.yaml \
        --prompt_mode agentic_videozoomer --video_channel image_url --no_mm_kwargs

PARAVT_EVAL_OUT="${PARAVT_EVAL_OUT}/charades" \
    bash scripts/run_eval.sh paravt/eval/configs/withtool.yaml \
        --datasets charades \
        --prompt_mode agentic_videozoomer --video_channel video_url --no_mm_kwargs
