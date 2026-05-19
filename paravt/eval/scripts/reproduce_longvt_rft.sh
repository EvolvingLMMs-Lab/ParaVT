#!/usr/bin/env bash
# LongVT-RFT-7B (agentic_longvt, prior work).
#
# NOTE: this row requires LONGVT_TOOL_PROMPT_SUFFIX to be filled in
# inside paravt/eval/driver.py (currently a placeholder).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

export PARAVT_EVAL_MODEL="${PARAVT_EVAL_MODEL:-longvideotool/LongVT-RFT}"
export PARAVT_EVAL_OUT="${PARAVT_EVAL_OUT:-./eval-results}/longvt-rft-7b"

PARAVT_EVAL_OUT="${PARAVT_EVAL_OUT}/mcq" \
    bash scripts/run_eval.sh paravt/eval/configs/withtool.yaml \
        --prompt_mode agentic_longvt --video_channel image_url --no_mm_kwargs

PARAVT_EVAL_OUT="${PARAVT_EVAL_OUT}/charades" \
    bash scripts/run_eval.sh paravt/eval/configs/withtool.yaml \
        --datasets charades \
        --prompt_mode agentic_longvt --video_channel video_url --no_mm_kwargs
