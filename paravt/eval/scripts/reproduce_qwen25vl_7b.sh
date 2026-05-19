#!/usr/bin/env bash
# Qwen2.5-VL-7B (direct, plain instruct backbone).
set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

export PARAVT_EVAL_MODEL="${PARAVT_EVAL_MODEL:-Qwen/Qwen2.5-VL-7B-Instruct}"
export PARAVT_EVAL_OUT="${PARAVT_EVAL_OUT:-./eval-results}/qwen2.5-vl-7b"

# Six MCQ benches — direct mode
PARAVT_EVAL_OUT="${PARAVT_EVAL_OUT}/mcq" \
    bash scripts/run_eval.sh paravt/eval/configs/notool.yaml \
        --prompt_mode direct --video_channel image_url --no_mm_kwargs

# Charades grounding — video_url channel
PARAVT_EVAL_OUT="${PARAVT_EVAL_OUT}/charades" \
    bash scripts/run_eval.sh paravt/eval/configs/notool.yaml \
        --datasets charades \
        --prompt_mode direct --video_channel video_url --no_mm_kwargs
