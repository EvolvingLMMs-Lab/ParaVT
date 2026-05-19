#!/usr/bin/env bash
# Reasoning baselines: Video-R1, VideoRFT, VideoChat-R1, Video-Thinker,
# Time-R1, ReWatch-R1 — all share --prompt_mode reasoning.
#
# Override PARAVT_EVAL_MODEL per row, e.g.:
#     PARAVT_EVAL_MODEL=Video-R1/Video-R1-7B \
#         bash paravt/eval/scripts/reproduce_reasoning_baselines.sh
#
# Defaults to Video-R1; the per-row HF ids are documented in
# eval/README.md (Per-row reproduce table).

set -euo pipefail
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

export PARAVT_EVAL_MODEL="${PARAVT_EVAL_MODEL:-Video-R1/Video-R1-7B}"
ROW_TAG="${PARAVT_EVAL_ROW:-$(basename "${PARAVT_EVAL_MODEL}")}"
export PARAVT_EVAL_OUT="${PARAVT_EVAL_OUT:-./eval-results}/reasoning-${ROW_TAG}"

PARAVT_EVAL_OUT="${PARAVT_EVAL_OUT}/mcq" \
    bash scripts/run_eval.sh paravt/eval/configs/withtool.yaml \
        --prompt_mode reasoning --max_turns 1 \
        --video_channel image_url --no_mm_kwargs

PARAVT_EVAL_OUT="${PARAVT_EVAL_OUT}/charades" \
    bash scripts/run_eval.sh paravt/eval/configs/withtool.yaml \
        --datasets charades \
        --prompt_mode reasoning --max_turns 1 \
        --video_channel video_url --no_mm_kwargs
