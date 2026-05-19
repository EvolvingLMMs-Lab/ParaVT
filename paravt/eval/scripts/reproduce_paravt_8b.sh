#!/usr/bin/env bash
# ParaVT-8B (Ours).
# Six MCQ benches via image_url; charades via video_url.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "${REPO_ROOT}"

export PARAVT_EVAL_MODEL="${PARAVT_EVAL_MODEL:-ParaVT/ParaVT-8B}"
export PARAVT_EVAL_OUT="${PARAVT_EVAL_OUT:-./eval-results}/paravt-8b"

# Six MCQ benches — image_url channel
PARAVT_EVAL_OUT="${PARAVT_EVAL_OUT}/mcq" \
    bash scripts/run_eval.sh paravt/eval/configs/withtool.yaml \
        --video_channel image_url

# Charades grounding — video_url channel + agentic_minimal prompt.
# The minimal tool-call envelope produces tighter temporal spans on
# Charades-STA than the verbose agentic_general scaffolding.
PARAVT_EVAL_OUT="${PARAVT_EVAL_OUT}/charades" \
    bash scripts/run_eval.sh paravt/eval/configs/withtool.yaml \
        --datasets charades \
        --prompt_mode agentic_minimal \
        --video_channel video_url
