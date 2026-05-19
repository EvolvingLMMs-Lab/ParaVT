#!/usr/bin/env bash
# Run every headline-table row in sequence on a single 8-GPU box.
#
# Each row writes to ./eval-results/<row_tag>/{mcq,charades}/...; re-runs
# are resume-friendly (existing summary JSONs are skipped). For 4-GPU
# boxes pass `--num_gpus 4` and the driver auto-uses tp1 + dp4.
#
# Placeholder rows (VideoZoomer / SAGE / LongVT-RFT) are gated behind
# PARAVT_RUN_STUBS=1 and require their per-baseline system prompts to
# be filled in inside paravt/eval/driver.py — see eval/README.md.

set -uo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

run_row() {
    local script="$1"
    echo
    echo "============================================================"
    echo "[batch_reproduce] ${script}"
    echo "============================================================"
    bash "${HERE}/${script}" || echo "[batch_reproduce] WARN ${script} returned non-zero"
}

# Directly runnable rows (3 prompts shipped verbatim: direct / reasoning / agentic_general)
run_row reproduce_qwen25vl_7b.sh
run_row reproduce_qwen3vl_8b.sh
run_row reproduce_paravt_8b.sh
run_row reproduce_conan_7b.sh

# Reasoning baselines — same script, different model id.
for model in \
    "Video-R1/Video-R1-7B" \
    "QiWang98/VideoRFT" \
    "OpenGVLab/VideoChat-R1_7B" \
    "ShijianW01/Video-Thinker-7B" \
    "Boshenxx/Time-R1-7B" \
    "zcccccz/ReWatch-R1"; do
    PARAVT_EVAL_MODEL="${model}" \
        run_row reproduce_reasoning_baselines.sh
done

# Placeholder rows (require system prompts to be filled in).
if [[ "${PARAVT_RUN_STUBS:-0}" == "1" ]]; then
    run_row reproduce_videozoomer_7b.sh
    run_row reproduce_sage_7b.sh
    run_row reproduce_longvt_rft.sh
else
    echo
    echo "[batch_reproduce] Skipping VideoZoomer / SAGE / LongVT-RFT rows."
    echo "Fill VIDEOZOOMER_AGENTIC_SYSTEM, SAGE_AGENTIC_SYSTEM, and"
    echo "LONGVT_TOOL_PROMPT_SUFFIX inside paravt/eval/driver.py, then"
    echo "rerun with PARAVT_RUN_STUBS=1."
fi
