#!/usr/bin/env bash
# ParaVT evaluation launcher.
#
# Iterates the recipe's `datasets:` list one at a time, killing vLLM
# between iterations so the next dataset starts from a clean session.
# This is the only safe way to run the eval suite end-to-end: streaming
# a single vLLM session through five or more long-video benchmarks
# back-to-back drops connections silently past the fifth dataset (we
# measured MLVU 37 %, MMVU 62 %, Charades-STA 62 % API errors on a
# 7-dataset csv run), and the driver's per-dataset error counter does
# not catch mid-call connection drops.
#
# Usage:
#     bash scripts/run_eval.sh paravt/eval/configs/withtool.yaml
#     bash scripts/run_eval.sh paravt/eval/configs/notool.yaml
#     bash scripts/run_eval.sh paravt/eval/configs/withtool.yaml \
#         --prompt_mode reasoning --max_turns 1               # reasoning baseline
#     bash scripts/run_eval.sh paravt/eval/configs/withtool.yaml \
#         --datasets charades --video_channel video_url       # one dataset only
#
# Overrides after the recipe path are forwarded to every per-dataset
# invocation. Pass `--datasets <csv>` to restrict the loop to a subset
# of the recipe's list.
#
# The recipe's `prompt_mode` field is the source of truth (one of
# `direct`, `reasoning`, `agentic_general`, `agentic_videozoomer`,
# `agentic_sage`, `agentic_longvt`). The legacy `no_tool: true` recipe
# field is still accepted and maps to `--prompt_mode direct`.
#
# Required env vars (or set them in .secrets.env):
#     PARAVT_EVAL_MODEL   model under test (HF id or snapshot path)
#     PARAVT_EVAL_OUT     output directory for per-sample JSON
#
# Each invocation lands per-dataset JSON under
# `${PARAVT_EVAL_OUT}/<mode>/<dataset>.json` plus a rolled-up
# `summary_all.json`. The driver is resume-friendly: re-running the
# same `output_dir` skips datasets whose summary is already on disk.

set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RECIPE="${1:-paravt/eval/configs/withtool.yaml}"
[[ $# -gt 0 ]] && shift

if [[ ! -f "${RECIPE}" ]]; then
    echo "[run_eval] recipe not found: ${RECIPE}"
    exit 1
fi

if [[ -z "${PARAVT_NO_AUTO_ACTIVATE:-}" ]] && [[ -f .venv-eval/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv-eval/bin/activate
fi
[[ -f .secrets.env ]] && source .secrets.env

# A `--datasets <csv>` override on the command line replaces the
# recipe's datasets list; otherwise we iterate the recipe's list. We
# still parse the YAML once for the rest of the args.
CLI_DATASETS=""
PASSTHROUGH_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --datasets)
            CLI_DATASETS="$2"; shift 2 ;;
        --datasets=*)
            CLI_DATASETS="${1#--datasets=}"; shift ;;
        *)
            PASSTHROUGH_ARGS+=("$1"); shift ;;
    esac
done

# Discard stderr (antlr4 / OmegaConf can spam stdout/stderr "ANTLR runtime
# version disagree" warnings when the lmms-eval-pinned antlr4-python3-runtime
# 4.7.2 collides with OmegaConf's 4.9.3-generated parser; those warnings break
# the read -r token split). `tail -1` keeps only the actual data line — the
# python script writes one print() to stdout, so any extra lines must be
# noise above it.
read -r PROMPT_MODE VIDEO_CHANNEL NO_MM_KWARGS MODEL_PATH RECIPE_DATASETS OUTPUT_DIR NUM_GPUS BASE_PORT MAIN_NFRAMES MAX_TURNS MAX_PARALLEL WORKERS_PER_SHARD SMOKE_TEST < <(python -c "
import sys
from omegaconf import OmegaConf
c = OmegaConf.load('${RECIPE}')
mode = c.get('prompt_mode', None)
if mode is None:
    mode = 'direct' if bool(c.get('no_tool', False)) else 'agentic_general'
print(
    mode,
    c.get('video_channel', 'image_url'),
    str(bool(c.get('no_mm_kwargs', True))),
    c.model_path, c.datasets, c.output_dir, c.num_gpus, c.base_port,
    c.main_nframes, c.get('max_turns', 3), c.get('max_parallel', 5),
    c.workers_per_shard, c.smoke_test,
)
" 2>/dev/null | tail -1)

DATASETS="${CLI_DATASETS:-${RECIPE_DATASETS}}"
IFS=',' read -r -a DS_ARR <<< "${DATASETS}"

echo "[run_eval] recipe=${RECIPE} prompt=${PROMPT_MODE} model=${MODEL_PATH}"
echo "[run_eval] iterating ${#DS_ARR[@]} dataset(s) one at a time:"
printf '  - %s\n' "${DS_ARR[@]}"

run_one() {
    local ds="$1"
    local args=(
        --model_path "${MODEL_PATH}"
        --datasets "${ds}"
        --output_dir "${OUTPUT_DIR}"
        --num_gpus "${NUM_GPUS}"
        --base_port "${BASE_PORT}"
        --main_nframes "${MAIN_NFRAMES}"
        --max_turns "${MAX_TURNS}"
        --max_parallel "${MAX_PARALLEL}"
        --workers_per_shard "${WORKERS_PER_SHARD}"
        --prompt_mode "${PROMPT_MODE}"
        --video_channel "${VIDEO_CHANNEL}"
    )
    [[ "${NO_MM_KWARGS}" == "True" ]] && args+=(--no_mm_kwargs)
    [[ "${SMOKE_TEST}" == "True" ]] && args+=(--smoke_test)
    python -m paravt.eval.driver "${args[@]}" "${PASSTHROUGH_ARGS[@]}"
}

for ds in "${DS_ARR[@]}"; do
    ds_clean="$(echo "${ds}" | xargs)"
    echo
    echo "========================================================"
    echo "[run_eval] dataset: ${ds_clean}"
    echo "========================================================"

    run_one "${ds_clean}" \
        || echo "[run_eval] WARN ${ds_clean} returned non-zero; continuing"

    # Tear down lingering vLLM servers so the next dataset starts on a
    # clean session. Safe because the driver has already exited; we
    # just want to make sure no zombie schedulers remain.
    if (( ${#DS_ARR[@]} > 1 )); then
        echo "[run_eval] tearing down vLLM session before next dataset..."
        pkill -9 -f "vllm.entrypoints" 2>/dev/null || true
        pkill -9 -f "sglang::scheduler" 2>/dev/null || true
        sleep 8
    fi
done

echo
echo "[run_eval] done."
