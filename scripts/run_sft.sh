#!/usr/bin/env bash
# ParaVT SFT launcher (lmms-engine FSDP2 trainer).
#
# Usage:
#     bash scripts/run_sft.sh [recipe.yaml]                # default: qwen3vl_8b
#     bash scripts/run_sft.sh paravt/sft/configs/<custom>.yaml
#
# Required env vars (set in .secrets.env or your shell):
#     PARAVT_FILEROOT     output directory for checkpoints
#     PARAVT_BASE_MODEL   HF snapshot or Hub id (e.g. Qwen/Qwen3-VL-8B-Instruct)
#     PARAVT_SFT_DATA     path to data manifest YAML (see paravt/sft/configs/data_manifest.example.yaml)
#
# Optional:
#     N_GPUS              GPUs per node, defaults to nproc-detected count
#     MASTER_PORT         torchrun master port, defaults 29500

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

RECIPE="${1:-paravt/sft/configs/qwen3vl_8b.yaml}"
[[ $# -gt 0 ]] && shift
if [[ ! -f "${RECIPE}" ]]; then
    echo "[run_sft] recipe not found: ${RECIPE}"
    exit 1
fi

if [[ -z "${PARAVT_NO_AUTO_ACTIVATE:-}" ]] && [[ -f .venv-sft/bin/activate ]]; then
    # shellcheck disable=SC1091
    source .venv-sft/bin/activate
fi
[[ -f .secrets.env ]] && source .secrets.env

N_GPUS="${N_GPUS:-$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | wc -l)}"
[[ "${N_GPUS}" -gt 0 ]] || { echo "[run_sft] no GPUs detected; set N_GPUS manually"; exit 1; }
MASTER_PORT="${MASTER_PORT:-29500}"

# Convert YAML recipe to lmms-engine Hydra overrides via OmegaConf.
# - Each override is emitted on its own NUL-separated line, so any value
#   containing whitespace stays one shell argument (mapfile -d '' below).
# - List values are emitted in Hydra's bracket-list syntax `[a,b,c]`
#   (NOT YAML `- a\n- b`), which is the only form Hydra's override grammar
#   accepts on the command line.
# - None / empty containers are skipped — Hydra rejects bare `key=` overrides.
# - The unquoted-OVERRIDES word-splitting trap (which lets `report_to: ["wandb"]`
#   leak `wandb` as its own bare override → `missing EQUAL` parse error) is
#   eliminated by mapfile + array expansion `"${OVERRIDES[@]}"`.
ABS_RECIPE="$(cd "$(dirname "${RECIPE}")" && pwd)/$(basename "${RECIPE}")"
# NUL-separated stream so values containing whitespace stay one shell token.
mapfile -d '' -t OVERRIDES < <(python - "${ABS_RECIPE}" <<'PYEOF'
import json
import sys
from omegaconf import OmegaConf
cfg = OmegaConf.to_container(OmegaConf.load(sys.argv[1]), resolve=True)
def hydra_value(v):
    if isinstance(v, list):
        # Hydra list literal: [a,b,c] — string elements JSON-quoted for safety.
        return "[" + ",".join(json.dumps(x) if isinstance(x, str) else str(x) for x in v) + "]"
    if isinstance(v, bool):
        return "true" if v else "false"
    if v is None:
        return "null"
    return str(v)
def flat(prefix, node):
    if isinstance(node, dict):
        if not node:
            return  # skip empty container — Hydra rejects bare `key=` overrides
        for k, vv in node.items():
            yield from flat(f"{prefix}.{k}" if prefix else k, vv)
    else:
        sys.stdout.write(f"{prefix}={hydra_value(node)}\0")
for _ in flat("", cfg): pass
PYEOF
)

# Stay at REPO_ROOT so relative paths in the recipe (e.g. the SFT data
# manifest at `paravt/sft/configs/data_manifest.example.yaml`) resolve
# correctly. lmms_engine is editable-installed in .venv-sft, so
# `python -m lmms_engine.launch.cli` works from any cwd.
#
# `hydra.job.chdir=False` prevents Hydra from auto-chdir-ing into
# outputs/<date>/<time>/ before main() runs (Hydra 1.2 still defaults
# to True under some setups).
# `hydra.run.dir=.` keeps any Hydra-side bookkeeping inside the cwd
# rather than spawning an outputs/ subtree.
exec torchrun \
    --nproc_per_node="${N_GPUS}" \
    --nnodes=1 \
    --node_rank=0 \
    --master_addr=127.0.0.1 \
    --master_port="${MASTER_PORT}" \
    -m lmms_engine.launch.cli \
    hydra.job.chdir=False \
    hydra.run.dir=. \
    "${OVERRIDES[@]}" \
    "$@"
