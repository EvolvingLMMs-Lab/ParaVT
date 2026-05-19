#!/usr/bin/env bash
# ParaVT environment setup — creates uv-managed venvs for SFT, RL, and eval.
#
# Usage:
#     bash scripts/setup_env.sh [sft|rl|eval|all]
#
# Prerequisites:
#     - uv (https://github.com/astral-sh/uv) installed and on PATH
#     - Python 3.12 available
#     - CUDA 12.6 toolchain matching torch 2.9.1+cu126
#     - cuDNN >= 9.10 (lock files); the RL venv post-upgrades to 9.16
#
# Why the cuDNN handling looks indirect: PyTorch 2.9.1+cu126 hard-pins
# nvidia-cudnn-cu12==9.10.2.21, so all three locks pin 9.10. The RL venv
# uses SGLang multimodal rollout which hits a known nn.Conv3d perf
# regression on cuDNN <9.15 (pytorch/pytorch#168167). build_rl_env()
# below post-installs 9.16.0.29 so the RL venv ends up at 9.16; SFT and
# eval do not need the upgrade and stay on the lock-pinned 9.10.
#
# After this script finishes, activate the venv that matches your workload:
#     source .venv-sft/bin/activate   # for SFT cold-start training
#     source .venv-rl/bin/activate    # for RL training and rollouts
#     source .venv-eval/bin/activate  # for evaluation
#
# Each venv is independent because torch/sglang/vllm version pins disagree
# across workloads. This mirrors the production setup we used to produce the
# paper numbers and avoids dependency conflicts.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

if ! command -v uv >/dev/null 2>&1; then
    echo "[setup_env] uv not found. Install via: curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi

PYTHON_VERSION="${PYTHON_VERSION:-3.12}"
WHICH="${1:-all}"

UV_INDEX_FLAGS=(
    --index-strategy unsafe-best-match
    --index-url https://pypi.org/simple
    --extra-index-url https://download.pytorch.org/whl/cu126
)

# Extract a single package's exact pin from a lock file (e.g. "torch==2.8.0").
# Used so the pre-install pass picks the same version the lock will resolve to,
# avoiding a pass-1/pass-2 ABI mismatch that breaks pre-built native wheels
# (e.g. flash-attn compiled against the pass-1 torch but loaded against the
# pass-2 torch).
_torch_pin_from_lock() {
    grep -E "^torch==" "$1" | head -1
}

build_sft_env() {
    local venv=".venv-sft"
    echo "[setup_env] creating ${venv} (Python ${PYTHON_VERSION}, SFT stack via lmms-engine)"
    if [[ ! -f paravt/sft/lmms-engine/pyproject.toml ]]; then
        echo "[setup_env] paravt/sft/lmms-engine missing — re-clone the repo with --recurse-submodules"
        exit 1
    fi
    local torch_pin
    torch_pin="$(_torch_pin_from_lock requirements/sft.lock)"
    if [[ -z "${torch_pin}" ]]; then
        echo "[setup_env] could not find torch pin in requirements/sft.lock"; exit 1
    fi
    uv venv "${venv}" --python "${PYTHON_VERSION}"
    # shellcheck disable=SC1091
    source "${venv}/bin/activate"
    # Two-pass install. Pass 1: install the **exact** torch version the lock
    # pins, plus the build toolchain. Pass 2: lock-install with
    # --no-build-isolation so flash-attn / deepspeed / liger-kernel /
    # mamba-ssm / causal-conv1d (none of which declare torch in
    # build-system.requires) build against the venv's torch.
    # The torch version MUST match between the two passes — otherwise pass 2
    # downgrades torch but flash-attn's pre-built .so was compiled for the
    # pass-1 torch ABI, and import fails with `undefined symbol _ZNK3c10...`.
    echo "[setup_env] using torch pin from sft.lock: ${torch_pin}"
    uv pip install "${UV_INDEX_FLAGS[@]}" "${torch_pin}" setuptools wheel packaging ninja
    uv pip install "${UV_INDEX_FLAGS[@]}" --no-build-isolation -r requirements/sft.lock
    uv pip install -e ./paravt/sft/lmms-engine
    deactivate
    echo "[setup_env] ${venv} ready."
}

build_rl_env() {
    local venv=".venv-rl"
    echo "[setup_env] creating ${venv} (Python ${PYTHON_VERSION}, RL stack)"
    local torch_pin
    torch_pin="$(_torch_pin_from_lock requirements/rl.lock)"
    if [[ -z "${torch_pin}" ]]; then
        echo "[setup_env] could not find torch pin in requirements/rl.lock"; exit 1
    fi
    uv venv "${venv}" --python "${PYTHON_VERSION}"
    # shellcheck disable=SC1091
    source "${venv}/bin/activate"
    # Same two-pass pattern as build_sft_env. See the comment there for why
    # the torch pin must match between pass 1 and pass 2.
    echo "[setup_env] using torch pin from rl.lock: ${torch_pin}"
    uv pip install "${UV_INDEX_FLAGS[@]}" "${torch_pin}" setuptools wheel packaging ninja
    uv pip install "${UV_INDEX_FLAGS[@]}" --no-build-isolation -r requirements/rl.lock
    uv pip install -e .
    uv pip install -e ./paravt/rl/areal
    # Post-install: force-upgrade cuDNN to 9.16.0.29.
    #
    # PyTorch's official torch==2.9.1+cu126 wheel pins nvidia-cudnn-cu12 to
    # 9.10.2.21 as a strict dependency, so requirements/rl.lock can only
    # pin 9.10.2.21 to keep uv's resolver happy. But cuDNN < 9.15 has a
    # known nn.Conv3d perf regression that hangs SGLang's multimodal
    # rollout for tens of minutes (see README). The fix has to be a
    # post-install override that pip/uv won't try to re-resolve. The
    # standalone 9.16 wheel is ABI-compatible with the 9.10 wheel that
    # torch was linked against, so the import path keeps working but the
    # actual runtime kernels come from 9.16.
    echo "[setup_env] post-install: force-upgrading nvidia-cudnn-cu12 to 9.16.0.29"
    uv pip install "${UV_INDEX_FLAGS[@]}" --reinstall-package nvidia-cudnn-cu12 \
        "nvidia-cudnn-cu12==9.16.0.29"
    deactivate
    echo "[setup_env] ${venv} ready."
}

build_eval_env() {
    local venv=".venv-eval"
    echo "[setup_env] creating ${venv} (Python ${PYTHON_VERSION}, eval stack)"
    # The eval driver (paravt/eval/driver.py) is self-contained: it
    # talks to vLLM directly + loads benchmarks via the `datasets` library
    # (lmms-lab/Video-MME, longvideobench, lmms-lab/charades_sta, ...).
    # The full dependency set lives in requirements/eval.lock (vllm
    # 0.17.x, torch 2.10.x, transformers, qwen-vl-utils, datasets, ...).
    # lmms-eval is also vendored at paravt/eval/lmms-eval/ for users who
    # want to compare against vanilla lmms-eval baselines under their
    # default harness; we install it editable so its tasks are available
    # in the eval venv.
    if [[ ! -f paravt/eval/lmms-eval/pyproject.toml ]]; then
        echo "[setup_env] paravt/eval/lmms-eval missing — re-clone the repo with --recurse-submodules"
        exit 1
    fi
    uv venv "${venv}" --python "${PYTHON_VERSION}"
    # shellcheck disable=SC1091
    source "${venv}/bin/activate"
    uv pip install "${UV_INDEX_FLAGS[@]}" -r requirements/eval.lock
    uv pip install -e .
    uv pip install -e ./paravt/eval/lmms-eval
    deactivate
    echo "[setup_env] ${venv} ready."
}

case "${WHICH}" in
    sft)  build_sft_env ;;
    rl)   build_rl_env ;;
    eval) build_eval_env ;;
    all)  build_sft_env; build_rl_env; build_eval_env ;;
    *)    echo "Usage: $0 [sft|rl|eval|all]"; exit 1 ;;
esac

echo "[setup_env] done."
