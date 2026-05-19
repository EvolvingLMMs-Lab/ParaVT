# lmms-eval pin

ParaVT vendors a copy of [`EvolvingLMMs-Lab/lmms-eval`](https://github.com/EvolvingLMMs-Lab/lmms-eval) at the commit below under `paravt/eval/lmms-eval/`. We currently ship the vendor unmodified — `paravt/eval/driver.py` is the eval driver that produces the headline numbers, and lmms-eval is bundled alongside it for users who want to run vanilla lmms-eval baselines under the upstream harness in the same eval venv.

If you upgrade or fork lmms-eval and need to maintain a diff against this pin, drop the patches into this directory next to this README following the same naming convention used under `patches/{areal,lmms-engine}/` (`NNNN-short-slug.patch`).

## Pin

| Repo | Commit | Date |
|---|---|---|
| `EvolvingLMMs-Lab/lmms-eval` | `d6cc2b567db7aeacbbf1af11b6595676e784d1d3` | 2026-05-02 |

## Patches

_None at the current pin._

## Release-side trim

The vendored copy at `paravt/eval/lmms-eval/` keeps the upstream Python package layout intact (`lmms_eval/`, `pyproject.toml`, `setup.py`, `LICENSE`, `README.md`, `configs/`) and drops local-dev artifacts (CI/git tooling, test suite, docs/examples sources, community-management files). The full upstream tree is preserved on the `paravt-full` branch and at https://github.com/EvolvingLMMs-Lab/lmms-eval.
