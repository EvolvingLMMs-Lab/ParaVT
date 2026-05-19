# `paravt/`

The Python package + per-workload subtree. Each workload (SFT, RL,
eval) lives under its own subdirectory with the same layout: a
top-level `README.md`, a `configs/` directory of YAML recipes, the
ParaVT-specific Python code, and a vendored framework alongside.

| Subdir | Workload | Vendored framework | Reproduce script |
|---|---|---|---|
| [`sft/`](sft) | Cold-start supervised fine-tuning | [`lmms-engine/`](sft/lmms-engine) | [`scripts/run_sft.sh`](../scripts/run_sft.sh) |
| [`rl/`](rl) | PARA-GRPO RL training | [`areal/`](rl/areal) | [`scripts/run_rl.sh`](../scripts/run_rl.sh) |
| [`eval/`](eval) | Video-benchmark evaluation | [`lmms-eval/`](eval/lmms-eval) | [`scripts/run_eval.sh`](../scripts/run_eval.sh) |

Per-workload READMEs describe the recipe knobs, the smoke-test
configuration, and the pieces that may need adjusting for a different
cluster shape:

- [`sft/README.md`](sft/README.md) — SFT cold-start recipe + data manifest.
- [`rl/README.md`](rl/README.md) — PARA-GRPO components + vanilla-GRPO baseline.
- [`eval/README.md`](eval/README.md) — seven prompt modes, per-row reproduce table, locked-in protocol.

## Python module layout

```text
paravt/
├── __init__.py                       # lazy / empty (eval venv must import without RL deps)
├── sft/                              # configs + vendored lmms-engine; no Python — SFT goes
│                                     #   through scripts/run_sft.sh -> torchrun lmms-engine
├── rl/
│   ├── train.py                      # RL training entry — invoked by `areal.launcher.local`
│   ├── trainer.py                    # HierarchicalPPOTrainer
│   ├── workflow.py                   # HierarchicalAgentWorkflow + GRPO config
│   ├── actor.py                      # HierarchicalPPOActor
│   ├── config.py                     # `paravt:` YAML block → env-var bridge
│   ├── rewards/                      # one function per file (format / tool / task / llm-judge)
│   ├── subagents/                    # crop_video subagent + registry
│   ├── utils/                        # tool-call parser
│   ├── configs/                      # RL YAML recipes
│   └── areal/                        # vendored framework (editable-installed by setup_env.sh)
└── eval/
    ├── driver.py                     # `python -m paravt.eval.driver` — the eval CLI
    ├── utils.py                      # shared infra (loaders, vLLM lifecycle, scoring)
    ├── configs/                      # eval YAML recipes
    ├── scripts/                      # per-row reproduce shell scripts
    └── lmms-eval/                    # vendored framework (editable-installed by setup_env.sh)
```

The flat `paravt/rl/{actor,config,train,trainer,workflow}.py` files
are ParaVT-specific glue that subclasses or composes AReaL APIs (e.g.
`HierarchicalPPOActor` extends AReaL's actor, `HierarchicalAgentGRPOConfig`
extends AReaL's `GRPOConfig`). The vendored AReaL bundle at
`paravt/rl/areal/` is unmodified upstream.
