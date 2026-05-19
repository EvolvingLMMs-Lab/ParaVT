# `patches/`

Diffs we maintain on top of the three vendored frameworks. Each
subdirectory documents the upstream commit pin and lists the patches
applied; the patches themselves are saved as `NNNN-short-slug.patch`
files alongside the README so a downstream fork can reapply them
against a fresh upstream clone.

| Subdir | Workload | Vendor location | Upstream | Pinned commit | Patches |
|---|---|---|---|---|---|
| [`lmms-engine/`](lmms-engine) | SFT  | [`paravt/sft/lmms-engine/`](../paravt/sft/lmms-engine) | [EvolvingLMMs-Lab/lmms-engine](https://github.com/EvolvingLMMs-Lab/lmms-engine) | `e445004` | 3 |
| [`areal/`](areal)             | RL   | [`paravt/rl/areal/`](../paravt/rl/areal) | [inclusionAI/AReaL](https://github.com/inclusionAI/AReaL) | `d0bf079` | 1 |
| [`lmms-eval/`](lmms-eval)     | Eval | [`paravt/eval/lmms-eval/`](../paravt/eval/lmms-eval) | [EvolvingLMMs-Lab/lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) | `d6cc2b5` | 0 (clean vendor) |

The vendored copies under `paravt/{sft,rl,eval}/` already have these
patches applied; you do not need to reapply them. The diffs are
preserved here so a maintainer who upgrades the upstream pin can see
exactly what we changed and decide whether each patch still applies.

The release branch trims local-dev artifacts (`.github/`, `cicd/`,
`test/`, `CLAUDE.md`, `CITATION.cff`, `CODE_OF_CONDUCT.md`, etc.) from
the vendored copies — see the per-vendor README for what was kept vs
trimmed. The `paravt-full` sibling branch keeps the upstream layout
verbatim.
