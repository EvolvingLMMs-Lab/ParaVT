"""ParaVT YAML config -> env-var bridge.

PARA-GRPO method knobs are documented as YAML fields under the ``paravt:``
block of an AReaL recipe. The training code reads them via
``os.environ`` for backwards compatibility with the original launch-script
flow. This module performs the YAML -> env translation once at startup,
so users can edit ``paravt/rl/configs/paragrpo_8b.yaml`` and never touch
shell scripts.

Usage::

    from paravt.rl.config import apply_paravt_config
    apply_paravt_config("paravt/rl/configs/paragrpo_8b.yaml")
    # then proceed with the regular AReaL training entry point
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf


@dataclass
class ConstrainedConfig:
    """Constrained-generation gates inside the Exploration Anchoring component.

    ``think_prefix`` fixes the first tokens of every response to ``<think>\\n``
    so rollouts cannot blind-call a tool or jump straight to ``<answer>``;
    ``answer_suffix`` adds an ``<answer>``-tag component to the format reward
    so closure of the answer block still earns credit when the rest of the
    format is imperfect. Together they leave reasoning and tool-calling
    strategies free to vary while keeping the format parser non-trivial to
    game.
    """

    think_prefix: bool = True
    answer_suffix: bool = True


@dataclass
class GatingConfig:
    """nFrames Gating — the ratio-gating component of PARA-GRPO.

    Randomly samples the overview frame budget K ~ Uniform({4, 8, 16, 32, 64})
    per prompt and shares it across all rollouts in the same GRPO group.
    Reduced budgets hide part of the evidence so that calling
    ``crop_video`` can earn higher reward than skipping it; the full
    64-frame budget keeps direct answering possible. Restores a
    call-vs-skip advantage on the gated subset rather than forcing tool
    calls on every prompt.
    """

    nframes_gating: bool = True


@dataclass
class AnchoringConfig:
    """Selective Anchoring — the parseability-anchoring component of PARA-GRPO.

    ``weight`` is the outer coefficient ``λ_anchor`` in
    ``R_fmt = R_base_fmt + λ_anchor · R_anchor``. The selective-anchor
    reward ``R_anchor`` rewards parseable format only at the structural
    tokens most prone to collapse: ``+0.4`` for ``</think>`` correctly
    closed, ``+0.3`` for the full ``<think> → </think> → <answer>`` flow,
    and ``-0.3`` if ``<think>`` is opened but never closed. Setting
    ``weight=0.0`` disables the anchor entirely (vanilla baseline).
    """

    weight: float = 0.5


@dataclass
class RewardConfig:
    """Outer-task reward configuration."""

    mode: str = "f1"  # one of: f1, exact, llm
    format_weight: float = 1.0


@dataclass
class ParaVTConfig:
    """The ``paravt:`` YAML block. See ``paravt/rl/configs/paragrpo_8b.yaml``."""

    constrained: ConstrainedConfig = field(default_factory=ConstrainedConfig)
    gating: GatingConfig = field(default_factory=GatingConfig)
    anchoring: AnchoringConfig = field(default_factory=AnchoringConfig)
    reward: RewardConfig = field(default_factory=RewardConfig)


def _bool_to_env(value: bool) -> str:
    return "1" if value else "0"


def paravt_block_to_env(cfg: ParaVTConfig | DictConfig) -> dict[str, str]:
    """Translate a ParaVT YAML block into env-var assignments.

    Returns a dict that can be merged into ``os.environ``. Idempotent.
    """
    if isinstance(cfg, ParaVTConfig):
        cfg = OmegaConf.structured(cfg)

    return {
        "THINK_PREFIX": _bool_to_env(bool(cfg.constrained.think_prefix)),
        "ANSWER_SUFFIX": _bool_to_env(bool(cfg.constrained.answer_suffix)),
        "NFRAMES_GATING": _bool_to_env(bool(cfg.gating.nframes_gating)),
        "ANCHOR_WEIGHT": f"{float(cfg.anchoring.weight)}",
        "REWARD_MODE": str(cfg.reward.mode),
        "FORMAT_WEIGHT": f"{float(cfg.reward.format_weight)}",
    }


def apply_paravt_config(yaml_path: str | os.PathLike) -> dict[str, str]:
    """Load a recipe YAML, extract its ``paravt:`` block, and inject the
    matching env vars into ``os.environ``.

    Returns the dict of env-var assignments (for logging). Missing
    ``paravt:`` block is treated as defaults.
    """
    path = Path(yaml_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Recipe not found: {path}")

    full = OmegaConf.load(path)
    paravt_block = full.get("paravt", None)
    if paravt_block is None:
        struct = OmegaConf.structured(ParaVTConfig())
    else:
        struct = OmegaConf.merge(OmegaConf.structured(ParaVTConfig()), paravt_block)

    env = paravt_block_to_env(struct)
    os.environ.update(env)
    return env
