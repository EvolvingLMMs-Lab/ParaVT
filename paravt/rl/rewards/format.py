"""Constrained-format reward.

Scores how disciplined the completion is about the
``<think>...</think><answer>...</answer>`` template that the parser depends
on. Three knobs (all set from the recipe's ``paravt:`` block, surfaced as
env vars at launch time):

* ``ANSWER_SUFFIX``: when on, ``<answer>`` / ``</answer>`` carry more weight
  than ``<think>``, mirroring the constrained decoding mode.
* ``ANCHOR_WEIGHT``: scales an extra bonus for ``</think>`` closure plus a
  full think→answer flow. Paid by a penalty for unclosed ``<think>``. This
  is the "selective anchoring" lever.
* implicit: ``think→tool`` ordering bonus and balanced-tag bonus apply
  unconditionally.
"""

from __future__ import annotations

import os
import re

from paravt.rl.rewards.utils import (
    has_parseable_tool_call,
    has_substantive_think,
    think_before_tool,
)

_ANSWER_BODY_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)


def _has_nonempty_answer(completion: str) -> bool:
    """True iff at least one <answer>...</answer> pair brackets non-whitespace content."""
    for body in _ANSWER_BODY_RE.findall(completion):
        if body.strip():
            return True
    return False


def _bool_env(key: str) -> bool:
    return os.environ.get(key, "") == "1"


def _float_env(key: str, default: float = 0.0) -> float:
    raw = os.environ.get(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


_BALANCED_TAGS = (
    ("<think>", "</think>"),
    ("<tool_call>", "</tool_call>"),
    ("<tool_code>", "</tool_code>"),
    ("<answer>", "</answer>"),
)


def compute_format_reward(completion: str) -> float:
    """Continuous format reward in ``[0, 1]``.

    The exact weightings replicate the production trial that produced the
    paper's PARA-GRPO numbers; tweaking them belongs in a config-time
    experiment, not in code. See ``paravt/rl/configs/paragrpo_8b.yaml``.
    """
    answer_suffix = _bool_env("ANSWER_SUFFIX")
    anchor_weight = _float_env("ANCHOR_WEIGHT", 0.0)

    has_think = has_substantive_think(completion)
    has_tool = has_parseable_tool_call(completion)
    has_answer_body = _has_nonempty_answer(completion)
    # The <answer>...</answer> credit is gated on the brackets containing
    # non-whitespace content. Without this gate, a model could learn to emit
    # <think></think><answer></answer> and harvest most of the format reward
    # without producing useful structure (the empty-tag exploit that PARA-GRPO
    # is meant to suppress in the first place).
    score = 0.0

    if answer_suffix:
        # Constrained-decoding mode: answer tags carry the bulk of credit.
        if has_think:
            score += 0.2
        if has_answer_body:
            if "<answer>" in completion:
                score += 0.3
            if "</answer>" in completion:
                score += 0.2
    else:
        # Default mode: think dominates.
        if has_think:
            score += 0.3
        if has_answer_body:
            if "<answer>" in completion:
                score += 0.2
            if "</answer>" in completion:
                score += 0.1

    # Selective anchoring on the structural-token closure that the
    # post-training-format-collapse failure mode keeps dropping.
    if anchor_weight > 0:
        anchor_bonus = 0.0
        if "</think>" in completion and has_think:
            anchor_bonus += 0.4
        if (
            "</think>" in completion
            and has_think
            and has_answer_body
            and "</answer>" in completion
        ):
            anchor_bonus += 0.3
        if "<think>" in completion and "</think>" not in completion:
            anchor_bonus -= 0.3
        score += anchor_weight * anchor_bonus

    if has_think and has_tool and think_before_tool(completion):
        score += 0.3

    if score > 0 and all(
        completion.count(open_) == completion.count(close) for open_, close in _BALANCED_TAGS
    ):
        score += 0.1

    return min(score, 1.0)
