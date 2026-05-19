"""Combine the per-aspect reward terms into the scalar reward AReaL consumes.

This module owns the only piece of reward logic that reads PARA-GRPO env
vars (``REWARD_MODE``, ``FORMAT_WEIGHT``). Adding a new reward term
means: write the term in its own module, then either include it in
:func:`compose_reward` (if it's universal) or gate it behind a new
recipe knob.
"""

from __future__ import annotations

import os
from typing import Any

from paravt.rl.rewards.format import compute_format_reward
from paravt.rl.rewards.llm_judge import llm_judge_reward
from paravt.rl.rewards.task_metrics import compute_f1, compute_mcq_acc, compute_temporal_iou
from paravt.rl.rewards.tool import compute_tool_reward
from paravt.rl.rewards.utils import extract_answer, normalize_text


def _float_env(key: str, default: float) -> float:
    raw = os.environ.get(key)
    if not raw:
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _compute_acc_reward(
    sol: str,
    answer: str,
    question_type: str,
    prompt: str,
) -> float:
    """Dispatch task-accuracy computation based on question type / reward mode."""
    if not sol or not answer:
        return 0.0

    # Fast path: exact normalized match works for any question type.
    if normalize_text(sol) == normalize_text(answer):
        return 1.0

    if question_type == "mcq":
        return compute_mcq_acc(sol, answer)
    if question_type == "tvg":
        return compute_temporal_iou(sol, answer)

    mode = os.environ.get("REWARD_MODE", "f1")
    if mode == "exact":
        return 0.0
    if mode == "llm":
        question = ""
        if "<|vision_end|>" in prompt:
            question = prompt.split("<|vision_end|>")[-1].split("<|im_end|>")[0].strip()
        return llm_judge_reward(question, sol, answer)
    return compute_f1(sol, answer)


def compose_reward(
    prompt: str,
    completions: str,
    prompt_ids: list[int],
    completion_ids: list[int],
    answer: str,
    **kwargs: Any,
) -> tuple[float, dict[str, float]]:
    """Combined reward = task accuracy + format + tool.

    Signature matches AReaL's ``reward_fn`` contract.
    Returns ``(scalar_reward, info_dict)``; ``info_dict`` is logged per step.
    """
    question_type = kwargs.get("question_type", "oe")

    # Short-circuit a known degenerate completion shape (long <|im_start|>
    # spam with no real content) before scoring it.
    if completions.count("<|im_start|>") >= 5 and len(completions) < 300:
        return 0.0, {
            "acc_reward": 0.0,
            "format_reward": 0.0,
            "tool_reward": 0.0,
        }

    sol = extract_answer(completions)
    acc_reward = _compute_acc_reward(sol, answer, question_type, prompt)
    format_reward = compute_format_reward(completions)
    tool_reward = compute_tool_reward(completions, acc_reward)

    format_weight = _float_env("FORMAT_WEIGHT", 0.5)
    final = acc_reward + format_weight * format_reward + tool_reward

    return final, {
        "acc_reward": acc_reward,
        "format_reward": format_reward,
        "tool_reward": tool_reward,
    }
