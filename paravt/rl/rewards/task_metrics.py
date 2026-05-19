"""Task-specific accuracy reward terms.

Each function takes ``(prediction, ground_truth)`` strings and returns a
scalar reward in ``[0, 1]``. They are the dispatch targets that
:func:`paravt.rewards.compose.compose_reward` selects between based on
``question_type`` (or the ``REWARD_MODE`` env var fallback).

Add a new task here by writing one function and wiring it up in
``compose._compute_acc_reward``.
"""

from __future__ import annotations

import re

from paravt.rl.rewards.utils import normalize_text


def compute_f1(prediction: str, ground_truth: str) -> float:
    """Token-level F1 (SQuAD-style) over normalized strings."""
    pred_tokens = normalize_text(prediction).split()
    gt_tokens = normalize_text(ground_truth).split()
    if not pred_tokens or not gt_tokens:
        return 0.0
    common = set(pred_tokens) & set(gt_tokens)
    if not common:
        return 0.0
    precision = len(common) / len(pred_tokens)
    recall = len(common) / len(gt_tokens)
    return 2 * precision * recall / (precision + recall)


def compute_temporal_iou(prediction: str, ground_truth: str) -> float:
    """Temporal IoU between predicted and ground-truth ``[start, end]`` ranges.

    Accepts ``[s, e]``, ``s, e``, or surrounding text — uses the first pair
    of decimals. Returns 0 on parse failure or zero-area intervals.
    """

    def parse_range(text: str) -> tuple[float, float] | None:
        match = re.search(r"\[?\s*([\d.]+)\s*,\s*([\d.]+)\s*\]?", text)
        if not match:
            return None
        return float(match.group(1)), float(match.group(2))

    pred = parse_range(prediction)
    gt = parse_range(ground_truth)
    if pred is None or gt is None:
        return 0.0
    pred_start, pred_end = pred
    gt_start, gt_end = gt
    if pred_end <= pred_start or gt_end <= gt_start:
        return 0.0
    inter = max(0.0, min(pred_end, gt_end) - max(pred_start, gt_start))
    union = (pred_end - pred_start) + (gt_end - gt_start) - inter
    return inter / union if union > 0 else 0.0


_MCQ_PATTERNS = [
    re.compile(r"^\(?([A-D])\)?$"),
    re.compile(r"[Tt]he answer is\s*\(?([A-D])\)?"),
    re.compile(r"[Oo]ption\s*\(?([A-D])\)?"),
    re.compile(r"^([A-D])[.):]"),
    re.compile(r"\b([A-D])\b"),
]


def compute_mcq_acc(prediction: str, ground_truth: str) -> float:
    """Multiple-choice accuracy against ground truth ``A`` / ``B`` / ``C`` / ``D``.

    Tolerates wrappings like ``(A)``, ``Option A``, ``The answer is A``.
    Returns 1.0 on match, 0.0 otherwise.
    """
    gt = ground_truth.strip().upper()
    if len(gt) != 1 or gt not in "ABCD":
        return 0.0
    pred = prediction.strip()
    if pred.upper() == gt:
        return 1.0
    for pattern in _MCQ_PATTERNS:
        match = pattern.search(pred)
        if match and match.group(1).upper() == gt:
            return 1.0
    return 0.0
