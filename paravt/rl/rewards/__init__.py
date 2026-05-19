"""ParaVT reward functions.

Public surface
--------------

The single user-facing entry point is :func:`compose_reward`, which is the
:py:type:`reward_fn` argument expected by ``HierarchicalAgentWorkflow``.

Internal modules:

* :mod:`paravt.rl.rewards.utils` — string-extraction and tag-parsing helpers
  shared across reward terms.
* :mod:`paravt.rl.rewards.task_metrics` — task-specific accuracy: token F1,
  temporal IoU, MCQ option matching.
* :mod:`paravt.rl.rewards.format` — constrained-format reward that scores
  ``<think>``/``<answer>`` discipline, gated by the ``constrained`` and
  ``anchoring`` levers in the recipe.
* :mod:`paravt.rl.rewards.tool` — tool-call reward based on ``<tool_call>``
  parseability and execution evidence.
* :mod:`paravt.rl.rewards.llm_judge` — optional LLM-as-judge accuracy.
* :mod:`paravt.rl.rewards.compose` — combines the above into the final scalar
  reward consumed by the trainer.

Adding a new task accuracy metric is a one-file change: drop a function in
``task_metrics.py`` (or a new module) and register it in
``compose._compute_acc_reward`` next to ``mcq``/``tvg``.
"""

from paravt.rl.rewards.compose import compose_reward
from paravt.rl.rewards.format import compute_format_reward
from paravt.rl.rewards.llm_judge import APIUsageTracker, llm_judge_reward
from paravt.rl.rewards.task_metrics import (
    compute_f1,
    compute_mcq_acc,
    compute_temporal_iou,
)
from paravt.rl.rewards.tool import compute_tool_reward
from paravt.rl.rewards.utils import (
    extract_answer,
    has_parseable_tool_call,
    has_substantive_think,
    normalize_text,
    think_before_tool,
)

__all__ = [
    # public reward dispatcher
    "compose_reward",
    # individual reward terms (exposed for tests + custom recipes)
    "compute_format_reward",
    "compute_tool_reward",
    "compute_f1",
    "compute_mcq_acc",
    "compute_temporal_iou",
    "llm_judge_reward",
    # helpers
    "extract_answer",
    "normalize_text",
    "has_parseable_tool_call",
    "has_substantive_think",
    "think_before_tool",
    # observability
    "APIUsageTracker",
]
