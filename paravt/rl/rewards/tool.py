"""Tool-call reward.

Graduated four-level signal that distinguishes:

1. No tool attempt — zero reward.
2. Parseable tool call (model emitted a real call, not just the tags).
3. Tool was actually executed (``<tool_response>`` is in the completion).
4. Tool-assisted correct answer (combined with ``acc_reward > 0``).

This shaping prevents the empty-``<tool_code></tool_code>`` exploit and
heavily rewards trajectories where the tool actually contributed to a
correct outcome.
"""

from __future__ import annotations

from paravt.rl.rewards.utils import has_parseable_tool_call


def compute_tool_reward(completion: str, acc_reward: float) -> float:
    """Return tool-call reward in ``[0, 1]``."""
    if not has_parseable_tool_call(completion):
        return 0.0
    reward = 0.1
    if "<tool_response>" in completion:
        reward = 0.3
    if acc_reward > 0:
        reward += 0.5
    return min(reward, 1.0)
