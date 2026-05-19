"""Subagent tools for ParaVT.

Subagents are small specialist tools that the main agent can call mid-rollout.
Each subagent shares the main model's weights but has an independent reward
signal. New subagents register themselves with ``SUBAGENT_REGISTRY`` via
:class:`SubagentToolBase`.
"""

from paravt.rl.subagents.base import (
    SUBAGENT_REGISTRY,
    SubagentToolBase,
    ToolCallStatus,
    ToolDescription,
)
from paravt.rl.subagents.crop_video.cropped_video import CroppedVideoSubagent

__all__ = [
    "SUBAGENT_REGISTRY",
    "SubagentToolBase",
    "ToolCallStatus",
    "ToolDescription",
    "CroppedVideoSubagent",
]
