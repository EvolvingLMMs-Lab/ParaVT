"""Subagent Manager for Hierarchical Agents"""

from typing import Any

import torch
from areal.utils import logging

from paravt.rl.subagents.base import SUBAGENT_REGISTRY, SubagentToolBase, ToolCallStatus
from paravt.rl.utils.tool_parser import ExtractedToolCallInformation, Hermes2ProToolParser

logger = logging.getLogger("SubagentManager")


class SubagentManager:
    def __init__(
        self,
        enable_subagents: str,
        subagents_kwargs: dict[str, Any],
    ):
        """Initialize subagent manager

        Args:
            enable_subagents: Semicolon-separated list of subagent names (e.g., "crop_video;analyzer")
            subagents_kwargs: Nested dict mapping subagent names to init kwargs
        """
        self.enable_subagents = enable_subagents
        self.subagents_kwargs = subagents_kwargs
        self.subagents: dict[str, SubagentToolBase] = {}

        self._initialize_subagents()

    def _initialize_subagents(self):
        """Initialize enabled subagents from registry"""
        for name in self.enable_subagents.split(";"):
            name = name.strip()
            if not name:
                continue

            if name not in SUBAGENT_REGISTRY:
                logger.warning(f"Subagent {name} not found in registry, skipping")
                continue

            tool_class = SUBAGENT_REGISTRY[name]
            kwargs = self.subagents_kwargs.get(name, {})
            subagent = tool_class(**kwargs)
            self.subagents[name] = subagent
            logger.info(f"Initialized subagent: {name}")

        self.engine = None
        self.gconfig = None
        self.tokenizer = None

    def set_execution_context(self, engine, gconfig, tokenizer, processor=None):
        """Set shared execution context for all subagents"""
        self.engine = engine
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.processor = processor
        self.tool_parser = Hermes2ProToolParser(tokenizer)

        for subagent in self.subagents.values():
            subagent.set_execution_context(engine, gconfig, tokenizer, processor)

    def parse_mcp_tool_call(self, text: str) -> ExtractedToolCallInformation:
        """Parse MCP format function call from text

        MCP format:
            <tool_name>
            {"name": "function-name", "arguments": {...}}
            </tool_name>

        Returns:
            ExtractedToolCallInformation
        """
        assert self.tool_parser is not None, "Tool parser not initialized"
        return self.tool_parser.extract_tool_calls(text, None)

    def get_tool_schema(self) -> list[dict[str, Any]]:
        """Generate OpenAI-style tool schema for all registered subagents"""
        schemas = []
        for _name, subagent in self.subagents.items():
            desc = subagent.get_description()
            schemas.append(
                {
                    "type": "function",
                    "function": {
                        "name": desc.name,
                        "description": desc.description,
                        "parameters": desc.parameters,
                    },
                }
            )
        return schemas

    async def execute_subagent(
        self, tool_name: str, parameters: dict[str, Any], data: dict[str, Any]
    ) -> dict[str, torch.Tensor] | None:
        """Execute subagent and return trajectory with reward

        Args:
            tool_name: Name of the subagent tool to execute
            parameters: Function call arguments
            data: Additional context data

        Returns:
            Complete trajectory dict with reward, or None on failure
        """
        if tool_name not in self.subagents:
            logger.warning(f"Subagent not found: {tool_name}")
            return None

        subagent = self.subagents[tool_name]
        trajectory, status = await subagent.execute(parameters, data)

        # All subagent trajectories share the main agent's reward signal —
        # the main reward is broadcast to all sub-trajectories during advantage
        # computation in HierarchicalPPOActor.prepare_data. There is therefore
        # no per-subagent reward function on this code path.
        if status == ToolCallStatus.SUCCESS:
            return trajectory
        return None
