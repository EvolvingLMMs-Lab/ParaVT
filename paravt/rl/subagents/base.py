"""Subagent Tool Base Classes for Hierarchical Agents"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum
from typing import Any

import torch
from areal.api.io_struct import ModelRequest
from areal.utils import logging

logger = logging.getLogger("SubagentTool")


SUBAGENT_REGISTRY: dict[str, type] = {}


def register_subagent(name: str):
    """Decorator to register a subagent class

    Usage:
        @register_subagent("crop_video")
        class CroppedVideoSubagent(SubagentToolBase):
            ...
    """

    def decorator(cls: type):
        SUBAGENT_REGISTRY[name] = cls
        logger.info(f"Registered subagent: {name} -> {cls.__name__}")
        return cls

    return decorator


class ToolCallStatus(Enum):
    SUCCESS = "success"
    ERROR = "error"
    NOT_FOUND = "not_found"


@dataclass
class ToolDescription:
    name: str
    description: str
    parameters: dict[str, str]
    required: list[str]


class SubagentToolBase(ABC):
    def __init__(
        self,
        timeout: int = 30,
        debug_mode: bool = False,
    ):
        self.timeout = timeout
        self.debug_mode = debug_mode
        self.engine = None
        self.gconfig = None
        self.tokenizer = None

    def set_execution_context(self, engine, gconfig, tokenizer, processor=None):
        self.engine = engine
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        self.processor = processor

    @abstractmethod
    def get_tool_name(self) -> str:
        pass

    @abstractmethod
    def get_description(self) -> ToolDescription:
        pass

    @abstractmethod
    def build_prompt(self, task: str, **kwargs) -> str:
        pass

    async def execute(
        self, parameters: dict[str, Any], data: dict[str, Any] | None = None
    ) -> tuple[dict[str, torch.Tensor], ToolCallStatus]:
        if not self.engine:
            raise RuntimeError(
                "Subagent engine not set. Call set_execution_context first."
            )

        task = parameters.get("task", "")
        prompt = self.build_prompt(task, **(data or {}))

        input_ids = self.tokenizer.encode(prompt, add_special_tokens=False)
        req = ModelRequest(
            input_ids=input_ids, gconfig=self.gconfig, tokenizer=self.tokenizer
        )
        resp = await self.engine.agenerate(req)

        trajectory = {
            "input_ids": torch.tensor(
                input_ids + resp.output_tokens, dtype=torch.int32
            ).unsqueeze(0),
            "logprobs": torch.tensor(
                [0.0] * len(input_ids) + resp.output_logprobs, dtype=torch.float32
            ).unsqueeze(0),
            "loss_mask": torch.tensor(
                [0] * len(input_ids) + [1] * len(resp.output_tokens), dtype=torch.int32
            ).unsqueeze(0),
            "versions": torch.tensor(
                [-1] * len(input_ids) + resp.output_versions, dtype=torch.int32
            ).unsqueeze(0),
            "attention_mask": torch.ones(
                len(input_ids) + len(resp.output_tokens), dtype=torch.bool
            ).unsqueeze(0),
            "subagent_type": self.get_tool_name(),
        }

        if self.debug_mode:
            logger.debug(f"Subagent {self.get_tool_name()} execution complete:")
            logger.debug(f"  Prompt: {prompt[:200]}...")
            logger.debug(f"  Output: {resp.output_text[:200]}...")

        return trajectory, ToolCallStatus.SUCCESS
