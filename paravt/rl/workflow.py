"""Hierarchical Agent Workflow with Tool-Calling Subagents"""

import asyncio
import json
import os
import random
import uuid
from collections.abc import Callable
from copy import deepcopy
from typing import Any, cast

import numpy as np
import torch
from areal.api.cli_args import GRPOConfig, dataclass, field
from areal.api.engine_api import InferenceEngine
from areal.api.io_struct import ModelRequest
from areal.api.reward_api import AsyncRewardWrapper
from areal.api.workflow_api import RolloutWorkflow
from areal.utils import logging, stats_tracker
from areal.utils.image import image2base64
from PIL import Image
from qwen_vl_utils import fetch_video
from transformers import AutoProcessor

from paravt.rl.subagents.manager import SubagentManager

# PARA-GRPO recipe knobs surfaced as env vars by paravt.rl.config.
# Read at call time (not module import) so Hydra CLI overrides like
# paravt.gating.nframes_gating=false reach the workflow even though
# scripts/run_rl.sh exports the YAML defaults via apply_paravt_config()
# before train.py runs Hydra. train.py:main() re-exports os.environ
# from the merged config, and the helpers below pick up the new value
# every time they are called.
NFRAMES_CHOICES = [4, 8, 16, 32, 64]


def _env_flag(key: str) -> bool:
    return os.environ.get(key, "") == "1"


def _nframes_gating() -> bool:
    return _env_flag("NFRAMES_GATING")


def _think_prefix() -> bool:
    return _env_flag("THINK_PREFIX")


def _answer_suffix() -> bool:
    return _env_flag("ANSWER_SUFFIX")


def _resolve_nframes(
    cache_key: tuple | None, gating_on: bool, default: int
) -> int:
    """Return the per-prompt frame budget.

    When NFRAMES_GATING is on we want every n_samples rollout sharing the same
    prompt to see the same frame count (the per-prompt-not-per-sample invariant
    the paper relies on). The grouped rollout workflow runs n_samples
    coroutines in parallel under asyncio.gather sharing one workflow instance,
    so a naive  mutated on instance state would race. Seeding
    a local Random on hash(cache_key) makes the call deterministic per prompt
    AND reproducible across runs without needing an asyncio.Lock.
    """
    if not gating_on:
        return default
    if cache_key is None:
        return random.choice(NFRAMES_CHOICES)
    return random.Random(hash(cache_key)).choice(NFRAMES_CHOICES)


logger = logging.getLogger("HierarchicalAgentWorkflow")

MAX_TOOL_RESPONSE_TOKENS = 512


@dataclass
class HierarchicalAgentConfig:
    """Configuration for hierarchical agent workflow

    This configuration controls:
    - Max number of turns for main agent
    - Subagent tool configurations
    - Prompt templates for each agent
    """

    max_turns: int = 3
    """Maximum number of turns for main agent generation"""

    enable_subagents: str = field(default="crop_video")

    subagents_kwargs: dict[str, Any] = field(default_factory=dict)

    nframes: int = 32
    """Number of frames to sample from the video"""

    fps: int | None = None
    """Frames per second to sample from the video"""

    max_pixels: int = 360 * 420
    """Maximum number of pixels in the video"""

    min_pixels: int = 28 * 28
    """Minimum number of pixels in the video"""

    max_frames: int = 128
    """Maximum number of frames in the video"""


@dataclass
class HierarchicalAgentGRPOConfig(GRPOConfig):
    hierarchical_agent: HierarchicalAgentConfig = field(
        default_factory=HierarchicalAgentConfig
    )
    # The 'paravt:' YAML block — opaque to AReaL; consumed by paravt.rl.config.
    paravt: Any = field(default_factory=dict)


class HierarchicalAgentWorkflow(RolloutWorkflow):
    """Workflow for hierarchical agent training with tool-calling subagents"""

    def __init__(
        self,
        gconfig,
        tokenizer,
        processor,
        reward_fn,
        config: HierarchicalAgentConfig,
    ):
        """Initialize hierarchical agent workflow

        Args:
            gconfig: Generation hyperparameters
            tokenizer: Tokenizer for encoding/decoding
            processor: Processor for encoding/decoding
            reward_fn: Main agent reward function
            config: Hierarchical agent configuration
        """
        super().__init__()
        self.gconfig = gconfig
        self.tokenizer = tokenizer
        if isinstance(processor, str):
            processor = AutoProcessor.from_pretrained(processor)
        self.processor = processor
        self.config = config

        self.subagent_manager = SubagentManager(
            enable_subagents=config.enable_subagents,
            subagents_kwargs=config.subagents_kwargs,
        )

        self.async_main_reward_fn = AsyncRewardWrapper(reward_fn)

        # Cache video preprocessing across n_samples rollouts.
        # Same data item spawns 8 rollouts via GroupedRolloutWorkflow;
        # video decode (~2s), processor (~5s), image2base64 (~1s) are identical
        # across rollouts, so caching avoids 7x redundant CPU work.
        self._episode_cache = {}

    @property
    def tools_system_message(self) -> str:
        """Generate system message for tool schemas"""
        tool_schema = self.subagent_manager.get_tool_schema()
        if not tool_schema:
            return ""

        tools_str = "\n".join([json.dumps(tool) for tool in tool_schema])
        return (
            "# Tools\n\n"
            "You may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n"
            "<tools>\n"
            f"{tools_str}\n"
            "</tools>\n\n"
            "For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n"
            "<tool_call>\n"
            '{"name": <function-name>, "arguments": <args-json-object>}\n'
            "</tool_call>"
        )

    def _init_trajectory(self) -> dict[str, Any]:
        """Initialize empty trajectory dict"""
        return {
            "input_ids": torch.zeros((1, 0), dtype=torch.int32),
            "logprobs": torch.zeros((1, 0), dtype=torch.float32),
            "loss_mask": torch.zeros((1, 0), dtype=torch.int32),
            "versions": torch.zeros((1, 0), dtype=torch.int32),
            "attention_mask": torch.zeros((1, 0), dtype=torch.bool),
            # "subagent_type": "main_agent",
        }

    def _add_to_trajectory(
        self,
        trajectory: dict[str, Any],
        new_tokens: list[int],
        new_logprobs: list[float],
        new_versions: list[int],
        loss_mask: list[int],
        attention_mask: list[bool],
    ) -> dict[str, Any]:
        trajectory["input_ids"] = torch.cat(
            [
                trajectory["input_ids"],
                torch.tensor(new_tokens, dtype=torch.int32).unsqueeze(0),
            ],
            dim=1,
        )
        trajectory["logprobs"] = torch.cat(
            [
                trajectory["logprobs"],
                torch.tensor(new_logprobs, dtype=torch.float32).unsqueeze(0),
            ],
            dim=1,
        )
        trajectory["loss_mask"] = torch.cat(
            [
                trajectory["loss_mask"],
                torch.tensor(loss_mask, dtype=torch.int32).unsqueeze(0),
            ],
            dim=1,
        )
        trajectory["versions"] = torch.cat(
            [
                trajectory["versions"],
                torch.tensor(new_versions, dtype=torch.int32).unsqueeze(0),
            ],
            dim=1,
        )
        trajectory["attention_mask"] = torch.cat(
            [
                trajectory["attention_mask"],
                torch.tensor(attention_mask, dtype=torch.bool).unsqueeze(0),
            ],
            dim=1,
        )
        return trajectory

    def _append_generation(
        self, trajectory: dict[str, Any], req: ModelRequest, resp
    ) -> dict[str, Any]:
        """Append generated tokens to trajectory with loss_mask=1"""
        new_tokens = resp.output_tokens
        new_logprobs = resp.output_logprobs
        new_versions = resp.output_versions
        loss_mask = [1] * len(new_tokens)
        attention_mask = [True] * len(new_tokens)

        trajectory = self._add_to_trajectory(
            trajectory,
            new_tokens,
            new_logprobs,
            new_versions,
            loss_mask,
            attention_mask,
        )

        return trajectory

    def _append_tool_output(
        self, trajectory: dict[str, Any], tool_output: str
    ) -> dict[str, Any]:
        """Append tool output to trajectory with loss_mask=0"""
        output_ids = self.tokenizer.encode(tool_output, add_special_tokens=False)
        output_len = len(output_ids)

        trajectory = self._add_to_trajectory(
            trajectory,
            output_ids,
            [0.0] * output_len,
            [-1] * output_len,
            [0] * output_len,
            [True] * output_len,
        )
        return trajectory

    def _calculate_timestamps(self, video_metadata: dict[str, Any]):
        indices = video_metadata["frames_indices"]
        if not isinstance(indices, list):
            indices = indices.tolist()
        fps = video_metadata["fps"]
        # Note this is a hardcode value for Qwen3-VL, should only be used for Qwen3-VL
        merge_size = 2
        if len(indices) % merge_size != 0:
            indices.extend(
                indices[-1] for _ in range(merge_size - len(indices) % merge_size)
            )
        timestamps = [idx / fps for idx in indices]
        # timestamps = [(timestamps[i] + timestamps[i + merge_size - 1]) / 2 for i in range(0, len(timestamps), merge_size)]
        return timestamps

    def prepare_video_contents(
        self, video_path: str, nframes_used: int
    ) -> list[dict[str, Any]]:
        images = []
        openai_video_contents = []
        hf_video_contents = []
        if "file://" in video_path:
            video_path = video_path.replace("file://", "")
        video_dict = {
            "type": "video",
            "video": f"file://{video_path}",
            "nframes": nframes_used,
            "max_pixels": self.config.max_pixels,
            "min_pixels": self.config.min_pixels,
        }
        # If fps is provided, use fps, otherwise use nframes
        if self.config.fps:
            video_dict.update(
                {"fps": self.config.fps, "max_frames": self.config.max_frames}
            )
            video_dict.pop("nframes")
        final_video, fps = fetch_video(
            video_dict, return_video_sample_fps=True, return_video_metadata=True
        )
        frames, video_metadata = final_video
        timestamps = self._calculate_timestamps(video_metadata)
        for frame, timestamp in zip(frames, timestamps, strict=False):
            image = Image.fromarray(frame.permute(1, 2, 0).numpy().astype(np.uint8))
            images.append(image)
            text_content = {
                "type": "text",
                "text": f"<{timestamp:.1f} seconds>",
            }
            openai_video_contents.append(text_content)
            hf_video_contents.append(text_content)
            openai_video_contents.append(
                {
                    "type": "image_url",
                    "image_url": {"url": ""},
                }
            )
            hf_video_contents.append({"type": "image"})
        return images, openai_video_contents, hf_video_contents

    def _init_episode_inputs(self, data: dict[str, Any]):
        """Initialize episode inputs with caching across n_samples rollouts.

        Video decode (~2s), processor (~5s), and image2base64 (~1s) produce
        identical results for the same data item. GroupedRolloutWorkflow spawns
        n_samples=8 rollouts per item; caching avoids 7x redundant CPU work.

        Returns:
            Tuple of (byte_images, processed_input, openai_messages, hf_messages,
                       input_ids, images). Mutable structures (messages, input_ids)
                       are deep-copied so each rollout gets its own instance.
        """
        video_paths = data.get("video_paths")
        cache_key = tuple(video_paths) if video_paths else None

        # Resolve per-prompt frame count once via a deterministic seeding on
        # cache_key. Race-free under asyncio.gather; reproducible across runs;
        # consumed by prepare_video_contents and the reward emit below via a
        # local variable rather than instance state.
        nframes_used = _resolve_nframes(
            cache_key, gating_on=_nframes_gating(), default=self.config.nframes
        )

        if cache_key and cache_key in self._episode_cache:
            cached = self._episode_cache[cache_key]
            return (
                cached["byte_images"],
                cached["processed_input"],
                deepcopy(cached["openai_messages"]),
                deepcopy(cached["hf_messages"]),
                list(cached["input_ids"]),
                cached["images"],
                nframes_used,
            )

        processor_callable = cast(Callable[..., dict[str, Any]], self.processor)
        images = []
        hf_messages = deepcopy(data["messages"])

        # Only inject dynamic tool schema if parquet system prompt doesn't already have one.
        # SFT-aligned parquet already includes <tools> block — avoid double injection.
        existing_system_text = ""
        if hf_messages and hf_messages[0]["role"] == "system":
            content = hf_messages[0]["content"]
            if isinstance(content, list) and content:
                existing_system_text = content[0].get("text", "") if isinstance(content[0], dict) else str(content[0])
            elif isinstance(content, str):
                existing_system_text = content

        tools_system = self.tools_system_message
        if tools_system and "<tools>" not in existing_system_text:
            if hf_messages[0]["role"] == "system":
                existing_content = hf_messages[0]["content"]
                hf_messages[0]["content"] = [
                    {
                        "type": "text",
                        "text": existing_content[0]["text"] + "\n\n" + tools_system,
                    }
                ]
            else:
                hf_messages.insert(
                    0,
                    {
                        "role": "system",
                        "content": [{"type": "text", "text": tools_system}],
                    },
                )

        openai_messages = []
        new_hf_messages = []
        video_count = 0
        for _idx, msg in enumerate(hf_messages):
            role = msg["role"]
            content = msg["content"]

            new_hf_content = []
            new_openai_content = []

            for item in content:
                content_type = item["type"]
                if content_type == "video":
                    video_path = video_paths[video_count]
                    frames, openai_video_contents, hf_video_contents = (
                        self.prepare_video_contents(video_path, nframes_used)
                    )
                    images.extend(frames)
                    new_openai_content.extend(openai_video_contents)
                    new_hf_content.extend(hf_video_contents)
                    video_count += 1
                else:
                    new_openai_content.append(item)
                    new_hf_content.append(item)
            new_hf_messages.append(
                {
                    "role": role,
                    "content": new_hf_content,
                }
            )
            openai_messages.append(
                {
                    "role": role,
                    "content": new_openai_content,
                }
            )

        hf_messages = new_hf_messages
        prompt = self.processor.apply_chat_template(
            hf_messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        if _think_prefix():
            prompt = prompt + "<think>\n"
        elif _answer_suffix():
            # If no THINK_PREFIX but ANSWER_SUFFIX, still guide format
            prompt = prompt + "<think>\n"

        processed_input = processor_callable(
            images=images,
            text=prompt,
            padding=False,
            return_tensors="pt",
        )

        input_ids: list[int] = processed_input["input_ids"].tolist()[0]
        byte_images = image2base64(images)

        # Store in cache. Bounded to 32 entries; pop the oldest insertion
        # (FIFO) when full. Concurrent overwrites by sibling rollouts in the
        # same group are fine — the value is the same data and the only
        # races are key-space, not value-space.
        if cache_key:
            if len(self._episode_cache) >= 32:
                self._episode_cache.pop(next(iter(self._episode_cache)))
            self._episode_cache[cache_key] = {
                "images": images,
                "byte_images": byte_images,
                "processed_input": processed_input,
                "openai_messages": openai_messages,
                "hf_messages": hf_messages,
                "input_ids": input_ids,
            }
            logger.info(
                f"Cached episode inputs (cache_size={len(self._episode_cache)})"
            )

        return (
            byte_images,
            processed_input,
            deepcopy(openai_messages),
            deepcopy(hf_messages),
            list(input_ids),
            images,
            nframes_used,
        )

    async def arun_episode(
        self, engine: InferenceEngine, data: dict[str, Any]
    ) -> dict[str, Any]:
        """Run one episode of hierarchical agent interaction

        Returns:
            dict[str, torch.Tensor]: Concatenated trajectories from main and subagents
        """
        self.subagent_manager.set_execution_context(
            engine, self.gconfig, self.tokenizer, self.processor
        )

        main_trajectory = self._init_trajectory()
        subagent_trajectories = []

        eot_token = self.tokenizer.convert_ids_to_tokens(self.tokenizer.eos_token_id)

        # Initialize episode inputs (cached across n_samples rollouts)
        byte_images, processed_input, openai_messages, hf_messages, input_ids, images, nframes_used = (
            self._init_episode_inputs(data)
        )

        main_trajectory["input_ids"] = torch.tensor(
            input_ids, dtype=torch.int32
        ).unsqueeze(0)
        main_trajectory["logprobs"] = torch.zeros(
            (1, len(input_ids)), dtype=torch.float32
        )
        main_trajectory["loss_mask"] = torch.zeros(
            (1, len(input_ids)), dtype=torch.int32
        )
        main_trajectory["versions"] = torch.full(
            (1, len(input_ids)), -1, dtype=torch.int32
        )
        main_trajectory["attention_mask"] = torch.ones(
            (1, len(input_ids)), dtype=torch.bool
        )
        episode_id = torch.tensor([hash(uuid.uuid4().hex)], dtype=torch.int64)

        turn = 0
        while turn <= self.config.max_turns:
            # Generate main agent response
            req = ModelRequest(
                rid=uuid.uuid4().hex,
                input_ids=input_ids,
                image_data=byte_images,
                vision_msg_vllm=[openai_messages],
                gconfig=self.gconfig.new(n_samples=1),
                tokenizer=self.tokenizer,
                processor=self.processor,
            )

            resp = await engine.agenerate(req)
            main_trajectory = self._append_generation(main_trajectory, req, resp)
            input_ids = main_trajectory["input_ids"][0].tolist()

            output_text = self.tokenizer.decode(resp.output_tokens)
            output_text_no_eot = output_text.replace(eot_token, "")
            openai_messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": output_text_no_eot}],
                }
            )
            hf_messages.append(
                {
                    "role": "assistant",
                    "content": [{"type": "text", "text": output_text_no_eot}],
                }
            )

            # Detect and execute MCP tool call
            extraced_tool_calls = self.subagent_manager.parse_mcp_tool_call(output_text)
            tools_called = extraced_tool_calls.tools_called
            tool_calls = extraced_tool_calls.tool_calls

            if tools_called:
                # Execute all tool calls in parallel with asyncio.gather
                async_tasks = []
                for tool_call in tool_calls:
                    tool_name = tool_call.function.name
                    tool_arguments = json.loads(tool_call.function.arguments)
                    async_tasks.append(
                        self.subagent_manager.execute_subagent(
                            tool_name, tool_arguments, data
                        )
                    )
                results = await asyncio.gather(*async_tasks, return_exceptions=True)

                # Process results sequentially (order matches tool_calls)
                for tool_call, result in zip(tool_calls, results, strict=False):
                    if isinstance(result, Exception):
                        logger.warning(f"Parallel tool call {tool_call.function.name} failed: {result}")
                        main_trajectory = self._append_tool_output(
                            main_trajectory,
                            f"[Error: {tool_call.function.name} failed: {result}]",
                        )
                        continue

                    subagent_trajectory = result
                    if subagent_trajectory:
                        subagent_trajectories.append(subagent_trajectory)

                        # Extract subagent completion as tool output
                        tool_output_text = self.subagent_manager._extract_completion(
                            subagent_trajectory
                        )
                        tool_output_text = tool_output_text.replace(eot_token, "")
                        # Truncate tool response to prevent excessively long outputs
                        response_ids = self.tokenizer.encode(tool_output_text, add_special_tokens=False)
                        if len(response_ids) > MAX_TOOL_RESPONSE_TOKENS:
                            response_ids = response_ids[:MAX_TOOL_RESPONSE_TOKENS]
                            tool_output_text = self.tokenizer.decode(response_ids, skip_special_tokens=True) + " [truncated]"
                        tool_output_text = (
                            f"<tool_response>\n{tool_output_text}\n</tool_response>"
                        )
                        hf_messages.append(
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": tool_output_text}],
                            }
                        )
                        openai_messages.append(
                            {
                                "role": "user",
                                "content": [{"type": "text", "text": tool_output_text}],
                            }
                        )

                        # Prepare the messages that align for current input,
                        # TODO: Assume no multimodal outputs from the tool call right now
                        curr_prompt = self.tokenizer.apply_chat_template(
                            hf_messages, tokenize=False, add_generation_prompt=True
                        )
                        if _think_prefix():
                            curr_prompt = curr_prompt + "<think>\n"
                        if (
                            _answer_suffix()
                            and turn >= self.config.max_turns - 1
                            and not _think_prefix()
                        ):
                            # Last turn, no <think> prefix: append an answer
                            # prefix to guide format. When THINK_PREFIX is on,
                            # we already appended <think> above, so skip.
                            curr_prompt = curr_prompt + "Based on my analysis, "
                        curr_processed_input = self.processor(
                            text=curr_prompt,
                            images=images,
                            padding=False,
                            return_tensors="pt",
                        )
                        curr_token = curr_processed_input["input_ids"].tolist()[0]
                        extend_input_ids = curr_token[len(input_ids) :]

                        input_ids.extend(extend_input_ids)
                        loss_mask = [0] * len(extend_input_ids)
                        versions = [-1] * len(extend_input_ids)
                        main_trajectory = self._add_to_trajectory(
                            main_trajectory,
                            extend_input_ids,
                            [0.0] * len(extend_input_ids),
                            versions,
                            loss_mask,
                            [True] * len(extend_input_ids),
                        )
                    else:
                        # Failed execution, append error message
                        main_trajectory = self._append_tool_output(
                            main_trajectory,
                            f"[Error: {tool_call.function.name} failed]",
                        )
            else:
                # No tool call, stop generation, reach eos token
                break

            turn += 1

        # Track tool call stats for WandB monitoring
        _tracker = stats_tracker.get("tool_stats")
        _tracker.scalar(tool_call_count=float(len(subagent_trajectories)))
        _tracker.scalar(tool_call_success=float(sum(
            1 for t in subagent_trajectories
            if t.get("input_ids") is not None and t["input_ids"].numel() > 0
        )))
        _tracker.scalar(tool_turns_used=float(turn))

        # Compute main agent reward
        prompt = self.tokenizer.decode(main_trajectory["input_ids"][0])
        loss_mask = main_trajectory["loss_mask"][0]
        completion_ids = main_trajectory["input_ids"][0][loss_mask == 1]
        completion = self.tokenizer.decode(completion_ids)

        reward_result = await self.async_main_reward_fn(
            prompt=prompt,
            completions=completion,
            prompt_ids=main_trajectory["input_ids"][0],
            completion_ids=completion_ids,
            nframes_used=nframes_used,
            **data,
        )

        # Support both tuple (reward, breakdown) and plain float return
        if isinstance(reward_result, tuple):
            main_reward, reward_breakdown = reward_result
        else:
            main_reward = reward_result
            reward_breakdown = {}

        # Log reward breakdown to WandB
        if reward_breakdown:
            _reward_tracker = stats_tracker.get("reward_breakdown")
            for key, value in reward_breakdown.items():
                _reward_tracker.scalar(**{key: float(value)})

        main_trajectory["rewards"] = torch.tensor([main_reward], dtype=torch.float32)
        multi_modal_input = [
            {
                "pixel_values": processed_input["pixel_values"],
            }
        ]
        if "image_grid_thw" in processed_input:
            multi_modal_input[0]["image_grid_thw"] = processed_input["image_grid_thw"]
            image_grid_thw_shape = processed_input["image_grid_thw"].shape

        main_trajectory["multi_modal_input"] = multi_modal_input

        pixel_values_shape = processed_input["pixel_values"].shape

        # Return main trajectory with subagent trajectories as a separate list
        if len(subagent_trajectories) == 0:
            subagent_trajectories = [self._init_trajectory()]
            subagent_trajectories[0]["multi_modal_input"] = []
            pixel_values = torch.zeros(
                (0, *pixel_values_shape[1:]),
                dtype=processed_input["pixel_values"].dtype,
                device=processed_input["pixel_values"].device,
            )
            subagent_trajectories[0]["multi_modal_input"] = [
                {
                    "pixel_values": pixel_values,
                }
            ]
            if "image_grid_thw" in processed_input:
                image_grid_thw = torch.zeros(
                    (0, *image_grid_thw_shape[1:]),
                    dtype=processed_input["image_grid_thw"].dtype,
                    device=processed_input["image_grid_thw"].device,
                )
                subagent_trajectories[0]["multi_modal_input"][0]["image_grid_thw"] = (
                    image_grid_thw
                )

        for traj in subagent_trajectories:
            traj["episode_id"] = episode_id
            traj["rewards"] = main_trajectory["rewards"]
        main_trajectory["subagent_trajectories"] = subagent_trajectories
        main_trajectory["input_ids"] = main_trajectory["input_ids"].long()
        main_trajectory["episode_id"] = episode_id
        return main_trajectory
