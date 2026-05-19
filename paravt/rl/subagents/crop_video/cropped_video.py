import os
import uuid
from typing import Any

import torch
from areal.api.io_struct import ModelRequest
from areal.utils import logging
from areal.utils.image import image2base64
from PIL import Image
from qwen_vl_utils import fetch_video
from torchvision.transforms.functional import to_pil_image

from paravt.rl.subagents.base import (
    SubagentToolBase,
    ToolCallStatus,
    ToolDescription,
    register_subagent,
)

logger = logging.getLogger("CroppedVideoSubagent")


@register_subagent("crop_video")
class CroppedVideoSubagent(SubagentToolBase):
    """Subagent that processes cropped video segments.

    When called, this subagent:
    1. Receives a video path and time range (start, end)
    2. The sampling API (provided externally) extracts fixed number of frames
    3. The frames are passed as images to the subagent model
    4. The subagent summarizes the video segment
    """

    def __init__(
        self,
        **kwargs,
    ):
        super().__init__(**kwargs)

    def get_tool_name(self) -> str:
        return "crop_video"

    def get_description(self) -> ToolDescription:
        return ToolDescription(
            name="crop_video",
            description=(
                "Analyze a cropped video segment by sampling fixed number of frames "
                "and providing a summary of the content. Use this when you need to "
                "understand what happens in a specific time range of a video."
            ),
            parameters={
                "video_path": {
                    "type": "string",
                    "description": "Path to the video file",
                },
                "start_time": {
                    "type": "number",
                    "description": "Start time in seconds",
                },
                "end_time": {
                    "type": "number",
                    "description": "End time in seconds",
                },
            },
            required=["video_path", "start_time", "end_time"],
        )

    def build_prompt(self, task: str, **kwargs) -> str:
        start_time = kwargs.get("start_time", 0)
        end_time = kwargs.get("end_time", 0)

        # Format time as MM:SS for readability
        def format_time(seconds: float) -> str:
            mins = int(seconds // 60)
            secs = int(seconds % 60)
            return f"{mins:02d}:{secs:02d}"

        start_str = format_time(start_time)
        end_str = format_time(end_time)
        duration = end_time - start_time

        prompt = f"""You are analyzing a video segment from {start_str} to {end_str} (duration: {duration:.1f} seconds).

The images shown are frames sampled from this video segment. Carefully examine each frame in sequence to understand the temporal progression of events.

Your task: Provide a concise summary of what happens in this video segment.

Guidelines:
- Describe the key actions, events, or changes that occur across the frames
- Note any important objects, people, or elements visible in the scene
- Capture the temporal flow (what happens first, then, finally)
- Focus on factual observations rather than interpretations

Summary:"""

        return prompt

    def validate_parameters(
        self, parameters: dict[str, Any], resolved_video_path: str | None = None
    ) -> tuple[bool, str]:
        start = parameters.get("start_time", parameters.get("start"))
        end = parameters.get("end_time", parameters.get("end"))
        if start is None:
            return False, "start_time (or start) is required"
        if end is None:
            return False, "end_time (or end) is required"
        if start >= end:
            return False, "start must be less than end"
        check_path = resolved_video_path or parameters.get("video_path", "")
        if check_path and not os.path.exists(check_path):
            return False, f"video_path does not exist: {check_path}"
        # Validate and clamp timestamps against video duration
        if check_path and os.path.exists(check_path):
            try:
                import subprocess
                result = subprocess.run(
                    ["ffprobe", "-v", "quiet", "-show_entries",
                     "format=duration", "-of", "csv=p=0", check_path],
                    capture_output=True, text=True, timeout=10
                )
                if result.returncode == 0 and result.stdout.strip():
                    duration = float(result.stdout.strip())
                    if start >= duration:
                        return False, f"start_time ({start:.1f}s) exceeds video duration ({duration:.1f}s)"
                    if end > duration:
                        # Clamp end_time to video duration
                        parameters["end_time"] = duration
                        logger.info(f"Clamped end_time from {end:.1f}s to {duration:.1f}s")
            except Exception as e:
                logger.warning(f"Could not get video duration: {e}")
        return True, ""

    async def execute(
        self, parameters: dict[str, Any], data: dict[str, Any] | None = None
    ) -> tuple[dict[str, torch.Tensor], ToolCallStatus]:
        """Execute cropped video subagent with video frame sampling

        Args:
            parameters: dict containing video_path, start, end, num_frames
            data: additional context data

        Returns:
            trajectory dict and status
        """
        if not self.engine:
            raise RuntimeError(
                "Subagent engine not set. Call set_execution_context first."
            )

        # Resolve video path: prefer data["video_paths"] over model-generated path
        video_path = parameters.get("video_path", "")
        if data and "video_paths" in data and data["video_paths"]:
            video_path = data["video_paths"][0]
        elif not os.path.isabs(video_path):
            for root in [os.environ.get("PARAVT_VIDEO_ROOT", "./data/videos")]:
                candidate = os.path.join(root, os.path.basename(video_path))
                if os.path.exists(candidate):
                    video_path = candidate
                    break

        is_valid, error_msg = self.validate_parameters(parameters, video_path)
        if not is_valid:
            logger.warning(f"Parameter validation failed: {error_msg}")
            return {}, ToolCallStatus.ERROR

        start_time = parameters.get("start_time", parameters.get("start"))
        end_time = parameters.get("end_time", parameters.get("end"))

        task = parameters.get("task", "")
        prompt = self.build_prompt(task, start_time=start_time, end_time=end_time)

        frame_images: list[Image.Image] = []

        # Use video_start/video_end to crop video to the requested time range.
        # Wrap in try/except so a corrupt video / ffmpeg backend failure /
        # degenerate range falls back to ToolCallStatus.ERROR (the same path
        # validate_parameters takes) instead of crashing the trainer subprocess.
        try:
            frames_tensor = fetch_video(
                {
                    "type": "video",
                    "video": f"file://{video_path}",
                    "fps": 1,
                    "min_frames": 1,
                    "max_frames": 16,
                    "min_pixels": 784,
                    "max_pixels": 50176,
                    "video_start": start_time,
                    "video_end": end_time,
                },
                return_video_sample_fps=False,
                return_video_metadata=False,
            )
        except Exception as fetch_err:
            logger.warning(
                f"fetch_video failed for {video_path} "
                f"[{start_time:.2f}s-{end_time:.2f}s]: {fetch_err}"
            )
            return {}, ToolCallStatus.ERROR
        if self.debug_mode:
            logger.info(f"  Video crop: {start_time:.1f}s - {end_time:.1f}s, frames: {frames_tensor.shape[0]}")

        for tensor in frames_tensor:
            tensor = tensor.to(torch.uint8)
            image = to_pil_image(tensor)
            frame_images.append(image)

        byte_images = image2base64(frame_images)
        openai_messages = [{"role": "user", "content": []}]
        hf_messages = [{"role": "user", "content": []}]
        for _image in frame_images:
            openai_messages[0]["content"].append(
                {"type": "image_url", "image_url": {"url": ""}}
            )
            hf_messages[0]["content"].append({"type": "image"})
        openai_messages[0]["content"].append({"type": "text", "text": prompt})
        hf_messages[0]["content"].append({"type": "text", "text": prompt})
        prompt = self.processor.apply_chat_template(
            hf_messages, tokenize=False, add_generation_prompt=True
        )

        processed_input = self.processor(
            images=frame_images, text=prompt, padding=False, return_tensors="pt"
        )
        input_ids = processed_input["input_ids"].tolist()[0]

        req = ModelRequest(
            rid=uuid.uuid4().hex,
            input_ids=input_ids,
            gconfig=self.gconfig.new(n_samples=1),
            tokenizer=self.tokenizer,
            image_data=byte_images if byte_images else None,
            processor=self.processor,
            vision_msg_vllm=[openai_messages],
        )
        resp = await self.engine.agenerate(req)
        output_text = self.tokenizer.decode(resp.output_tokens)

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
            "multi_modal_input": [
                {
                    "pixel_values": processed_input["pixel_values"],
                }
            ],
        }
        if "image_grid_thw" in processed_input:
            trajectory["multi_modal_input"][0]["image_grid_thw"] = processed_input[
                "image_grid_thw"
            ]

        if self.debug_mode:
            logger.info(f"Subagent {self.get_tool_name()} execution complete:")
            logger.info(f"  Prompt: {prompt[:200]}...")
            logger.info(f"  Output: {output_text[:200]}...")
            logger.info(f"  Frames sampled: {len(frame_images)}")

        return trajectory, ToolCallStatus.SUCCESS
