"""ParaVT video-benchmark evaluation driver.

One driver, seven prompt modes — controlled by ``--prompt_mode``. The
mode set covers the native protocols used by each baseline class so
the right prompt is wired automatically per row:

* **direct** — single forward pass; minimal "answer in
  ``<answer>...</answer>``" instruction. Right for plain instruct
  backbones (Qwen2.5-VL-7B).
* **reasoning** — single forward pass; the standard
  ``<think>...</think><answer>...</answer>`` contract baked into
  Video-R1, VideoRFT, VideoChat-R1, Video-Thinker, Time-R1, ReWatch-R1.
* **agentic_general** — multi-turn ``crop_video`` dispatch with the
  agentic prompt that wraps ``<think>`` / ``<tool_call>`` / ``<answer>``
  and accepts multiple tool calls per turn. Right for Qwen3-VL-8B (tool),
  Conan-7B, and **ParaVT-8B (Ours)** on MCQ benchmarks.
* **agentic_minimal** — same ``crop_video`` schema as
  ``agentic_general`` under a lighter prompt envelope (no verbose
  framing, just the bare tool block). Used for Charades-STA temporal
  grounding because the lighter envelope produces tighter spans than
  the verbose scaffolding.
* **agentic_videozoomer** — multi-turn ``video_zoom`` dispatch using
  VideoZoomer-7B's native tool protocol.
* **agentic_sage** — multi-turn dispatch using SAGE-7B's Context-VLM
  JSON tool schema.
* **agentic_longvt** — multi-turn ``crop_video`` dispatch using
  LongVT-RFT-7B's iMCoTT prompt suffix.

Backwards compat: ``--no_tool`` is preserved as an alias for
``--prompt_mode direct``.

Channel selection (``--video_channel``):
  - ``image_url`` (default for MCQ): client-side base64 PNG frames.
  - ``video_url``: file:// path → vLLM mm-processor decodes server-side.
    Recommended for charades temporal grounding — the model gets
    ``video_metadata`` (fps, frame indices) from vLLM, which an
    ``image_url`` channel cannot pass through.

The driver disables ``--mm-processor-kwargs`` by default
(``--no_mm_kwargs``, ON by default) because injecting a
``max_pixels=50176`` envelope on top of client-side base64 frames
re-resizes them to 224x224 in vLLM, which discards visual detail and
shifts behavior off-trend. Pass ``--mm_kwargs`` to re-enable the
vLLM-side kwargs for custom ``video_url`` configs.

Usage::

    # ParaVT-8B
    python -m paravt.eval.driver \\
        --model_path ParaVT/ParaVT-8B \\
        --datasets videomme --output_dir ./eval-results/paravt \\
        --prompt_mode agentic_general --max_turns 3 --max_parallel 5 \\
        --num_gpus 8 --main_nframes 64 \\
        --video_channel image_url --no_mm_kwargs

    # Reasoning baseline (Video-R1 / VideoChat-R1 / ...)
    python -m paravt.eval.driver \\
        --model_path Video-R1/Video-R1-7B \\
        --datasets videomme --output_dir ./eval-results/video-r1 \\
        --prompt_mode reasoning --max_turns 1 \\
        --num_gpus 8 --main_nframes 64 \\
        --video_channel image_url --no_mm_kwargs

    # Direct baseline (Qwen2.5-VL-7B-Instruct)
    python -m paravt.eval.driver \\
        --model_path Qwen/Qwen2.5-VL-7B-Instruct \\
        --datasets videomme --output_dir ./eval-results/qwen2.5-vl \\
        --prompt_mode direct --max_turns 1 \\
        --num_gpus 8 --main_nframes 64 \\
        --video_channel image_url --no_mm_kwargs
"""

from __future__ import annotations

import functools
import json
import os
import random
import re
import time
from typing import Any

from paravt.eval import utils

# ─── Tool schema (matches SFT training) ──────────────────────────────────────
TOOL_SCHEMA = json.dumps(
    {
        "type": "function",
        "function": {
            "name": "crop_video",
            "description": "Analyze a cropped video segment by sampling frames from a specific time range.",
            "parameters": {
                "type": "object",
                "properties": {
                    "video_path": {"type": "string", "description": "Path to the video file"},
                    "start_time": {"type": "number", "description": "Start time in seconds"},
                    "end_time": {"type": "number", "description": "End time in seconds"},
                },
                "required": ["video_path", "start_time", "end_time"],
            },
        },
    }
)

# ─── System prompts (one per --prompt_mode value) ────────────────────────────
# Kept side-by-side here so a reader can diff the seven prompts directly.

DIRECT_SYSTEM = (
    "You are a helpful video understanding assistant. Watch the video "
    "carefully and answer the question. Provide your final answer in "
    "<answer></answer> tags."
)

REASONING_SYSTEM = (
    "You are a helpful assistant. When the user asks a question, your "
    "response must include two parts: first, the reasoning process "
    "enclosed in <think>...</think> tags, then the final answer "
    "enclosed in <answer>...</answer> tags."
)

# The default agentic prompt — preserves the parallel-dispatch
# capability that is ParaVT's headline contribution and keeps the
# verbatim <think>/<tool_call>/<answer> contract used at training time.
GENERAL_AGENTIC_SYSTEM = """You are a video understanding agent that can analyze videos by cropping and examining specific segments.

Available tool:
<tools>
{tool_schema}
</tools>

For tool calls, return a JSON object within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": "crop_video", "arguments": {{"video_path": "...", "start_time": 0, "end_time": 10}}}}
</tool_call>

You may emit **multiple <tool_call> blocks in one turn** to analyze different segments in parallel — this is more efficient than sequential calls and is the recommended dispatch shape for this agent.

Format strictly as:
<think>your reasoning here</think>
<tool_call>...</tool_call>  (zero or more, parallel if multiple)
<answer>your final answer</answer>

Always wrap your reasoning in <think> tags and your final answer in <answer> tags."""

# A minimal tool-call envelope (no Workflow / Important / Format
# scaffolding). Useful when the verbose agentic_general scaffolding adds
# noise on tasks that benefit from concise outputs — temporal-grounding
# benchmarks like Charades-STA in particular tend to land tighter spans
# under this lighter envelope.
MINIMAL_AGENTIC_SYSTEM = """# Tools

You may call one or more functions to assist with the user query.

You are provided with function signatures within <tools></tools> XML tags:
<tools>
{tool_schema}
</tools>

For each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:
<tool_call>
{{"name": <function-name>, "arguments": <args-json-object>}}
</tool_call>"""


# TODO(release): paste VideoZoomer-7B's verbatim system prompt + native
# <video_zoom> tool protocol here. Ships as a stub; the runtime guard in
# _system_message() raises if --prompt_mode agentic_videozoomer is used
# before this is filled in.
VIDEOZOOMER_AGENTIC_SYSTEM = "FILL_FROM_VIDEOZOOMER"

# TODO(release): paste SAGE-7B's verbatim system prompt + Context-VLM
# JSON 5-tool schema here. Ships as a stub; runtime guard in
# _system_message() raises if --prompt_mode agentic_sage is used before
# this is filled in.
SAGE_AGENTIC_SYSTEM = "FILL_FROM_SAGE"

# TODO(release): paste LongVT-RFT-7B's verbatim TOOL_PROMPT suffix
# (sourced from `lmms_eval_tasks/videomme/utils.py` in
# EvolvingLMMs-Lab/LongVT). Ships as a stub; runtime guard in
# _system_message() raises if --prompt_mode agentic_longvt is used
# before this is filled in.
LONGVT_TOOL_PROMPT_SUFFIX = "FILL_FROM_LONGVT"


PROMPT_MODES = (
    "direct",
    "reasoning",
    "agentic_general",
    "agentic_minimal",
    "agentic_videozoomer",
    "agentic_sage",
    "agentic_longvt",
)
SINGLE_TURN_MODES = ("direct", "reasoning")


# Default max output tokens. 2048 keeps long-CoT baselines (SAGE,
# Conan) from hitting truncation that drops their <answer>. Temperature
# is 0 across all modes, so the extra headroom changes nothing for
# shorter responses.
DEFAULT_MAX_TOKENS = 2048

# Per-call OpenAI client timeout. Long videos under multi-turn agentic
# dispatch can run ~10 min end-to-end; the OpenAI library's 30 s default
# would hard-fail.
OPENAI_TIMEOUT_S = 1800.0


def parse_tool_calls(text: str) -> list[dict[str, Any]]:
    """Parse ``<tool_call>{json}</tool_call>`` and the leaked
    ``<tool_code>name(args)</tool_code>`` form Qwen-VL sometimes emits
    because of its pre-training prior. Without the second branch a model
    that defaults to the Python-style tag silently contributes zero tool
    calls to the rollout.
    """
    calls: list[dict[str, Any]] = []
    for m in re.findall(r"<tool_call>\s*(.*?)\s*</tool_call>", text, re.DOTALL):
        try:
            calls.append(json.loads(m))
        except Exception:
            pass
    for m in re.findall(r"<tool_code>\s*(.*?)\s*</tool_code>", text, re.DOTALL):
        body = m.strip()
        if not body or body.startswith("<|im_end|>"):
            continue
        match = re.match(r"crop_video\s*\((.*)\)\s*", body, re.DOTALL)
        if not match:
            continue
        argstr = match.group(1)
        args: dict[str, Any] = {}
        for kv in re.split(r",\s*(?=[a-zA-Z_]+\s*=)", argstr):
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            v = v.strip().strip("\"'")
            try:
                args[k.strip()] = float(v)
            except ValueError:
                args[k.strip()] = v
        if args:
            calls.append({"name": "crop_video", "arguments": args})
    return calls


def parse_video_zoom_calls(text: str) -> list[dict[str, Any]]:
    """Parse VideoZoomer's native ``<video_zoom>`` blocks.

    Accepts either of two surface forms VideoZoomer-7B has been observed
    to emit:

      * ``<video_zoom>{"start_time": ..., "end_time": ...}</video_zoom>``
      * ``<video_zoom>start=10.0 end=20.0</video_zoom>``

    Both shapes are converted to the canonical ``crop_video``-style
    dispatch dict so the agentic loop downstream can stay shared with
    ``agentic_general``.
    """
    calls: list[dict[str, Any]] = []
    for m in re.findall(r"<video_zoom>\s*(.*?)\s*</video_zoom>", text, re.DOTALL):
        body = m.strip()
        if not body:
            continue
        try:
            args = json.loads(body)
            if isinstance(args, dict):
                calls.append({"name": "crop_video", "arguments": args})
                continue
        except Exception:
            pass
        kv_args: dict[str, Any] = {}
        for kv in re.split(r"\s+|,\s*", body):
            if "=" not in kv:
                continue
            k, v = kv.split("=", 1)
            try:
                kv_args[k.strip()] = float(v.strip().strip("\"'"))
            except ValueError:
                kv_args[k.strip()] = v.strip().strip("\"'")
        if kv_args:
            calls.append({"name": "crop_video", "arguments": kv_args})
    return calls


def _completion_with_retry(
    client, model_name: str, messages: list[dict[str, Any]], max_tokens: int
):
    """Three-retry chain with linear-jitter backoff. Connection drops on
    long-stream sessions are common; the retry chain converts transients
    into successes."""
    last_exc: Exception | None = None
    for attempt in range(3):
        try:
            return client.chat.completions.create(
                model=model_name,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0,
            )
        except Exception as e:  # noqa: BLE001
            last_exc = e
            time.sleep(1.5 * (attempt + 1) + random.uniform(0, 0.5))
    if last_exc is not None:
        raise last_exc
    raise RuntimeError("unreachable")


def _score(sample: dict[str, Any], final: str) -> tuple[Any, float, Any]:
    if sample["is_mcq"]:
        pred = utils.extract_letter(final)
        correct: Any = pred == sample["answer"]
        score = 1.0 if correct else 0.0
        return correct, score, pred
    pred_ts = utils.extract_time_from_response(final)
    gt = [sample["gt_start"], sample["gt_end"]]
    if pred_ts:
        return None, utils.temporal_iou(pred_ts, gt), f"{pred_ts[0]:.1f}-{pred_ts[1]:.1f}"
    return None, 0.0, final[:200]


def _make_result(
    sample: dict[str, Any], final: str, pred: Any, correct: Any, score: float, tool_calls: int
) -> dict[str, Any]:
    result: dict[str, Any] = {
        "id": sample["id"],
        "answer": str(sample["answer"]),
        "pred": pred,
        "correct": correct,
        "tool_calls": tool_calls,
        "score": score,
        "full_response": final[:500],
    }
    for extra in ("key", "config", "duration"):
        if extra in sample:
            result[extra] = sample[extra]
    if not sample["is_mcq"]:
        result["pred_text"] = final[:500]
    return result


def _system_message(prompt_mode: str) -> str | None:
    """Return the system-prompt string for the requested mode.

    Raises ValueError if a placeholder baseline prompt
    (VIDEOZOOMER/SAGE/LONGVT) hasn't been filled in yet — running
    against an unset placeholder would silently score nonsense, so we
    fail loud instead.
    """
    if prompt_mode == "direct":
        return DIRECT_SYSTEM
    if prompt_mode == "reasoning":
        return REASONING_SYSTEM
    if prompt_mode == "agentic_general":
        return GENERAL_AGENTIC_SYSTEM.format(tool_schema=TOOL_SCHEMA)
    if prompt_mode == "agentic_minimal":
        return MINIMAL_AGENTIC_SYSTEM.format(tool_schema=TOOL_SCHEMA)
    if prompt_mode == "agentic_videozoomer":
        if VIDEOZOOMER_AGENTIC_SYSTEM == "FILL_FROM_VIDEOZOOMER":
            raise ValueError(
                "VIDEOZOOMER_AGENTIC_SYSTEM is a placeholder. Paste "
                "VideoZoomer-7B's native system prompt and <video_zoom> "
                "tool protocol into paravt/eval/driver.py."
            )
        return VIDEOZOOMER_AGENTIC_SYSTEM
    if prompt_mode == "agentic_sage":
        if SAGE_AGENTIC_SYSTEM == "FILL_FROM_SAGE":
            raise ValueError(
                "SAGE_AGENTIC_SYSTEM is a placeholder. Paste SAGE-7B's "
                "native system prompt and Context-VLM JSON 5-tool schema "
                "into paravt/eval/driver.py."
            )
        return SAGE_AGENTIC_SYSTEM
    if prompt_mode == "agentic_longvt":
        if LONGVT_TOOL_PROMPT_SUFFIX == "FILL_FROM_LONGVT":
            raise ValueError(
                "LONGVT_TOOL_PROMPT_SUFFIX is a placeholder. Paste "
                "LongVT-RFT-7B's TOOL_PROMPT suffix from "
                "EvolvingLMMs-Lab/LongVT lmms_eval_tasks/videomme/utils.py "
                "into paravt/eval/driver.py."
            )
        return LONGVT_TOOL_PROMPT_SUFFIX
    raise ValueError(f"unknown prompt_mode: {prompt_mode!r}")


def _is_agentic(prompt_mode: str) -> bool:
    return prompt_mode.startswith("agentic_")


def _video_user_content(
    vpath: str, q_text: str, video_channel: str, nframes: int
) -> list[dict[str, Any]]:
    """Build the user-turn content list for the initial video.

    ``image_url`` channel: client-side base64 PNG frames — the default
    for MCQ benches.

    ``video_url`` channel: file:// → vLLM server-side decode. Used for
    charades temporal grounding because the model needs vLLM's
    ``video_metadata`` (fps, frame indices) to anchor its time
    predictions.
    """
    if video_channel == "video_url":
        return [
            {
                "type": "video_url",
                "video_url": {"url": f"file://{os.path.abspath(vpath)}"},
            },
            {"type": "text", "text": q_text},
        ]
    # image_url: client-side base64 frames
    frames = utils.get_frames(vpath, nframes=nframes)
    content: list[dict[str, Any]] = [
        {"type": "image_url", "image_url": {"url": utils.pil_to_b64(img)}}
        for img in frames
    ]
    content.append({"type": "text", "text": q_text})
    return content


def eval_one(
    client,
    model_name: str,
    sample: dict[str, Any],
    nframes: int,
    max_turns: int,
    prompt_mode: str = "agentic_general",
    max_parallel: int = 5,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    video_channel: str = "image_url",
) -> dict[str, Any]:
    """Evaluate one sample under one of the seven prompt modes.

    Single-turn modes (``direct``, ``reasoning``) skip the dispatch loop
    entirely. The four ``agentic_*`` modes run the multi-turn dispatch
    up to ``max_turns`` rounds before forcing a final answer, capped to
    ``max_parallel`` tool calls per turn.
    """
    try:
        vpath = sample["video_path"]
        if not os.path.exists(vpath):
            return {**utils.base_result(sample), "error": f"video not found: {vpath[-60:]}"}

        q_text = sample["question"]
        if _is_agentic(prompt_mode) and "video path" not in q_text.lower():
            q_text = f"{q_text} The video path for this video is: {os.path.basename(vpath)}"
        user_content = _video_user_content(vpath, q_text, video_channel, nframes)

        system_msg = _system_message(prompt_mode)
        base_messages: list[dict[str, Any]] = []
        if system_msg is not None:
            base_messages.append({"role": "system", "content": system_msg})
        base_messages.append({"role": "user", "content": user_content})

        if not _is_agentic(prompt_mode):
            try:
                resp = _completion_with_retry(
                    client, model_name, base_messages, max_tokens=max_tokens
                )
                final = resp.choices[0].message.content or ""
            except Exception as e:
                return {**utils.base_result(sample), "error": f"api error: {str(e)[:120]}"}
            correct, score, pred = _score(sample, final)
            return _make_result(sample, final, pred, correct, score, tool_calls=0)

        # agentic multi-turn dispatch
        messages = list(base_messages)
        tool_calls = 0
        final = ""
        for turn in range(max_turns + 1):
            try:
                resp = _completion_with_retry(
                    client, model_name, messages, max_tokens=max_tokens
                )
                response = resp.choices[0].message.content or ""
            except Exception as e:
                final = f"[API error: {e}]"
                break

            if prompt_mode == "agentic_videozoomer":
                tcs = parse_video_zoom_calls(response)
            else:
                tcs = parse_tool_calls(response)
            if not tcs:
                final = response
                break
            if max_parallel > 0:
                tcs = tcs[:max_parallel]

            messages.append({"role": "assistant", "content": response})
            for tc in tcs:
                tool_calls += 1
                args = tc.get("arguments", {})
                st = utils.parse_time_val(args.get("start_time", args.get("start", 0)))
                et = utils.parse_time_val(args.get("end_time", args.get("end", st + 10)))
                cropped = utils.crop_frames(vpath, st, et)
                if cropped:
                    tc_c: list[dict[str, Any]] = [
                        {"type": "image_url", "image_url": {"url": utils.pil_to_b64(img)}}
                        for img in cropped
                    ]
                    tc_c.append(
                        {
                            "type": "text",
                            "text": (
                                f"<tool_response>\n{len(cropped)} frames "
                                f"from {st:.1f}s to {et:.1f}s.\n</tool_response>"
                            ),
                        }
                    )
                    messages.append({"role": "user", "content": tc_c})
                else:
                    messages.append(
                        {
                            "role": "user",
                            "content": (
                                f"<tool_response>\n[Error cropping {st:.1f}s-{et:.1f}s]\n</tool_response>"
                            ),
                        }
                    )

            if turn == max_turns - 1:
                if sample["is_mcq"]:
                    force = "Based on the video and cropped segments above, answer with ONLY the letter."
                    force_max = 64
                else:
                    force = (
                        "Based on the video and any cropped segments, provide your final answer "
                        "about when the event happens. Use the format: "
                        "'The event happens in the X.X - Y.Y seconds.'"
                    )
                    force_max = 128
                messages.append({"role": "user", "content": force})
                try:
                    resp = _completion_with_retry(
                        client, model_name, messages, max_tokens=force_max
                    )
                    final = resp.choices[0].message.content or final
                except Exception:
                    pass
                break

        correct, score, pred = _score(sample, final)
        return _make_result(sample, final, pred, correct, score, tool_calls=tool_calls)
    except Exception as e:
        return {**utils.base_result(sample), "error": str(e)[:200]}


def main() -> None:
    args = _parse_args_with_prompt_mode()
    bound_eval_one = functools.partial(
        eval_one,
        prompt_mode=args.prompt_mode,
        max_parallel=args.max_parallel,
        max_tokens=args.max_tokens,
        video_channel=args.video_channel,
    )
    utils.run_eval(
        args,
        eval_one_fn=bound_eval_one,
        driver_name=args.prompt_mode,
    )


def _parse_args_with_prompt_mode():
    """Local arg parser so ``--prompt_mode``, ``--video_channel``, and the
    legacy ``--no_tool`` alias are recognized alongside the shared
    :func:`utils.parse_args` flags."""
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True)
    parser.add_argument(
        "--datasets",
        required=True,
        help="Comma-separated subset of the seven Tab1 benches: "
        "videomme, videomme_wsub, longvideobench, lvbench, mlvu, mmvu, charades.",
    )
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--num_gpus", type=int, default=8)
    parser.add_argument("--base_port", type=int, default=8000)
    parser.add_argument("--main_nframes", type=int, default=64)
    parser.add_argument(
        "--max_turns",
        type=int,
        default=3,
        help="Number of tool-dialog turns under any --prompt_mode "
        "agentic_*; ignored for direct/reasoning",
    )
    parser.add_argument("--workers_per_shard", type=int, default=4)
    parser.add_argument(
        "--smoke_test",
        action="store_true",
        help="Quick test: 1 GPU, 5 samples per dataset",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of samples per dataset (overrides --smoke_test slice)",
    )
    parser.add_argument(
        "--prompt_mode",
        choices=PROMPT_MODES,
        default="agentic_general",
        help="System-prompt + dispatch shape for the model under test. "
        "One per baseline class; see eval/README.md for the per-row "
        "mapping.",
    )
    parser.add_argument(
        "--video_channel",
        choices=("image_url", "video_url"),
        default="image_url",
        help="Channel the initial video travels through. 'image_url' "
        "is the default for MCQ benches; 'video_url' is recommended "
        "for charades because the model needs vLLM's video_metadata "
        "(fps, frame indices) to anchor temporal predictions.",
    )
    parser.add_argument(
        "--no_mm_kwargs",
        dest="no_mm_kwargs",
        action="store_true",
        default=True,
        help="Disable vLLM --mm-processor-kwargs. ON by default; pass "
        "--mm_kwargs to re-enable for custom video_url configs that "
        "need an explicit pixel-envelope override.",
    )
    parser.add_argument(
        "--mm_kwargs",
        dest="no_mm_kwargs",
        action="store_false",
        help="Re-enable vLLM --mm-processor-kwargs (paired inverse of "
        "--no_mm_kwargs).",
    )
    parser.add_argument(
        "--no_tool",
        action="store_true",
        help="Deprecated alias for --prompt_mode direct (kept for "
        "backward compat with the prior release).",
    )
    parser.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Outer shard index for cross-machine split (0..num_shards-1)",
    )
    parser.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total outer shards for cross-machine split",
    )
    parser.add_argument(
        "--max_parallel",
        type=int,
        default=5,
        help="Cap on tool calls accepted per turn (set <=0 to disable)",
    )
    parser.add_argument(
        "--reuse_servers",
        action="store_true",
        help="Assume vLLM servers at base_port..base_port+num_gpus are already up; "
        "skip launch + skip kill on exit.",
    )
    parser.add_argument(
        "--max_pixels",
        type=int,
        default=50176,
        help="Per-frame max pixels passed to vLLM mm-processor when "
        "--no_mm_kwargs is disabled (default 50176 = 224x224 matches "
        "paravt/rl/configs/_base.yaml).",
    )
    parser.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=131072,
        help="vLLM --max-num-batched-tokens (prefill batch ceiling). "
        "Raised from 65536 -> 131072 to absorb long-sequence batch-"
        "scheduling edges on 64-frame Qwen2.5-VL inputs.",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=DEFAULT_MAX_TOKENS,
        help="Max output tokens per generation request. Default 2048 "
        "keeps long-CoT baselines (SAGE, Conan) from hitting truncation "
        "that drops their <answer> tag.",
    )
    parser.add_argument(
        "--vllm_host",
        type=str,
        default="localhost",
        help="Hostname of the vLLM servers. Defaults to localhost; set "
        "this when the eval driver runs on a different machine than the "
        "vLLM workers (e.g. SSH-forwarded ports).",
    )
    args = parser.parse_args()

    if args.no_tool and _is_agentic(args.prompt_mode):
        args.prompt_mode = "direct"
    elif args.no_tool and args.prompt_mode != "direct":
        print(
            f"[driver] WARN --no_tool passed alongside --prompt_mode "
            f"{args.prompt_mode!r}; ignoring --no_tool.",
        )
    return args


if __name__ == "__main__":
    main()
