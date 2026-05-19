"""ParaVT single-shot inference demo.

Loads a ParaVT checkpoint via vLLM and answers a single question about
a video. The system prompt is selected by ``--prompt_mode``; this is
the same enum the eval driver (``paravt/eval/driver.py``) uses, so a
demo run reproduces the same per-baseline framing used in the headline
table.

Usage:
    # No-tool overview answer (default — direct system prompt)
    python scripts/inference.py \\
        --video /path/to/video.mp4 \\
        --question "What does the speaker do after the demo?" \\
        --model ParaVT/ParaVT-8B

    # Reasoning baseline (<think>/<answer> contract)
    python scripts/inference.py \\
        --video /path/to/video.mp4 \\
        --question "Where does the cat go?" \\
        --model Video-R1/Video-R1-7B \\
        --prompt_mode reasoning

    # ParaVT agentic (single-turn variant — emits <tool_call> blocks
    # but does not actually dispatch them; for the full multi-turn
    # crop_video loop use paravt/eval/driver.py).
    python scripts/inference.py \\
        --video /path/to/video.mp4 \\
        --question "..." --prompt_mode agentic_general

The legacy ``--no_tool`` flag is kept as an alias for
``--prompt_mode direct``.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--question", required=True)
    parser.add_argument(
        "--model",
        default=os.environ.get("PARAVT_EVAL_MODEL", "ParaVT/ParaVT-8B"),
    )
    parser.add_argument("--nframes", type=int, default=64)
    parser.add_argument("--max_pixels", type=int, default=50176)
    parser.add_argument(
        "--min_pixels",
        type=int,
        default=784,
        help="Lower bound on per-frame pixel count, matches paravt/rl/configs/_base.yaml.",
    )
    parser.add_argument("--max_tokens", type=int, default=512)
    parser.add_argument(
        "--prompt_mode",
        default="direct",
        choices=(
            "direct",
            "reasoning",
            "agentic_general",
            "agentic_videozoomer",
            "agentic_sage",
            "agentic_longvt",
        ),
        help="System-prompt shape; see paravt/eval/README.md for the "
        "per-baseline mapping.",
    )
    parser.add_argument(
        "--no_tool",
        action="store_true",
        help="Deprecated alias for --prompt_mode direct.",
    )
    args = parser.parse_args()

    if args.no_tool:
        args.prompt_mode = "direct"

    if not args.video.is_file():
        raise FileNotFoundError(args.video)

    # Lazy imports so --help works without GPU dependencies.
    from qwen_vl_utils import process_vision_info
    from transformers import AutoProcessor
    from vllm import LLM, SamplingParams

    # Reuse the eval driver's prompt strings so the demo never drifts
    # from what the headline numbers were generated under.
    from paravt.eval.driver import _system_message  # type: ignore[attr-defined]

    system_prompt = _system_message(args.prompt_mode)

    processor = AutoProcessor.from_pretrained(args.model)
    llm = LLM(model=args.model, dtype="bfloat16", trust_remote_code=True)

    user_content = [
        {
            "type": "video",
            "video": str(args.video),
            "max_pixels": args.max_pixels,
                    "min_pixels": args.min_pixels,
            "nframes": args.nframes,
        },
        {"type": "text", "text": args.question},
    ]
    messages = []
    if system_prompt is not None:
        messages.append({"role": "system", "content": system_prompt})
    messages.append({"role": "user", "content": user_content})

    prompt = processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)
    image_inputs, video_inputs = process_vision_info(messages)

    out = llm.generate(
        {
            "prompt": prompt,
            "multi_modal_data": {"video": video_inputs},
        },
        SamplingParams(temperature=0.0, max_tokens=args.max_tokens),
    )
    print(out[0].outputs[0].text)


if __name__ == "__main__":
    main()
