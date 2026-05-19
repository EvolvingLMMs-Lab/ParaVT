"""ParaVT RL training dataset adapter.

Reads a ParaVT parquet file and yields the dict shape that
``HierarchicalAgentWorkflow`` expects. Skips rows whose backing video
files are missing on disk so a partially synced data tree does not crash
training.
"""

from __future__ import annotations

import json
import os
import sys

from areal.utils import logging
from datasets import Dataset

logger = logging.getLogger("ParaVT.Dataset")


def load_paravt_rl_dataset(parquet_path: str, video_root: str) -> Dataset:
    """Load a ParaVT RL parquet and resolve video paths against ``video_root``.

    Each row of the parquet must have ``prompt`` (list of role/content
    dicts), ``videos`` (list with ``{"video": <path>}``), and
    ``extra_info`` (JSON-encoded answer + question_type).

    Returns a :class:`datasets.Dataset` of dicts with keys ``messages``,
    ``video_paths``, ``answer``, and ``question_type``.
    """
    raw = Dataset.from_parquet(parquet_path)

    def gen():
        for item in raw:
            messages = []
            for content in item["prompt"]:
                text = content["content"]
                role = content["role"]
                if "<video>" in text:
                    text = text.replace("<video>", "")
                    messages.append(
                        {
                            "role": role,
                            "content": [
                                {"type": "video"},
                                {"type": "text", "text": text},
                            ],
                        }
                    )
                else:
                    messages.append(
                        {"role": role, "content": [{"type": "text", "text": text}]}
                    )

            video_paths: list[str] = []
            for video_info in item["videos"]:
                vpath = video_info["video"]
                if vpath.startswith("file://"):
                    vpath = vpath[len("file://") :]
                full_path = os.path.join(video_root, vpath)
                if not os.path.exists(full_path):
                    # Avoid logger.warning here: closure-over-logger trips up
                    # HuggingFace datasets fingerprint hashing.
                    print(f"[paravt.data] missing video, skipping: {full_path}", file=sys.stderr)
                    continue
                video_paths.append(full_path)

            if not video_paths:
                continue

            extra_info = item["extra_info"]
            if isinstance(extra_info, str):
                extra_info = json.loads(extra_info)

            yield {
                "messages": messages,
                "video_paths": video_paths,
                "answer": extra_info.get("answer", ""),
                "question_type": extra_info.get("question_type", "oe"),
            }

    dataset = Dataset.from_generator(gen)
    logger.info(f"Loaded {len(dataset)} training samples from {parquet_path}")
    return dataset
