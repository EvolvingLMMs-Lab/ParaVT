"""Shared infrastructure for the :mod:`paravt.eval.driver` driver.

The driver itself (one ``eval_one`` callable, six prompt modes, two
video channels) lives in :mod:`paravt.eval.driver`. Everything else
(dataset loaders, vLLM lifecycle, shard fan-out, MCQ/IoU scoring, JSON
output) is shared and lives here.

Usage from a custom driver::

    from paravt.eval import utils

    def eval_one(client, model_name, sample, nframes, max_turns):
        ...
        return result_dict

    def main():
        args = utils.parse_args()
        utils.run_eval(args, eval_one_fn=eval_one, driver_name="my-driver")
"""

from __future__ import annotations

import argparse
import ast
import base64
import json
import os
import re
import subprocess
import sys
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
from io import BytesIO
from multiprocessing import Process
from typing import Any

import numpy as np
from openai import OpenAI
from PIL import Image
from qwen_vl_utils.vision_process import fetch_video


# ─── Frame helpers ───────────────────────────────────────────────────────────
def pil_to_b64(img: Image.Image) -> str:
    buf = BytesIO()
    img.save(buf, format="JPEG", quality=85)
    return f"data:image/jpeg;base64,{base64.b64encode(buf.getvalue()).decode()}"


def get_frames(video_path: str, nframes: int = 64) -> list[Image.Image]:
    """Sample ``nframes`` evenly spaced frames from a video, fail-soft."""
    try:
        frames = fetch_video(
            {"type": "video", "video": f"file://{video_path}", "nframes": nframes},
            return_video_sample_fps=False,
            return_video_metadata=False,
        )
        return [Image.fromarray(t.permute(1, 2, 0).numpy().astype(np.uint8)) for t in frames]
    except Exception:
        return []


def crop_frames(
    video_path: str, st: float, et: float, max_frames: int = 16
) -> list[Image.Image]:
    """Crop a temporal segment of a video and sample frames at 1 fps (fail-soft)."""
    try:
        frames = fetch_video(
            {
                "type": "video",
                "video": f"file://{video_path}",
                "fps": 1,
                "min_frames": 1,
                "max_frames": max_frames,
                "min_pixels": 784,
                "max_pixels": 50176,
                "video_start": st,
                "video_end": et,
            },
            return_video_sample_fps=False,
            return_video_metadata=False,
        )
        return [Image.fromarray(t.permute(1, 2, 0).numpy().astype(np.uint8)) for t in frames]
    except Exception:
        return []


# ─── Answer / time extractors ────────────────────────────────────────────────
def parse_time_val(val: Any) -> float:
    """Accept seconds (int/float/str) or ``HH:MM:SS`` / ``MM:SS`` strings."""
    if isinstance(val, (int, float)):
        return float(val)
    try:
        return float(val)
    except Exception:
        pass
    if isinstance(val, str) and ":" in val:
        parts = val.split(":")
        try:
            if len(parts) == 2:
                return float(parts[0]) * 60 + float(parts[1])
            if len(parts) == 3:
                return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
        except Exception:
            pass
    return 0.0


def extract_letter(text: str) -> str:
    """Parse the option letter (A–J) out of an MCQ response with graceful fallback."""
    text = text.strip()
    m = re.search(r"<answer>(.*?)</answer>", text, re.DOTALL)
    if m:
        inner = m.group(1).strip()
        if len(inner) == 1 and inner in "ABCDEFGHIJ":
            return inner
        lm = re.search(r"\b([A-J])\b", inner)
        if lm:
            return lm.group(1)
    if len(text) == 1 and text in "ABCDEFGHIJ":
        return text
    m = re.search(r"(?:answer|Answer)\s*(?:is|:)\s*([A-J])", text)
    if m:
        return m.group(1)
    m = re.search(r"\b([A-J])\b", text)
    if m:
        return m.group(1)
    return text[:1] if text else ""


def extract_time_from_response(text: str) -> list[float] | None:
    """Pull a ``[start, end]`` pair out of a temporal-grounding response.

    Tries ``mm:ss[.f]`` before bare seconds so timestamps like
    ``<time>00:19.7 - 00:25.6</time>`` parse as 19.7s/25.6s rather than
    19.7s/0.0s (the bare-seconds regex would otherwise pick up ``19.7-00``
    and silently drop the minutes part of the second timestamp).
    """
    if not text or not text.strip():
        return None
    tag = re.search(r"<time>([^<]+)</time>", text)
    targets = ([tag.group(1)] if tag else []) + [text]
    for t in targets:
        m = re.search(
            r"(\d+):(\d+(?:\.\d+)?)\s*[-–—]\s*(\d+):(\d+(?:\.\d+)?)", t
        )
        if m:
            start = int(m.group(1)) * 60 + float(m.group(2))
            end = int(m.group(3)) * 60 + float(m.group(4))
            if end > start and end < 3600:
                return [start, end]
        m = re.search(r"(\d+\.?\d*)\s*[-–—]\s*(\d+\.?\d*)", t)
        if m:
            start, end = float(m.group(1)), float(m.group(2))
            if end > start:
                return [start, end]
    starts = re.findall(
        r"(?:start(?:ing)?|begin(?:ning)?)\s*(?:time)?[:\s]*(\d+\.?\d*)", text, re.I
    )
    ends = re.findall(
        r"(?:end(?:ing)?|finish(?:ing)?)\s*(?:time)?[:\s]*(\d+\.?\d*)", text, re.I
    )
    if starts and ends:
        return [float(starts[0]), float(ends[0])]
    nums = re.findall(r"(\d+\.?\d*)", text)
    if len(nums) >= 2:
        return [float(nums[0]), float(nums[1])]
    return None


def temporal_iou(a: list[float], b: list[float]) -> float:
    inter = max(0.0, min(a[1], b[1]) - max(a[0], b[0]))
    union = max(a[1], b[1]) - min(a[0], b[0])
    return inter / union if union > 0 else 0.0


def parse_srt(srt_path: str) -> str:
    """Return concatenated subtitle text from an SRT file (empty on failure)."""
    if not os.path.exists(srt_path):
        return ""
    try:
        with open(srt_path, encoding="utf-8", errors="ignore") as f:
            content = f.read()
        lines = []
        for block in re.split(r"\n\s*\n", content.strip()):
            parts = block.strip().split("\n")
            if len(parts) >= 3:
                for line in parts[2:]:
                    clean = re.sub(r"<[^>]+>", "", line).strip()
                    if clean:
                        lines.append(clean)
        return "\n".join(lines) if lines else ""
    except Exception:
        return ""


def base_result(sample: dict[str, Any]) -> dict[str, Any]:
    """Empty result dict used for failed samples."""
    return {
        "id": sample.get("id", "?"),
        "answer": str(sample.get("answer", "")),
        "pred": "",
        "correct": False,
        "tool_calls": 0,
        "score": 0.0,
    }


# ─── Dataset loaders ─────────────────────────────────────────────────────────
def load_samples(dataset_name: str, hf_home: str, prompt_mode: str = "agentic_general") -> list[dict[str, Any]]:
    """Load and normalize a benchmark to ParaVT's per-sample dict shape.

    Each sample carries ``id``, ``video_path``, ``question``, ``answer``,
    ``is_mcq``, plus dataset-specific extras (``duration``, ``config``,
    ``key``, ``gt_start``, ``gt_end``).
    """
    from datasets import load_dataset as hf_load

    mcq_instr = (
        "Think step by step, use crop_video if needed, then provide your "
        "answer in <answer>X</answer> tags where X is the option letter."
        if prompt_mode == "direct"
        else "Answer with the option letter only."
    )

    if dataset_name == "videomme":
        ds = hf_load(
            "lmms-lab/Video-MME",
            "videomme",
            split="test",
            cache_dir=os.path.join(hf_home, "datasets"),
        )
        video_dir = os.path.join(hf_home, "videomme", "data")
        samples = []
        for item in ds:
            opts = item.get("options", [])
            if isinstance(opts, list):
                opts_str = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(opts))
            else:
                opts_str = str(opts)
            question = f"Question: {item['question']}\nOptions:\n{opts_str}\n{mcq_instr}"
            samples.append(
                {
                    "id": item["videoID"],
                    "video_path": os.path.join(video_dir, f"{item['videoID']}.mp4"),
                    "question": question,
                    "answer": item.get("answer", ""),
                    "is_mcq": True,
                    "duration": item.get("duration", ""),
                }
            )
        return samples

    if dataset_name == "videomme_wsub":
        ds = hf_load(
            "lmms-lab/Video-MME",
            "videomme",
            split="test",
            cache_dir=os.path.join(hf_home, "datasets"),
        )
        video_dir = os.path.join(hf_home, "videomme", "data")
        sub_dir = os.path.join(hf_home, "videomme", "subtitle")
        samples = []
        n_with_sub = 0
        for item in ds:
            opts = item.get("options", [])
            if isinstance(opts, list):
                opts_str = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(opts))
            else:
                opts_str = str(opts)
            sub_text = parse_srt(os.path.join(sub_dir, f"{item['videoID']}.srt"))
            sub_prefix = (
                f"This video's subtitles are listed below:\n{sub_text}\n\n" if sub_text else ""
            )
            if sub_text:
                n_with_sub += 1
            question = f"{sub_prefix}Question: {item['question']}\nOptions:\n{opts_str}\n{mcq_instr}"
            samples.append(
                {
                    "id": item["videoID"],
                    "video_path": os.path.join(video_dir, f"{item['videoID']}.mp4"),
                    "question": question,
                    "answer": item.get("answer", ""),
                    "is_mcq": True,
                    "duration": item.get("duration", ""),
                }
            )
        print(f"  videomme_wsub: {n_with_sub}/{len(samples)} samples have subtitles")
        return samples


    if dataset_name == "longvideobench":
        ds = hf_load(
            "longvideobench/LongVideoBench",
            split="validation",
            cache_dir=os.path.join(hf_home, "datasets"),
        )
        video_dir = os.path.join(hf_home, "datasets", "longvideobench", "videos")
        samples = []
        for item in ds:
            opts = []
            for i in range(5):
                opt = item.get(f"option{i}", "")
                if opt and opt != "N/A":
                    opts.append(opt)
            opts_str = "\n".join(f"{chr(65 + i)}. {o}" for i, o in enumerate(opts))
            answer = chr(65 + item["correct_choice"]) if item.get("correct_choice", -1) >= 0 else "?"
            samples.append(
                {
                    "id": str(item.get("id", item.get("video_id", ""))),
                    "video_path": os.path.join(video_dir, item["video_path"]),
                    "question": f"Question: {item['question']}\nOptions:\n{opts_str}\n{mcq_instr}",
                    "answer": answer,
                    "is_mcq": True,
                }
            )
        return samples

    if dataset_name == "lvbench":
        ds = hf_load("lmms-lab/LVBench", split="train", cache_dir=os.path.join(hf_home, "datasets"))
        video_dir = os.path.join(hf_home, "lvbench")
        samples = []
        for item in ds:
            samples.append(
                {
                    "id": item.get("uid", ""),
                    "video_path": os.path.join(video_dir, item["video_path"]),
                    "question": f"{item['question']}\n{mcq_instr}",
                    "answer": item.get("answer", ""),
                    "is_mcq": True,
                }
            )
        return samples

    if dataset_name == "mlvu":
        # Upstream MLVU/MVLU was restructured upstream and dropped the M-Eval
        # config; sy1998/MLVU_dev preserves the QA schema (video_name +
        # question + candidates + answer + task_type) the eval driver expects.
        ds = hf_load(
            "sy1998/MLVU_dev",
            split="test",
            cache_dir=os.path.join(hf_home, "datasets"),
        )
        # Followers must extract the dataset's video_part_*.zip into MLVU/video.
        # Override with PARAVT_MLVU_VIDEO_DIR if videos live elsewhere.
        video_dir = os.environ.get(
            "PARAVT_MLVU_VIDEO_DIR", os.path.join(hf_home, "MLVU", "video")
        )
        samples = []
        for i, item in enumerate(ds):
            opts = item.get("candidates", [])
            if isinstance(opts, list):
                opts_str = "\n".join(f"{chr(65 + j)}. {o}" for j, o in enumerate(opts))
            else:
                opts_str = str(opts)
            answer_text = item.get("answer", "")
            answer_letter = "?"
            if isinstance(opts, list):
                for j, o in enumerate(opts):
                    if str(o).strip() == str(answer_text).strip():
                        answer_letter = chr(65 + j)
                        break
            samples.append(
                {
                    "id": f"{item.get('task_type', 'mlvu')}-{i}",
                    "video_path": os.path.join(video_dir, item["video_name"]),
                    "question": f"Question: {item['question']}\nOptions:\n{opts_str}\n{mcq_instr}",
                    "answer": answer_letter,
                    "is_mcq": True,
                    "config": item.get("task_type", ""),
                }
            )
        return samples

    if dataset_name == "mmvu":
        ds = hf_load(
            "yale-nlp/MMVU",
            split="validation",
            cache_dir=os.path.join(hf_home, "datasets"),
        )
        # yale-nlp/MMVU stores the video field as a full huggingface.co URL
        # (e.g. .../resolve/main/videos/Art/0.mp4). The actual mp4s live in
        # the dataset snapshot under <snapshot>/videos/<subject>/<id>.mp4 —
        # snapshot_download materializes them with allow_patterns=[videos/**]
        # so we only fetch the videos, not the upstream LFS metadata.
        # Override with PARAVT_MMVU_VIDEO_DIR if videos live elsewhere.
        env_dir = os.environ.get("PARAVT_MMVU_VIDEO_DIR")
        if env_dir:
            video_root = env_dir
        else:
            from huggingface_hub import snapshot_download

            # First call materializes ~30 GB of mp4s under
            # $HF_HOME/hub/datasets--yale-nlp--MMVU/snapshots/<sha>/videos.
            # Set PARAVT_MMVU_VIDEO_DIR to skip this for offline / shared
            # video stores.
            video_root = snapshot_download(
                repo_id="yale-nlp/MMVU",
                repo_type="dataset",
                allow_patterns=["videos/**"],
                cache_dir=os.path.join(hf_home, "hub"),
            )
        samples = []
        for item in ds:
            choices = item.get("choices", {}) or {}
            opts_str = "\n".join(
                f"{k}. {v}" for k, v in sorted(choices.items()) if v not in (None, "")
            )
            video_url = item["video"]
            if "/main/" in video_url:
                video_rel = video_url.split("/main/", 1)[1]
            elif video_url.startswith(("http://", "https://")):
                raise ValueError(
                    f"unexpected mmvu video field (no /main/ marker): {video_url!r}"
                )
            else:
                video_rel = video_url
            samples.append(
                {
                    "id": item.get("id", ""),
                    "video_path": os.path.join(video_root, video_rel),
                    "question": f"Question: {item['question']}\nOptions:\n{opts_str}\n{mcq_instr}",
                    "answer": str(item.get("answer", "")).strip(),
                    "is_mcq": True,
                    "config": item.get("subject", ""),
                }
            )
        return samples

    if dataset_name == "charades":
        ds = hf_load(
            "lmms-lab/charades_sta",
            split="test",
            cache_dir=os.path.join(hf_home, "datasets"),
        )
        video_dir = os.path.join(hf_home, "charades_sta", "Charades_v1_480")
        samples = []
        for i, item in enumerate(ds):
            caption = item["caption"]
            ts = item["timestamp"]
            samples.append(
                {
                    "id": i,
                    "key": f"{item['video']}>>>{caption}>>>{ts}",
                    "video_path": os.path.join(video_dir, item["video"]),
                    "question": (
                        "Please find the visual event described by a sentence in the video, "
                        "determining its starting and ending times. The format should be: "
                        "'The event happens in the start time - end time'. "
                        "For example, The event 'person turn a light on' happens in the 24.3 - 30.4 seconds. "
                        f'Now I will give you the textual sentence: "{caption}". '
                        "Please return its start time and end time."
                    ),
                    "answer": ts,
                    "is_mcq": False,
                    "gt_start": float(ts[0]),
                    "gt_end": float(ts[1]),
                }
            )
        return samples

    raise ValueError(f"Unknown dataset: {dataset_name}")


# ─── vLLM server lifecycle ───────────────────────────────────────────────────
def launch_servers(
    model: str,
    n: int = 8,
    base_port: int = 8000,
    nframes: int = 64,
    max_pixels: int = 50176,
    max_num_batched_tokens: int = 131072,
    log_prefix: str = "vllm",
    disable_mm_kwargs: bool = True,
) -> list[subprocess.Popen]:
    """Spawn ``n`` independent vLLM servers, one per GPU.

    Maps through the outer ``CUDA_VISIBLE_DEVICES`` so callers can share GPUs
    across runs.

    * ``--allowed-local-media-path /`` lets vLLM open ``file://`` mp4s
      directly when the driver uses ``--video_channel video_url``.
    * ``--mm-processor-kwargs`` is **disabled by default**
      (``disable_mm_kwargs=True``) because injecting a
      ``max_pixels=50176`` envelope on top of client-side base64 frames
      re-resizes them to 224x224 inside vLLM, which discards visual
      detail and shifts behavior off-trend. Pass
      ``disable_mm_kwargs=False`` only for the rare custom configs that
      need the kwarg envelope on ``video_url`` inputs.
    * ``--max-num-batched-tokens 131072`` defaults wide enough to absorb
      long-sequence batch-scheduling edges on 64-frame Qwen2.5-VL
      inputs. A single 64-frame Qwen2.5-VL long-video input generates
      up to ~43 K visual tokens; smaller caps return 400 BadRequest on
      every long-video sample. Raising the cap carries no fixed memory
      cost (it only widens the prefill ceiling).
    """
    procs: list[subprocess.Popen] = []
    outer_cvd = os.environ.get("CUDA_VISIBLE_DEVICES", None)
    visible_gpus = outer_cvd.split(",") if outer_cvd is not None else [str(i) for i in range(8)]
    if outer_cvd is not None and n > len(visible_gpus):
        raise ValueError(
            f"--num_gpus={n} exceeds the {len(visible_gpus)} GPUs visible via "
            f"CUDA_VISIBLE_DEVICES={outer_cvd!r}; vLLM would silently fall back to "
            f"physical GPU ids outside the mask."
        )
    for gpu in range(n):
        cmd = [
            sys.executable,
            "-m",
            "vllm.entrypoints.openai.api_server",
            "--model",
            model,
            "--trust-remote-code",
            "--dtype",
            "bfloat16",
            "--max-model-len",
            "65536",
            "--max-num-batched-tokens",
            str(max_num_batched_tokens),
            "--gpu-memory-utilization",
            "0.85",
            "--limit-mm-per-prompt",
            '{"video":1,"image":1024}',
            "--allowed-local-media-path",
            "/",
            "--port",
            str(base_port + gpu),
        ]
        if not disable_mm_kwargs:
            mm_kwargs = {
                "num_frames": nframes,
                "min_pixels": 784,
                "max_pixels": max_pixels,
            }
            cmd.extend(["--mm-processor-kwargs", json.dumps(mm_kwargs)])
        env = os.environ.copy()
        physical_gpu = visible_gpus[gpu] if gpu < len(visible_gpus) else str(gpu)
        env["CUDA_VISIBLE_DEVICES"] = physical_gpu
        log = open(f"/tmp/{log_prefix}_gpu{physical_gpu}_pid{os.getpid()}.log", "w", encoding="utf-8")
        p = subprocess.Popen(cmd, env=env, stdout=log, stderr=log)
        # Hang the log handle off the Popen so kill_servers can close it.
        p._paravt_log = log  # type: ignore[attr-defined]
        procs.append(p)
        mm_tag = "no_mm_kwargs" if disable_mm_kwargs else f"mm_kwargs(nframes={nframes})"
        print(f"vLLM GPU {physical_gpu} port {base_port + gpu} PID {p.pid} ({mm_tag})")
    return procs


def wait_servers(n: int = 8, base_port: int = 8000, timeout: int = 900) -> bool:
    """Poll each server's ``/v1/models`` endpoint until all are ready."""
    import requests

    ready = [False] * n
    t0 = time.time()
    _host = os.environ.get("PARAVT_VLLM_HOST", "localhost")
    while time.time() - t0 < timeout:
        for g in range(n):
            if ready[g]:
                continue
            try:
                if (
                    requests.get(f"http://{_host}:{base_port + g}/v1/models", timeout=3).status_code
                    == 200
                ):
                    ready[g] = True
                    print(f"  GPU {g} ready")
            except Exception:
                pass
        if all(ready):
            return True
        time.sleep(10)
    print(f"WARNING: {sum(not r for r in ready)} servers not ready")
    return False


def kill_servers(procs: list[subprocess.Popen]) -> None:
    for p in procs:
        try:
            p.terminate()
            p.wait(timeout=10)
        except Exception:
            try:
                p.kill()
            except Exception:
                pass
        log = getattr(p, "_paravt_log", None)
        if log is not None:
            try:
                log.close()
            except Exception:
                pass
    time.sleep(2)


# ─── Shard fan-out + per-dataset orchestration ───────────────────────────────
def run_shard(
    eval_one_fn: Callable,
    shard_id: int,
    total: int,
    port: int,
    model_name: str,
    samples: list[dict[str, Any]],
    nframes: int,
    max_turns: int,
    workers: int,
    output_path: str,
) -> None:
    """Execute ``eval_one_fn`` over a stride-shard of ``samples`` and dump JSON."""
    # OpenAI client default timeout is 30s; long-video multi-turn requests
    # routinely run ~10 min, so widen it generously. The vLLM scheduler
    # caps per-token wait independently, so a long client timeout is safe.
    client = OpenAI(
        api_key="EMPTY",
        base_url=f"http://{os.environ.get('PARAVT_VLLM_HOST', 'localhost')}:{port}/v1",
        timeout=1800.0,
    )
    shard = [samples[i] for i in range(shard_id, len(samples), total)]
    results: list[dict[str, Any]] = []
    t0 = time.time()
    print(f"[Shard {shard_id}] {len(shard)} samples on port {port}", flush=True)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        futs = {
            pool.submit(eval_one_fn, client, model_name, s, nframes, max_turns): i
            for i, s in enumerate(shard)
        }
        for i, f in enumerate(as_completed(futs)):
            results.append(f.result())
            if (i + 1) % 50 == 0:
                valid = [r for r in results if r.get("score") is not None]
                avg = sum(r["score"] for r in valid) / max(len(valid), 1)
                tc = sum(r.get("tool_calls", 0) for r in results) / max(len(results), 1)
                rate = (i + 1) / max(time.time() - t0, 1) * 3600
                print(
                    f"  [Shard {shard_id}] [{i + 1}/{len(shard)}] "
                    f"avg_score={avg:.3f} avg_tc={tc:.2f} rate={rate:.0f}/h",
                    flush=True,
                )

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(results, f)
    print(f"[Shard {shard_id}] Done. {len(results)} results in {time.time() - t0:.0f}s", flush=True)


def run_dataset(
    eval_one_fn: Callable,
    dataset_name: str,
    model_name: str,
    samples: list[dict[str, Any]],
    num_gpus: int,
    base_port: int,
    nframes: int,
    max_turns: int,
    workers: int,
    output_path: str,
    shard_prefix: str = "paravt-eval",
) -> dict[str, Any]:
    """Fan ``samples`` out across ``num_gpus`` shards, score, write summary JSON."""
    print(f"\n{'=' * 60}\nDataset: {dataset_name} ({len(samples)} samples)\n{'=' * 60}", flush=True)
    t0 = time.time()
    shard_outputs = []
    shard_procs = []
    for s in range(num_gpus):
        out = f"/tmp/{shard_prefix}_{dataset_name}_shard_{s}_pid{os.getpid()}.json"
        shard_outputs.append(out)
        sp = Process(
            target=run_shard,
            args=(
                eval_one_fn,
                s,
                num_gpus,
                base_port + s,
                model_name,
                samples,
                nframes,
                max_turns,
                workers,
                out,
            ),
        )
        sp.start()
        shard_procs.append(sp)
    for sp in shard_procs:
        sp.join()

    all_r: list[dict[str, Any]] = []
    for out in shard_outputs:
        if os.path.exists(out):
            with open(out, encoding="utf-8") as f:
                all_r.extend(json.load(f))

    errors = sum(1 for r in all_r if "error" in r)
    avg_tc = sum(r.get("tool_calls", 0) for r in all_r) / max(len(all_r), 1)
    tool_rate = sum(1 for r in all_r if r.get("tool_calls", 0) > 0) / max(len(all_r), 1)
    summary: dict[str, Any] = {
        "dataset": dataset_name,
        "total": len(all_r),
        "errors": errors,
        "avg_tool_calls": avg_tc,
        "tool_usage_rate": tool_rate,
        "elapsed": time.time() - t0,
        "nframes": nframes,
    }

    is_mcq = samples[0]["is_mcq"] if samples else True
    if is_mcq:
        mcq = [r for r in all_r if r.get("correct") is not None]
        acc = sum(r["correct"] for r in mcq) / max(len(mcq), 1) if mcq else 0
        summary["accuracy"] = acc
        summary["mcq_total"] = len(mcq)
        if dataset_name == "videomme":
            for dur in ("short", "medium", "long"):
                dur_r = [
                    r
                    for r in all_r
                    if r.get("duration") == dur and r.get("correct") is not None
                ]
                if dur_r:
                    summary[f"acc_{dur}"] = sum(r["correct"] for r in dur_r) / len(dur_r)
                    summary[f"n_{dur}"] = len(dur_r)
        print(f"\n=== {dataset_name} Results ===")
        print(f"Accuracy: {acc:.4f} ({sum(r['correct'] for r in mcq)}/{len(mcq)})")
        for tag in ("acc_short", "acc_medium", "acc_long"):
            if tag in summary:
                n_tag = summary.get(f"n_{tag.split('_', 1)[1]}", 0)
                print(f"  {tag}: {summary[tag]:.4f} (n={n_tag})")
    else:
        valid = [r for r in all_r if "error" not in r]
        ious = [r["score"] for r in valid]
        miou = sum(ious) / max(len(ious), 1)
        r03 = sum(1 for v in ious if v >= 0.3) / max(len(ious), 1)
        r05 = sum(1 for v in ious if v >= 0.5) / max(len(ious), 1)
        r07 = sum(1 for v in ious if v >= 0.7) / max(len(ious), 1)
        summary.update({"mIoU": miou, "R@0.3": r03, "R@0.5": r05, "R@0.7": r07})
        submission = {
            r["key"]: r["pred_text"] for r in all_r if "key" in r and r.get("pred_text")
        }
        summary["submission_count"] = len(submission)
        print(f"\n=== {dataset_name} Results ===")
        print(f"R@0.3: {r03:.4f}, R@0.5: {r05:.4f}, R@0.7: {r07:.4f}, mIoU: {miou:.4f}")

    print(f"Tool Usage: {tool_rate:.2%}, Avg Calls: {avg_tc:.2f}, Errors: {errors}")
    print(f"Elapsed: {time.time() - t0:.0f}s", flush=True)

    os.makedirs(os.path.dirname(output_path) or ".", exist_ok=True)
    output: dict[str, Any] = {"summary": summary, "results": all_r}
    if not is_mcq:
        output["submission"] = submission  # type: ignore[has-type]
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print(f"Saved to {output_path}")

    for out in shard_outputs:
        try:
            os.remove(out)
        except OSError:
            pass
    return summary


# ─── CLI ─────────────────────────────────────────────────────────────────────
def parse_args() -> argparse.Namespace:
    """Common CLI shared by both eval drivers."""
    p = argparse.ArgumentParser()
    p.add_argument("--model_path", required=True)
    p.add_argument(
        "--datasets",
        required=True,
        help="Comma-separated subset of: videomme, videomme_wsub, "
        "longvideobench, lvbench, mlvu, mmvu, charades",
    )
    p.add_argument("--output_dir", required=True)
    p.add_argument("--num_gpus", type=int, default=8)
    p.add_argument(
        "--vllm_host",
        type=str,
        default="localhost",
        help="Hostname for the vLLM servers spawned per GPU. Defaults to "
        "localhost; override when the eval driver runs on a different "
        "machine than the vLLM workers.",
    )
    p.add_argument("--base_port", type=int, default=8000)
    p.add_argument("--main_nframes", type=int, default=64)
    p.add_argument("--max_turns", type=int, default=3, help="Only used by eval driver")
    p.add_argument("--workers_per_shard", type=int, default=4)
    p.add_argument(
        "--smoke_test",
        action="store_true",
        help="Quick test: 1 GPU, 5 samples per dataset",
    )
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Cap the number of samples per dataset (overrides smoke_test slice)",
    )
    p.add_argument(
        "--shard_id",
        type=int,
        default=0,
        help="Outer shard index for cross-machine split (0..num_shards-1)",
    )
    p.add_argument(
        "--num_shards",
        type=int,
        default=1,
        help="Total outer shards for cross-machine split",
    )
    p.add_argument(
        "--max_parallel",
        type=int,
        default=5,
        help="Cap on tool calls accepted per turn (only used by eval driver)",
    )
    p.add_argument(
        "--reuse_servers",
        action="store_true",
        help="Assume vLLM servers at base_port..base_port+num_gpus are already up; "
        "skip launch + skip kill on exit. Lets multi-config sweeps share one server.",
    )
    p.add_argument(
        "--max_pixels",
        type=int,
        default=50176,
        help="Per-frame max pixels passed to vLLM mm-processor "
        "(default 50176 = 224x224 matches paravt/rl/configs/_base.yaml)",
    )
    p.add_argument(
        "--max_num_batched_tokens",
        type=int,
        default=131072,
        help="vLLM --max-num-batched-tokens (prefill batch ceiling). "
        "Default 131072 absorbs long-sequence batch-scheduling edges on "
        "64-frame Qwen2.5-VL inputs.",
    )
    p.add_argument(
        "--no_mm_kwargs",
        action="store_true",
        default=True,
        help="Disable vLLM --mm-processor-kwargs. Default ON because the "
        "kwarg envelope re-resizes image_url frames to 224x224 inside "
        "vLLM, discarding visual detail and shifting behavior off-trend.",
    )
    p.add_argument(
        "--video_channel",
        choices=("image_url", "video_url"),
        default="image_url",
        help="Channel the initial video travels through (used only by "
        "eval driver).",
    )
    return p.parse_args()


def run_eval(
    args: argparse.Namespace,
    eval_one_fn: Callable,
    driver_name: str,
) -> None:
    """End-to-end driver: load samples, launch servers, run shards, write JSON."""
    if args.smoke_test:
        args.num_gpus = 1
        args.workers_per_shard = 1

    hf_home = os.environ.get("HF_HOME", os.path.expanduser("~/.cache/huggingface"))
    os.environ.setdefault("HF_HOME", hf_home)

    datasets_list = [d.strip() for d in args.datasets.split(",")]
    print(f"=== ParaVT eval ({driver_name}) ===")
    print(f"Model: {args.model_path}")
    print(f"Datasets: {datasets_list}")
    print(f"GPUs: {args.num_gpus}, nframes: {args.main_nframes}")

    shard_id = getattr(args, "shard_id", 0)
    num_shards = getattr(args, "num_shards", 1)

    all_samples: dict[str, list[dict[str, Any]]] = {}
    for ds_name in datasets_list:
        print(f"\nLoading {ds_name}...")
        samples = load_samples(ds_name, hf_home, prompt_mode=args.prompt_mode)
        # Narrow first (limit / smoke), then split across outer shards. This
        # ordering keeps cross-machine sharding deterministic regardless of
        # how many machines we run on.
        if args.limit is not None and args.limit > 0:
            samples = samples[: args.limit]
        elif args.smoke_test:
            samples = samples[:5]
        if num_shards > 1:
            full_n = len(samples)
            samples = samples[shard_id::num_shards]
            print(f"  outer shard {shard_id}/{num_shards}: {len(samples)}/{full_n} samples")
        all_samples[ds_name] = samples
        missing = sum(1 for s in samples if not os.path.exists(s["video_path"]))
        print(f"  {ds_name}: {len(samples)} samples, {missing} videos missing")

    reuse_servers = getattr(args, "reuse_servers", False)
    if reuse_servers:
        print("[run_eval] --reuse_servers set: skipping launch_servers, "
              "expecting vLLM up on base_port..base_port+num_gpus already")
        procs = []
    else:
        procs = launch_servers(
            args.model_path,
            args.num_gpus,
            args.base_port,
            nframes=args.main_nframes,
            max_pixels=getattr(args, "max_pixels", 50176),
            max_num_batched_tokens=getattr(args, "max_num_batched_tokens", 131072),
            log_prefix=f"vllm-{driver_name}",
            disable_mm_kwargs=getattr(args, "no_mm_kwargs", True),
        )
    try:
        if not wait_servers(args.num_gpus, args.base_port):
            print("ERROR: servers not ready")
            return
        import requests

        # Fall back to localhost when called from driver.py (whose argparse
        # doesn't expose --vllm_host). The custom-driver path in this file's
        # argparse does carry it.
        vllm_host = getattr(args, "vllm_host", "localhost")
        os.environ["PARAVT_VLLM_HOST"] = vllm_host
        model_name = requests.get(
            f"http://{vllm_host}:{args.base_port}/v1/models"
        ).json()["data"][0]["id"]
        print(f"\nModel loaded: {model_name}")

        all_summaries: dict[str, dict[str, Any]] = {}
        for ds_name in datasets_list:
            output_path = os.path.join(args.output_dir, f"{ds_name}.json")
            # Resume-friendly: if a prior run for this dataset finished and its
            # output_path is present, reuse that summary instead of re-running.
            # Delete the file (or pass a different --output_dir) to force a
            # fresh run.
            if os.path.exists(output_path):
                try:
                    with open(output_path, encoding="utf-8") as f:
                        prior = json.load(f)
                    if isinstance(prior, dict) and "summary" in prior:
                        print(f"\n[skip] {ds_name}: reusing existing {output_path}")
                        all_summaries[ds_name] = prior["summary"]
                        continue
                except (OSError, json.JSONDecodeError):
                    print(f"[warn] {output_path} is corrupt; rerunning {ds_name}")
            summary = run_dataset(
                eval_one_fn,
                ds_name,
                model_name,
                all_samples[ds_name],
                args.num_gpus,
                args.base_port,
                args.main_nframes,
                args.max_turns,
                args.workers_per_shard,
                output_path,
                shard_prefix=f"paravt-{driver_name}",
            )
            all_summaries[ds_name] = summary

        print(f"\n{'=' * 60}\n=== ALL RESULTS ({driver_name}) ===\n{'=' * 60}")
        for ds_name, s in all_summaries.items():
            if "accuracy" in s:
                print(f"{ds_name}: acc={s['accuracy']:.4f}, tools={s['avg_tool_calls']:.2f}")
            elif "mIoU" in s:
                print(
                    f"{ds_name}: mIoU={s['mIoU']:.4f}, R@0.5={s['R@0.5']:.4f}, "
                    f"tools={s['avg_tool_calls']:.2f}"
                )

        os.makedirs(args.output_dir, exist_ok=True)
        # Merge with any pre-existing summary_all.json. scripts/run_eval.sh
        # invokes the driver once per dataset (to tear down vLLM between
        # datasets), so each invocation only knows its own dataset and would
        # otherwise wipe prior summaries on overwrite. Per-dataset *.json
        # files survive regardless.
        summary_path = os.path.join(args.output_dir, "summary_all.json")
        merged_summaries: dict[str, Any] = {}
        if os.path.exists(summary_path):
            try:
                with open(summary_path, encoding="utf-8") as f:
                    merged_summaries = json.load(f)
            except Exception:
                merged_summaries = {}
        merged_summaries.update(all_summaries)
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(merged_summaries, f, indent=2)
        print(f"\nAll summaries merged into {summary_path}")
    finally:
        if not reuse_servers:
            kill_servers(procs)
