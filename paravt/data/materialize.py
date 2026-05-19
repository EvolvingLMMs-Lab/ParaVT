"""Materialize the sentinel paths inside ParaVT-Parquet against a local video root.

After you download ``ParaVT/ParaVT-Source`` and unpack its zips, point
this script at the directory and it will rewrite every sentinel-form
path in the parquets back to an absolute path under that root.

Usage (after ``huggingface-cli download ParaVT/ParaVT-Source --local-dir
./paravt-source-root`` and unzipping)::

    python -m paravt.data.materialize \\
        --root ./paravt-source-root \\
        --parquet-dir ./paravt-parquet-downloaded \\
        --output-dir  ./paravt-parquet-materialized

The materialized parquets carry absolute ``file://`` URIs that any
downstream pipeline (lmms-engine, AReaL) can consume directly.  By
default the script also reports any path whose target file is missing
on disk so you notice if a zip wasn't unpacked.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

from paravt.data.sanitize import SENTINEL_RULES


SENTINEL_PREFIXES: tuple[str, ...] = tuple(sentinel for _, sentinel in SENTINEL_RULES)


def materialize_path(rel: str, root: Path, missing: Counter) -> str:
    """Return ``file://<root>/<rel>`` if *rel* is a sentinel path; else passthrough."""
    if not rel:
        return rel
    raw = rel[len("file://"):] if rel.startswith("file://") else rel
    if not any(raw.startswith(p) for p in SENTINEL_PREFIXES):
        return rel  # unknown shape — leave alone
    full = (root / raw).resolve()
    if not full.exists():
        missing[raw.split("/", 1)[0]] += 1
    return "file://" + str(full)


def _walk_messages(messages: Any, root: Path, missing: Counter) -> Any:
    if isinstance(messages, str):
        decoded = json.loads(messages)
        return json.dumps(_walk_messages(decoded, root, missing), ensure_ascii=False)
    if isinstance(messages, list):
        return [_walk_messages(turn, root, missing) for turn in messages]
    if isinstance(messages, dict):
        out = dict(messages)
        for kind in ("video_url", "image_url"):
            if out.get("type") == kind:
                sub = out.get(kind) or {}
                if "url" in sub and sub["url"]:
                    out[kind] = {**sub, "url": materialize_path(sub["url"], root, missing)}
        if "content" in out:
            out["content"] = _walk_messages(out["content"], root, missing)
        if "messages" in out:
            out["messages"] = _walk_messages(out["messages"], root, missing)
        return out
    return messages


def _walk_videos(videos: Any, root: Path, missing: Counter) -> Any:
    if not isinstance(videos, list):
        return videos
    out: list[Any] = []
    for item in videos:
        if isinstance(item, dict) and "video" in item:
            out.append({**item, "video": materialize_path(item["video"], root, missing)})
        else:
            out.append(item)
    return out


def materialize_parquet(src: Path, dst: Path, root: Path) -> dict[str, int]:
    table = pq.read_table(str(src))
    columns = {name: table.column(name).to_pylist() for name in table.schema.names}
    missing: Counter = Counter()

    if "messages" in columns:
        columns["messages"] = [_walk_messages(m, root, missing) for m in columns["messages"]]
    if "videos" in columns:
        columns["videos"] = [_walk_videos(v, root, missing) for v in columns["videos"]]

    new_table = pa.table(columns, schema=table.schema)
    dst.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(new_table, str(dst))
    return dict(missing)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, required=True,
                    help="Directory where ParaVT-Source zips were extracted.")
    ap.add_argument("--parquet-dir", type=Path, required=True,
                    help="Directory containing the (downloaded, still-sentinel) parquets.")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Where to write the materialized parquets.")
    args = ap.parse_args(argv)

    if not args.root.exists():
        print(f"--root does not exist: {args.root}", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)
    report: dict[str, dict[str, int]] = {}

    parquets = sorted(args.parquet_dir.glob("paravt_*.parquet"))
    if not parquets:
        print(f"no parquets matched paravt_*.parquet under {args.parquet_dir}", file=sys.stderr)
        return 1

    for src in parquets:
        dst = args.output_dir / src.name
        missing = materialize_parquet(src, dst, args.root)
        report[src.name] = missing
        if missing:
            print(f"[warn] {src.name}: missing files per bucket -> {missing}", file=sys.stderr)
        else:
            print(f"[done] {src.name}")

    audit = args.output_dir / "materialize_report.json"
    audit.write_text(json.dumps(report, indent=2))
    print(f"[done] audit -> {audit}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
