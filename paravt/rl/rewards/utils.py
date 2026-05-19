"""Shared helpers for ParaVT reward functions: answer extraction, tag parsing."""

from __future__ import annotations

import json
import re

_TOOL_CALL_RE = re.compile(r"<tool_call>(.*?)(?:</tool_call>|$)", re.DOTALL)
_TOOL_CODE_RE = re.compile(r"<tool_code>(.*?)(?:</tool_code>|$)", re.DOTALL)
_ANSWER_RE = re.compile(r"<answer>(.*?)</answer>", re.DOTALL)
_THINK_CLOSED_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_THINK_OPEN_RE = re.compile(r"<think>(.*?)(?:<tool_|<answer>|<\|im_end\|>|$)", re.DOTALL)
_TOOL_CALL_BLOCK_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_TOOL_CODE_BLOCK_RE = re.compile(r"<tool_code>.*?</tool_code>", re.DOTALL)
_TOOL_RESP_BLOCK_RE = re.compile(r"<tool_response>.*?</tool_response>", re.DOTALL)


def extract_answer(completion: str) -> str:
    """Pull the model's final answer out of a completion with graceful fallback.

    Priority:
        1. ``<answer>...</answer>`` tags (canonical case).
        2. Content after ``</think>`` with tool-related blocks stripped.
        3. Last non-empty line.

    Returns the extracted string (may be empty if nothing parseable was
    found). Empty answers should yield zero accuracy reward downstream.
    """
    m = _ANSWER_RE.search(completion)
    if m:
        return m.group(1).strip()

    if "</think>" in completion:
        tail = completion.split("</think>", 1)[1]
        tail = _TOOL_CALL_BLOCK_RE.sub("", tail)
        tail = _TOOL_CODE_BLOCK_RE.sub("", tail)
        tail = _TOOL_RESP_BLOCK_RE.sub("", tail)
        tail = tail.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()
        if tail:
            return tail

    # Keep single-character lines: bare-letter MCQ answers ("B", "A", ...)
    # are common when a model skips the <answer> wrapper and we want to credit
    # them with the right letter rather than zero out the gradient signal.
    lines = [
        line.strip()
        for line in completion.splitlines()
        if line.strip()
        and not line.strip().startswith("<|im_")
        and not line.strip().startswith("<tool_")
    ]
    if lines:
        return lines[-1]
    return ""


def normalize_text(text: str) -> str:
    """Normalize for matching: lowercase, strip articles + punctuation, collapse whitespace."""
    text = text.strip().lower()
    text = re.sub(r"\b(a|an|the)\b", " ", text)
    text = re.sub(r"[^\w\s]", "", text)
    return " ".join(text.split())


def has_parseable_tool_call(completion: str) -> bool:
    """True iff the completion contains a tool call with non-empty content.

    Blocks the empty-``<tool_code></tool_code>`` exploit where the model
    games the format reward without actually calling a tool.
    """
    for match in _TOOL_CALL_RE.findall(completion):
        body = match.strip()
        if not body or body.startswith("<|im_end|>"):
            continue
        try:
            data = json.loads(body)
            if isinstance(data, dict) and "name" in data:
                return True
        except (json.JSONDecodeError, ValueError):
            pass

    for match in _TOOL_CODE_RE.findall(completion):
        body = match.strip()
        if not body or body.startswith("<|im_end|>"):
            continue
        for line in body.splitlines():
            line = line.strip()
            if line and "(" in line and not line.startswith("#"):
                return True
    return False


def has_substantive_think(completion: str, min_chars: int = 10) -> bool:
    """True iff the completion contains a ``<think>`` block with >= ``min_chars`` of content."""
    match = _THINK_CLOSED_RE.search(completion) or _THINK_OPEN_RE.search(completion)
    if not match:
        return False
    return len(match.group(1).strip()) >= min_chars


def think_before_tool(completion: str) -> bool:
    """True iff a ``<think>`` tag precedes the first tool call in the completion."""
    think_pos = completion.find("<think>")
    if think_pos == -1:
        return False
    candidates = [completion.find(tag) for tag in ("<tool_call>", "<tool_code>")]
    candidates = [pos for pos in candidates if pos != -1]
    if not candidates:
        return False
    return think_pos < min(candidates)
