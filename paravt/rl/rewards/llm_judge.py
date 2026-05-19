"""LLM-as-judge accuracy reward.

Engaged when the recipe sets ``paravt.reward.mode: llm``. Falls back to
token-F1 on API failures so a flaky judge can't crash a training run.

The judge endpoint, key, and model name are read from the environment so
secrets never enter the repo. See ``.secrets.env.example`` for the full
list (``LLM_JUDGE_API_KEY`` / ``LLM_JUDGE_BASE_URL`` / ``LLM_JUDGE_MODEL``).
"""

from __future__ import annotations

import os
import re
import threading
import time
from typing import Any

from areal.utils import logging

from paravt.rl.rewards.task_metrics import compute_f1

logger = logging.getLogger("ParaVT.LLMJudge")


class APIUsageTracker:
    """Thread-safe counter for LLM-judge calls. Logs a summary every 10 calls."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self.total_calls = 0
        self.total_input_tokens = 0
        self.total_output_tokens = 0
        self.total_errors = 0
        self.total_latency = 0.0
        self._last_log_calls = 0

    def record(
        self, input_tokens: int, output_tokens: int, latency: float, error: bool = False
    ) -> None:
        with self._lock:
            self.total_calls += 1
            self.total_input_tokens += input_tokens
            self.total_output_tokens += output_tokens
            self.total_latency += latency
            if error:
                self.total_errors += 1
            if self.total_calls - self._last_log_calls >= 10:
                self._last_log_calls = self.total_calls
                self._log_summary()

    def _log_summary(self) -> None:
        avg_latency = self.total_latency / max(self.total_calls, 1)
        # gpt-4o-mini list price; override by adjusting if you switch models.
        est_cost = (self.total_input_tokens * 0.15 + self.total_output_tokens * 0.60) / 1e6
        logger.info(
            f"[LLM Judge] calls={self.total_calls} errors={self.total_errors} "
            f"input_tok={self.total_input_tokens} output_tok={self.total_output_tokens} "
            f"avg_latency={avg_latency:.2f}s est_cost=${est_cost:.4f}"
        )

    def log_final(self) -> None:
        with self._lock:
            self._log_summary()


_tracker = APIUsageTracker()
_client: Any = None


def _get_client():
    global _client
    if _client is None:
        from openai import OpenAI

        _client = OpenAI(
            api_key=os.environ.get("LLM_JUDGE_API_KEY"),
            base_url=os.environ.get("LLM_JUDGE_BASE_URL"),
        )
        logger.info(
            "Initialized LLM-judge client: model=%s base_url=%s",
            os.environ.get("LLM_JUDGE_MODEL", "gpt-4o-mini"),
            os.environ.get("LLM_JUDGE_BASE_URL"),
        )
    return _client


def llm_judge_reward(question: str, prediction: str, ground_truth: str) -> float:
    """LLM-judged semantic match between ``prediction`` and ``ground_truth``.

    Returns a scalar in ``{0.0, 0.5, 1.0}`` per the judge's instruction; on
    any API failure, falls back to :func:`compute_f1` so training is never
    blocked by a flaky external service.
    """
    client = _get_client()
    model = os.environ.get("LLM_JUDGE_MODEL", "gpt-4o-mini")
    t0 = time.time()
    input_tokens = output_tokens = 0
    try:
        resp = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": (
                        f"Question: {question}\n"
                        f"Ground truth answer: {ground_truth}\n"
                        f"Model prediction: {prediction}\n\n"
                        "Is the prediction semantically equivalent to the ground truth? "
                        "Reply with a single number: 1.0 if correct, 0.5 if partially correct, 0.0 if wrong."
                    ),
                }
            ],
            max_tokens=8,
            temperature=0,
        )
        score_str = resp.choices[0].message.content.strip()
        m = re.search(r"[\d.]+", score_str)
        if m is None:
            raise ValueError(f"non-numeric judge reply: {score_str!r}")
        score = float(m.group())
        if hasattr(resp, "usage") and resp.usage:
            input_tokens = getattr(resp.usage, "prompt_tokens", 0)
            output_tokens = getattr(resp.usage, "completion_tokens", 0)
        latency = time.time() - t0
        _tracker.record(input_tokens, output_tokens, latency)
        return min(max(score, 0.0), 1.0)
    except Exception as exc:  # noqa: BLE001 — judge failure must not crash training
        latency = time.time() - t0
        _tracker.record(input_tokens, output_tokens, latency, error=True)
        logger.warning(f"LLM judge failed in {latency:.1f}s: {exc}; falling back to F1")
        return compute_f1(prediction, ground_truth)


def get_tracker() -> APIUsageTracker:
    """Module-level tracker singleton (used by trainer for end-of-run logging)."""
    return _tracker
