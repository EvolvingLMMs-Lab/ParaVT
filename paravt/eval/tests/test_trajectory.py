"""Dependency-free CPU tests for ``--save_trajectory`` (no GPU / vLLM / model).

The eval driver's heavy imports (openai, qwen-vl-utils, PIL decode) live in
``paravt.eval.utils``; here we stub that module and feed a scripted fake
OpenAI client, then exercise the *real* ``eval_one`` / ``_make_result`` /
``_trajectory_view`` / ``_redact_content`` logic. Runs under pytest or as a
plain script::

    python paravt/eval/tests/test_trajectory.py
"""
import json
import os
import re
import sys
import types

FAKE_B64 = "data:image/png;base64," + "A" * 256  # must never leak into a trajectory


def _stub_utils():
    utils = types.ModuleType("paravt.eval.utils")
    utils.base_result = lambda s: {
        "id": s.get("id", "?"), "answer": str(s.get("answer", "")),
        "pred": "", "correct": False, "tool_calls": 0, "score": 0.0,
    }
    utils.get_frames = lambda video_path, nframes=64: ["FRAME"] * nframes
    utils.pil_to_b64 = lambda img: FAKE_B64
    utils.crop_frames = lambda video_path, st, et: ["CROP"] * 4
    utils.parse_time_val = lambda x: (float(x) if str(x).replace(".", "", 1).lstrip("-").isdigit() else 0.0)
    utils.extract_time_from_response = lambda text: None
    utils.temporal_iou = lambda pred, gt: 0.0

    def extract_letter(text):
        m = re.search(r"<answer>\s*([A-Z])", text)
        if m:
            return m.group(1)
        found = re.findall(r"\b([A-F])\b", text)
        return found[-1] if found else ""

    utils.extract_letter = extract_letter
    return utils


def _load_driver():
    """Load the real driver.py under a stubbed ``paravt.eval.utils``."""
    if "paravt.eval.driver" in sys.modules:
        return sys.modules["paravt.eval.driver"]
    utils = _stub_utils()
    pkg = sys.modules.get("paravt") or types.ModuleType("paravt")
    pkg.__path__ = getattr(pkg, "__path__", [])
    evalpkg = types.ModuleType("paravt.eval")
    evalpkg.__path__ = []
    evalpkg.utils = utils
    sys.modules["paravt"] = pkg
    sys.modules["paravt.eval"] = evalpkg
    sys.modules["paravt.eval.utils"] = utils

    path = os.path.join(os.path.dirname(__file__), "..", "driver.py")
    driver = types.ModuleType("paravt.eval.driver")
    driver.__file__ = path
    sys.modules["paravt.eval.driver"] = driver
    with open(path) as f:
        exec(compile(f.read(), path, "exec"), driver.__dict__)
    return driver


DRIVER = _load_driver()
HERE = os.path.abspath(__file__)  # an existing path so os.path.exists(video_path) is True


class _FakeClient:
    def __init__(self, scripted):
        self.scripted = list(scripted)
        outer = self

        class _Comp:
            def create(self, **kw):
                content = outer.scripted.pop(0)
                return types.SimpleNamespace(
                    choices=[types.SimpleNamespace(message=types.SimpleNamespace(content=content))]
                )

        self.chat = types.SimpleNamespace(completions=_Comp())


def _mcq(ans="B"):
    return {"id": "v1", "video_path": HERE, "question": "Q? A.x B.y", "answer": ans, "is_mcq": True}


_TURN0 = ('<think>The key moment is near the start, crop it.</think>\n'
          '<tool_call>\n{"name": "crop_video", "arguments": {"start_time": 0, "end_time": 10}}\n</tool_call>')
_ANSWER = '<think>Now it is clear.</think>\n<answer>B</answer>'


def test_agentic_trajectory_captures_reasoning_that_full_response_drops():
    r = DRIVER.eval_one(_FakeClient([_TURN0, _ANSWER]), "m", _mcq("B"), nframes=8, max_turns=3,
                        prompt_mode="agentic_general", save_trajectory=True)
    blob = json.dumps(r["trajectory"])
    assert r["tool_calls"] == 1
    assert [m["role"] for m in r["trajectory"]] == ["system", "user", "assistant", "user", "assistant"]
    assert "key moment is near the start" in blob       # turn-0 reasoning IS captured
    assert "crop_video" in blob and "<tool_response>" in blob
    assert r["trajectory"][-1]["content"].endswith("<answer>B</answer>")
    # the omission this fixes: full_response keeps ONLY the final turn
    assert r["full_response"] == _ANSWER
    assert "key moment is near the start" not in r["full_response"]


def test_trajectory_has_no_base64():
    r = DRIVER.eval_one(_FakeClient([_TURN0, _ANSWER]), "m", _mcq("B"), nframes=64, max_turns=3,
                        prompt_mode="agentic_general", save_trajectory=True)
    assert FAKE_B64 not in json.dumps(r)
    assert "<64 frame(s) omitted>" in r["trajectory"][1]["content"]


def test_off_by_default_is_backward_compatible():
    r = DRIVER.eval_one(_FakeClient([_TURN0, _ANSWER]), "m", _mcq("B"), nframes=8, max_turns=3,
                        prompt_mode="agentic_general", save_trajectory=False)
    assert "trajectory" not in r
    assert "full_response" in r and r["tool_calls"] == 1


def test_flag_is_additive_and_invariant():
    scripted = [_TURN0, _ANSWER]
    off = DRIVER.eval_one(_FakeClient(list(scripted)), "m", _mcq("B"), nframes=8, max_turns=3,
                          prompt_mode="agentic_general", save_trajectory=False)
    on = DRIVER.eval_one(_FakeClient(list(scripted)), "m", _mcq("B"), nframes=8, max_turns=3,
                         prompt_mode="agentic_general", save_trajectory=True)
    assert {k: v for k, v in on.items() if k != "trajectory"} == off


def test_forced_final_is_answer_only_but_reasoning_is_in_trajectory():
    tcall = ('<think>One more crop.</think>\n<tool_call>\n'
             '{"name": "crop_video", "arguments": {"start_time": 1, "end_time": 2}}\n</tool_call>')
    r = DRIVER.eval_one(_FakeClient([tcall, tcall, tcall, "B"]), "m", _mcq("B"), nframes=8, max_turns=3,
                        prompt_mode="agentic_general", save_trajectory=True)
    assert r["full_response"] == "B"                    # forced "ONLY the letter" closer
    assert "One more crop" in json.dumps(r["trajectory"])  # earlier reasoning preserved


def test_redact_content_collapses_media_keeps_text():
    content = [{"type": "image_url", "image_url": {"url": FAKE_B64}} for _ in range(8)]
    content.append({"type": "text", "text": "<tool_response>\n8 frames from 0.0s to 5.0s.\n</tool_response>"})
    red = DRIVER._redact_content(content)
    assert isinstance(red, str)
    assert red.startswith("<8 frame(s) omitted>")
    assert "8 frames from 0.0s to 5.0s" in red and "AAAA" not in red
    assert DRIVER._redact_content("plain") == "plain"


def test_redact_content_labels_video_channel():
    content = [{"type": "video_url", "video_url": {"url": "file:///x.mp4"}},
               {"type": "text", "text": "When does X happen?"}]
    red = DRIVER._redact_content(content)
    assert red.startswith("<video omitted>")        # a whole video, not "1 frame"
    assert "file://" not in red and "When does X happen?" in red


def test_max_turns_zero_writes_no_empty_assistant_turn():
    tcall = ('<think>crop.</think>\n<tool_call>\n'
             '{"name": "crop_video", "arguments": {"start_time": 0, "end_time": 5}}\n</tool_call>')
    r = DRIVER.eval_one(_FakeClient([tcall]), "m", _mcq("B"), nframes=8, max_turns=0,
                        prompt_mode="agentic_general", save_trajectory=True)
    assert {"role": "assistant", "content": ""} not in r["trajectory"]


if __name__ == "__main__":
    fns = [v for k, v in sorted(globals().items()) if k.startswith("test_") and callable(v)]
    failures = 0
    for fn in fns:
        try:
            fn()
            print("PASS  " + fn.__name__)
        except AssertionError as e:
            failures += 1
            print("FAIL  " + fn.__name__ + "  -- " + repr(e))
    print("\n%d passed, %d failed" % (len(fns) - failures, failures))
    sys.exit(1 if failures else 0)
