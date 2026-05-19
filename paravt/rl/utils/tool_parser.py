"""Compatibility shim for tool parsing classes originally from vllm.
Avoids direct vllm import which may fail due to torch ABI mismatch."""

import ast
import json
import re
from dataclasses import dataclass, field


@dataclass
class FunctionCall:
    name: str
    arguments: str

@dataclass
class ToolCall:
    type: str = "function"
    function: FunctionCall = None

@dataclass
class ExtractedToolCallInformation:
    tools_called: bool = False
    tool_calls: list = field(default_factory=list)
    content: str | None = None

# Default positional arg mapping for known tools
_POSITIONAL_ARG_MAP = {
    "crop_video": ["video_path", "start_time", "end_time"],
}

def _parse_python_function_call(code_text: str) -> list[dict]:
    """Parse Python function call syntax into list of {name, arguments} dicts.

    Handles:
      - crop_video(video_path="x", start_time=10, end_time=20)  # keyword args
      - crop_video("x", 10, 20)  # positional args (mapped via _POSITIONAL_ARG_MAP)
      - print(crop_video(...))  # unwraps print wrapper
      - Stray trailing quotes: crop_video(..., end_time=20.0')  # common model error
    """
    code_text = code_text.strip()
    if not code_text:
        return []

    # Handle multiple statements (one per line)
    results = []
    for line in code_text.split("\n"):
        line = line.strip()
        if not line:
            continue
        try:
            tree = ast.parse(line, mode="eval")
        except SyntaxError:
            # Try fixing common model syntax errors
            cleaned = line
            # Remove stray quotes after closing paren: func(...)')  or func(...)'
            cleaned = re.sub(r"\)\s*['\"].*$", ")", cleaned)
            # Remove stray quotes before closing paren: ...=20.0')
            cleaned = re.sub(r"(['\"])(\s*\))", r"\2", cleaned)
            if cleaned != line:
                try:
                    tree = ast.parse(cleaned, mode="eval")
                except SyntaxError:
                    continue
            else:
                continue

        call = tree.body
        # Unwrap print() wrapper
        if isinstance(call, ast.Call) and isinstance(call.func, ast.Name) and call.func.id == "print":
            if call.args and isinstance(call.args[0], ast.Call):
                call = call.args[0]

        if not isinstance(call, ast.Call):
            continue

        # Get function name
        if isinstance(call.func, ast.Name):
            func_name = call.func.id
        elif isinstance(call.func, ast.Attribute):
            func_name = call.func.attr
        else:
            continue

        arguments = {}

        # Extract keyword arguments
        for kw in call.keywords:
            if kw.arg is None:
                continue
            try:
                arguments[kw.arg] = ast.literal_eval(kw.value)
            except (ValueError, TypeError):
                pass

        # Extract positional arguments (map to known param names)
        if call.args and not arguments:
            param_names = _POSITIONAL_ARG_MAP.get(func_name, [])
            for i, arg in enumerate(call.args):
                try:
                    val = ast.literal_eval(arg)
                    if i < len(param_names):
                        arguments[param_names[i]] = val
                except (ValueError, TypeError):
                    pass

        if func_name and arguments:
            results.append({"name": func_name, "arguments": arguments})

    return results


class Hermes2ProToolParser:
    """Tool call parser supporting both <tool_call> JSON and <tool_code> Python formats."""
    def __init__(self, tokenizer=None):
        self.tool_call_start_token = "<tool_call>"
        self.tool_call_end_token = "</tool_call>"
        self.tool_call_regex = re.compile(
            r"<tool_call>(.*?)</tool_call>|<tool_call>(.*)", re.DOTALL
        )
        self.tool_code_regex = re.compile(
            r"<tool_code>(.*?)</tool_code>|<tool_code>(.*)", re.DOTALL
        )

    def extract_tool_calls(self, model_output: str, request=None) -> ExtractedToolCallInformation:
        # 1. Try <tool_call> JSON format first (SFT-trained format)
        if self.tool_call_start_token in model_output:
            try:
                function_call_tuples = self.tool_call_regex.findall(model_output)
                raw_function_calls = [
                    json.loads(match[0] if match[0] else match[1])
                    for match in function_call_tuples
                ]
                tool_calls = [
                    ToolCall(
                        type="function",
                        function=FunctionCall(
                            name=function_call["name"],
                            arguments=json.dumps(
                                function_call["arguments"], ensure_ascii=False
                            ),
                        ),
                    )
                    for function_call in raw_function_calls
                ]
                content = model_output[: model_output.find(self.tool_call_start_token)]
                return ExtractedToolCallInformation(
                    tools_called=True,
                    tool_calls=tool_calls,
                    content=content if content else None,
                )
            except Exception:
                pass

        # 2. Try <tool_code> Python function call format (Qwen3-VL native format)
        if "<tool_code>" in model_output:
            try:
                code_matches = self.tool_code_regex.findall(model_output)
                tool_calls = []
                for match in code_matches:
                    code_text = match[0] if match[0] else match[1]
                    parsed = _parse_python_function_call(code_text)
                    for call_info in parsed:
                        tool_calls.append(
                            ToolCall(
                                type="function",
                                function=FunctionCall(
                                    name=call_info["name"],
                                    arguments=json.dumps(
                                        call_info["arguments"], ensure_ascii=False
                                    ),
                                ),
                            )
                        )
                if tool_calls:
                    content = model_output[: model_output.find("<tool_code>")]
                    return ExtractedToolCallInformation(
                        tools_called=True,
                        tool_calls=tool_calls,
                        content=content if content else None,
                    )
            except Exception:
                pass

        return ExtractedToolCallInformation(
            tools_called=False, tool_calls=[], content=model_output
        )
