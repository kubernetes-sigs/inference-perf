# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Generated tool names and arguments, excluding IDs and protocol/template wrappers."""

from __future__ import annotations

import json
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from inference_perf.utils.custom_tokenizer import CustomTokenizer


def chat_tool_calls(message: Any) -> list[dict[str, Any]]:
    """Normalize valid calls, using their position when a server omits the index.

    Ignore malformed entries and fields without discarding sibling text or calls.
    Both streamed accounting and replay reconstruction consume this representation.
    """
    if not isinstance(message, dict) or not isinstance(message.get("tool_calls"), list):
        return []
    calls = []
    for position, call in enumerate(message["tool_calls"]):
        if not isinstance(call, dict):
            continue
        function = call.get("function", {})
        # ID-only deltas can explicitly carry a null function. Keep their metadata.
        if function is None:
            function = {}
        index = call.get("index", position)
        if not isinstance(function, dict) or type(index) is not int or index < 0:
            continue
        calls.append(
            {
                "index": index,
                "id": call.get("id") if isinstance(call.get("id"), str) else "",
                "type": "function",
                "function": {
                    key: function.get(key) if isinstance(function.get(key), str) else "" for key in ("name", "arguments")
                },
            }
        )
    return calls


def chat_tool_call_deltas(data: dict[str, Any]) -> list[dict[str, Any]]:
    choices = data.get("choices")
    if isinstance(choices, list) and choices and isinstance(choices[0], dict):
        return chat_tool_calls(choices[0].get("delta"))
    return []


def tool_argument_deltas(data: dict[str, Any]) -> list[tuple[int, str]]:
    """Argument fragments keyed by call/block index; empty strings mark tool turns."""
    if data.get("choices"):
        return [(call["index"], call["function"]["arguments"]) for call in chat_tool_call_deltas(data)]
    index = data.get("index", 0)
    if type(index) is not int or index < 0:
        return []
    if data.get("type") == "content_block_start":
        block = data.get("content_block")
        if isinstance(block, dict) and block.get("type") == "tool_use":
            # The empty input object starts a streamed block; it is not a generated fragment.
            value = block.get("input")
            return [(index, json.dumps(value, ensure_ascii=False) if isinstance(value, dict) and value else "")]
    if data.get("type") == "content_block_delta":
        delta = data.get("delta")
        if isinstance(delta, dict) and delta.get("type") == "input_json_delta":
            fragment = delta.get("partial_json")
            return [(index, fragment if isinstance(fragment, str) else "")]
    return []


def tool_name_deltas(data: dict[str, Any]) -> list[tuple[int, str]]:
    """Generated name fragments, keyed independently from argument fragments."""
    if data.get("choices"):
        return [(call["index"], call["function"]["name"]) for call in chat_tool_call_deltas(data)]
    index = data.get("index", 0)
    block = data.get("content_block")
    if (
        data.get("type") == "content_block_start"
        and type(index) is int
        and index >= 0
        and isinstance(block, dict)
        and block.get("type") == "tool_use"
        and isinstance(block.get("name"), str)
    ):
        return [(index, block["name"])]
    return []


def message_tool_texts(message: dict[str, Any] | None) -> list[str]:
    """Names and arguments counted separately, including converted Messages."""
    return [call["function"][key] for call in chat_tool_calls(message) for key in ("name", "arguments")]


def count_tool_tokens(tokenizer: CustomTokenizer, texts: Iterable[str]) -> int:
    """Tokenize each assembled name and argument string separately, without special tokens."""
    return sum(tokenizer.count_tokens(argument, add_special_tokens=False) for argument in texts if argument)
