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
"""Generated tool arguments, excluding names, IDs and protocol/template wrappers."""

from __future__ import annotations

import json
from typing import Any, Iterable, TYPE_CHECKING

if TYPE_CHECKING:
    from inference_perf.utils.custom_tokenizer import CustomTokenizer


def tool_argument_deltas(data: dict[str, Any]) -> list[tuple[int, str]]:
    """Argument fragments keyed by call/block index; empty strings mark tool turns."""
    if choices := data.get("choices"):
        calls = (choices[0].get("delta") or {}).get("tool_calls") or []
        return [(call.get("index", i), (call.get("function") or {}).get("arguments") or "") for i, call in enumerate(calls)]
    if data.get("type") == "content_block_start":
        block = data.get("content_block") or {}
        if block.get("type") == "tool_use":
            # The empty input object starts a streamed block; it is not a generated fragment.
            value = block.get("input")
            return [(data.get("index", 0), json.dumps(value, ensure_ascii=False) if value else "")]
    if data.get("type") == "content_block_delta":
        delta = data.get("delta") or {}
        if delta.get("type") == "input_json_delta":
            return [(data.get("index", 0), delta.get("partial_json") or "")]
    return []


def message_tool_arguments(message: dict[str, Any] | None) -> list[str]:
    """Arguments from an OpenAI-shaped assistant message, including converted Messages."""
    return [(call.get("function") or {}).get("arguments") or "" for call in (message or {}).get("tool_calls") or []]


def count_tool_tokens(tokenizer: CustomTokenizer, arguments: Iterable[str]) -> int:
    """Tokenize each assembled call separately, without sequence-start special tokens."""
    return sum(tokenizer.count_tokens(argument, add_special_tokens=False) for argument in arguments if argument)
