# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared replay graph domain types."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Literal, Optional


@dataclass
class ReplayMessage:
    role: str
    text: str


class ComplexReplayMessage(ReplayMessage):
    """Replay message carrying the original structured message representation."""

    def __init__(self, role: str, message_info: dict[str, Any], raw_reconstructed_text: str):
        super().__init__(role=role, text=raw_reconstructed_text)
        self.message_info = message_info

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, ComplexReplayMessage):
            return NotImplemented
        return self.role == other.role and self.text == other.text and self.message_info == other.message_info

    def __hash__(self) -> int:
        return hash((self.role, self.text))


@dataclass
class InputSegment:
    """One segment of an LLM call's input prompt, at message granularity.

    type:
        "shared"  — leading messages identical to a predecessor call's messages
                    (KV cache hit opportunity — these tokens are already cached)
        "output"  — a single assistant message whose content is a predecessor's output
                    (injected result; at replay time, substitute with actual generated text)
        "unique"  — messages unique to this call (no predecessor shares them)

    message_count: how many messages this segment covers
    token_count: estimated or recorded token count for this segment
    source_event_id: which predecessor event this segment comes from (shared/output only)
    """

    type: Literal["shared", "output", "unique"]
    message_count: int
    token_count: int
    source_event_id: Optional[str] = None


@dataclass
class GraphCall:
    """An LLM call within a graph event, ready for replay."""

    call_id: str
    model: str
    # Stored as plain dicts at graph-build time (not ReplayMessage objects) because
    # the graph is serialisable to JSON. Dict format: {"role": str, "content": str}.
    messages: List[Dict[str, Any]]
    expected_output: str
    input_segments: List[InputSegment]
    total_input_tokens: int
    expected_output_tokens: int
    temperature: Optional[float]
    max_tokens_recorded: Optional[int]
    tool_definitions: Optional[List[Dict[str, Any]]] = None
    # True when the recorded output was a tool call (not plain text). Used at
    # replay time to force tool_choice so the live model can't return plain text
    # where the original trace had a tool call — which would leave role:tool
    # messages in the successor with dangling tool_call_id references.
    expected_output_is_tool_call: bool = False
    # Function names from the recorded tool calls, in order. Used to force the
    # specific function when there is exactly one tool call; with multiple calls
    # we fall back to "required" since vLLM only accepts one name at a time.
    expected_output_tool_names: Optional[List[str]] = None


@dataclass
class GraphEvent:
    """An event in the replay graph. Contains exactly one LLM call."""

    event_id: str
    call: GraphCall
    predecessor_event_ids: List[str]
    predecessor_dependency_types: Dict[str, str]
    wait_ms: int
    t_start_ms: int
    t_end_ms: int


@dataclass
class ReplayGraph:
    """The complete replay graph for one combined trace/session source."""

    events: Dict[str, GraphEvent]
    root_event_ids: List[str]
    source_file: str
