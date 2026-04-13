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
from typing import Any, Dict, List, Optional


@dataclass
class ReplayMessage:
    role: str
    text: Any


class ComplexReplayMessage(ReplayMessage):
    """Replay message carrying the original structured message representation."""

    def __init__(self, role: str, message_info: dict[str, Any], raw_reconstructed_text: str):
        super().__init__(role=role, text=raw_reconstructed_text)
        self.message_info = message_info


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

    type: str  # "shared" | "output" | "unique"
    message_count: int
    token_count: int
    source_event_id: Optional[str] = None


@dataclass
class GraphCall:
    """An LLM call within a graph event, ready for replay."""

    call_id: str
    model: str
    messages: List[ReplayMessage]
    expected_output: str
    input_segments: List[InputSegment]
    total_input_tokens: int
    expected_output_tokens: int
    temperature: Optional[float]
    max_tokens_recorded: Optional[int]


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
