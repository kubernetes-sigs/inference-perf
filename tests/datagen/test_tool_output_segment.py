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

"""Tests for the `tool_output` InputSegment primitive.

`tool_output` replaces ONLY the `content` of a recorded role:"tool" message
with a predecessor event's live output TEXT, preserving `role` and
`tool_call_id`. It is used for sub-agent fan-out merges.
"""

from __future__ import annotations

from inference_perf.datagen.replay_graph_types import InputSegment
from inference_perf.datagen.replay_graph_session_datagen import (
    EventOutputRegistry,
    SessionChatCompletionAPIData,
    WorkerSessionTracker,
)


def _make_api_data(event_id, registry, tracker, original_messages, input_segments, predecessor_event_ids):
    # SessionChatCompletionAPIData is a pydantic model; `messages`, `max_tokens`,
    # `worker_tracker`, `completion_queue`, and `total_events_in_session` have no
    # defaults and must be supplied, following the pattern used in
    # tests/test_otel_replay_datagen.py. `messages` itself is irrelevant here
    # since _build_messages_with_substitution reads from `original_messages`.
    return SessionChatCompletionAPIData(
        messages=[],
        max_tokens=50,
        event_id=event_id,
        registry=registry,
        worker_tracker=tracker,
        completion_queue=None,
        total_events_in_session=1,
        predecessor_event_ids=predecessor_event_ids,
        input_segments=input_segments,
        original_messages=original_messages,
    )


def test_tool_output_replaces_content_preserves_role_and_id():
    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()

    # dispatch event produced an assistant tool-call message
    registry.record(
        "sessX:dispatch1",
        "irrelevant",
        messages=[],
        output_message={
            "role": "assistant",
            "tool_calls": [{"id": "call_A", "type": "function", "function": {"name": "dispatch_agent", "arguments": "{}"}}],
        },
    )
    # child terminal event produced live answer TEXT
    registry.record(
        "sessX:child1",
        "the child's live answer text",
        messages=[],
        output_message={"role": "assistant", "content": "the child's live answer text"},
    )

    original_messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "call_A", "type": "function", "function": {"name": "dispatch_agent", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_A", "content": "PLACEHOLDER"},
    ]
    ev = _make_api_data(
        event_id="sessX:merge",
        registry=registry,
        tracker=tracker,
        original_messages=original_messages,
        input_segments=[
            InputSegment(type="output", message_count=1, token_count=5, source_event_id="sessX:dispatch1"),
            InputSegment(type="tool_output", message_count=1, token_count=5, source_event_id="sessX:child1"),
        ],
        predecessor_event_ids=["sessX:dispatch1", "sessX:child1"],
    )

    result = ev._build_messages_with_substitution()

    tool_msg = result[1]
    assert tool_msg["role"] == "tool"  # role preserved
    assert tool_msg["tool_call_id"] == "call_A"  # id preserved
    assert tool_msg["content"] == "the child's live answer text"  # content replaced with TEXT


def test_tool_output_guard_non_tool_role_falls_back():
    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()

    registry.record(
        "sessX:child1",
        "answer",
        messages=[],
        output_message={"role": "assistant", "content": "answer"},
    )
    original_messages = [{"role": "assistant", "content": "not a tool msg"}]
    ev = _make_api_data(
        event_id="sessX:e",
        registry=registry,
        tracker=tracker,
        original_messages=original_messages,
        input_segments=[InputSegment(type="tool_output", message_count=1, token_count=5, source_event_id="sessX:child1")],
        predecessor_event_ids=["sessX:child1"],
    )

    result = ev._build_messages_with_substitution()

    assert result[0]["role"] == "assistant"  # unchanged
    assert result[0]["content"] == "not a tool msg"  # recorded content kept (guard fired)


def test_tool_output_unavailable_output_falls_back():
    registry = EventOutputRegistry()  # nothing recorded for the source
    tracker = WorkerSessionTracker()

    original_messages = [{"role": "tool", "tool_call_id": "call_A", "content": "PLACEHOLDER"}]
    ev = _make_api_data(
        event_id="sessX:e",
        registry=registry,
        tracker=tracker,
        original_messages=original_messages,
        input_segments=[InputSegment(type="tool_output", message_count=1, token_count=5, source_event_id="sessX:missing")],
        predecessor_event_ids=["sessX:missing"],
    )

    result = ev._build_messages_with_substitution()

    assert result[0]["content"] == "PLACEHOLDER"  # fell back, no crash
