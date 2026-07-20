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


def test_output_and_shared_segments_unchanged_by_tool_output_addition():
    """A graph with NO tool_output segment must substitute exactly as before —
    the new branch is additive and inert on the OTel/Weka path."""
    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()

    registry.record("sessY:e1", "live-out", messages=[], output_message={"role": "assistant", "content": "live-out"})
    original_messages = [{"role": "assistant", "content": "PLACEHOLDER"}]
    ev = _make_api_data(
        event_id="sessY:e2",
        registry=registry,
        tracker=tracker,
        original_messages=original_messages,
        input_segments=[InputSegment(type="output", message_count=1, token_count=5, source_event_id="sessY:e1")],
        predecessor_event_ids=["sessY:e1"],
    )
    result = ev._build_messages_with_substitution()
    # output segment still substitutes the WHOLE message (assistant), as before
    assert result[0]["role"] == "assistant"
    assert result[0]["content"] == "live-out"


def test_multiple_tool_output_segments_do_not_double_advance_cursor():
    """Regression test for the fan-out merge cursor bug.

    A merge event whose input_segments alternate output/tool_output —
    [output(dispatch1), tool_output(child1), output(dispatch2), tool_output(child2)] —
    over 4 original_messages. Before the fix, the tool_output branch's success
    path incremented `cursor` itself AND fell through to the shared loop-tail
    increment, advancing cursor by 2 instead of 1 per tool_output segment. That
    mis-slices the later segments in `self.original_messages[cursor : cursor +
    seg.message_count]`, eventually producing an empty `seg_msgs` list so
    `seg_msgs[0]` raises IndexError. This reproduces the crash reported for ALL
    sub-agent fan-out replay.
    """
    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()

    registry.record(
        "sessZ:dispatch1",
        "irrelevant",
        messages=[],
        output_message={
            "role": "assistant",
            "tool_calls": [{"id": "call_A", "type": "function", "function": {"name": "dispatch_agent", "arguments": "{}"}}],
        },
    )
    registry.record(
        "sessZ:child1",
        "child1 live answer",
        messages=[],
        output_message={"role": "assistant", "content": "child1 live answer"},
    )
    registry.record(
        "sessZ:dispatch2",
        "irrelevant",
        messages=[],
        output_message={
            "role": "assistant",
            "tool_calls": [{"id": "call_B", "type": "function", "function": {"name": "dispatch_agent", "arguments": "{}"}}],
        },
    )
    registry.record(
        "sessZ:child2",
        "child2 live answer",
        messages=[],
        output_message={"role": "assistant", "content": "child2 live answer"},
    )

    original_messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "call_A", "type": "function", "function": {"name": "dispatch_agent", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_A", "content": "PLACEHOLDER_A"},
        {
            "role": "assistant",
            "tool_calls": [{"id": "call_B", "type": "function", "function": {"name": "dispatch_agent", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_B", "content": "PLACEHOLDER_B"},
    ]

    ev = _make_api_data(
        event_id="sessZ:merge",
        registry=registry,
        tracker=tracker,
        original_messages=original_messages,
        input_segments=[
            InputSegment(type="output", message_count=1, token_count=5, source_event_id="sessZ:dispatch1"),
            InputSegment(type="tool_output", message_count=1, token_count=5, source_event_id="sessZ:child1"),
            InputSegment(type="output", message_count=1, token_count=5, source_event_id="sessZ:dispatch2"),
            InputSegment(type="tool_output", message_count=1, token_count=5, source_event_id="sessZ:child2"),
        ],
        predecessor_event_ids=["sessZ:dispatch1", "sessZ:child1", "sessZ:dispatch2", "sessZ:child2"],
    )

    result = ev._build_messages_with_substitution()  # must not raise IndexError

    assert len(result) == 4

    tool_msg_a = result[1]
    assert tool_msg_a["role"] == "tool"
    assert tool_msg_a["tool_call_id"] == "call_A"
    assert tool_msg_a["content"] == "child1 live answer"

    tool_msg_b = result[3]
    assert tool_msg_b["role"] == "tool"
    assert tool_msg_b["tool_call_id"] == "call_B"
    assert tool_msg_b["content"] == "child2 live answer"


def test_bad_tool_call_handling_inherited_by_session_replay_base():
    from inference_perf.config.datagen.replay import SessionReplayConfig, BadToolCallHandling

    cfg = SessionReplayConfig()
    assert cfg.bad_tool_call_handling == BadToolCallHandling.NONE
