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

"""Tests for the `async_report` InputSegment primitive.

`async_report` replaces ONLY the `content` of a recorded role:"user" message with
a predecessor event's live output TEXT, preserving `role`. It models an async
sub-agent notification: the child's report arrives as its own user-role message,
decoupled from the dispatch tool_call_id entirely (the dispatch's own tool result
is a static content-free ack, so it needs no substitution).

It replaces the former `tool_output` segment type, which injected the child's
report into a role:"tool" slot -- a shape that did not match how a real async
Agent tool behaves.
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


def test_async_report_replaces_content_preserves_user_role():
    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()

    # spawn event produced an assistant tool-call message (the K dispatch calls)
    registry.record(
        "sessX:spawn",
        "irrelevant",
        messages=[],
        output_message={
            "role": "assistant",
            "tool_calls": [{"id": "call_A", "type": "function", "function": {"name": "dispatch_agent", "arguments": "{}"}}],
        },
    )
    # child terminal event produced live report TEXT
    registry.record(
        "sessX:child1",
        "the child's live report text",
        messages=[],
        output_message={"role": "assistant", "content": "the child's live report text"},
    )

    original_messages = [
        {
            "role": "assistant",
            "tool_calls": [{"id": "call_A", "type": "function", "function": {"name": "dispatch_agent", "arguments": "{}"}}],
        },
        # the dispatch's tool result: a STATIC ack, never substituted
        {"role": "tool", "tool_call_id": "call_A", "content": "Async agent launched successfully."},
        # the async notification carrying the child's report
        {"role": "user", "content": "PLACEHOLDER_ASYNC_REPORT"},
    ]
    ev = _make_api_data(
        event_id="sessX:notify0",
        registry=registry,
        tracker=tracker,
        original_messages=original_messages,
        input_segments=[
            InputSegment(type="output", message_count=1, token_count=5, source_event_id="sessX:spawn"),
            InputSegment(type="unique", message_count=1, token_count=5, source_event_id=None),
            InputSegment(type="async_report", message_count=1, token_count=5, source_event_id="sessX:child1"),
        ],
        predecessor_event_ids=["sessX:spawn", "sessX:child1"],
    )

    result = ev._build_messages_with_substitution()

    notif = result[2]
    assert notif["role"] == "user"  # role preserved
    assert "tool_call_id" not in notif  # no tool_call_id involved at all
    # content replaced with the child's live TEXT, wrapped in the notification envelope
    assert notif["content"] == ("<task-notification>\n<result>\nthe child's live report text\n</result>\n</task-notification>")
    # the report body is recoverable from inside <result>
    body = notif["content"].split("<result>\n", 1)[1].split("\n</result>", 1)[0]
    assert body == "the child's live report text"
    # no tool-use-id is emitted (reports arrive in completion order, so positional
    # pairing with a dispatch call would stamp the wrong id)
    assert "<tool-use-id>" not in notif["content"]

    # the dispatch's static ack is untouched by substitution
    assert result[1]["role"] == "tool"
    assert result[1]["content"] == "Async agent launched successfully."


def test_async_report_guard_non_user_role_falls_back():
    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()

    registry.record(
        "sessX:child1",
        "report",
        messages=[],
        output_message={"role": "assistant", "content": "report"},
    )
    # A role:"tool" message is now the WRONG target for async_report (it was the
    # right one for the removed `tool_output` type), so the guard must fire.
    original_messages = [{"role": "tool", "tool_call_id": "call_A", "content": "static ack"}]
    ev = _make_api_data(
        event_id="sessX:e",
        registry=registry,
        tracker=tracker,
        original_messages=original_messages,
        input_segments=[InputSegment(type="async_report", message_count=1, token_count=5, source_event_id="sessX:child1")],
        predecessor_event_ids=["sessX:child1"],
    )

    result = ev._build_messages_with_substitution()

    assert result[0]["role"] == "tool"  # unchanged
    assert result[0]["content"] == "static ack"  # recorded content kept (guard fired)


def test_async_report_unavailable_output_falls_back():
    registry = EventOutputRegistry()  # nothing recorded for the source
    tracker = WorkerSessionTracker()

    original_messages = [{"role": "user", "content": "PLACEHOLDER_ASYNC_REPORT"}]
    ev = _make_api_data(
        event_id="sessX:e",
        registry=registry,
        tracker=tracker,
        original_messages=original_messages,
        input_segments=[InputSegment(type="async_report", message_count=1, token_count=5, source_event_id="sessX:missing")],
        predecessor_event_ids=["sessX:missing"],
    )

    result = ev._build_messages_with_substitution()

    assert result[0]["content"] == "PLACEHOLDER_ASYNC_REPORT"  # fell back, no crash


def test_output_and_shared_segments_unchanged_by_async_report_addition():
    """A graph with NO async_report segment must substitute exactly as before —
    the branch is additive and inert on the OTel/Weka path."""
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


def test_multiple_async_report_segments_do_not_double_advance_cursor():
    """Regression test for the segment-cursor class of bug.

    An event whose input_segments interleave output/unique/async_report over 5
    original_messages. If the async_report branch's success path incremented
    `cursor` itself AND fell through to the shared loop-tail increment, cursor
    would advance by 2 instead of 1 per async_report segment. That mis-slices the
    later segments in `self.original_messages[cursor : cursor + seg.message_count]`,
    eventually producing an empty `seg_msgs` list so `seg_msgs[0]` raises
    IndexError. This is the exact failure mode that broke ALL sub-agent fan-out
    replay under the former `tool_output` segment.
    """
    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()

    registry.record(
        "sessZ:spawn",
        "irrelevant",
        messages=[],
        output_message={
            "role": "assistant",
            "tool_calls": [
                {"id": "call_A", "type": "function", "function": {"name": "dispatch_agent", "arguments": "{}"}},
                {"id": "call_B", "type": "function", "function": {"name": "dispatch_agent", "arguments": "{}"}},
            ],
        },
    )
    registry.record(
        "sessZ:child1",
        "child1 live report",
        messages=[],
        output_message={"role": "assistant", "content": "child1 live report"},
    )
    registry.record(
        "sessZ:child2",
        "child2 live report",
        messages=[],
        output_message={"role": "assistant", "content": "child2 live report"},
    )

    original_messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "call_A", "type": "function", "function": {"name": "dispatch_agent", "arguments": "{}"}},
                {"id": "call_B", "type": "function", "function": {"name": "dispatch_agent", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "call_A", "content": "ack A"},
        {"role": "tool", "tool_call_id": "call_B", "content": "ack B"},
        {"role": "user", "content": "PLACEHOLDER_ASYNC_REPORT_1"},
        {"role": "user", "content": "PLACEHOLDER_ASYNC_REPORT_2"},
    ]

    ev = _make_api_data(
        event_id="sessZ:notify1",
        registry=registry,
        tracker=tracker,
        original_messages=original_messages,
        input_segments=[
            InputSegment(type="output", message_count=1, token_count=5, source_event_id="sessZ:spawn"),
            InputSegment(type="unique", message_count=2, token_count=5, source_event_id=None),
            InputSegment(type="async_report", message_count=1, token_count=5, source_event_id="sessZ:child1"),
            InputSegment(type="async_report", message_count=1, token_count=5, source_event_id="sessZ:child2"),
        ],
        predecessor_event_ids=["sessZ:spawn", "sessZ:child1", "sessZ:child2"],
    )

    result = ev._build_messages_with_substitution()  # must not raise IndexError

    assert len(result) == 5

    # each async_report slot got its OWN child's report, in order (no cursor drift),
    # each wrapped in its own notification envelope
    def _body(msg):
        return msg["content"].split("<result>\n", 1)[1].split("\n</result>", 1)[0]

    assert result[3]["role"] == "user"
    assert _body(result[3]) == "child1 live report"
    assert result[4]["role"] == "user"
    assert _body(result[4]) == "child2 live report"
    # one envelope per slot -- not one envelope wrapping both
    for m in (result[3], result[4]):
        assert m["content"].count("<task-notification>") == 1
        assert m["content"].count("<result>") == 1


def test_async_report_id_rewrite_still_applies_to_static_acks():
    """The `output` segment's tool_call_id post-pass must still rewrite the STATIC
    ack results to the live dispatch call ids.

    The acks carry no substituted content, but their `tool_call_id` values are the
    RECORDED ones and must be rewritten to the live model's actual call ids —
    otherwise every ack dangles. This is what keeps inv #3 true now that the
    child reports no longer ride the tool slots.
    """
    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()

    registry.record(
        "sessW:spawn",
        "irrelevant",
        messages=[],
        output_message={
            "role": "assistant",
            "tool_calls": [
                {"id": "LIVE_1", "type": "function", "function": {"name": "dispatch_agent", "arguments": "{}"}},
                {"id": "LIVE_2", "type": "function", "function": {"name": "dispatch_agent", "arguments": "{}"}},
            ],
        },
    )
    registry.record("sessW:child1", "r1", messages=[], output_message={"role": "assistant", "content": "r1"})

    original_messages = [
        {
            "role": "assistant",
            "tool_calls": [
                {"id": "recorded_1", "type": "function", "function": {"name": "dispatch_agent", "arguments": "{}"}},
                {"id": "recorded_2", "type": "function", "function": {"name": "dispatch_agent", "arguments": "{}"}},
            ],
        },
        {"role": "tool", "tool_call_id": "recorded_1", "content": "ack"},
        {"role": "tool", "tool_call_id": "recorded_2", "content": "ack"},
        {"role": "user", "content": "PLACEHOLDER_ASYNC_REPORT"},
    ]
    ev = _make_api_data(
        event_id="sessW:notify0",
        registry=registry,
        tracker=tracker,
        original_messages=original_messages,
        input_segments=[
            InputSegment(type="output", message_count=1, token_count=5, source_event_id="sessW:spawn"),
            InputSegment(type="unique", message_count=2, token_count=5, source_event_id=None),
            InputSegment(type="async_report", message_count=1, token_count=5, source_event_id="sessW:child1"),
        ],
        predecessor_event_ids=["sessW:spawn", "sessW:child1"],
    )

    result = ev._build_messages_with_substitution()

    call_ids = [tc["id"] for tc in result[0]["tool_calls"]]
    assert call_ids == ["LIVE_1", "LIVE_2"], "live dispatch calls substituted in"
    tool_ids = [m["tool_call_id"] for m in result if m.get("role") == "tool"]
    assert tool_ids == ["LIVE_1", "LIVE_2"], "static acks rewritten to the live call ids (no dangling)"


def test_bad_tool_call_handling_inherited_by_session_replay_base():
    from inference_perf.config.datagen.replay import SessionReplayConfig, BadToolCallHandling

    cfg = SessionReplayConfig()
    assert cfg.bad_tool_call_handling == BadToolCallHandling.NONE


def test_notification_envelope_shape_and_omissions():
    """The envelope wraps the report body and omits the fields we deliberately skip.

    Real harness notifications carry <task-id>, <tool-use-id>, <output-file>,
    <status> and a usage block alongside <result>. We reproduce the RESULT wrapper
    only. <tool-use-id> in particular is omitted on purpose: reports are delivered in
    child COMPLETION order, not dispatch order, so pairing a report with a dispatch
    call by position would stamp an id that does not belong to that child.
    """
    from inference_perf.datagen.replay_graph_session_datagen import _wrap_async_notification

    wrapped = _wrap_async_notification("REPORT BODY")
    assert wrapped == "<task-notification>\n<result>\nREPORT BODY\n</result>\n</task-notification>"
    # the body round-trips out of the envelope
    assert wrapped.split("<result>\n", 1)[1].split("\n</result>", 1)[0] == "REPORT BODY"
    # deliberately absent
    for omitted in ("<tool-use-id>", "<task-id>", "<output-file>", "<status>", "<usage>"):
        assert omitted not in wrapped, f"{omitted} must not be emitted"


def test_notification_envelope_survives_multiline_and_markup_reports():
    """A child report is free-form model text: it may be multi-line, contain markdown
    tables, or even mention tag-like text. The envelope must still delimit it so the
    body is recoverable."""
    from inference_perf.datagen.replay_graph_session_datagen import _wrap_async_notification

    body = "## Findings\n\n| a | b |\n|---|---|\n| 1 | 2 |\n\nMentions <result> in prose."
    wrapped = _wrap_async_notification(body)
    assert wrapped.startswith("<task-notification>\n<result>\n")
    assert wrapped.endswith("\n</result>\n</task-notification>")
    # slicing on the OUTER delimiters recovers the body even though it mentions <result>
    inner = wrapped[len("<task-notification>\n<result>\n") : -len("\n</result>\n</task-notification>")]
    assert inner == body


def test_dispatch_description_documents_the_envelope_and_ordering():
    """The dispatch tool definition is the ONLY place the async contract is stated
    (no directive message is injected after a notification), so it must describe both
    the envelope the reports arrive in and their completion-order delivery."""
    from inference_perf.datagen.synthetic_agentic import (
        DISPATCH_AGENT_DESCRIPTION,
        DISPATCH_AGENT_TOOL_DEF,
    )

    desc = DISPATCH_AGENT_DESCRIPTION
    assert "<task-notification>" in desc and "<result>" in desc, "envelope shape documented"
    assert "completion order" in desc.lower(), "delivery ordering documented"
    # reports arrive individually, not batched — the orchestrator must expect one at a time
    assert "one at a time" in desc.lower(), "per-report (non-batched) delivery documented"
    # advertised in BOTH schema positions so any client shape picks it up
    assert DISPATCH_AGENT_TOOL_DEF["description"] == desc
    assert DISPATCH_AGENT_TOOL_DEF["function"]["description"] == desc
