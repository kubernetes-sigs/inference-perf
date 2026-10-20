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

"""Unit tests for _event_to_toolace() in synthetic_agentic_to_replay_graph.py."""

import json
from typing import Any, Dict, List, Optional

from inference_perf.datagen.synthetic_agentic.synthetic_agentic_to_replay_graph import _event_to_toolace
from inference_perf.datagen.replay.replay_graph_types import GraphCall, GraphEvent, InputSegment, ReplayGraph

_EMPTY_GRAPH = ReplayGraph(events={}, root_event_ids=[], source_file="test")


def _event(call: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    event: Dict[str, Any] = {"call": call}
    event.update(extra)
    return event


def _graph_call(
    messages: List[Dict[str, Any]],
    expected_output: str = "",
    input_segments: Optional[List[InputSegment]] = None,
    expected_output_is_tool_call: bool = False,
    expected_output_tool_names: Optional[List[str]] = None,
) -> GraphCall:
    return GraphCall(
        call_id="c",
        model="",
        messages=messages,
        expected_output=expected_output,
        input_segments=input_segments or [],
        total_input_tokens=0,
        expected_output_tokens=0,
        temperature=None,
        max_tokens_recorded=None,
        expected_output_is_tool_call=expected_output_is_tool_call,
        expected_output_tool_names=expected_output_tool_names,
    )


def _graph(events: Dict[str, GraphCall]) -> ReplayGraph:
    """Build a ReplayGraph from {event_id: GraphCall}, filling in trivial GraphEvent fields."""
    return ReplayGraph(
        events={
            eid: GraphEvent(
                event_id=eid,
                call=call,
                predecessor_event_ids=[],
                predecessor_dependency_types={},
                wait_ms=0,
                t_start_ms=0,
                t_end_ms=0,
            )
            for eid, call in events.items()
        },
        root_event_ids=[],
        source_file="test",
    )


def test_plain_conversation_becomes_human_and_gpt_turns() -> None:
    call = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ],
    }
    record = _event_to_toolace("evt-1", _event(call), _EMPTY_GRAPH)

    assert record["system"] == "You are a helpful assistant."
    assert record["conversations"] == [
        {"from": "human", "value": "hello"},
        {"from": "gpt", "value": "hi there"},
    ]
    assert record["tools"] == "[]"


def test_no_leading_system_message_leaves_system_empty() -> None:
    call = {"messages": [{"role": "user", "content": "hello"}]}
    record = _event_to_toolace("evt-1", _event(call), _EMPTY_GRAPH)

    assert record["system"] == ""
    assert record["conversations"] == [{"from": "human", "value": "hello"}]


def test_tool_call_and_observation_turns() -> None:
    call = {
        "messages": [
            {"role": "system", "content": "sys"},
            {"role": "user", "content": "what's the weather?"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'},
                    }
                ],
            },
            {"role": "tool", "tool_call_id": "call_1", "content": "sunny, 72F"},
            {"role": "assistant", "content": "It's sunny and 72F in NYC."},
        ],
        "tool_definitions": [{"name": "get_weather", "parameters": {}}],
    }
    record = _event_to_toolace("evt-1", _event(call), _EMPTY_GRAPH)

    assert record["conversations"] == [
        {"from": "human", "value": "what's the weather?"},
        {
            "from": "function_call",
            "value": json.dumps([{"name": "get_weather", "arguments": '{"city": "NYC"}'}]),
        },
        {
            "from": "observation",
            "value": json.dumps([{"name": "get_weather", "results": "sunny, 72F"}]),
        },
        {"from": "gpt", "value": "It's sunny and 72F in NYC."},
    ]
    assert record["tools"] == json.dumps([{"name": "get_weather", "parameters": {}}])


def test_multiple_tool_calls_matched_by_tool_call_id() -> None:
    call = {
        "messages": [
            {"role": "user", "content": "compare cities"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'}},
                    {"id": "call_2", "function": {"name": "get_weather", "arguments": '{"city": "LA"}'}},
                ],
            },
            {"role": "tool", "tool_call_id": "call_2", "content": "sunny, 80F"},
            {"role": "tool", "tool_call_id": "call_1", "content": "rainy, 55F"},
        ],
    }
    record = _event_to_toolace("evt-1", _event(call), _EMPTY_GRAPH)

    function_calls = [c for c in record["conversations"] if c["from"] == "function_call"]
    observation = next(c for c in record["conversations"] if c["from"] == "observation")

    assert len(function_calls) == 1
    # Results are matched to the tool name via tool_call_id, regardless of message order.
    assert json.loads(observation["value"]) == [
        {"name": "get_weather", "results": "rainy, 55F"},
        {"name": "get_weather", "results": "sunny, 80F"},
    ]


def test_tool_call_without_matching_id_falls_back_positionally() -> None:
    call = {
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "function": {"name": "lookup", "arguments": "{}"}}],
            },
            # tool_call_id doesn't match any tool_calls[].id -> positional fallback.
            {"role": "tool", "tool_call_id": "unknown_id", "content": "result"},
        ],
    }
    record = _event_to_toolace("evt-1", _event(call), _EMPTY_GRAPH)

    observation = next(c for c in record["conversations"] if c["from"] == "observation")
    assert json.loads(observation["value"]) == [{"name": "", "results": "result"}]


def test_multiple_unkeyed_tool_results_are_not_collapsed() -> None:
    call = {
        "messages": [
            {"role": "user", "content": "compare cities"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'}},
                    {"id": "call_2", "function": {"name": "get_weather", "arguments": '{"city": "LA"}'}},
                ],
            },
            # Neither tool message carries a tool_call_id (e.g. an OTel part with no
            # recorded id) -- both must survive in message order, not collapse onto "".
            {"role": "tool", "content": "rainy, 55F"},
            {"role": "tool", "content": "sunny, 80F"},
        ],
    }
    record = _event_to_toolace("evt-1", _event(call), _EMPTY_GRAPH)

    observation = next(c for c in record["conversations"] if c["from"] == "observation")
    assert json.loads(observation["value"]) == [
        {"name": "", "results": "rainy, 55F"},
        {"name": "", "results": "sunny, 80F"},
    ]


def test_one_empty_result_does_not_blank_other_matched_names() -> None:
    call = {
        "messages": [
            {"role": "user", "content": "compare cities"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [
                    {"id": "call_1", "function": {"name": "get_weather", "arguments": '{"city": "NYC"}'}},
                    {"id": "call_2", "function": {"name": "get_weather", "arguments": '{"city": "LA"}'}},
                ],
            },
            # call_2's result is legitimately empty, but both ids are still present
            # and matchable -- names should not be discarded for either result.
            {"role": "tool", "tool_call_id": "call_1", "content": "rainy, 55F"},
            {"role": "tool", "tool_call_id": "call_2", "content": ""},
        ],
    }
    record = _event_to_toolace("evt-1", _event(call), _EMPTY_GRAPH)

    observation = next(c for c in record["conversations"] if c["from"] == "observation")
    assert json.loads(observation["value"]) == [
        {"name": "get_weather", "results": "rainy, 55F"},
        {"name": "get_weather", "results": ""},
    ]


def test_assistant_tool_call_with_no_following_tool_messages_emits_no_observation() -> None:
    call = {
        "messages": [
            {"role": "user", "content": "hi"},
            {
                "role": "assistant",
                "content": "",
                "tool_calls": [{"id": "call_1", "function": {"name": "lookup", "arguments": "{}"}}],
            },
        ],
    }
    record = _event_to_toolace("evt-1", _event(call), _EMPTY_GRAPH)

    assert [c["from"] for c in record["conversations"]] == ["human", "function_call"]


def test_expected_output_appended_as_final_gpt_turn() -> None:
    call = {
        "messages": [{"role": "user", "content": "hi"}],
        "expected_output": "final answer",
    }
    record = _event_to_toolace("evt-1", _event(call), _EMPTY_GRAPH)

    assert record["conversations"][-1] == {"from": "gpt", "value": "final answer"}


def test_empty_expected_output_adds_no_extra_turn() -> None:
    call = {
        "messages": [{"role": "user", "content": "hi"}],
        "expected_output": "",
    }
    record = _event_to_toolace("evt-1", _event(call), _EMPTY_GRAPH)

    assert record["conversations"] == [{"from": "human", "value": "hi"}]


def test_metadata_captures_event_and_call_fields() -> None:
    call = {
        "messages": [{"role": "user", "content": "hi"}],
        "expected_output_tokens": 42,
        "input_segments": [{"type": "unique"}],
        "temperature": 0.7,
        "model": "test-model",
    }
    event = _event(
        call,
        predecessor_event_ids=["evt-0"],
        predecessor_dependency_types={"evt-0": "shared"},
    )
    record = _event_to_toolace("evt-1", event, _EMPTY_GRAPH)

    assert record["metadata"] == {
        "event_id": "evt-1",
        "predecessor_event_ids": ["evt-0"],
        "predecessor_dependency_types": {"evt-0": "shared"},
        "expected_output_tokens": 42,
        "input_segments": [{"type": "unique"}],
        "temperature": 0.7,
        "model": "test-model",
    }


def test_metadata_defaults_when_predecessors_missing() -> None:
    call = {"messages": [{"role": "user", "content": "hi"}]}
    record = _event_to_toolace("evt-1", _event(call), _EMPTY_GRAPH)

    assert record["metadata"]["predecessor_event_ids"] == []
    assert record["metadata"]["predecessor_dependency_types"] == {}
    assert record["metadata"]["input_segments"] == []


def test_expected_output_is_tool_call_flag_included_only_when_true() -> None:
    call = {
        "messages": [{"role": "user", "content": "hi"}],
        "expected_output_is_tool_call": True,
        "expected_output_tool_names": ["get_weather"],
    }
    record = _event_to_toolace("evt-1", _event(call), _EMPTY_GRAPH)

    assert record["metadata"]["expected_output_is_tool_call"] is True
    assert record["metadata"]["expected_output_tool_names"] == ["get_weather"]


def test_expected_output_is_tool_call_flag_omitted_when_falsy() -> None:
    call = {
        "messages": [{"role": "user", "content": "hi"}],
        "expected_output_is_tool_call": False,
    }
    record = _event_to_toolace("evt-1", _event(call), _EMPTY_GRAPH)

    assert "expected_output_is_tool_call" not in record["metadata"]
    assert "expected_output_tool_names" not in record["metadata"]


def test_stray_tool_message_without_preceding_tool_call_is_skipped() -> None:
    call = {
        "messages": [
            {"role": "user", "content": "hi"},
            {"role": "tool", "tool_call_id": "x", "content": "orphan"},
            {"role": "assistant", "content": "done"},
        ],
    }
    record = _event_to_toolace("evt-1", _event(call), _EMPTY_GRAPH)

    assert record["conversations"] == [
        {"from": "human", "value": "hi"},
        {"from": "gpt", "value": "done"},
    ]


def test_output_segment_placeholder_resolved_to_source_expected_output() -> None:
    # evt-0's real answer must replace the PLACEHOLDER_PRIOR_ANSWER text sitting
    # in evt-1's recorded `output` segment slot, exactly like replay-time
    # substitution (_build_messages_with_substitution) does.
    graph = _graph(
        {
            "evt-0": _graph_call(
                messages=[{"role": "user", "content": "task"}],
                expected_output="the real prior answer",
            ),
            "evt-1": _graph_call(
                messages=[
                    {"role": "user", "content": "task"},
                    {"role": "assistant", "content": "PLACEHOLDER_PRIOR_ANSWER"},
                    {"role": "user", "content": "follow up"},
                ],
                input_segments=[
                    InputSegment(type="shared", message_count=1, token_count=0, source_event_id="evt-0"),
                    InputSegment(type="output", message_count=1, token_count=0, source_event_id="evt-0"),
                    InputSegment(type="unique", message_count=1, token_count=0, source_event_id=None),
                ],
            ),
        }
    )
    call = {
        "messages": graph.events["evt-1"].call.messages,
        "input_segments": [
            {"type": "shared", "message_count": 1, "source_event_id": "evt-0"},
            {"type": "output", "message_count": 1, "source_event_id": "evt-0"},
            {"type": "unique", "message_count": 1},
        ],
    }
    record = _event_to_toolace("evt-1", _event(call), graph)

    assert record["conversations"] == [
        {"from": "human", "value": "task"},
        {"from": "gpt", "value": "the real prior answer"},
        {"from": "human", "value": "follow up"},
    ]


def test_async_report_segment_placeholder_resolved_to_source_expected_output() -> None:
    graph = _graph(
        {
            "child-term": _graph_call(
                messages=[{"role": "user", "content": "child task"}],
                expected_output="child's real report",
            ),
            "evt-1": _graph_call(
                messages=[
                    {"role": "user", "content": "delegating"},
                    {"role": "user", "content": "PLACEHOLDER_ASYNC_REPORT"},
                ],
                input_segments=[
                    InputSegment(type="unique", message_count=1, token_count=0, source_event_id=None),
                    InputSegment(type="async_report", message_count=1, token_count=0, source_event_id="child-term"),
                ],
            ),
        }
    )
    call = {
        "messages": graph.events["evt-1"].call.messages,
        "input_segments": [
            {"type": "unique", "message_count": 1},
            {"type": "async_report", "message_count": 1, "source_event_id": "child-term"},
        ],
    }
    record = _event_to_toolace("evt-1", _event(call), graph)

    assert record["conversations"] == [
        {"from": "human", "value": "delegating"},
        {"from": "human", "value": "child's real report"},
    ]


def test_output_segment_with_unresolvable_source_keeps_recorded_message() -> None:
    # source_event_id points at an event absent from the graph -- leave the
    # recorded placeholder message alone rather than crashing or blanking it.
    call = {
        "messages": [{"role": "assistant", "content": "PLACEHOLDER_PRIOR_ANSWER"}],
        "input_segments": [{"type": "output", "message_count": 1, "source_event_id": "missing-evt"}],
    }
    record = _event_to_toolace("evt-1", _event(call), _EMPTY_GRAPH)

    assert record["conversations"] == [{"from": "gpt", "value": "PLACEHOLDER_PRIOR_ANSWER"}]


def test_tool_call_arguments_recovered_from_output_successor() -> None:
    # evt-1's recorded output is a dispatch_agent tool call (expected_output_is_tool_call).
    # evt-2 (the dispatch_ack) consumes it via an `output` segment whose message IS
    # the materialized call with real arguments -- that's what must be rendered,
    # not a name-only "{}" placeholder.
    real_tool_calls = [
        {"id": "call_1", "function": {"name": "dispatch_agent", "arguments": '{"objective": "investigate X"}'}},
        {"id": "call_2", "function": {"name": "dispatch_agent", "arguments": '{"objective": "investigate Y"}'}},
    ]
    graph = _graph(
        {
            "evt-1": _graph_call(
                messages=[{"role": "user", "content": "delegate"}],
                expected_output_is_tool_call=True,
                expected_output_tool_names=["dispatch_agent", "dispatch_agent"],
            ),
            "evt-2": _graph_call(
                messages=[
                    {"role": "user", "content": "delegate"},
                    {"role": "assistant", "tool_calls": real_tool_calls},
                ],
                input_segments=[
                    InputSegment(type="shared", message_count=1, token_count=0, source_event_id="evt-1"),
                    InputSegment(type="output", message_count=1, token_count=0, source_event_id="evt-1"),
                ],
            ),
        }
    )
    call = {
        "messages": [{"role": "user", "content": "delegate"}],
        "expected_output_is_tool_call": True,
        "expected_output_tool_names": ["dispatch_agent", "dispatch_agent"],
    }
    record = _event_to_toolace("evt-1", _event(call), graph)

    function_call = record["conversations"][-1]
    assert function_call["from"] == "function_call"
    assert json.loads(function_call["value"]) == [
        {"name": "dispatch_agent", "arguments": '{"objective": "investigate X"}'},
        {"name": "dispatch_agent", "arguments": '{"objective": "investigate Y"}'},
    ]


def test_tool_call_arguments_fall_back_to_empty_when_no_successor() -> None:
    # evt-1 is a session terminal (nothing sources it via an `output` segment) --
    # fall back to a name-only call with empty arguments rather than failing.
    call = {
        "messages": [{"role": "user", "content": "hi"}],
        "expected_output_is_tool_call": True,
        "expected_output_tool_names": ["get_weather"],
    }
    record = _event_to_toolace("evt-1", _event(call), _EMPTY_GRAPH)

    function_call = record["conversations"][-1]
    assert function_call["from"] == "function_call"
    assert json.loads(function_call["value"]) == [{"name": "get_weather", "arguments": "{}"}]
