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
from typing import Any, Dict

from inference_perf.datagen.synthetic_agentic.synthetic_agentic_to_replay_graph import _event_to_toolace


def _event(call: Dict[str, Any], **extra: Any) -> Dict[str, Any]:
    event: Dict[str, Any] = {"call": call}
    event.update(extra)
    return event


def test_plain_conversation_becomes_human_and_gpt_turns() -> None:
    call = {
        "messages": [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"},
            {"role": "assistant", "content": "hi there"},
        ],
    }
    record = _event_to_toolace("evt-1", _event(call))

    assert record["system"] == "You are a helpful assistant."
    assert record["conversations"] == [
        {"from": "human", "value": "hello"},
        {"from": "gpt", "value": "hi there"},
    ]
    assert record["tools"] == "[]"


def test_no_leading_system_message_leaves_system_empty() -> None:
    call = {"messages": [{"role": "user", "content": "hello"}]}
    record = _event_to_toolace("evt-1", _event(call))

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
    record = _event_to_toolace("evt-1", _event(call))

    assert record["conversations"] == [
        {"from": "human", "value": "what's the weather?"},
        {
            "from": "function_call",
            "value": json.dumps({"name": "get_weather", "arguments": '{"city": "NYC"}'}),
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
    record = _event_to_toolace("evt-1", _event(call))

    function_calls = [c for c in record["conversations"] if c["from"] == "function_call"]
    observation = next(c for c in record["conversations"] if c["from"] == "observation")

    assert len(function_calls) == 2
    # Results are matched to the tool name via tool_call_id, regardless of message order.
    assert json.loads(observation["value"]) == [
        {"name": "get_weather", "results": "sunny, 80F"},
        {"name": "get_weather", "results": "rainy, 55F"},
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
    record = _event_to_toolace("evt-1", _event(call))

    observation = next(c for c in record["conversations"] if c["from"] == "observation")
    assert json.loads(observation["value"]) == [{"name": "lookup", "results": "result"}]


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
    record = _event_to_toolace("evt-1", _event(call))

    assert [c["from"] for c in record["conversations"]] == ["human", "function_call"]


def test_expected_output_appended_as_final_gpt_turn() -> None:
    call = {
        "messages": [{"role": "user", "content": "hi"}],
        "expected_output": "final answer",
    }
    record = _event_to_toolace("evt-1", _event(call))

    assert record["conversations"][-1] == {"from": "gpt", "value": "final answer"}


def test_empty_expected_output_adds_no_extra_turn() -> None:
    call = {
        "messages": [{"role": "user", "content": "hi"}],
        "expected_output": "",
    }
    record = _event_to_toolace("evt-1", _event(call))

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
    record = _event_to_toolace("evt-1", event)

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
    record = _event_to_toolace("evt-1", _event(call))

    assert record["metadata"]["predecessor_event_ids"] == []
    assert record["metadata"]["predecessor_dependency_types"] == {}
    assert record["metadata"]["input_segments"] == []


def test_expected_output_is_tool_call_flag_included_only_when_true() -> None:
    call = {
        "messages": [{"role": "user", "content": "hi"}],
        "expected_output_is_tool_call": True,
        "expected_output_tool_names": ["get_weather"],
    }
    record = _event_to_toolace("evt-1", _event(call))

    assert record["metadata"]["expected_output_is_tool_call"] is True
    assert record["metadata"]["expected_output_tool_names"] == ["get_weather"]


def test_expected_output_is_tool_call_flag_omitted_when_falsy() -> None:
    call = {
        "messages": [{"role": "user", "content": "hi"}],
        "expected_output_is_tool_call": False,
    }
    record = _event_to_toolace("evt-1", _event(call))

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
    record = _event_to_toolace("evt-1", _event(call))

    assert record["conversations"] == [
        {"from": "human", "value": "hi"},
        {"from": "gpt", "value": "done"},
    ]
