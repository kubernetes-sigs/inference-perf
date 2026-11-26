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

"""Replay handling for a predecessor turn the server stopped for length (#1072).

A reasoning model can spend the recorded `max_tokens` budget on reasoning and
stop with `finish_reason: "length"` before producing the recorded shape: no
visible text, or no tool call where the recording made one. That truncated
message must not silently become the successor's history. The opt-in
`truncated_output_handling` knob substitutes the recorded assistant message
(`use_recorded`) or fails the successor (`fail`); the default `none` leaves
the truncated message in place, exactly as upstream main does.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator, Dict, List, Mapping
from unittest.mock import AsyncMock, MagicMock

import pytest

from inference_perf.apis.base import LazyLoadInferenceAPIData
from inference_perf.apis.chat import ChatMessage
from inference_perf.config import APIConfig, APIType
from inference_perf.config.datagen.replay import SessionReplayConfig, TruncatedOutputHandling
from inference_perf.datagen.replay.replay_graph_session_datagen import (
    EventOutputRegistry,
    ReplayGraphSessionGeneratorBase,
    ReplaySessionEvent,
    SessionAnthropicMessagesAPIData,
    SessionChatCompletionAPIData,
    SessionFailureCause,
    WorkerSessionTracker,
)
from inference_perf.datagen.replay.replay_graph_types import InputSegment

PREDECESSOR = "session_0:event_0"
SUCCESSOR = "session_0:event_1"
RECORDED_TOOL_CALL: Dict[str, object] = {
    "id": "call_recorded",
    "type": "function",
    "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
}
MALFORMED_TOOL_CALL: Dict[str, object] = {
    "id": "call_recorded",
    "type": "function",
    "function": {"name": "get_weather", "arguments": "</parameter></function>"},
}
RECORDED_ANSWER = "The weather in Paris is sunny."


def _tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.count_tokens = lambda text, **kw: max(1, len((text or "").split()))
    return tokenizer


def _config(streaming: bool, api_type: APIType = APIType.Chat) -> APIConfig:
    return APIConfig(type=api_type, streaming=streaming)


def _recorded_chat_message(recorded_message: Dict[str, object]) -> ChatMessage:
    """The recorded assistant turn as the wire ChatMessage of a replayed event."""
    raw_content = recorded_message.get("content")
    raw_tool_calls = recorded_message.get("tool_calls")
    return ChatMessage(
        role=str(recorded_message.get("role")),
        content=raw_content if isinstance(raw_content, str) else None,
        tool_calls=raw_tool_calls if isinstance(raw_tool_calls, list) else None,
    )


def _make_api_data(
    registry: EventOutputRegistry,
    tracker: WorkerSessionTracker,
    *,
    recorded_message: Dict[str, object],
    expected_tool_call: bool,
    handling: TruncatedOutputHandling,
    disable_output_substitution: bool = False,
) -> SessionChatCompletionAPIData:
    """Two-event chain: event_0 produces a turn; event_1 replays it from the output segment."""
    return SessionChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="What is the weather?"), _recorded_chat_message(recorded_message)],
        max_tokens=50,
        event_id=SUCCESSOR,
        registry=registry,
        worker_tracker=tracker,
        completion_queue=None,
        total_events_in_session=2,
        predecessor_event_ids=[PREDECESSOR],
        input_segments=[
            InputSegment(type="unique", message_count=1, token_count=5),
            InputSegment(type="output", message_count=1, token_count=10, source_event_id=PREDECESSOR),
        ],
        original_messages=[{"role": "user", "content": "What is the weather?"}, recorded_message],
        override_tool_call_max_tokens=False,
        expected_output_is_tool_call=expected_tool_call,
        truncated_output_handling=handling,
        disable_output_substitution=disable_output_substitution,
    )


def _record_live_turn(
    registry: EventOutputRegistry,
    message: Dict[str, object],
    *,
    finish_reason: str = "length",
    text: str = "",
) -> None:
    """Seed the registry with the predecessor's live message and its finish reason."""
    registry.record(PREDECESSOR, text, [], output_message=message, finish_reason=finish_reason)


def _unary_response(body: Mapping[str, object]) -> MagicMock:
    response = MagicMock()
    response.json = AsyncMock(return_value=body)
    return response


def _streaming_response(frames: List[Dict[str, object]]) -> MagicMock:
    payload = b"".join(f"data: {json.dumps(frame)}\n\n".encode() for frame in frames) + b"data: [DONE]\n\n"

    async def iter_any() -> AsyncGenerator[bytes, None]:
        yield payload

    response = MagicMock()
    response.status = 200
    response.headers = {"content-type": "text/event-stream"}
    response.content.iter_any = iter_any
    return response


def _process_response_api_data(registry: EventOutputRegistry) -> SessionChatCompletionAPIData:
    return SessionChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="hi")],
        max_tokens=50,
        event_id=PREDECESSOR,
        registry=registry,
        worker_tracker=WorkerSessionTracker(),
        completion_queue=None,
        total_events_in_session=1,
        predecessor_event_ids=[],
    )


# `truncated_output_handling` defaults to `none`, so an unset config keeps the
# pre-fix wire behavior for every replay type that inherits SessionReplayConfig.
def test_default_handling_is_none() -> None:
    assert SessionReplayConfig().truncated_output_handling == TruncatedOutputHandling.NONE


# A recorded tool call whose live counterpart never arrived: `use_recorded`
# replaces the empty live turn with the recorded assistant message, so the
# successor's recorded role:tool result keeps a valid tool_call_id.
@pytest.mark.asyncio
async def test_truncated_tool_call_substitutes_recorded_message() -> None:
    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()
    _record_live_turn(registry, {"role": "assistant", "content": "", "reasoning_content": "thinking"})

    api_data = _make_api_data(
        registry,
        tracker,
        recorded_message={"role": "assistant", "content": None, "tool_calls": [RECORDED_TOOL_CALL]},
        expected_tool_call=True,
        handling=TruncatedOutputHandling.USE_RECORDED,
    )
    await api_data.wait_for_predecessors_and_substitute()

    assert not api_data.skip_request
    assert not registry.is_event_failed(SUCCESSOR)
    assert api_data.messages[1].tool_calls == [RECORDED_TOOL_CALL]
    assert tracker.get_session_recorded_substitution_event_ids("session_0") == ["event_0"]
    assert api_data._substitution_failure_reason is None


# A recorded text answer replaced by an empty live turn (reasoning consumed the
# budget) is repaired the same way, so the next turn sees a full answer.
@pytest.mark.asyncio
async def test_truncated_text_substitutes_recorded_message() -> None:
    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()
    _record_live_turn(registry, {"role": "assistant", "content": "", "reasoning_content": "thinking"})

    api_data = _make_api_data(
        registry,
        tracker,
        recorded_message={"role": "assistant", "content": RECORDED_ANSWER},
        expected_tool_call=False,
        handling=TruncatedOutputHandling.USE_RECORDED,
    )
    await api_data.wait_for_predecessors_and_substitute()

    assert not api_data.skip_request
    assert api_data.messages[1].content == RECORDED_ANSWER
    assert tracker.get_session_recorded_substitution_event_ids("session_0") == ["event_0"]


# `fail` scores the truncated turn as the failure it is instead of repairing it:
# the successor is skipped, the event is failed, and the cause is reportable.
@pytest.mark.asyncio
async def test_truncated_turn_fails_event_when_handling_fail() -> None:
    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()
    _record_live_turn(registry, {"role": "assistant", "content": ""})

    api_data = _make_api_data(
        registry,
        tracker,
        recorded_message={"role": "assistant", "content": None, "tool_calls": [RECORDED_TOOL_CALL]},
        expected_tool_call=True,
        handling=TruncatedOutputHandling.FAIL,
    )
    await api_data.wait_for_predecessors_and_substitute()

    assert api_data.skip_request
    assert registry.is_event_failed(SUCCESSOR)
    assert api_data._substitution_failure_cause == SessionFailureCause.TRUNCATED_OUTPUT
    assert api_data._substitution_failure_reason is not None
    assert "truncated" in api_data._substitution_failure_reason
    assert tracker.get_session_recorded_substitution_event_ids("session_0") == []


# Default `none` must not intervene: the truncated live turn is appended as-is,
# matching upstream main byte for byte.
@pytest.mark.asyncio
async def test_truncated_turn_passes_through_when_handling_none() -> None:
    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()
    _record_live_turn(registry, {"role": "assistant", "content": ""})

    api_data = _make_api_data(
        registry,
        tracker,
        recorded_message={"role": "assistant", "content": None, "tool_calls": [RECORDED_TOOL_CALL]},
        expected_tool_call=True,
        handling=TruncatedOutputHandling.NONE,
    )
    await api_data.wait_for_predecessors_and_substitute()

    assert not api_data.skip_request
    assert not registry.is_event_failed(SUCCESSOR)
    assert api_data.messages[1].tool_calls is None
    assert api_data.messages[1].content == ""
    assert tracker.get_session_recorded_substitution_event_ids("session_0") == []


# With `disable_output_substitution` the recorded messages go on the wire, so
# there is no truncated live turn to repair and the knob must stay inert.
@pytest.mark.asyncio
async def test_handling_is_inert_when_substitution_is_disabled() -> None:
    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()
    _record_live_turn(registry, {"role": "assistant", "content": ""})

    api_data = _make_api_data(
        registry,
        tracker,
        recorded_message={"role": "assistant", "content": None, "tool_calls": [RECORDED_TOOL_CALL]},
        expected_tool_call=True,
        handling=TruncatedOutputHandling.FAIL,
        disable_output_substitution=True,
    )
    await api_data.wait_for_predecessors_and_substitute()

    assert not api_data.skip_request
    assert not registry.is_event_failed(SUCCESSOR)
    assert api_data.messages[1].tool_calls == [RECORDED_TOOL_CALL]
    assert tracker.get_session_recorded_substitution_event_ids("session_0") == []


# A length-stopped turn that still made the recorded tool call is a usable
# answer; it must be kept, not overwritten with the recording.
@pytest.mark.asyncio
async def test_length_stop_with_recorded_shape_is_not_substituted() -> None:
    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()
    live_call = {
        "id": "call_live",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"location": "Paris"}'},
    }
    _record_live_turn(registry, {"role": "assistant", "tool_calls": [live_call]})

    api_data = _make_api_data(
        registry,
        tracker,
        recorded_message={"role": "assistant", "content": None, "tool_calls": [RECORDED_TOOL_CALL]},
        expected_tool_call=True,
        handling=TruncatedOutputHandling.USE_RECORDED,
    )
    await api_data.wait_for_predecessors_and_substitute()

    assert api_data.messages[1].tool_calls == [live_call]
    assert tracker.get_session_recorded_substitution_event_ids("session_0") == []


# Only the length stop is a truncation signal. A normal stop with empty content
# keeps today's behavior (the empty turn is replayed as recorded).
@pytest.mark.asyncio
async def test_non_length_stop_is_not_substituted() -> None:
    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()
    _record_live_turn(registry, {"role": "assistant", "content": ""}, finish_reason="stop")

    api_data = _make_api_data(
        registry,
        tracker,
        recorded_message={"role": "assistant", "content": RECORDED_ANSWER},
        expected_tool_call=False,
        handling=TruncatedOutputHandling.USE_RECORDED,
    )
    await api_data.wait_for_predecessors_and_substitute()

    assert api_data.messages[1].content == ""
    assert tracker.get_session_recorded_substitution_event_ids("session_0") == []


# A partial answer is still the model's answer. The issue is about turns with
# none of the recorded shape; replacing visible text would change the workload.
@pytest.mark.asyncio
async def test_truncated_partial_text_is_not_substituted() -> None:
    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()
    _record_live_turn(registry, {"role": "assistant", "content": "The weather in"}, text="The weather in")

    api_data = _make_api_data(
        registry,
        tracker,
        recorded_message={"role": "assistant", "content": RECORDED_ANSWER},
        expected_tool_call=False,
        handling=TruncatedOutputHandling.USE_RECORDED,
    )
    await api_data.wait_for_predecessors_and_substitute()

    assert api_data.messages[1].content == "The weather in"
    assert tracker.get_session_recorded_substitution_event_ids("session_0") == []


# If the trace itself carries malformed tool arguments, substituting them would
# only move the broken turn one hop; the event fails like the existing path.
@pytest.mark.asyncio
async def test_truncated_turn_with_malformed_recorded_fallback_fails() -> None:
    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()
    _record_live_turn(registry, {"role": "assistant", "content": ""})

    api_data = _make_api_data(
        registry,
        tracker,
        recorded_message={"role": "assistant", "content": None, "tool_calls": [MALFORMED_TOOL_CALL]},
        expected_tool_call=True,
        handling=TruncatedOutputHandling.USE_RECORDED,
    )
    await api_data.wait_for_predecessors_and_substitute()

    assert api_data.skip_request
    assert registry.is_event_failed(SUCCESSOR)
    assert api_data._substitution_failure_cause == SessionFailureCause.RECORDED_FALLBACK_MALFORMED
    assert tracker.get_session_recorded_substitution_event_ids("session_0") == []


# Anthropic spells the length stop `max_tokens`; replay must treat it like
# OpenAI's `length` so the same mitigation works on that API.
@pytest.mark.asyncio
async def test_anthropic_max_tokens_stop_is_truncated() -> None:
    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()
    _record_live_turn(registry, {"role": "assistant", "content": ""}, finish_reason="max_tokens")

    api_data = _make_api_data(
        registry,
        tracker,
        recorded_message={"role": "assistant", "content": None, "tool_calls": [RECORDED_TOOL_CALL]},
        expected_tool_call=True,
        handling=TruncatedOutputHandling.USE_RECORDED,
    )
    await api_data.wait_for_predecessors_and_substitute()

    assert api_data.messages[1].tool_calls == [RECORDED_TOOL_CALL]
    assert tracker.get_session_recorded_substitution_event_ids("session_0") == ["event_0"]


# The unary chat path must keep the server's finish reason with the registered
# message, otherwise substitution cannot distinguish a truncated turn.
@pytest.mark.asyncio
async def test_unary_chat_process_response_records_finish_reason() -> None:
    registry = EventOutputRegistry()
    api_data = _process_response_api_data(registry)
    body = {
        "choices": [
            {
                "message": {"role": "assistant", "content": "", "reasoning_content": "thinking"},
                "finish_reason": "length",
            }
        ]
    }

    await api_data.process_response(_unary_response(body), _config(streaming=False), _tokenizer())

    assert registry.get_finish_reason_by_event_id(PREDECESSOR) == "length"
    # The stored assistant message is the wire message: no finish_reason key.
    assert "finish_reason" not in (registry.get_message_by_event_id(PREDECESSOR) or {})


# The streaming path receives finish_reason on the last content frame, so the
# parser must surface it to the replay registry.
@pytest.mark.asyncio
async def test_streaming_chat_process_response_records_finish_reason() -> None:
    registry = EventOutputRegistry()
    api_data = _process_response_api_data(registry)
    frames: List[Dict[str, object]] = [
        {"choices": [{"delta": {"reasoning_content": "thinking"}, "finish_reason": None}]},
        {"choices": [{"delta": {}, "finish_reason": "length"}]},
    ]

    await api_data.process_response(_streaming_response(frames), _config(streaming=True), _tokenizer())

    assert registry.get_finish_reason_by_event_id(PREDECESSOR) == "length"


# The generator must hand the configured handling to every materialized event;
# otherwise the knob would parse but never reach the substitution path.
def test_load_lazy_data_propagates_truncated_handling() -> None:
    gen = object.__new__(ReplayGraphSessionGeneratorBase)
    gen.replay_config = SessionReplayConfig(truncated_output_handling=TruncatedOutputHandling.FAIL)
    gen.output_registry = EventOutputRegistry()
    gen.worker_tracker = WorkerSessionTracker()
    gen.session_completion_queue = None
    gen.session_graph_state = {}
    gen.num_workers = 1
    gen.all_events = [
        ReplaySessionEvent(
            call_id="call_0",
            event_id="session_0:event_0",
            session_index=0,
            t_start_ms=0,
            t_end_ms=1,
            model="test-model",
            messages=[{"role": "user", "content": "hi"}],
            expected_output="ok",
            input_segments=[InputSegment(type="unique", message_count=1, token_count=1)],
            expected_output_tokens=2,
            temperature=None,
            max_tokens_recorded=None,
        )
    ]

    event = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0))

    assert isinstance(event, SessionChatCompletionAPIData)
    assert event.truncated_output_handling == TruncatedOutputHandling.FAIL


# Anthropic unary carries the stop reason as `stop_reason`, not finish_reason.
@pytest.mark.asyncio
async def test_anthropic_process_response_records_stop_reason() -> None:
    registry = EventOutputRegistry()
    api_data = SessionAnthropicMessagesAPIData(
        messages=[ChatMessage(role="user", content="hi")],
        event_id=PREDECESSOR,
        registry=registry,
        worker_tracker=WorkerSessionTracker(),
        completion_queue=None,
        total_events_in_session=1,
    )

    await api_data.process_response(
        _unary_response({"content": [], "stop_reason": "max_tokens"}),
        _config(streaming=False, api_type=APIType.AnthropicMessages),
        _tokenizer(),
    )

    assert registry.get_finish_reason_by_event_id(PREDECESSOR) == "max_tokens"
