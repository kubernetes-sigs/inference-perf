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

"""Tests for tool-call output capture and structured substitution."""

from typing import Any, Dict, List, Optional

import pytest
from inference_perf.datagen.replay_graph_session_datagen import (
    EventOutputRegistry,
    SessionChatCompletionAPIData,
    SessionInferenceInfo,
)
from inference_perf.datagen.replay_graph_types import InputSegment


_TOOL_CALLS = [
    {
        "id": "call_abc",
        "type": "function",
        "function": {"name": "get_weather", "arguments": '{"location":"Paris"}'},
    }
]


class TestEventOutputRegistryStructured:
    def test_record_with_tool_call_message_stored(self) -> None:
        registry = EventOutputRegistry()
        msg = {"role": "assistant", "tool_calls": _TOOL_CALLS}
        registry.record("evt1", "", [], output_message=msg)
        assert registry.get_message_by_event_id("evt1") == msg

    def test_record_text_response_no_message_entry(self) -> None:
        registry = EventOutputRegistry()
        registry.record("evt1", "hello", [])
        assert registry.get_message_by_event_id("evt1") is None
        assert registry.get_output_by_event_id("evt1") == "hello"

    def test_record_text_response_with_explicit_message(self) -> None:
        registry = EventOutputRegistry()
        msg = {"role": "assistant", "content": "hello"}
        registry.record("evt1", "hello", [], output_message=msg)
        assert registry.get_message_by_event_id("evt1") == msg
        assert registry.get_output_by_event_id("evt1") == "hello"

    def test_cleanup_removes_output_message(self) -> None:
        registry = EventOutputRegistry()
        msg = {"role": "assistant", "tool_calls": _TOOL_CALLS}
        registry.record("sess:evt1", "", [], output_message=msg)
        registry._event_output_message.pop("sess:evt1", None)
        assert registry.get_message_by_event_id("sess:evt1") is None

    def test_get_message_absent_returns_none(self) -> None:
        registry = EventOutputRegistry()
        assert registry.get_message_by_event_id("nonexistent") is None


class TestSubstitutionWithToolCalls:
    """Tests for _build_messages_with_substitution when predecessor emitted tool calls."""

    def _make_api_data(
        self,
        registry: EventOutputRegistry,
        original_messages: List[Dict[str, Any]],
        input_segments: List[InputSegment],
    ) -> SessionChatCompletionAPIData:
        from inference_perf.datagen.replay_graph_session_datagen import WorkerSessionTracker
        from inference_perf.apis.chat import ChatMessage

        chat_messages = [ChatMessage(role=m["role"], content=m.get("content")) for m in original_messages]
        return SessionChatCompletionAPIData(
            messages=chat_messages,
            max_tokens=100,
            event_id="sess:evt2",
            registry=registry,
            worker_tracker=WorkerSessionTracker(),
            completion_queue=None,
            total_events_in_session=2,
            predecessor_event_ids=["sess:evt1"],
            input_segments=input_segments,
            original_messages=original_messages,
        )

    def test_output_segment_injects_structured_tool_call_message(self) -> None:
        registry = EventOutputRegistry()
        tool_msg = {"role": "assistant", "tool_calls": _TOOL_CALLS}
        registry.record("sess:evt1", "", [], output_message=tool_msg)

        original_messages = [
            {"role": "user", "content": "What is the weather?"},
            {"role": "assistant", "content": "<recorded>"},  # will be replaced
        ]
        segments = [
            InputSegment(type="unique", message_count=1, token_count=10),
            InputSegment(type="output", message_count=1, token_count=5, source_event_id="sess:evt1"),
        ]
        api_data = self._make_api_data(registry, original_messages, segments)
        result = api_data._build_messages_with_substitution()

        assert len(result) == 2
        assert result[0] == {"role": "user", "content": "What is the weather?"}
        assert result[1] == {"role": "assistant", "tool_calls": _TOOL_CALLS}
        assert "content" not in result[1]

    def test_output_segment_falls_back_to_text_when_no_structured_message(self) -> None:
        registry = EventOutputRegistry()
        registry.record("sess:evt1", "The answer is 42.", [])

        original_messages = [
            {"role": "user", "content": "What is 6x7?"},
            {"role": "assistant", "content": "<recorded>"},
        ]
        segments = [
            InputSegment(type="unique", message_count=1, token_count=10),
            InputSegment(type="output", message_count=1, token_count=5, source_event_id="sess:evt1"),
        ]
        api_data = self._make_api_data(registry, original_messages, segments)
        result = api_data._build_messages_with_substitution()

        assert len(result) == 2
        assert result[1]["content"] == "The answer is 42."

    def test_output_segment_uses_recorded_when_no_output_at_all(self) -> None:
        registry = EventOutputRegistry()
        # Record with empty text and no structured message
        registry.record("sess:evt1", "", [])

        original_messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "<recorded fallback>"},
        ]
        segments = [
            InputSegment(type="unique", message_count=1, token_count=10),
            InputSegment(type="output", message_count=1, token_count=5, source_event_id="sess:evt1"),
        ]
        api_data = self._make_api_data(registry, original_messages, segments)
        result = api_data._build_messages_with_substitution()

        assert result[1]["content"] == "<recorded fallback>"

    def test_mixed_content_and_tool_calls_in_structured_message(self) -> None:
        """A response that has both text content and tool_calls is preserved as-is."""
        registry = EventOutputRegistry()
        mixed_msg = {"role": "assistant", "content": "Let me check that.", "tool_calls": _TOOL_CALLS}
        registry.record("sess:evt1", "Let me check that.", [], output_message=mixed_msg)

        original_messages = [
            {"role": "user", "content": "What is the weather?"},
            {"role": "assistant", "content": "<recorded>"},
        ]
        segments = [
            InputSegment(type="unique", message_count=1, token_count=10),
            InputSegment(type="output", message_count=1, token_count=5, source_event_id="sess:evt1"),
        ]
        api_data = self._make_api_data(registry, original_messages, segments)
        result = api_data._build_messages_with_substitution()

        assert result[1] == mixed_msg


class TestToolChoiceInjection:
    """Tests for tool_choice injection in to_payload when expected output was a tool call."""

    def _make_api_data(
        self,
        tool_definitions: Optional[List[Dict[str, Any]]],
        expected_output_is_tool_call: bool,
        expected_output_tool_names: Optional[List[str]],
    ) -> SessionChatCompletionAPIData:
        from inference_perf.datagen.replay_graph_session_datagen import WorkerSessionTracker
        from inference_perf.apis.chat import ChatMessage

        return SessionChatCompletionAPIData(
            messages=[ChatMessage(role="user", content="What is the weather?")],
            max_tokens=100,
            tool_definitions=tool_definitions,
            event_id="sess:evt1",
            registry=EventOutputRegistry(),
            worker_tracker=WorkerSessionTracker(),
            completion_queue=None,
            total_events_in_session=1,
            expected_output_is_tool_call=expected_output_is_tool_call,
            expected_output_tool_names=expected_output_tool_names,
        )

    @pytest.mark.asyncio
    async def test_single_tool_call_forces_specific_function(self) -> None:
        tool_defs = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        ]
        api_data = self._make_api_data(tool_defs, True, ["get_weather"])
        payload = await api_data.to_payload("model", 100, False, False)
        assert payload["tool_choice"] == {"type": "function", "function": {"name": "get_weather"}}

    @pytest.mark.asyncio
    async def test_multiple_tool_calls_use_required(self) -> None:
        tool_defs = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
            {
                "type": "function",
                "name": "get_time",
                "description": "Get time",
                "parameters": {"type": "object", "properties": {}, "required": []},
            },
        ]
        api_data = self._make_api_data(tool_defs, True, ["get_weather", "get_time"])
        payload = await api_data.to_payload("model", 100, False, False)
        assert payload["tool_choice"] == "required"

    @pytest.mark.asyncio
    async def test_no_tool_choice_when_not_tool_call_output(self) -> None:
        tool_defs = [
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get weather",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        ]
        api_data = self._make_api_data(tool_defs, False, None)
        payload = await api_data.to_payload("model", 100, False, False)
        assert "tool_choice" not in payload

    @pytest.mark.asyncio
    async def test_no_tool_choice_when_no_tool_definitions(self) -> None:
        # Even if the recorded output was a tool call, don't set tool_choice when
        # there are no tool_definitions in the request (nothing to choose from).
        api_data = self._make_api_data(None, True, ["get_weather"])
        payload = await api_data.to_payload("model", 100, False, False)
        assert "tool_choice" not in payload

    @pytest.mark.asyncio
    async def test_recorded_tool_not_in_definitions_falls_back_to_required(self) -> None:
        # The recorded output named a tool that isn't in this call's tool_definitions.
        # This happens with real traces where tool lists and outputs don't always agree.
        # vLLM rejects tool_choice by name when the name isn't in tools, so we must
        # fall back to "required" rather than cause a 500.
        tool_defs = [
            {
                "type": "function",
                "name": "finish",
                "description": "Finish",
                "parameters": {"type": "object", "properties": {}, "required": []},
            }
        ]
        api_data = self._make_api_data(tool_defs, True, ["some_other_tool_not_in_list"])
        payload = await api_data.to_payload("model", 100, False, False)
        assert payload["tool_choice"] == "required"


class TestToolCallIdRewriting:
    """Tests for tool_call_id rewriting in role:tool messages after output substitution."""

    def _make_api_data(
        self,
        registry: EventOutputRegistry,
        original_messages: List[Dict[str, Any]],
        input_segments: List[InputSegment],
        expected_output_is_tool_call: bool = True,
    ) -> SessionChatCompletionAPIData:
        from inference_perf.datagen.replay_graph_session_datagen import WorkerSessionTracker
        from inference_perf.apis.chat import ChatMessage

        chat_messages = [
            ChatMessage(role=m["role"], content=m.get("content"), tool_calls=m.get("tool_calls")) for m in original_messages
        ]
        return SessionChatCompletionAPIData(
            messages=chat_messages,
            max_tokens=100,
            event_id="sess:evt2",
            registry=registry,
            worker_tracker=WorkerSessionTracker(),
            completion_queue=None,
            total_events_in_session=2,
            predecessor_event_ids=["sess:evt1"],
            input_segments=input_segments,
            original_messages=original_messages,
            expected_output_is_tool_call=expected_output_is_tool_call,
        )

    def test_tool_call_ids_rewritten_to_live_ids(self) -> None:
        from inference_perf.datagen.replay_graph_session_datagen import EventOutputRegistry
        from inference_perf.datagen.replay_graph_types import InputSegment

        registry = EventOutputRegistry()
        live_tool_calls = [
            {"id": "live_id_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"loc":"Paris"}'}}
        ]
        registry.record("sess:evt1", "", [], output_message={"role": "assistant", "tool_calls": live_tool_calls})

        original_messages = [
            {"role": "user", "content": "What is the weather?"},
            # recorded assistant message (will be replaced by live output)
            {"role": "assistant", "content": "<recorded>"},
            # role:tool result referencing the OLD recorded tool_call_id
            {"role": "tool", "tool_call_id": "recorded_id_old", "content": '{"temp": 22}'},
        ]
        segments = [
            InputSegment(type="unique", message_count=1, token_count=10),
            InputSegment(type="output", message_count=1, token_count=5, source_event_id="sess:evt1"),
            InputSegment(type="unique", message_count=1, token_count=5),
        ]
        api_data = self._make_api_data(registry, original_messages, segments)
        result = api_data._build_messages_with_substitution()

        assert len(result) == 3
        # The assistant message uses live tool calls
        assert result[1]["tool_calls"] == live_tool_calls
        # The role:tool message has its ID rewritten to the live ID
        assert result[2]["role"] == "tool"
        assert result[2]["tool_call_id"] == "live_id_1"

    def test_multiple_tool_call_ids_rewritten_in_order(self) -> None:
        from inference_perf.datagen.replay_graph_session_datagen import EventOutputRegistry
        from inference_perf.datagen.replay_graph_types import InputSegment

        registry = EventOutputRegistry()
        live_tool_calls = [
            {"id": "live_1", "type": "function", "function": {"name": "f1", "arguments": "{}"}},
            {"id": "live_2", "type": "function", "function": {"name": "f2", "arguments": "{}"}},
        ]
        registry.record("sess:evt1", "", [], output_message={"role": "assistant", "tool_calls": live_tool_calls})

        original_messages = [
            {"role": "user", "content": "Go"},
            {"role": "assistant", "content": "<recorded>"},
            {"role": "tool", "tool_call_id": "old_1", "content": "result1"},
            {"role": "tool", "tool_call_id": "old_2", "content": "result2"},
        ]
        segments = [
            InputSegment(type="unique", message_count=1, token_count=5),
            InputSegment(type="output", message_count=1, token_count=5, source_event_id="sess:evt1"),
            InputSegment(type="unique", message_count=2, token_count=10),
        ]
        api_data = self._make_api_data(registry, original_messages, segments)
        result = api_data._build_messages_with_substitution()

        assert result[2]["tool_call_id"] == "live_1"
        assert result[3]["tool_call_id"] == "live_2"

    def test_non_tool_unique_messages_not_rewritten(self) -> None:
        """Messages after the tool results in the unique segment must not be touched."""
        from inference_perf.datagen.replay_graph_session_datagen import EventOutputRegistry
        from inference_perf.datagen.replay_graph_types import InputSegment

        registry = EventOutputRegistry()
        live_tool_calls = [{"id": "live_1", "type": "function", "function": {"name": "f1", "arguments": "{}"}}]
        registry.record("sess:evt1", "", [], output_message={"role": "assistant", "tool_calls": live_tool_calls})

        original_messages = [
            {"role": "user", "content": "Go"},
            {"role": "assistant", "content": "<recorded>"},
            {"role": "tool", "tool_call_id": "old_1", "content": "result"},
            {"role": "user", "content": "Follow up"},
        ]
        segments = [
            InputSegment(type="unique", message_count=1, token_count=5),
            InputSegment(type="output", message_count=1, token_count=5, source_event_id="sess:evt1"),
            InputSegment(type="unique", message_count=2, token_count=10),
        ]
        api_data = self._make_api_data(registry, original_messages, segments)
        result = api_data._build_messages_with_substitution()

        assert result[2]["tool_call_id"] == "live_1"
        # The user follow-up after the tool result must be untouched
        assert result[3] == {"role": "user", "content": "Follow up"}

    def test_warning_logged_when_plain_text_replaces_tool_call(self, caplog: Any) -> None:
        """A warning must be emitted when the live model returns text where a tool call was expected."""
        import logging
        from inference_perf.datagen.replay_graph_session_datagen import EventOutputRegistry
        from inference_perf.datagen.replay_graph_types import InputSegment

        registry = EventOutputRegistry()
        # Live model returned plain text, no structured message stored
        registry.record("sess:evt1", "some plain text", [])

        original_messages = [
            {"role": "user", "content": "Go"},
            {"role": "assistant", "content": "<recorded tool call>"},
        ]
        segments = [
            InputSegment(type="unique", message_count=1, token_count=5),
            InputSegment(type="output", message_count=1, token_count=5, source_event_id="sess:evt1"),
        ]
        api_data = self._make_api_data(registry, original_messages, segments, expected_output_is_tool_call=True)
        with caplog.at_level(logging.WARNING, logger="inference_perf.datagen.replay_graph_session_datagen"):
            api_data._build_messages_with_substitution()

        assert any("plain text" in r.message for r in caplog.records)


class TestCausalDepToolCallIds:
    """Fix 25: get_causal_dep returns CAUSAL_TOOL_CALL_IDS_MATCHED when OTel parts format
    matches OpenAI tool_calls format via shared tool-call IDs."""

    def _make_raw_call(self, out_message: Any, messages: List[Any]) -> "Any":
        from inference_perf.datagen.otel_trace_to_replay_graph import RawCall

        return RawCall(
            call_id="span1",
            trace_id="trace1",
            t_start_ms=0,
            t_end_ms=100,
            model="test",
            messages=messages,
            out_message=out_message,
            prompt_tokens=None,
            completion_tokens=None,
            temperature=None,
            max_tokens_recorded=None,
        )

    def test_otel_parts_format_matched_by_id_in_openai_format(self) -> None:
        """OTel parts output (a) is matched against OpenAI tool_calls input (b) via IDs."""
        from inference_perf.datagen.otel_trace_to_replay_graph import get_causal_dep, DEPENDENCY_TYPE
        from inference_perf.datagen.replay_graph_types import ComplexReplayMessage, ReplayMessage

        # Call A output: OTel "parts" format with a tool call carrying an ID
        out_parts = [{"type": "tool_call", "name": "bash", "id": "call_xyz", "arguments": "{}"}]
        out_msg = ComplexReplayMessage(
            role="assistant",
            message_info={"parts": out_parts, "parts_text": "<|tool_call|>bash<|end|>"},
            raw_reconstructed_text="<|tool_call|>bash<|end|>",
        )

        # Call B input: OpenAI tool_calls format — same ID, different text representation
        b_user = ReplayMessage(role="user", text="What is the weather?")
        b_assistant = ComplexReplayMessage(
            role="assistant",
            message_info={
                "tool_calls": [{"id": "call_xyz", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]
            },
            raw_reconstructed_text='{"tool_calls": [{"id": "call_xyz"}]}',
        )

        call_a = self._make_raw_call(out_msg, [])
        call_b = self._make_raw_call(None, [b_user, b_assistant])

        dep = get_causal_dep(call_a, call_b)
        assert dep == DEPENDENCY_TYPE.CAUSAL_TOOL_CALL_IDS_MATCHED

    def test_no_match_when_ids_differ(self) -> None:
        """No dependency when tool-call IDs don't overlap."""
        from inference_perf.datagen.otel_trace_to_replay_graph import get_causal_dep
        from inference_perf.datagen.replay_graph_types import ComplexReplayMessage

        out_parts = [{"type": "tool_call", "name": "bash", "id": "call_aaa", "arguments": "{}"}]
        out_msg = ComplexReplayMessage(
            role="assistant",
            message_info={"parts": out_parts, "parts_text": "<|tool_call|>bash<|end|>"},
            raw_reconstructed_text="<|tool_call|>bash<|end|>",
        )
        b_assistant = ComplexReplayMessage(
            role="assistant",
            message_info={
                "tool_calls": [{"id": "call_zzz", "type": "function", "function": {"name": "bash", "arguments": "{}"}}]
            },
            raw_reconstructed_text="",
        )
        call_a = self._make_raw_call(out_msg, [])
        call_b = self._make_raw_call(None, [b_assistant])

        dep = get_causal_dep(call_a, call_b)
        assert dep is None


class TestToolChoiceEmptyDefinitions:
    """Fix 26: tool_choice must NOT be set when tool_definitions is [] (empty list, not None)."""

    @pytest.mark.asyncio
    async def test_no_tool_choice_when_tool_definitions_is_empty_list(self) -> None:
        from inference_perf.datagen.replay_graph_session_datagen import (
            SessionChatCompletionAPIData,
            WorkerSessionTracker,
            EventOutputRegistry,
        )
        from inference_perf.apis.chat import ChatMessage

        api_data = SessionChatCompletionAPIData(
            messages=[ChatMessage(role="user", content="Go")],
            max_tokens=100,
            tool_definitions=[],  # empty list — nothing to choose from
            event_id="sess:evt1",
            registry=EventOutputRegistry(),
            worker_tracker=WorkerSessionTracker(),
            completion_queue=None,
            total_events_in_session=1,
            expected_output_is_tool_call=True,
            expected_output_tool_names=["some_tool"],
        )
        payload = await api_data.to_payload("model", 100, False, False)
        assert "tool_choice" not in payload


class TestToolCallIdRewriteWithInterveningMessage:
    """Fix 27: tool_call_id rewrite skips intervening non-tool messages correctly."""

    def test_rewrite_skips_intervening_non_tool_message(self) -> None:
        """A non-tool message between the assistant turn and role:tool must not shift the rewrite index."""
        from inference_perf.datagen.replay_graph_session_datagen import (
            SessionChatCompletionAPIData,
            WorkerSessionTracker,
            EventOutputRegistry,
        )
        from inference_perf.datagen.replay_graph_types import InputSegment
        from inference_perf.apis.chat import ChatMessage

        registry = EventOutputRegistry()
        live_tool_calls = [{"id": "live_1", "type": "function", "function": {"name": "f1", "arguments": "{}"}}]
        registry.record("sess:evt1", "", [], output_message={"role": "assistant", "tool_calls": live_tool_calls})

        # Non-tool message sits between the assistant and the role:tool message.
        # This is an unusual but valid trace layout.
        original_messages = [
            {"role": "user", "content": "Go"},
            {"role": "assistant", "content": "<recorded>"},
            {"role": "user", "content": "intermediate non-tool message"},
            {"role": "tool", "tool_call_id": "old_1", "content": "result"},
        ]
        segments = [
            InputSegment(type="unique", message_count=1, token_count=5),
            InputSegment(type="output", message_count=1, token_count=5, source_event_id="sess:evt1"),
            InputSegment(type="unique", message_count=2, token_count=10),
        ]

        chat_messages = [
            ChatMessage(role=m["role"], content=m.get("content"), tool_calls=m.get("tool_calls")) for m in original_messages
        ]
        api_data = SessionChatCompletionAPIData(
            messages=chat_messages,
            max_tokens=100,
            event_id="sess:evt2",
            registry=registry,
            worker_tracker=WorkerSessionTracker(),
            completion_queue=None,
            total_events_in_session=2,
            predecessor_event_ids=["sess:evt1"],
            input_segments=segments,
            original_messages=original_messages,
            expected_output_is_tool_call=True,
        )
        result = api_data._build_messages_with_substitution()

        assert len(result) == 4
        # The role:tool message at index 3 must have the live ID rewritten
        assert result[3]["role"] == "tool"
        assert result[3]["tool_call_id"] == "live_1"
        # The intermediate non-tool message must be untouched
        assert result[2] == {"role": "user", "content": "intermediate non-tool message"}


class TestSessionInferenceInfoOutputMessage:
    def test_output_message_field_present(self) -> None:
        info = SessionInferenceInfo(
            input_tokens=10,
            output_tokens=5,
            output_text=None,
            output_message={"role": "assistant", "tool_calls": _TOOL_CALLS},
        )
        assert info.output_message is not None
        assert info.output_message["tool_calls"] == _TOOL_CALLS

    def test_output_message_defaults_none(self) -> None:
        info = SessionInferenceInfo(input_tokens=10, output_tokens=5)
        assert info.output_message is None
