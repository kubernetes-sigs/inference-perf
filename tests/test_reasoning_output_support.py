#!/usr/bin/env python3

# Copyright 2025 The Kubernetes Authors.
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

"""
Tests for reasoning output support in OTEL trace replay.

Exercises the actual production code in SessionChatCompletionAPIData.process_response
to verify correct handling of reasoning_content from reasoning models.
"""

from __future__ import annotations

import json
from typing import Any, AsyncGenerator, Dict, List, Optional, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponse

from inference_perf.apis.chat import ChatMessage
from inference_perf.datagen.replay_graph_session_datagen import (
    EventOutputRegistry,
    SessionChatCompletionAPIData,
    SessionInferenceInfo,
    WorkerSessionTracker,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.count_tokens = lambda text: max(1, len((text or "").split()))
    return tok


def _make_config(streaming: bool = False) -> MagicMock:
    cfg = MagicMock()
    cfg.streaming = streaming
    return cfg


def _make_non_streaming_response(
    content: str = "",
    reasoning_content: Optional[str] = None,
    tool_calls: Optional[List[Dict[str, Any]]] = None,
) -> MagicMock:
    message: Dict[str, Any] = {"role": "assistant", "content": content}
    if reasoning_content is not None:
        message["reasoning_content"] = reasoning_content
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    response = MagicMock()
    response.json = AsyncMock(return_value={"choices": [{"message": message}]})
    return response


class FakeStreamingResponse:
    """Minimal aiohttp ClientResponse stand-in that yields preset SSE bytes."""

    def __init__(self, chunks: List[bytes]) -> None:
        self.status = 200
        self.headers = {"content-type": "text/event-stream"}
        self._chunks = chunks
        self.content = self._make_content()

    def _make_content(self) -> MagicMock:
        content = MagicMock()

        async def iter_any() -> AsyncGenerator[bytes, None]:
            for chunk in self._chunks:
                yield chunk

        content.iter_any = iter_any
        return content


def _build_sse_stream(deltas: List[Dict[str, Any]], completion_tokens: Optional[int] = None) -> bytes:
    """Build an OpenAI-style SSE byte stream from a list of delta dicts."""
    parts: List[bytes] = []
    for delta in deltas:
        chunk = json.dumps({"choices": [{"delta": delta}]})
        parts.append(f"data: {chunk}\n\n".encode())
    if completion_tokens is not None:
        usage_chunk = json.dumps({"choices": [], "usage": {"completion_tokens": completion_tokens}})
        parts.append(f"data: {usage_chunk}\n\n".encode())
    parts.append(b"data: [DONE]\n\n")
    return b"".join(parts)


def _make_streaming_response(deltas: List[Dict[str, Any]], completion_tokens: Optional[int] = None) -> ClientResponse:
    sse_bytes = _build_sse_stream(deltas, completion_tokens)
    return cast(ClientResponse, FakeStreamingResponse([sse_bytes]))


def _make_session_api_data(
    event_id: str = "session_1:event_0",
    registry: Optional[EventOutputRegistry] = None,
    tracker: Optional[WorkerSessionTracker] = None,
) -> SessionChatCompletionAPIData:
    if registry is None:
        registry = EventOutputRegistry()
    if tracker is None:
        tracker = WorkerSessionTracker()
    return SessionChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="Hello")],
        max_tokens=500,
        event_id=event_id,
        registry=registry,
        worker_tracker=tracker,
        completion_queue=None,
        total_events_in_session=1,
        predecessor_event_ids=[],
    )


# ---------------------------------------------------------------------------
# Non-streaming tests
# ---------------------------------------------------------------------------


class TestReasoningNonStreaming:
    """Test reasoning_content handling via the real non-streaming process_response path."""

    @pytest.mark.asyncio
    async def test_reasoning_combined_with_content(self) -> None:
        registry = EventOutputRegistry()
        api_data = _make_session_api_data(registry=registry)
        response = _make_non_streaming_response(
            content="The answer is 42.",
            reasoning_content="Let me think. ",
        )

        info = await api_data.process_response(response, _make_config(), _make_tokenizer())

        assert isinstance(info, SessionInferenceInfo)
        assert info.output_text == "Let me think. The answer is 42."
        assert registry.get_output_by_event_id("session_1:event_0") == "Let me think. The answer is 42."

    @pytest.mark.asyncio
    async def test_no_reasoning_content(self) -> None:
        registry = EventOutputRegistry()
        api_data = _make_session_api_data(registry=registry)
        response = _make_non_streaming_response(content="Hello there!")

        info = await api_data.process_response(response, _make_config(), _make_tokenizer())

        assert info.output_text == "Hello there!"
        assert registry.get_output_by_event_id("session_1:event_0") == "Hello there!"

    @pytest.mark.asyncio
    async def test_empty_reasoning_content(self) -> None:
        registry = EventOutputRegistry()
        api_data = _make_session_api_data(registry=registry)
        response = _make_non_streaming_response(content="Answer", reasoning_content="")

        info = await api_data.process_response(response, _make_config(), _make_tokenizer())

        # Empty string is falsy, so reasoning should not be prepended
        assert info.output_text == "Answer"

    @pytest.mark.asyncio
    async def test_null_reasoning_content(self) -> None:
        """Explicit None in the response dict should be handled like absent."""
        response = MagicMock()
        response.json = AsyncMock(
            return_value={"choices": [{"message": {"role": "assistant", "content": "Answer", "reasoning_content": None}}]}
        )
        registry = EventOutputRegistry()
        api_data = _make_session_api_data(registry=registry)

        info = await api_data.process_response(response, _make_config(), _make_tokenizer())

        assert info.output_text == "Answer"

    @pytest.mark.asyncio
    async def test_reasoning_with_tool_calls(self) -> None:
        registry = EventOutputRegistry()
        api_data = _make_session_api_data(registry=registry)
        tool_calls = [{"id": "call_1", "type": "function", "function": {"name": "get_data", "arguments": "{}"}}]
        response = _make_non_streaming_response(
            content="Here you go.",
            reasoning_content="Let me check. ",
            tool_calls=tool_calls,
        )

        info = await api_data.process_response(response, _make_config(), _make_tokenizer())

        assert info.output_text == "Let me check. Here you go."
        msg = info.output_message
        assert msg is not None
        assert "tool_calls" in msg
        assert msg["tool_calls"] == tool_calls

    @pytest.mark.asyncio
    async def test_reasoning_only_no_content(self) -> None:
        registry = EventOutputRegistry()
        api_data = _make_session_api_data(registry=registry)
        response = _make_non_streaming_response(content="", reasoning_content="Just thinking...")

        info = await api_data.process_response(response, _make_config(), _make_tokenizer())

        assert info.output_text == "Just thinking..."


# ---------------------------------------------------------------------------
# Streaming tests
# ---------------------------------------------------------------------------


class TestReasoningStreaming:
    """Test reasoning_content handling via the real streaming process_response path."""

    @pytest.mark.asyncio
    async def test_streaming_reasoning_chunks_combined(self) -> None:
        registry = EventOutputRegistry()
        api_data = _make_session_api_data(registry=registry)
        deltas = [
            {"reasoning_content": "Let me "},
            {"reasoning_content": "think. "},
            {"content": "Answer: "},
            {"content": "42"},
        ]
        response = _make_streaming_response(deltas, completion_tokens=8)

        info = await api_data.process_response(response, _make_config(streaming=True), _make_tokenizer())

        assert isinstance(info, SessionInferenceInfo)
        assert info.output_text == "Let me think. Answer: 42"
        assert registry.get_output_by_event_id("session_1:event_0") == "Let me think. Answer: 42"

    @pytest.mark.asyncio
    async def test_streaming_without_reasoning(self) -> None:
        registry = EventOutputRegistry()
        api_data = _make_session_api_data(registry=registry)
        deltas = [
            {"content": "Hello "},
            {"content": "there!"},
        ]
        response = _make_streaming_response(deltas, completion_tokens=2)

        info = await api_data.process_response(response, _make_config(streaming=True), _make_tokenizer())

        assert info.output_text == "Hello there!"
        assert registry.get_output_by_event_id("session_1:event_0") == "Hello there!"

    @pytest.mark.asyncio
    async def test_streaming_reasoning_only_no_content(self) -> None:
        registry = EventOutputRegistry()
        api_data = _make_session_api_data(registry=registry)
        deltas = [
            {"reasoning_content": "Deep thought..."},
        ]
        response = _make_streaming_response(deltas, completion_tokens=3)

        info = await api_data.process_response(response, _make_config(streaming=True), _make_tokenizer())

        assert info.output_text == "Deep thought..."


# ---------------------------------------------------------------------------
# Registry integration tests — verify output_message structure
# ---------------------------------------------------------------------------


class TestReasoningRegistryIntegration:
    """Test that output_message stored in registry has correct structure."""

    @pytest.mark.asyncio
    async def test_output_message_has_separate_fields(self) -> None:
        """Non-streaming: output_message should have reasoning_content and output_content."""
        registry = EventOutputRegistry()
        api_data = _make_session_api_data(registry=registry)
        response = _make_non_streaming_response(
            content="The answer.",
            reasoning_content="Reasoning. ",
        )

        await api_data.process_response(response, _make_config(), _make_tokenizer())

        msg = registry.get_message_by_event_id("session_1:event_0")
        assert msg is not None
        assert msg["role"] == "assistant"
        assert msg["content"] == "Reasoning. The answer."
        assert msg["reasoning_content"] == "Reasoning. "
        assert msg["output_content"] == "The answer."

    @pytest.mark.asyncio
    async def test_streaming_output_message_has_separate_fields(self) -> None:
        """Streaming: output_message should have reasoning_content and output_content."""
        registry = EventOutputRegistry()
        api_data = _make_session_api_data(registry=registry)
        deltas = [
            {"reasoning_content": "Think. "},
            {"content": "Result."},
        ]
        response = _make_streaming_response(deltas, completion_tokens=4)

        await api_data.process_response(response, _make_config(streaming=True), _make_tokenizer())

        msg = registry.get_message_by_event_id("session_1:event_0")
        assert msg is not None
        assert msg["role"] == "assistant"
        assert msg["content"] == "Think. Result."
        assert msg["reasoning_content"] == "Think. "
        assert msg["output_content"] == "Result."

    @pytest.mark.asyncio
    async def test_registry_message_without_reasoning_has_no_extra_fields(self) -> None:
        """Without reasoning_content, output_message should not have extra fields."""
        registry = EventOutputRegistry()
        api_data = _make_session_api_data(registry=registry)
        response = _make_non_streaming_response(content="Plain answer.")

        await api_data.process_response(response, _make_config(), _make_tokenizer())

        msg = registry.get_message_by_event_id("session_1:event_0")
        assert msg is not None
        assert msg["role"] == "assistant"
        assert msg["content"] == "Plain answer."
        assert "reasoning_content" not in msg
        assert "output_content" not in msg


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
