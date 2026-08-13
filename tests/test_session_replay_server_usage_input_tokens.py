#!/usr/bin/env python3

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

"""Tests for server_usage-derived request_metrics.text.input_tokens in
SessionChatCompletionAPIData.process_response.

Exercises the real production code path (not mocks of process_response
itself) to verify that when the server reports usage.prompt_tokens, it
becomes the authoritative request_metrics.text.input_tokens — both for
the streaming and non-streaming branches — with a client-side tokenization
fallback (including tool_definitions) when the server doesn't report usage.
"""

from __future__ import annotations

from typing import Any, AsyncGenerator, Dict, List, Optional, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponse

from inference_perf.apis import UnaryResponseMetrics
from inference_perf.apis.chat import ChatMessage
from inference_perf.datagen.replay.replay_graph_session_datagen import (
    EventOutputRegistry,
    SessionChatCompletionAPIData,
    WorkerSessionTracker,
)


def _make_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.count_tokens = lambda text, **kw: max(1, len((text or "").split()))
    return tok


def _make_config(streaming: bool = False) -> MagicMock:
    cfg = MagicMock()
    cfg.streaming = streaming
    return cfg


def _make_non_streaming_response(content: str = "hi", usage: Optional[Dict[str, Any]] = None) -> MagicMock:
    body: Dict[str, Any] = {"choices": [{"message": {"role": "assistant", "content": content}}]}
    if usage is not None:
        body["usage"] = usage
    response = MagicMock()
    response.json = AsyncMock(return_value=body)
    return response


class _FakeStreamingResponse:
    """Minimal aiohttp ClientResponse stand-in that yields preset SSE bytes."""

    def __init__(self, chunks: List[bytes]) -> None:
        self.status = 200
        self.headers = {"content-type": "text/event-stream"}
        self.content = self._make_content(chunks)

    @staticmethod
    def _make_content(chunks: List[bytes]) -> MagicMock:
        content = MagicMock()

        async def iter_any() -> AsyncGenerator[bytes, None]:
            for chunk in chunks:
                yield chunk

        content.iter_any = iter_any
        return content


def _make_streaming_response(deltas: List[Dict[str, Any]], usage: Optional[Dict[str, Any]] = None) -> ClientResponse:
    import json

    parts: List[bytes] = []
    for delta in deltas:
        chunk = json.dumps({"choices": [{"delta": delta}]})
        parts.append(f"data: {chunk}\n\n".encode())
    if usage is not None:
        usage_chunk = json.dumps({"choices": [], "usage": usage})
        parts.append(f"data: {usage_chunk}\n\n".encode())
    parts.append(b"data: [DONE]\n\n")
    return cast(ClientResponse, _FakeStreamingResponse([b"".join(parts)]))


def _make_session_api_data(
    tool_definitions: Optional[List[Dict[str, Any]]] = None,
) -> SessionChatCompletionAPIData:
    return SessionChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="hi")],
        max_tokens=50,
        tool_definitions=tool_definitions,
        event_id="session_1:event_0",
        registry=EventOutputRegistry(),
        worker_tracker=WorkerSessionTracker(),
        completion_queue=None,
        total_events_in_session=1,
        predecessor_event_ids=[],
    )


class TestNonStreamingServerUsage:
    @pytest.mark.asyncio
    async def test_uses_server_prompt_tokens_when_present(self) -> None:
        api_data = _make_session_api_data()
        response = _make_non_streaming_response(usage={"prompt_tokens": 99, "completion_tokens": 3})

        info = await api_data.process_response(response, _make_config(streaming=False), _make_tokenizer())

        assert info.request_metrics.text.input_tokens == 99
        assert isinstance(info.response_metrics, UnaryResponseMetrics)
        assert info.response_metrics.server_usage == {"prompt_tokens": 99, "completion_tokens": 3}

    @pytest.mark.asyncio
    async def test_falls_back_to_client_tokenization_without_usage(self) -> None:
        api_data = _make_session_api_data()
        response = _make_non_streaming_response(usage=None)

        info = await api_data.process_response(response, _make_config(streaming=False), _make_tokenizer())

        # "hi" -> 1 token via the word-count fake tokenizer.
        assert info.request_metrics.text.input_tokens == 1

    @pytest.mark.asyncio
    async def test_fallback_counts_tool_definitions(self) -> None:
        """Client-side fallback must include tool_definitions — otherwise input
        tokens are undercounted whenever the server doesn't report usage."""
        api_data = _make_session_api_data(
            tool_definitions=[
                {
                    "type": "function",
                    "name": "get_weather",
                    "description": "five tokens for this one",
                    "parameters": {"type": "object", "properties": {}},
                }
            ]
        )
        response = _make_non_streaming_response(usage=None)

        info = await api_data.process_response(response, _make_config(streaming=False), _make_tokenizer())

        # 1 (message "hi") + 15 (json.dumps'd tool_definitions, whitespace-tokenized)
        assert info.request_metrics.text.input_tokens == 16


class TestStreamingServerUsage:
    @pytest.mark.asyncio
    async def test_uses_server_prompt_tokens_when_present(self) -> None:
        api_data = _make_session_api_data()
        response = _make_streaming_response(deltas=[{"content": "hello"}], usage={"prompt_tokens": 55, "completion_tokens": 1})

        info = await api_data.process_response(response, _make_config(streaming=True), _make_tokenizer())

        assert info.request_metrics.text.input_tokens == 55

    @pytest.mark.asyncio
    async def test_falls_back_to_client_tokenization_without_usage(self) -> None:
        api_data = _make_session_api_data()
        response = _make_streaming_response(deltas=[{"content": "hello"}], usage=None)

        info = await api_data.process_response(response, _make_config(streaming=True), _make_tokenizer())

        assert info.request_metrics.text.input_tokens == 1
