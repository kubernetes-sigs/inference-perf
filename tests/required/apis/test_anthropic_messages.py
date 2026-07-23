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

from collections.abc import AsyncGenerator
from typing import Any
from unittest.mock import AsyncMock, MagicMock

import pytest

from inference_perf.apis import AnthropicMessagesAPIData, ChatMessage
from inference_perf.config import APIConfig, APIType
from inference_perf.datagen.replay_graph_session_datagen import (
    EventOutputRegistry,
    SessionAnthropicMessagesAPIData,
    WorkerSessionTracker,
)


def _mock_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.count_tokens.side_effect = lambda text: len((text or "").split())
    return tokenizer


def _mock_response(payload: dict[str, Any]) -> MagicMock:
    response = MagicMock()
    response.json = AsyncMock(return_value=payload)
    return response


def _mock_stream_response(chunks: list[bytes]) -> MagicMock:
    response = MagicMock()
    content = MagicMock()
    response.content = content

    async def mock_iter_any() -> AsyncGenerator[bytes, None]:
        for chunk in chunks:
            yield chunk

    content.iter_any = mock_iter_any
    return response


@pytest.mark.asyncio
async def test_anthropic_messages_request_body_with_tools() -> None:
    data = AnthropicMessagesAPIData(
        messages=[
            ChatMessage(role="system", content="Be concise."),
            ChatMessage(role="user", content="What is the weather in Paris?"),
            ChatMessage(
                role="assistant",
                tool_calls=[
                    {
                        "id": "toolu_01",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                    }
                ],
            ),
            ChatMessage(role="tool", content="Sunny", tool_call_id="toolu_01"),
        ],
        tool_definitions=[
            {
                "type": "function",
                "name": "get_weather",
                "description": "Get current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
    )

    assert data.get_api_type() == APIType.AnthropicMessages
    assert data.get_route() == "/v1/messages"
    assert await data.to_request_body("claude-sonnet", 128, False, False) == {
        "model": "claude-sonnet",
        "system": "Be concise.",
        "messages": [
            {"role": "user", "content": "What is the weather in Paris?"},
            {
                "role": "assistant",
                "content": [
                    {
                        "type": "tool_use",
                        "id": "toolu_01",
                        "name": "get_weather",
                        "input": {"city": "Paris"},
                    }
                ],
            },
            {
                "role": "user",
                "content": [{"type": "tool_result", "tool_use_id": "toolu_01", "content": "Sunny"}],
            },
        ],
        "max_tokens": 128,
        "stream": False,
        "tools": [
            {
                "name": "get_weather",
                "description": "Get current weather",
                "input_schema": {
                    "type": "object",
                    "properties": {"city": {"type": "string"}},
                    "required": ["city"],
                },
            }
        ],
    }


@pytest.mark.asyncio
async def test_anthropic_messages_process_response_uses_anthropic_usage() -> None:
    data = AnthropicMessagesAPIData(messages=[ChatMessage(role="user", content="hello there")])
    response = _mock_response(
        {
            "content": [{"type": "text", "text": "hello back"}],
            "stop_reason": "end_turn",
            "usage": {"input_tokens": 3, "output_tokens": 5},
        }
    )

    info = await data.process_response(
        response=response,
        config=APIConfig(type=APIType.AnthropicMessages),
        tokenizer=_mock_tokenizer(),
    )

    assert info.request_metrics.text.input_tokens == 3
    assert info.response_metrics is not None
    assert info.response_metrics.output_tokens == 5
    assert info.extra_info["stop_reason"] == "end_turn"


@pytest.mark.asyncio
async def test_anthropic_messages_streaming_preserves_usage_across_events() -> None:
    data = AnthropicMessagesAPIData(messages=[ChatMessage(role="user", content="hello there")])
    response = _mock_stream_response(
        [
            b'data: {"type": "message_start", "message": {"usage": {"input_tokens": 3}}}\n\n',
            b'data: {"type": "content_block_start", "index": 0, "content_block": {"type": "text", "text": ""}}\n\n',
            b'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}}\n\n',
            b'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": " back"}}\n\n',
            b'data: {"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {"output_tokens": 5}}\n\n',
            b'data: {"type": "message_stop"}\n\n',
        ]
    )

    info = await data.process_response(
        response=response,
        config=APIConfig(type=APIType.AnthropicMessages, streaming=True),
        tokenizer=_mock_tokenizer(),
    )

    assert info.request_metrics.text.input_tokens == 3
    assert info.response_metrics is not None
    assert info.response_metrics.output_tokens == 5
    assert info.response_metrics.server_usage == {"input_tokens": 3, "output_tokens": 5}
    assert info.extra_info["output_text"] == "hello back"
    assert info.extra_info["output_message"] == {"role": "assistant", "content": "hello back"}


@pytest.mark.asyncio
async def test_session_anthropic_messages_records_tool_use_for_replay() -> None:
    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()
    data = SessionAnthropicMessagesAPIData(
        messages=[ChatMessage(role="user", content="Use the weather tool")],
        event_id="session_1:event_1",
        registry=registry,
        worker_tracker=tracker,
        completion_queue=None,
        total_events_in_session=1,
    )
    response = _mock_response(
        {
            "content": [
                {
                    "type": "tool_use",
                    "id": "toolu_01",
                    "name": "get_weather",
                    "input": {"city": "Paris"},
                }
            ],
            "stop_reason": "tool_use",
            "usage": {"input_tokens": 4, "output_tokens": 6},
        }
    )

    info = await data.process_response(
        response=response,
        config=APIConfig(type=APIType.AnthropicMessages),
        tokenizer=_mock_tokenizer(),
    )

    assert info.output_text is None
    assert registry.get_output_by_event_id("session_1:event_1") == ""
    assert registry.get_message_by_event_id("session_1:event_1") == {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "toolu_01",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
            }
        ],
    }
    assert tracker.is_event_completed("session_1", "event_1")


@pytest.mark.asyncio
async def test_session_anthropic_messages_streaming_records_tool_use_for_replay() -> None:
    registry = EventOutputRegistry()
    tracker = WorkerSessionTracker()
    data = SessionAnthropicMessagesAPIData(
        messages=[ChatMessage(role="user", content="Use the weather tool")],
        event_id="session_1:event_1",
        registry=registry,
        worker_tracker=tracker,
        completion_queue=None,
        total_events_in_session=1,
    )
    response = _mock_stream_response(
        [
            b'data: {"type": "message_start", "message": {"usage": {"input_tokens": 4}}}\n\n',
            (
                b'data: {"type": "content_block_start", "index": 0, '
                b'"content_block": {"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {}}}\n\n'
            ),
            (
                b'data: {"type": "content_block_delta", "index": 0, '
                b'"delta": {"type": "input_json_delta", "partial_json": "{\\"city\\":"}}\n\n'
            ),
            (
                b'data: {"type": "content_block_delta", "index": 0, '
                b'"delta": {"type": "input_json_delta", "partial_json": "\\"Paris\\"}"}}\n\n'
            ),
            b'data: {"type": "content_block_stop", "index": 0}\n\n',
            b'data: {"type": "message_delta", "delta": {"stop_reason": "tool_use"}, "usage": {"output_tokens": 6}}\n\n',
            b'data: {"type": "message_stop"}\n\n',
        ]
    )

    info = await data.process_response(
        response=response,
        config=APIConfig(type=APIType.AnthropicMessages, streaming=True),
        tokenizer=_mock_tokenizer(),
    )

    assert info.output_text is None
    assert info.request_metrics.text.input_tokens == 4
    assert info.response_metrics is not None
    assert info.response_metrics.output_tokens == 6
    assert info.response_metrics.server_usage == {"input_tokens": 4, "output_tokens": 6}
    assert registry.get_output_by_event_id("session_1:event_1") == ""
    assert registry.get_message_by_event_id("session_1:event_1") == {
        "role": "assistant",
        "tool_calls": [
            {
                "id": "toolu_01",
                "type": "function",
                "function": {"name": "get_weather", "arguments": '{"city":"Paris"}'},
            }
        ],
    }
    assert tracker.is_event_completed("session_1", "event_1")
