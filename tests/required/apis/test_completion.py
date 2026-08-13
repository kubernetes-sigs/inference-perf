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
from typing import AsyncGenerator, List, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponse

from inference_perf.apis import ChatCompletionAPIData, ChatMessage, UnaryResponseMetrics
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.config import APIConfig, APIType


def _make_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.count_tokens = lambda text, **kw: max(1, len((text or "").split()))
    return tok


def _make_config(streaming: bool) -> MagicMock:
    cfg = MagicMock()
    cfg.streaming = streaming
    return cfg


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


@pytest.mark.asyncio
async def test_completion_api_data() -> None:
    data = CompletionAPIData(prompt="Hello, world!")
    assert data.get_api_type() == APIType.Completion
    assert data.prompt == "Hello, world!"
    assert await data.to_request_body("test-model", 100, False, True) == {
        "model": "test-model",
        "prompt": "Hello, world!",
        "max_tokens": 100,
        "ignore_eos": False,
        "stream": True,
        "stream_options": {"include_usage": True},
    }


@pytest.mark.asyncio
async def test_chat_completion_api_data_with_tools() -> None:
    tool_defs = [
        {
            "type": "function",
            "name": "get_weather",
            "description": "Get the current weather",
            "parameters": {
                "type": "object",
                "properties": {"location": {"type": "string"}},
                "required": ["location"],
            },
        }
    ]
    data = ChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="What is the weather?")],
        tool_definitions=tool_defs,
    )
    payload = await data.to_request_body("test-model", 100, False, False)
    assert payload["tools"] == [
        {
            "type": "function",
            "function": {
                "name": "get_weather",
                "description": "Get the current weather",
                "parameters": {
                    "type": "object",
                    "properties": {"location": {"type": "string"}},
                    "required": ["location"],
                },
            },
        }
    ]
    # Other fields unaffected
    assert payload["model"] == "test-model"
    assert payload["messages"] == [{"role": "user", "content": "What is the weather?"}]


@pytest.mark.asyncio
async def test_chat_completion_api_data_without_tools_has_no_tools_key() -> None:
    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="Hello")])
    payload = await data.to_request_body("test-model", 100, False, False)
    assert "tools" not in payload


@pytest.mark.asyncio
async def test_chat_message_with_tool_calls_serialized_correctly() -> None:
    """Tool-call assistant messages must use tool_calls key, not content."""
    tool_calls = [
        {"id": "call_1", "type": "function", "function": {"name": "get_weather", "arguments": '{"location":"Paris"}'}}
    ]
    data = ChatCompletionAPIData(
        messages=[
            ChatMessage(role="user", content="What is the weather?"),
            ChatMessage(role="assistant", tool_calls=tool_calls),
        ]
    )
    payload = await data.to_request_body("test-model", 100, False, False)
    msgs = payload["messages"]
    assert msgs[0] == {"role": "user", "content": "What is the weather?"}
    assert msgs[1] == {"role": "assistant", "tool_calls": tool_calls}
    assert "content" not in msgs[1]


@pytest.mark.asyncio
async def test_chat_message_content_none_treated_as_empty() -> None:
    """content=None should serialize as empty string (not 'None')."""
    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content=None)])
    payload = await data.to_request_body("test-model", 100, False, False)
    assert payload["messages"][0] == {"role": "user", "content": ""}


@pytest.mark.asyncio
async def test_process_response_non_streaming_uses_server_prompt_tokens() -> None:
    """When the server reports usage.prompt_tokens, request_metrics.text.input_tokens
    is resolved from it rather than client-side tokenization."""
    data = CompletionAPIData(prompt="hi")
    tokenizer = _make_tokenizer()

    response = MagicMock()
    response.json = AsyncMock(
        return_value={
            "choices": [{"text": "hello there"}],
            "usage": {"prompt_tokens": 42, "completion_tokens": 2},
        }
    )

    info = await data.process_response(response, _make_config(streaming=False), tokenizer)

    assert info.request_metrics.text.input_tokens == 42
    assert isinstance(info.response_metrics, UnaryResponseMetrics)
    assert info.response_metrics.server_usage == {"prompt_tokens": 42, "completion_tokens": 2}


@pytest.mark.asyncio
async def test_process_response_non_streaming_falls_back_without_server_usage() -> None:
    """No usage in the response body falls back to client-side tokenization."""
    data = CompletionAPIData(prompt="one two three")
    tokenizer = _make_tokenizer()

    response = MagicMock()
    response.json = AsyncMock(return_value={"choices": [{"text": "hi"}]})

    info = await data.process_response(response, _make_config(streaming=False), tokenizer)

    assert info.request_metrics.text.input_tokens == 3


@pytest.mark.asyncio
async def test_process_response_non_streaming_no_choices_uses_server_prompt_tokens() -> None:
    """The no-choices early return still resolves input_tokens from server usage."""
    data = CompletionAPIData(prompt="hi")
    tokenizer = _make_tokenizer()

    response = MagicMock()
    response.json = AsyncMock(return_value={"choices": [], "usage": {"prompt_tokens": 7}})

    info = await data.process_response(response, _make_config(streaming=False), tokenizer)

    assert info.request_metrics.text.input_tokens == 7


@pytest.mark.asyncio
async def test_process_response_streaming_uses_server_prompt_tokens() -> None:
    """Streaming trailing usage chunk resolves input_tokens the same way as non-streaming."""
    data = CompletionAPIData(prompt="hi")
    tokenizer = _make_tokenizer()

    sse = (
        b'data: {"choices": [{"text": "hello"}]}\n\n'
        b'data: {"choices": [], "usage": {"prompt_tokens": 17, "completion_tokens": 1}}\n\n'
        b"data: [DONE]\n\n"
    )
    response = cast(ClientResponse, _FakeStreamingResponse([sse]))

    info = await data.process_response(response, _make_config(streaming=True), tokenizer)

    assert info.request_metrics.text.input_tokens == 17


@pytest.mark.asyncio
async def test_process_response_streaming_falls_back_without_server_usage() -> None:
    """No usage chunk in the stream falls back to client-side tokenization."""
    data = CompletionAPIData(prompt="one two three")
    tokenizer = _make_tokenizer()

    sse = b'data: {"choices": [{"text": "hi"}]}\n\ndata: [DONE]\n\n'
    response = cast(ClientResponse, _FakeStreamingResponse([sse]))

    info = await data.process_response(response, _make_config(streaming=True), tokenizer)

    assert info.request_metrics.text.input_tokens == 3
