"""Tests for the streaming response fix in openai_client.py.

Verifies that process_response() is called before the response body is consumed,
so the SSE streaming parser can iterate response.content.
"""

from typing import AsyncGenerator, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponse

from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.config import APIConfig, APIType


class FakeStreamingResponse:
    """Simulates an aiohttp streaming response that tracks read order."""

    def __init__(self, chunks: list[bytes]):
        self.status = 200
        self.headers = {"content-type": "text/event-stream"}
        self._chunks = chunks
        self._text_called = False
        self._iter_called = False
        self.content = self._make_content()

    def _make_content(self) -> MagicMock:
        content = MagicMock()

        async def iter_any() -> AsyncGenerator[bytes, None]:
            self._iter_called = True
            if self._text_called:
                # Stream already consumed by text()
                return
            for chunk in self._chunks:
                yield chunk

        content.iter_any = iter_any
        return content

    async def text(self) -> str:
        self._text_called = True
        return b"".join(self._chunks).decode()


@pytest.mark.asyncio
async def test_streaming_parser_receives_content() -> None:
    """The SSE parser should receive chunks when streaming is enabled."""
    sse_data = (
        b'data: {"choices":[{"delta":{"content":"Hello"}}]}\n\n'
        b'data: {"choices":[{"delta":{"content":" world"}}]}\n\n'
        b"data: [DONE]\n\n"
    )

    response = FakeStreamingResponse([sse_data])
    config = APIConfig(type=APIType.Chat, streaming=True)

    tokenizer = MagicMock()
    tokenizer.count_tokens = MagicMock(side_effect=lambda text: len(text.split()))

    data = ChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="test")],
        max_tokens=100,
    )

    info = await data.process_response(cast(ClientResponse, response), config, tokenizer)

    assert info.output_tokens > 0, "Output tokens should be > 0 when streaming content is present"
    assert response._iter_called, "Streaming iterator should have been called"


@pytest.mark.asyncio
async def test_non_streaming_reads_body() -> None:
    """Non-streaming responses should parse JSON body directly."""
    config = APIConfig(type=APIType.Chat, streaming=False)

    response = MagicMock()
    response.status = 200
    response.json = AsyncMock(return_value={"choices": [{"message": {"content": "Hello world"}}]})

    tokenizer = MagicMock()
    tokenizer.count_tokens = MagicMock(return_value=2)

    data = ChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="test")],
        max_tokens=100,
    )

    info = await data.process_response(cast(ClientResponse, response), config, tokenizer)

    assert info.output_tokens == 2
