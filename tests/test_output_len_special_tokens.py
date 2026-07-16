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
"""Regression test for special-token handling in client-derived output_len.

The server's completion_tokens counts generated tokens only; it never
includes a client-side BOS. Counting the generated text with
add_special_tokens=True adds one BOS per request for BOS-prepending
tokenizers (e.g. Llama-3.1), so client output_len would sit at N+1 against
the server's N and the two could never agree exactly.

This pins the asymmetric policy:
- output text is a continuation -> counted WITHOUT special tokens
- prompt text starts a sequence -> counted WITH special tokens (matching
  the server's prompt_tokens, which includes BOS)

The fake tokenizer mimics a BOS-prepending tokenizer: one token per
whitespace-separated word, plus one BOS when add_special_tokens=True.
"""

from typing import Any, AsyncGenerator, Dict, List, cast
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientResponse

from inference_perf.apis.base import StreamedResponseMetrics, UnaryResponseMetrics
from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.config import APIConfig, APIType


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


class FakeUnaryResponse:
    """Minimal aiohttp ClientResponse stand-in for non-streaming JSON."""

    def __init__(self, payload: Dict[str, Any]) -> None:
        self.status = 200
        self.headers = {"content-type": "application/json"}
        self._payload = payload

    async def json(self) -> Dict[str, Any]:
        return self._payload


def _bos_tokenizer() -> MagicMock:
    """One token per word; add_special_tokens=True prepends one BOS."""

    def count(text: str, add_special_tokens: bool = True) -> int:
        if text == "":
            return 0
        return len(text.split()) + (1 if add_special_tokens else 0)

    tokenizer = MagicMock()
    tokenizer.count_tokens = MagicMock(side_effect=count)
    return tokenizer


def _chat_sse(chunk_texts: List[str]) -> bytes:
    parts = [f'data: {{"choices":[{{"delta":{{"content":"{text}"}}}}]}}\n\n'.encode() for text in chunk_texts]
    parts.append(b"data: [DONE]\n\n")
    return b"".join(parts)


def _completion_sse(chunk_texts: List[str]) -> bytes:
    parts = [f'data: {{"choices":[{{"text":"{text}"}}]}}\n\n'.encode() for text in chunk_texts]
    parts.append(b"data: [DONE]\n\n")
    return b"".join(parts)


@pytest.mark.asyncio
async def test_chat_streaming_output_len_excludes_special_tokens() -> None:
    chunk_texts = ["aa bb", "cc dd", "ee"]
    response = FakeStreamingResponse([_chat_sse(chunk_texts)])
    tokenizer = _bos_tokenizer()
    config = APIConfig(type=APIType.Chat, streaming=True)
    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="p1 p2 p3")], max_tokens=100)

    info = await data.process_response(cast(ClientResponse, response), config, tokenizer)

    assert isinstance(info.response_metrics, StreamedResponseMetrics)
    # "aa bbcc ddee" -> 3 words, and no BOS for generated text.
    assert info.response_metrics.output_tokens == 3
    # Prompt-side counting keeps special tokens: 3 words + BOS.
    assert info.request_metrics.text is not None
    assert info.request_metrics.text.input_tokens == 4


@pytest.mark.asyncio
async def test_chat_unary_output_len_excludes_special_tokens() -> None:
    payload = {"choices": [{"message": {"content": "aa bb cc"}}]}
    response = FakeUnaryResponse(payload)
    tokenizer = _bos_tokenizer()
    config = APIConfig(type=APIType.Chat, streaming=False)
    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="p1 p2 p3")], max_tokens=100)

    info = await data.process_response(cast(ClientResponse, response), config, tokenizer)

    assert isinstance(info.response_metrics, UnaryResponseMetrics)
    assert info.response_metrics.output_tokens == 3
    assert info.request_metrics.text is not None
    assert info.request_metrics.text.input_tokens == 4


@pytest.mark.asyncio
async def test_completion_streaming_output_len_excludes_special_tokens() -> None:
    chunk_texts = ["aa bb", "cc dd", "ee"]
    response = FakeStreamingResponse([_completion_sse(chunk_texts)])
    tokenizer = _bos_tokenizer()
    config = APIConfig(type=APIType.Completion, streaming=True)
    data = CompletionAPIData(prompt="p1 p2 p3", max_tokens=100)

    info = await data.process_response(cast(ClientResponse, response), config, tokenizer)

    assert isinstance(info.response_metrics, StreamedResponseMetrics)
    assert info.response_metrics.output_tokens == 3
    assert info.request_metrics.text is not None
    assert info.request_metrics.text.input_tokens == 4


@pytest.mark.asyncio
async def test_completion_unary_output_len_excludes_special_tokens() -> None:
    payload = {"choices": [{"text": "aa bb cc"}]}
    response = FakeUnaryResponse(payload)
    tokenizer = _bos_tokenizer()
    config = APIConfig(type=APIType.Completion, streaming=False)
    data = CompletionAPIData(prompt="p1 p2 p3", max_tokens=100)

    info = await data.process_response(cast(ClientResponse, response), config, tokenizer)

    assert isinstance(info.response_metrics, UnaryResponseMetrics)
    assert info.response_metrics.output_tokens == 3
    assert info.request_metrics.text is not None
    assert info.request_metrics.text.input_tokens == 4
