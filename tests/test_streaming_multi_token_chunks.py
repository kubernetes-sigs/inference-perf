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
"""Regression test for issue #364.

Pre-#410, output_tokens was effectively len(response_chunks): each SSE chunk
was charged as one token regardless of how many tokens its delta contained.
For servers that batch multiple tokens per chunk (typical for vLLM under
load), this undercounted output_tokens by the average chunk size, producing
the 2x TPOT / 0.5x OTPS gap reported against `vllm bench`.

This test exercises the full pipeline (raw SSE bytes -> parse_sse_stream ->
ChatCompletionAPIData.process_response -> summarize_requests) with
multi-token chunks and asserts that output_tokens reflects actual token
counts, not chunk counts.
"""

from typing import AsyncGenerator, List, cast
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientResponse

from inference_perf.apis.base import (
    RequestLifecycleMetric,
    StreamedInferenceResponseInfo,
)
from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.config import APIConfig, APIType
from inference_perf.reportgen.base import summarize_requests


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


def _build_sse_stream(chunk_texts: List[str], completion_tokens: int) -> bytes:
    """Build an OpenAI-style SSE byte stream with one delta per chunk and a
    trailing usage chunk (as emitted with stream_options.include_usage=true).
    """
    parts = [f'data: {{"choices":[{{"delta":{{"content":"{text}"}}}}]}}\n\n'.encode() for text in chunk_texts]
    parts.append(f'data: {{"choices":[],"usage":{{"completion_tokens":{completion_tokens}}}}}\n\n'.encode())
    parts.append(b"data: [DONE]\n\n")
    return b"".join(parts)


@pytest.mark.asyncio
async def test_multi_token_chunks_count_tokens_not_chunks() -> None:
    """Each chunk's delta carries multiple tokens; output_tokens must equal
    the per-chunk token sum, not the chunk count.

    Pre-#410: output_tokens == len(chunks) == 4. TPOT computed against 4
    "tokens" overstates per-token latency by ~2x for 2-tokens-per-chunk
    streams. Post-#410 the per-chunk tokenizer count drives output_tokens,
    so output_tokens == 8.
    """
    chunk_texts = ["aa bb", "cc dd", "ee ff", "gg hh"]
    sse = _build_sse_stream(chunk_texts, completion_tokens=8)
    response = FakeStreamingResponse([sse])

    tokenizer = MagicMock()
    tokenizer.count_tokens = MagicMock(side_effect=lambda text: len(text.split()))

    config = APIConfig(type=APIType.Chat, streaming=True)
    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="prompt")], max_tokens=100)

    info = await data.process_response(cast(ClientResponse, response), config, tokenizer)

    assert isinstance(info.response_info, StreamedInferenceResponseInfo)
    assert info.response_info.server_usage == {"completion_tokens": 8}
    assert len(info.response_info.response_chunks) == len(chunk_texts)

    metric = RequestLifecycleMetric(
        scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="prompt", info=info, error=None
    )
    summarize_requests([metric], [50], tokenizer=tokenizer)

    assert isinstance(metric.info.response_info, StreamedInferenceResponseInfo)
    # Sum of per-chunk tokens, NOT chunk count. This is the assertion that
    # would fail under the pre-#410 implementation.
    assert metric.info.response_info.output_tokens == 8
    assert metric.info.response_info.output_tokens != len(chunk_texts)
    # Per-token timestamps expanded so TPOT/ITL distributions are accurate.
    assert len(metric.info.response_info.output_token_times) == 8


@pytest.mark.asyncio
async def test_multi_token_chunks_match_server_usage() -> None:
    """When server_usage.completion_tokens is present, the client-derived
    count and the server-reported count should agree for a stream where the
    tokenizer matches the server. This is what eliminates the gap against
    `vllm bench`, which reads completion_tokens directly.
    """
    # Simulate a long stream of multi-token chunks - the regime where #364
    # originally surfaced (1k input, 8k output, multi-token chunks).
    tokens_per_chunk = 4
    num_chunks = 64
    chunk_texts = [" ".join([f"t{i}_{j}" for j in range(tokens_per_chunk)]) for i in range(num_chunks)]
    expected_total = tokens_per_chunk * num_chunks

    sse = _build_sse_stream(chunk_texts, completion_tokens=expected_total)
    response = FakeStreamingResponse([sse])

    tokenizer = MagicMock()
    tokenizer.count_tokens = MagicMock(side_effect=lambda text: len(text.split()))

    config = APIConfig(type=APIType.Chat, streaming=True)
    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="prompt")], max_tokens=expected_total)

    info = await data.process_response(cast(ClientResponse, response), config, tokenizer)
    metric = RequestLifecycleMetric(
        scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="prompt", info=info, error=None
    )
    result = summarize_requests([metric], [50], tokenizer=tokenizer)

    assert isinstance(metric.info.response_info, StreamedInferenceResponseInfo)
    assert metric.info.response_info.output_tokens == expected_total
    assert metric.info.response_info.server_usage == {"completion_tokens": expected_total}

    # No mismatches between client tokenization and server usage when the
    # tokenizer matches; this is the normal-operation expectation.
    assert result is not None
    assert result.successes["token_count_mismatches"] == 0
