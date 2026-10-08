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
"""Regression tests for issue #559.

Reasoning models emit reasoning-channel tokens before (and, when the output
budget is exhausted, instead of) any content token. Pre-#559 only
delta.content chunks were timestamped, so TTFT was inflated to
time-to-first-content and null when the stream ended mid-reasoning, while
TPOT divided a content-only span by a server token count that includes
reasoning.

The server generates and counts reasoning tokens like any other
(vllm:time_to_first_token, usage.completion_tokens), so TTFT, TPOT, ITL and
output length count both channels. Time to first output token is the
content-only view: the first content token, after any reasoning.
"""

import json
from typing import Any, AsyncGenerator, List, Optional, cast
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientResponse

from inference_perf.apis.anthropic_messages import AnthropicMessagesAPIData
from inference_perf.apis.base import InferenceInfo, RequestLifecycleMetric, StreamedResponseMetrics, UnaryResponseMetrics
from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.config import APIConfig, APIType
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.base import summarize_requests


# A successful streamed request built from explicit timestamps. output_token_times
# is the merged timeline; chunk_times the content chunks; reasoning_chunk_times the
# reasoning-only chunks. No tokenizer, so the timestamps are used as given.
def make_streamed_metric(
    start_time: float,
    end_time: float,
    output_token_times: List[float],
    chunk_times: Optional[List[float]] = None,
    reasoning_chunk_times: Optional[List[float]] = None,
    output_tokens: int = 0,
    server_usage: Optional[dict[str, Any]] = None,
) -> RequestLifecycleMetric:
    """A successful streamed request with synthetic timestamps, bypassing the
    chunk re-parse (no tokenizer is passed to summarize_requests) so the
    arithmetic is tested against exact, controlled inputs."""
    return RequestLifecycleMetric(
        scheduled_time=start_time,
        start_time=start_time,
        end_time=end_time,
        request_data="prompt",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=1)),
            response_metrics=StreamedResponseMetrics(
                output_tokens=output_tokens,
                output_token_times=output_token_times,
                chunk_times=chunk_times or [],
                reasoning_chunk_times=reasoning_chunk_times or [],
                server_usage=server_usage,
            ),
        ),
        error=None,
    )


# Start 1.0, reasoning at 2.0 and 2.5, content at 5.0, 5.1, 5.2 (5 tokens).
# TTFT = 1.0 (first reasoning), TTFO = 4.0 (first content), TPOT = 3.2/4 = 0.8,
# ITL max = 2.5 (the reasoning-to-content gap is decode time too).
def test_all_latency_metrics_count_reasoning_tokens() -> None:
    """The inflated case: pre-#559 TTFT reported 4.0s, the whole reasoning
    phase. TTFT now anchors to the first reasoning token, and TPOT/ITL span
    the whole generation. TTFO keeps the content-only view."""
    metric = make_streamed_metric(
        start_time=1.0,
        end_time=6.0,
        output_token_times=[2.0, 2.5, 5.0, 5.1, 5.2],
        chunk_times=[5.0, 5.1, 5.2],
        reasoning_chunk_times=[2.0, 2.5],
        output_tokens=5,
    )
    latency = summarize_requests([metric], [50]).successes["latency"]

    assert latency["time_to_first_token"]["mean"] == pytest.approx(1.0)
    assert latency["time_to_first_output_token"]["mean"] == pytest.approx(4.0)
    assert latency["time_per_output_token"]["mean"] == pytest.approx(0.8)
    assert latency["inter_token_latency"]["max"] == pytest.approx(2.5)


# Start 1.0, reasoning only at 2.0, 2.5, 3.0 (3 tokens), no content.
# TTFT = 1.0, TPOT = ITL = 0.5, TTFO = None (no content ever arrived).
def test_reasoning_only_stream_has_no_first_output_token() -> None:
    """The null case: the output budget is exhausted mid-reasoning. Pre-#559
    TTFT, TPOT and ITL were all null; they are now defined, and TTFO is the
    metric that reports the missing content."""
    metric = make_streamed_metric(
        start_time=1.0,
        end_time=4.0,
        output_token_times=[2.0, 2.5, 3.0],
        reasoning_chunk_times=[2.0, 2.5, 3.0],
        output_tokens=3,
    )
    latency = summarize_requests([metric], [50]).successes["latency"]

    assert latency["time_to_first_token"]["mean"] == pytest.approx(1.0)
    assert latency["time_per_output_token"]["mean"] == pytest.approx(0.5)
    assert latency["inter_token_latency"]["mean"] == pytest.approx(0.5)
    assert latency["time_to_first_output_token"] is None


# Start 1.0, content at 3.0, 3.2, 3.4, no reasoning. Every metric matches the
# pre-#559 values: TTFT = 2.0, TPOT = ITL = 0.2. TTFO = TTFT = 2.0.
def test_content_only_stream_is_unchanged() -> None:
    """Non-reasoning models must see identical numbers, plus a TTFO that
    equals TTFT."""
    metric = make_streamed_metric(
        start_time=1.0,
        end_time=6.0,
        output_token_times=[3.0, 3.2, 3.4],
        chunk_times=[3.0, 3.2, 3.4],
        output_tokens=3,
    )
    latency = summarize_requests([metric], [50]).successes["latency"]

    assert latency["time_to_first_token"]["mean"] == pytest.approx(2.0)
    assert latency["time_to_first_output_token"]["mean"] == pytest.approx(2.0)
    assert latency["time_per_output_token"]["mean"] == pytest.approx(0.2)
    assert latency["inter_token_latency"]["mean"] == pytest.approx(0.2)


# Same stream as above but with only output_token_times set, as records built
# before the channel split are. TTFO falls back to that timeline: 2.0.
def test_first_output_token_for_records_without_chunk_times() -> None:
    """A record with no chunk_times and no reasoning predates the split, so
    its output_token_times is content."""
    metric = make_streamed_metric(start_time=1.0, end_time=6.0, output_token_times=[3.0, 3.2, 3.4], output_tokens=3)
    latency = summarize_requests([metric], [50]).successes["latency"]

    assert latency["time_to_first_output_token"]["mean"] == pytest.approx(2.0)


# One reasoning chunk at 2.0 and nothing else. TTFT and TTFO are both None.
def test_single_generation_event_is_not_streamable() -> None:
    """A single timestamped event is indistinguishable from a unary response,
    matching the pre-#559 guard against single-event streams."""
    metric = make_streamed_metric(
        start_time=1.0, end_time=4.0, output_token_times=[2.0], reasoning_chunk_times=[2.0], output_tokens=1
    )
    latency = summarize_requests([metric], [50]).successes["latency"]

    assert latency["time_to_first_token"] is None
    assert latency["time_to_first_output_token"] is None


# Timeline 2.0 to 5.2 (3.2s) over 5 tokens, server completion_tokens = 5 with
# use_server_output_tokens on. TPOT = 3.2 / 4 = 0.8, same as the client count.
def test_server_output_tokens_flag_divides_the_same_span() -> None:
    """Pre-#559 the flag divided the content-only span (5.0 to 5.2) by a
    server count that includes reasoning, deflating TPOT to 0.05. Numerator
    and denominator now cover the same tokens."""
    metric = make_streamed_metric(
        start_time=1.0,
        end_time=6.0,
        output_token_times=[2.0, 2.5, 5.0, 5.1, 5.2],
        chunk_times=[5.0, 5.1, 5.2],
        reasoning_chunk_times=[2.0, 2.5],
        output_tokens=3,
        server_usage={"completion_tokens": 5},
    )
    latency = summarize_requests([metric], [50], use_server_output_tokens=True).successes["latency"]

    assert latency["time_per_output_token"]["mean"] == pytest.approx(0.8)


# SSE bytes for a chat stream: one reasoning_content chunk per reasoning text,
# one content chunk per content text, then a usage chunk and [DONE].
def _build_reasoning_sse(reasoning_texts: List[str], content_texts: List[str], completion_tokens: int) -> bytes:
    parts = [f'data: {{"choices":[{{"delta":{{"reasoning_content":"{t}"}}}}]}}\n\n'.encode() for t in reasoning_texts]
    parts += [f'data: {{"choices":[{{"delta":{{"content":"{t}"}}}}]}}\n\n'.encode() for t in content_texts]
    parts.append(f'data: {{"choices":[],"usage":{{"completion_tokens":{completion_tokens}}}}}\n\n'.encode())
    parts.append(b"data: [DONE]\n\n")
    return b"".join(parts)


# SSE bytes from a list of JSON events, each framed as its own data: message.
def _build_sse(events: List[dict[str, Any]]) -> bytes:
    return b"".join(f"data: {json.dumps(event)}\n\n".encode() for event in events) + b"data: [DONE]\n\n"


# Tokenizer stub: one token per whitespace-separated word.
def _word_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.count_tokens = MagicMock(side_effect=lambda text, **kwargs: len(text.split()))
    return tokenizer


# An aiohttp ClientResponse stand-in that yields the given SSE bytes in one read,
# or returns the given JSON body for a unary response.
class FakeStreamingResponse:
    """Minimal aiohttp ClientResponse stand-in that yields preset SSE bytes."""

    def __init__(self, body: bytes = b"", json_body: Optional[dict[str, Any]] = None) -> None:
        self.status = 200
        self.content = MagicMock()
        self._json_body = json_body

        async def iter_any() -> AsyncGenerator[bytes, None]:
            yield body

        self.content.iter_any = iter_any

    async def json(self) -> Optional[dict[str, Any]]:
        return self._json_body


# 2 reasoning chunks (5 words) then 2 content chunks (4 words), server says 9.
# Client output_tokens = 9, the token timeline has 9 entries after correction,
# and nothing is flagged as a mismatch. Capped variant: 5 reasoning words only,
# still re-tokenized into 5 timeline entries.
@pytest.mark.asyncio
async def test_pipeline_counts_reasoning_in_output_and_timeline() -> None:
    """Full pipeline (SSE bytes -> process_response -> summarize_requests)
    with a tokenizer matching the server's count."""
    sse = _build_reasoning_sse(["think one two", " three four"], ["answer is", " four ok"], completion_tokens=9)
    tokenizer = _word_tokenizer()

    config = APIConfig(type=APIType.Chat, streaming=True)
    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="prompt")], max_tokens=100)
    info = await data.process_response(cast(ClientResponse, FakeStreamingResponse(sse)), config, tokenizer)

    response_metrics = info.response_metrics
    assert isinstance(response_metrics, StreamedResponseMetrics)
    assert response_metrics.output_tokens == 9
    assert len(response_metrics.reasoning_chunks) == 2
    assert len(response_metrics.chunk_times) == 2
    assert response_metrics.output_token_times == sorted(response_metrics.reasoning_chunk_times + response_metrics.chunk_times)

    metric = RequestLifecycleMetric(
        scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="prompt", info=info, error=None
    )
    result = summarize_requests([metric], [50], tokenizer=tokenizer)
    assert result.successes["token_count_mismatches"] == 0
    assert len(response_metrics.output_token_times) == 9
    assert response_metrics.output_token_times[0] == response_metrics.reasoning_chunk_times[0]

    capped_sse = _build_reasoning_sse(["think one two", " three four"], [], completion_tokens=5)
    capped_info = await data.process_response(cast(ClientResponse, FakeStreamingResponse(capped_sse)), config, tokenizer)
    assert isinstance(capped_info.response_metrics, StreamedResponseMetrics)
    assert capped_info.response_metrics.output_tokens == 5

    capped_metric = RequestLifecycleMetric(
        scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="prompt", info=capped_info, error=None
    )
    capped_result = summarize_requests([capped_metric], [50], tokenizer=tokenizer)
    assert capped_result.successes["latency"]["time_to_first_token"] is not None
    assert capped_result.successes["latency"]["time_to_first_output_token"] is None
    assert capped_result.successes["token_count_mismatches"] == 0
    # Re-tokenized per token, not left at one timestamp per chunk.
    assert len(capped_info.response_metrics.output_token_times) == 5


# One chunk carrying reasoning "a b" and content "c", then content "d".
# The first chunk is timestamped once, as content; output_tokens = 4 (a b c d).
@pytest.mark.asyncio
async def test_chunk_with_both_channels_counts_both() -> None:
    """A chunk bearing reasoning and content is recorded once on the
    timeline, and its reasoning still counts toward the output."""
    sse = _build_sse(
        [
            {"choices": [{"delta": {"reasoning_content": "a b", "content": "c"}}]},
            {"choices": [{"delta": {"content": " d"}}]},
        ]
    )
    config = APIConfig(type=APIType.Chat, streaming=True)
    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="prompt")], max_tokens=100)
    info = await data.process_response(cast(ClientResponse, FakeStreamingResponse(sse)), config, _word_tokenizer())

    response_metrics = info.response_metrics
    assert isinstance(response_metrics, StreamedResponseMetrics)
    assert response_metrics.output_tokens == 4
    assert len(response_metrics.chunk_times) == 2
    assert response_metrics.reasoning_chunks == []
    assert len(response_metrics.output_token_times) == 2


# Non-streamed chat: content "the answer" plus reasoning_content "x y z".
# output_tokens = 5.
@pytest.mark.asyncio
async def test_unary_chat_counts_reasoning() -> None:
    """The non-streamed path counts message.reasoning_content like the
    streamed path does."""
    body = {"choices": [{"message": {"content": "the answer", "reasoning_content": "x y z"}}]}
    config = APIConfig(type=APIType.Chat, streaming=False)
    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="prompt")], max_tokens=100)
    info = await data.process_response(cast(ClientResponse, FakeStreamingResponse(json_body=body)), config, _word_tokenizer())

    assert isinstance(info.response_metrics, UnaryResponseMetrics)
    assert info.response_metrics.output_tokens == 5


# Anthropic stream: a thinking block with two thinking_deltas, then a text block
# with two text_deltas, no usage. The timeline has 4 entries, thinking first;
# chunk_times has the 2 text deltas; output_tokens falls back to 3 + 2 words.
@pytest.mark.asyncio
async def test_anthropic_thinking_counts_in_timeline_and_output() -> None:
    """Messages API thinking deltas are the reasoning channel."""
    sse = _build_sse(
        [
            {"type": "content_block_start", "index": 0, "content_block": {"type": "thinking", "thinking": ""}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": "one two"}},
            {"type": "content_block_delta", "index": 0, "delta": {"type": "thinking_delta", "thinking": " three"}},
            {"type": "content_block_start", "index": 1, "content_block": {"type": "text", "text": ""}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": "hi"}},
            {"type": "content_block_delta", "index": 1, "delta": {"type": "text_delta", "text": " there"}},
        ]
    )
    config = APIConfig(type=APIType.AnthropicMessages, streaming=True)
    data = AnthropicMessagesAPIData(messages=[ChatMessage(role="user", content="prompt")], max_tokens=100)
    info = await data.process_response(cast(ClientResponse, FakeStreamingResponse(sse)), config, _word_tokenizer())

    response_metrics = info.response_metrics
    assert isinstance(response_metrics, StreamedResponseMetrics)
    assert len(response_metrics.reasoning_chunk_times) == 2
    assert len(response_metrics.chunk_times) == 2
    assert response_metrics.output_token_times[:2] == response_metrics.reasoning_chunk_times
    assert response_metrics.output_tokens == 5
    assert info.extra_info["output_text"] == "hi there"


# Non-streamed Anthropic, no usage: a thinking block "a b c" and a text block
# "done". The fallback output_tokens = 4.
@pytest.mark.asyncio
async def test_anthropic_unary_counts_thinking_without_usage() -> None:
    """The fallback count matches usage.output_tokens, which counts thinking."""
    body = {"content": [{"type": "thinking", "thinking": "a b c"}, {"type": "text", "text": "done"}]}
    config = APIConfig(type=APIType.AnthropicMessages, streaming=False)
    data = AnthropicMessagesAPIData(messages=[ChatMessage(role="user", content="prompt")], max_tokens=100)
    info = await data.process_response(cast(ClientResponse, FakeStreamingResponse(json_body=body)), config, _word_tokenizer())

    assert isinstance(info.response_metrics, UnaryResponseMetrics)
    assert info.response_metrics.output_tokens == 4
