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
delta.content chunks were timestamped, so time_to_first_token was inflated to
time-to-first-CONTENT (prefill plus the entire reasoning-decode phase), and
null when the stream ended while still in the reasoning channel.

TTFT must anchor to the first generated token of ANY channel, matching
server-side metrics (vllm:time_to_first_token). TPOT, ITL, and output-length
stay content-based: reasoning is "thinking", not user-facing output.
"""

from typing import AsyncGenerator, List, Optional, cast
from unittest.mock import MagicMock

import pytest
from aiohttp import ClientResponse

from inference_perf.apis.base import InferenceInfo, RequestLifecycleMetric, StreamedResponseMetrics
from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.config import APIConfig, APIType
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.base import summarize_requests


def make_streamed_metric(
    start_time: float,
    end_time: float,
    output_token_times: List[float],
    reasoning_chunk_times: Optional[List[float]] = None,
    output_tokens: int = 0,
) -> RequestLifecycleMetric:
    """A successful streamed request with synthetic timestamps, bypassing the
    chunk re-parse (no tokenizer is passed to summarize_requests) so the TTFT
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
                reasoning_chunk_times=reasoning_chunk_times or [],
            ),
        ),
        error=None,
    )


def test_ttft_anchors_to_reasoning_not_first_content() -> None:
    """The inflated case: reasoning starts at t=2.0 but content only at t=5.0.
    Pre-#559 TTFT reported 4.0s (time-to-first-content, i.e. prefill + the
    whole reasoning-decode phase); it must report 1.0s, the first generated
    token. ITL and TPOT keep ignoring the reasoning channel."""
    metric = make_streamed_metric(
        start_time=1.0,
        end_time=6.0,
        output_token_times=[5.0, 5.1, 5.2],
        reasoning_chunk_times=[2.0, 2.5],
        output_tokens=3,
    )
    result = summarize_requests([metric], [50])

    ttft = result.successes["latency"]["time_to_first_token"]
    assert ttft is not None
    assert ttft["mean"] == pytest.approx(1.0)
    # ITL only spans content gaps (5.0->5.1->5.2); the 2.5s reasoning->content
    # gap must not appear in it.
    itl = result.successes["latency"]["inter_token_latency"]
    assert itl is not None
    assert itl["max"] == pytest.approx(0.1)


def test_ttft_defined_for_reasoning_only_stream() -> None:
    """The null case: the output budget is exhausted mid-reasoning so no
    content token ever arrives. Pre-#559 TTFT was null; it must report the
    first reasoning token, while TPOT and ITL stay undefined (no content)."""
    metric = make_streamed_metric(
        start_time=1.0,
        end_time=4.0,
        output_token_times=[],
        reasoning_chunk_times=[2.0, 2.5, 3.0],
    )
    result = summarize_requests([metric], [50])

    ttft = result.successes["latency"]["time_to_first_token"]
    assert ttft is not None, "reasoning-only stream must still yield a TTFT"
    assert ttft["mean"] == pytest.approx(1.0)
    assert result.successes["latency"]["time_per_output_token"] is None
    assert result.successes["latency"]["inter_token_latency"] is None


def test_ttft_unchanged_for_content_only_stream() -> None:
    """Non-reasoning responses must keep the exact pre-#559 semantics."""
    metric = make_streamed_metric(
        start_time=1.0,
        end_time=6.0,
        output_token_times=[3.0, 3.2, 3.4],
        output_tokens=3,
    )
    result = summarize_requests([metric], [50])

    ttft = result.successes["latency"]["time_to_first_token"]
    assert ttft is not None
    assert ttft["mean"] == pytest.approx(2.0)


def test_ttft_undefined_below_two_generation_events() -> None:
    """A single timestamped event across both channels is indistinguishable
    from a unary response, so TTFT stays undefined, matching the pre-#559
    guard against single-event streams."""
    metric = make_streamed_metric(
        start_time=1.0,
        end_time=4.0,
        output_token_times=[],
        reasoning_chunk_times=[2.0],
    )
    result = summarize_requests([metric], [50])

    assert result.successes["latency"]["time_to_first_token"] is None


def _build_reasoning_sse(reasoning_texts: List[str], content_texts: List[str], completion_tokens: int) -> bytes:
    parts = [f'data: {{"choices":[{{"delta":{{"reasoning_content":"{t}"}}}}]}}\n\n'.encode() for t in reasoning_texts]
    parts += [f'data: {{"choices":[{{"delta":{{"content":"{t}"}}}}]}}\n\n'.encode() for t in content_texts]
    parts.append(f'data: {{"choices":[],"usage":{{"completion_tokens":{completion_tokens}}}}}\n\n'.encode())
    parts.append(b"data: [DONE]\n\n")
    return b"".join(parts)


class FakeStreamingResponse:
    """Minimal aiohttp ClientResponse stand-in that yields preset SSE bytes."""

    def __init__(self, body: bytes) -> None:
        self.status = 200
        self.content = MagicMock()

        async def iter_any() -> AsyncGenerator[bytes, None]:
            yield body

        self.content.iter_any = iter_any


@pytest.mark.asyncio
async def test_pipeline_output_len_content_only_and_accounting_includes_reasoning() -> None:
    """Full pipeline (SSE bytes -> process_response -> summarize_requests):
    client output_len must count content tokens only, while the token-count
    mismatch check must include reasoning tokens, since the server's
    completion_tokens counts them. With a tokenizer matching the server the
    request must therefore NOT flag as mismatched."""
    reasoning_texts = ["think one two", " three four"]  # 5 tokens
    content_texts = ["answer is", " four ok"]  # 4 tokens
    sse = _build_reasoning_sse(reasoning_texts, content_texts, completion_tokens=9)

    tokenizer = MagicMock()
    tokenizer.count_tokens = MagicMock(side_effect=lambda text, **kwargs: len(text.split()))

    config = APIConfig(type=APIType.Chat, streaming=True)
    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="prompt")], max_tokens=100)
    info = await data.process_response(cast(ClientResponse, FakeStreamingResponse(sse)), config, tokenizer)

    assert isinstance(info.response_metrics, StreamedResponseMetrics)
    # output_tokens (client) is content-only: reasoning is not user-facing output.
    assert info.response_metrics.output_tokens == 4
    assert len(info.response_metrics.reasoning_chunks) == 2
    assert len(info.response_metrics.chunk_times) == 2

    metric = RequestLifecycleMetric(
        scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="prompt", info=info, error=None
    )
    result = summarize_requests([metric], [50], tokenizer=tokenizer)
    assert result.successes["token_count_mismatches"] == 0

    # The capped variant: no content at all, TTFT still defined end-to-end.
    capped_sse = _build_reasoning_sse(["think one two", "three four"], [], completion_tokens=5)
    capped_info = await data.process_response(cast(ClientResponse, FakeStreamingResponse(capped_sse)), config, tokenizer)
    assert isinstance(capped_info.response_metrics, StreamedResponseMetrics)
    assert capped_info.response_metrics.output_tokens == 0

    capped_metric = RequestLifecycleMetric(
        scheduled_time=0.0, start_time=0.0, end_time=10.0, request_data="prompt", info=capped_info, error=None
    )
    capped_result = summarize_requests([capped_metric], [50], tokenizer=tokenizer)
    assert capped_result.successes["latency"]["time_to_first_token"] is not None
    assert capped_result.successes["token_count_mismatches"] == 0
