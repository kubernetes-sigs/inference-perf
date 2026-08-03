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
"""Integration test for issue #559 (#606 Integration tier, per-change lane).

Drives the full client pipeline (real HTTP request, real SSE wire bytes,
real timing) against an in-process fake reasoning model and asserts that
reported TTFT anchors to the first reasoning token, using the server's own
send timestamps as the oracle. The unit tests in tests/test_reasoning_ttft.py
pin the arithmetic with synthetic timestamps; this test pins the wiring: that
a reasoning delta on the wire actually reaches the TTFT calculation.
"""

import time
from typing import Tuple
from unittest.mock import MagicMock

import aiohttp
import pytest

from fake_openai_server import FakeOpenAIServer, ServedStream, StreamEvent

from inference_perf.apis.base import RequestLifecycleMetric, StreamedResponseMetrics
from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.config import APIConfig, APIType
from inference_perf.reportgen.base import ResponsesSummary, summarize_requests


def make_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.count_tokens = MagicMock(side_effect=lambda text, **kwargs: len(text.split()))
    return tokenizer


async def run_request(server: FakeOpenAIServer) -> Tuple[ResponsesSummary, RequestLifecycleMetric, ServedStream, float]:
    """One benchmark request against the fake, through the real parse and
    summarize path. Returns (summary, metric, what the server did, start)."""
    tokenizer = make_tokenizer()
    config = APIConfig(type=APIType.Chat, streaming=True)
    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="prompt")], max_tokens=100)

    async with aiohttp.ClientSession() as session:
        start = time.perf_counter()
        async with session.post(server.url, json={}) as response:
            info = await data.process_response(response, config, tokenizer)
        end = time.perf_counter()

    metric = RequestLifecycleMetric(
        scheduled_time=start, start_time=start, end_time=end, request_data="prompt", info=info, error=None
    )
    return summarize_requests([metric], [50], tokenizer=tokenizer), metric, server.served[-1], start


@pytest.mark.asyncio
async def test_ttft_anchors_to_first_reasoning_token_on_the_wire() -> None:
    """The inflated-TTFT case: reasoning streams promptly, content only after a
    long reasoning phase. TTFT must land at the first reasoning chunk, not be
    inflated by the reasoning-decode phase to the first content chunk."""
    reasoning_phase = [StreamEvent("reasoning", "Let me", 0.02), StreamEvent("reasoning", " think.", 0.02)]
    # The pause before content is the reasoning-decode phase TTFT must NOT
    # include. Large relative to loopback parse latency so the bound is robust.
    content_phase = [StreamEvent("content", "The answer", 0.35), StreamEvent("content", " is 4.", 0.02)]

    async with FakeOpenAIServer(reasoning_phase + content_phase, completion_tokens=7) as server:
        result, metric, served, start = await run_request(server)

    ttft_summary = result.successes["latency"]["time_to_first_token"]
    assert ttft_summary is not None
    ttft = ttft_summary["mean"]
    # The client cannot observe a chunk before the server sent it (same
    # perf_counter clock, same process), so TTFT is at least the first
    # reasoning send offset...
    assert ttft >= served.reasoning_send_times[0] - start - 1e-6
    # ...and anchoring to reasoning means it must come in strictly before the
    # first content chunk was even sent. Pre-#559 this read ~0.4s, not ~0.02s.
    assert ttft < served.content_send_times[0] - start

    # Output length stays content-based: reasoning is thinking, not output.
    assert isinstance(metric.info.response_metrics, StreamedResponseMetrics)
    assert metric.info.response_metrics.output_tokens == 4  # "The answer is 4."
    # Client accounting (content + reasoning) matches the server's
    # completion_tokens, so the mismatch detector stays quiet.
    assert result.successes["token_count_mismatches"] == 0


@pytest.mark.asyncio
async def test_ttft_reported_when_output_budget_exhausted_mid_reasoning() -> None:
    """The null-TTFT case: max_tokens below the reasoning length means the
    stream ends while still in the reasoning channel, so no content token ever
    arrives. TTFT must still be reported; TPOT and ITL stay undefined."""
    script = [
        StreamEvent("reasoning", "Thinking", 0.02),
        StreamEvent("reasoning", " quite", 0.02),
        StreamEvent("reasoning", " hard", 0.02),
    ]

    async with FakeOpenAIServer(script, completion_tokens=3) as server:
        result, metric, served, start = await run_request(server)

    ttft_summary = result.successes["latency"]["time_to_first_token"]
    assert ttft_summary is not None, "reasoning-only stream must yield a TTFT, not null"
    assert ttft_summary["mean"] >= served.reasoning_send_times[0] - start - 1e-6

    assert result.successes["latency"]["time_per_output_token"] is None
    assert result.successes["latency"]["inter_token_latency"] is None
    assert isinstance(metric.info.response_metrics, StreamedResponseMetrics)
    assert metric.info.response_metrics.output_tokens == 0
