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
"""Integration test for the token timeline built from returned token ids (#917).

Drives the real client against a fake server that streams one tool call the way
vLLM does with ``return_token_ids``: the tokens its tool parser swallows (the
wrapper before the call, the end token after it) arrive on chunks with no text,
each chunk carrying the ids it generated. The server's own send stamps are the
oracle for the timeline.
"""

from typing import List, Optional, Tuple

import pytest

from fake_openai_server import FakeOpenAIServer, ServedStream, StreamEvent
from openai_client_harness import make_tokenizer, run_request_against

from inference_perf.apis.base import RequestLifecycleMetric, StreamedResponseMetrics
from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.config import APIConfig, APIType
from inference_perf.reportgen.base import ResponsesSummary, summarize_requests

# One tool call, 8 generated tokens. The wrapper token goes out 20ms in and the
# first text the client can see 300ms after that, so a TTFT anchored to the
# wrapper and one anchored to the text are far apart. The harness tokenizer
# counts words, so the visible text (name plus two argument fragments) is 3.
TOOL_CALL_SCRIPT: List[StreamEvent] = [
    StreamEvent("hidden", "", 0.02, token_ids=(101,)),
    StreamEvent("tool", "", 0.3, token_ids=(102, 103), tool_name="get_weather"),
    StreamEvent("tool", '{"city":', 0.02, token_ids=(104, 105)),
    StreamEvent("tool", ' "Paris"}', 0.02, token_ids=(106, 107)),
    StreamEvent("hidden", "", 0.02, token_ids=(108,)),
]
GENERATED_TOKENS = 8


# One chat request through the real client against the fake, summarized.
# `return_token_ids` is the api config flag, `completion_tokens` what the server
# claims in usage. Returns (summary, the recorded metric, what the server did).
async def run_tool_turn(
    return_token_ids: bool, completion_tokens: Optional[int] = GENERATED_TOKENS
) -> Tuple[ResponsesSummary, RequestLifecycleMetric, ServedStream]:
    async with FakeOpenAIServer(TOOL_CALL_SCRIPT, completion_tokens=completion_tokens) as server:
        metric = await run_request_against(
            server.base_url,
            api_config=APIConfig(type=APIType.Chat, streaming=True, return_token_ids=return_token_ids),
            data=ChatCompletionAPIData(messages=[ChatMessage(role="user", content="weather in Paris?")], max_tokens=64),
        )
        served = server.served[-1]
    assert metric.error is None, f"request failed: {metric.error}"
    return summarize_requests([metric], [50], tokenizer=make_tokenizer()), metric, served


# Flag on, server sends 8 ids and says 8. The request body carries return_token_ids,
# the timeline has 8 stamps each at or after the server's send stamp for that id,
# no mismatch is flagged, and the summary counts 1 id request.
@pytest.mark.asyncio
async def test_timeline_comes_from_the_ids_the_server_sent() -> None:
    summary, metric, served = await run_tool_turn(return_token_ids=True)

    assert served.request_body["return_token_ids"] is True
    response_metrics = metric.info.response_metrics
    assert isinstance(response_metrics, StreamedResponseMetrics)
    assert response_metrics.token_ids_returned is True
    assert len(served.token_id_send_times) == GENERATED_TOKENS
    assert len(response_metrics.output_token_times) == GENERATED_TOKENS
    # same clock, same process: the client cannot see an id before the server sent it
    for seen, sent in zip(response_metrics.output_token_times, served.token_id_send_times, strict=True):
        assert seen >= sent - 1e-6
    assert response_metrics.output_token_times == sorted(response_metrics.output_token_times)

    assert summary.successes["requests_with_token_ids"] == 1
    assert summary.successes["tool_token_count_mismatches"] == 0
    assert summary.successes["token_count_mismatches"] == 0
    # output_len stays the client's count of the visible text
    assert response_metrics.output_tokens == 3


# Flag on. TTFT lands on the wrapper token, sent 300ms before the first visible
# tool text: at or after the wrapper's send stamp and before the first text was sent.
@pytest.mark.asyncio
async def test_ttft_anchors_to_a_token_the_tool_parser_swallowed() -> None:
    summary, metric, served = await run_tool_turn(return_token_ids=True)

    ttft = summary.successes["latency"]["time_to_first_token"]["mean"]
    wrapper_sent, first_text_sent = served.token_id_send_times[0], served.token_id_send_times[1]
    assert ttft >= wrapper_sent - metric.start_time - 1e-6
    assert ttft < first_text_sent - metric.start_time


# Flag on, server sends 8 ids but says 9. One token off is a mismatch on the id
# path. The same gap is inside the 16-token tolerance of the re-tokenized path.
@pytest.mark.asyncio
async def test_one_token_off_is_a_mismatch_with_ids() -> None:
    summary, _, _ = await run_tool_turn(return_token_ids=True, completion_tokens=GENERATED_TOKENS + 1)

    assert summary.successes["requests_with_token_ids"] == 1
    assert summary.successes["tool_token_count_mismatches"] == 1


# Flag off (the default). The body has no return_token_ids, the server sends no
# ids, and the timeline is the 3 re-tokenized words of visible text. TTFT then
# cannot land before the first visible text was sent. The summary counts 0 id
# requests, and 3 vs 8 is inside the tool tolerance so nothing is flagged.
@pytest.mark.asyncio
async def test_without_the_flag_the_timeline_is_retokenized() -> None:
    summary, metric, served = await run_tool_turn(return_token_ids=False)

    assert "return_token_ids" not in served.request_body
    assert served.token_id_send_times == []
    response_metrics = metric.info.response_metrics
    assert isinstance(response_metrics, StreamedResponseMetrics)
    assert response_metrics.token_ids_returned is False
    assert len(response_metrics.output_token_times) == 3

    assert summary.successes["requests_with_token_ids"] == 0
    assert summary.successes["tool_token_count_mismatches"] == 0
    ttft = summary.successes["latency"]["time_to_first_token"]["mean"]
    assert ttft >= 0.3
