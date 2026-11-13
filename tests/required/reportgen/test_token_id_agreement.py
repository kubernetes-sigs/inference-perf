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
"""Unit tests for the exact token count check on responses that returned token ids (#917)."""

from typing import Optional
from unittest.mock import MagicMock

from inference_perf.apis.base import InferenceInfo, RequestLifecycleMetric, StreamedResponseMetrics, UnaryResponseMetrics
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.base import correct_streamed_response_metrics, summarize_requests

# A tool-call fragment that the word-counting tokenizer below reads as 2 tokens.
TOOL_CHUNK = '{"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"city\\": \\"Paris\\"}"}}]}}]}'


# A tokenizer that counts whitespace-separated words, so "a b c" is 3 tokens.
def make_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.count_tokens = MagicMock(side_effect=lambda text, **kwargs: len(text.split()))
    return tokenizer


# A streamed tool turn from 0s to 1s whose one visible chunk arrived at 0.5s.
# `token_times` is the timeline it carries, `ids` says whether it came from token ids,
# `server_tokens` is the server's completion_tokens (None for no usage at all).
def tool_turn(token_times: list[float], ids: bool, server_tokens: Optional[int]) -> RequestLifecycleMetric:
    server_usage: Optional[dict[str, int]] = None if server_tokens is None else {"completion_tokens": server_tokens}
    info = InferenceInfo(
        request_metrics=RequestMetrics(text=Text(input_tokens=5)),
        response_metrics=StreamedResponseMetrics(
            output_tokens=2,
            output_token_times=token_times,
            tool_chunks=[TOOL_CHUNK],
            tool_chunk_times=[0.5],
            has_tool_calls=True,
            token_ids_returned=ids,
            server_usage=server_usage,
        ),
    )
    return RequestLifecycleMetric(scheduled_time=0.0, start_time=0.0, end_time=1.0, request_data="req", info=info, error=None)


# 8 id stamps, server says 8. No mismatch, and the 8 stamps are left alone even
# though re-tokenizing the visible text would give 2.
def test_id_timeline_is_kept_and_agrees_with_the_server() -> None:
    times = [0.1, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.9]
    metric = tool_turn(list(times), ids=True, server_tokens=8)

    assert correct_streamed_response_metrics(metric, make_tokenizer()) is False

    assert isinstance(metric.info.response_metrics, StreamedResponseMetrics)
    assert metric.info.response_metrics.output_token_times == times


# 8 id stamps on a tool turn, server says 9, then 7. Each is a mismatch: one token
# off is far inside the 16-token tolerance the re-tokenized path allows tool turns.
def test_id_count_is_compared_exactly_on_a_tool_turn() -> None:
    times = [0.1, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.9]

    assert correct_streamed_response_metrics(tool_turn(list(times), ids=True, server_tokens=9), make_tokenizer()) is True
    assert correct_streamed_response_metrics(tool_turn(list(times), ids=True, server_tokens=7), make_tokenizer()) is True


# The same turn without ids: the visible text re-tokenizes to 2 against a server
# count of 9. That is inside the tool tolerance, so the fallback reports no mismatch
# and rewrites the timeline to the 2 re-tokenized stamps.
def test_fallback_still_retokenizes_with_its_tolerance() -> None:
    metric = tool_turn([0.5], ids=False, server_tokens=9)

    assert correct_streamed_response_metrics(metric, make_tokenizer()) is False

    assert isinstance(metric.info.response_metrics, StreamedResponseMetrics)
    assert metric.info.response_metrics.output_token_times == [0.5, 0.5]


# 3 id stamps and no usage from the server. Nothing to compare against, so no mismatch.
def test_id_path_without_server_usage_is_not_a_mismatch() -> None:
    assert (
        correct_streamed_response_metrics(tool_turn([0.1, 0.2, 0.3], ids=True, server_tokens=None), make_tokenizer()) is False
    )


# 3 id stamps, server says 4, and no tokenizer at all. The id check needs no
# tokenizer, so the mismatch is still found.
def test_id_path_needs_no_tokenizer() -> None:
    assert correct_streamed_response_metrics(tool_turn([0.1, 0.2, 0.3], ids=True, server_tokens=4), None) is True


# Three successes: an id tool turn that agrees (8 vs 8), an id tool turn one token
# off (8 vs 9), and a re-tokenized tool turn. The summary counts 2 id requests and
# 1 tool mismatch, the off-by-one turn.
def test_summary_counts_id_requests_and_their_exact_mismatches() -> None:
    times = [0.1, 0.5, 0.5, 0.6, 0.6, 0.7, 0.7, 0.9]
    metrics = [
        tool_turn(list(times), ids=True, server_tokens=8),
        tool_turn(list(times), ids=True, server_tokens=9),
        tool_turn([0.5], ids=False, server_tokens=9),
    ]

    successes = summarize_requests(metrics, [50], tokenizer=make_tokenizer()).successes

    assert successes["requests_with_token_ids"] == 2
    assert successes["tool_token_count_mismatches"] == 1
    assert successes["token_count_mismatches"] == 0


# One unary success and one re-tokenized streamed success, neither with ids.
# The count is 0, present in the summary rather than missing.
def test_summary_reports_zero_id_requests_when_none_returned_ids() -> None:
    unary = RequestLifecycleMetric(
        scheduled_time=0.0,
        start_time=0.0,
        end_time=1.0,
        request_data="req",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=5)),
            response_metrics=UnaryResponseMetrics(output_tokens=2, server_usage={"completion_tokens": 2}),
        ),
        error=None,
    )

    successes = summarize_requests(
        [unary, tool_turn([0.5], ids=False, server_tokens=2)], [50], tokenizer=make_tokenizer()
    ).successes

    assert successes["requests_with_token_ids"] == 0
