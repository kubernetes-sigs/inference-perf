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
"""Tests for the BR0.2 ``build_results`` adapter.

The adapter is strictly results-only — it derives a ``Results`` from
inference-perf request metrics and nothing else. Run/scenario assembly is
the partial-report's job (see ``test_partial_report.py``).
"""

import json
import time
from typing import List, Optional, Tuple

import pytest

from inference_perf.apis import (
    ErrorResponseInfo,
    InferenceInfo,
    RequestLifecycleMetric,
    SessionLifecycleMetric,
    StreamedResponseMetrics,
    UnaryResponseMetrics,
)
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.br.v0_2 import build_results
from inference_perf.reportgen.br.v0_2.schema import Units
from inference_perf.utils.custom_tokenizer import CustomTokenizer


def _streaming_metric(start: float, output_tokens: int = 10, itl: float = 0.02, ttft: float = 0.05) -> RequestLifecycleMetric:
    times = [start + ttft + i * itl for i in range(output_tokens)]
    return RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=start - 0.001,
        start_time=start,
        end_time=times[-1] + 0.005,
        request_data="{}",
        response_data="ok",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=128)),
            response_metrics=StreamedResponseMetrics(
                response_chunks=[],
                chunk_times=times,
                output_tokens=output_tokens,
                output_token_times=times,
            ),
        ),
        error=None,
    )


def test_build_results_aggregate_request_counts_match_inputs() -> None:
    now = time.time()
    metrics: List[RequestLifecycleMetric] = [_streaming_metric(now + i * 0.1) for i in range(30)]
    metrics[2].error = ErrorResponseInfo(error_type="timeout", error_msg="boom")
    metrics[17].error = ErrorResponseInfo(error_type="timeout", error_msg="boom")

    results = build_results(metrics)
    assert results.request_performance is not None
    aggregate = results.request_performance.aggregate
    assert aggregate is not None
    assert aggregate.requests is not None
    assert aggregate.requests.total == 30
    assert aggregate.requests.failures == 2


def test_build_results_latency_calculations() -> None:
    now = time.time()
    metrics = [_streaming_metric(now + i * 0.1, output_tokens=11, itl=0.02, ttft=0.05) for i in range(10)]

    results = build_results(metrics)
    assert results.request_performance is not None
    aggregate = results.request_performance.aggregate
    assert aggregate is not None and aggregate.latency is not None

    latency = aggregate.latency
    assert latency.time_to_first_token is not None
    assert latency.time_per_output_token is not None
    assert latency.inter_token_latency is not None
    assert latency.time_to_first_token.mean == pytest.approx(0.05, abs=1e-6)
    # 10 inter-token gaps of 0.02s each; TPOT = (last - first) / (n - 1) = 0.20 / 10 = 0.02
    assert latency.time_per_output_token.mean == pytest.approx(0.02, abs=1e-6)
    assert latency.inter_token_latency.mean == pytest.approx(0.02, abs=1e-6)


# Ten streamed requests with one token each, arriving 0.05s after the request starts.
# TTFT mean = 0.05. TPOT and ITL are absent: neither exists with one token.
def test_build_results_single_token_streams_have_ttft() -> None:
    now = time.time()
    metrics = [_streaming_metric(now + i * 0.1, output_tokens=1, ttft=0.05) for i in range(10)]

    results = build_results(metrics)
    assert results.request_performance is not None
    aggregate = results.request_performance.aggregate
    assert aggregate is not None and aggregate.latency is not None

    latency = aggregate.latency
    assert latency.time_to_first_token is not None
    assert latency.time_to_first_token.mean == pytest.approx(0.05, abs=1e-6)
    assert latency.time_per_output_token is None
    assert latency.inter_token_latency is None


def test_build_results_aggregate_units() -> None:
    """The BR0.2 schema enforces unit compatibility per metric
    category. Guards that the adapter assigns units the schema accepts."""
    now = time.time()
    metrics = [_streaming_metric(now + i * 0.1) for i in range(10)]

    results = build_results(metrics)
    assert results.request_performance is not None
    aggregate = results.request_performance.aggregate
    assert (
        aggregate is not None
        and aggregate.requests is not None
        and aggregate.requests.input_length is not None
        and aggregate.latency is not None
        and aggregate.latency.time_to_first_token is not None
        and aggregate.latency.time_per_output_token is not None
        and aggregate.throughput is not None
        and aggregate.throughput.output_token_rate is not None
        and aggregate.throughput.request_rate is not None
    )
    assert aggregate.requests.input_length.units == Units.COUNT
    assert aggregate.latency.time_to_first_token.units == Units.S
    assert aggregate.latency.time_per_output_token.units == Units.S_PER_TOKEN
    assert aggregate.throughput.output_token_rate.units == Units.TOKEN_PER_S
    assert aggregate.throughput.request_rate.units == Units.QUERY_PER_S


def test_build_results_no_metrics_returns_empty_request_performance() -> None:
    results = build_results([])
    assert results.request_performance is None


def _unary_metric(
    start: float, end: float, output_tokens: int, server_completion_tokens: Optional[int] = None
) -> RequestLifecycleMetric:
    server_usage = {"completion_tokens": server_completion_tokens} if server_completion_tokens is not None else None
    return RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=start - 0.001,
        start_time=start,
        end_time=end,
        request_data="{}",
        response_data="ok",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=128)),
            response_metrics=UnaryResponseMetrics(output_tokens=output_tokens, server_usage=server_usage),
        ),
        error=None,
    )


def test_build_results_ntpot_for_non_streaming_requests() -> None:
    """NTPOT is derived from request latency and output_tokens, so it must be
    populated for non-streaming (unary) requests too, matching
    summarize_requests. Regression: an early version derived the count from
    the streamed token timeline and reported all-zero NTPOT for unary runs."""
    now = time.time()
    metrics = [_unary_metric(now + i, now + i + 2.0, output_tokens=10) for i in range(4)]

    results = build_results(metrics)
    assert results.request_performance is not None
    aggregate = results.request_performance.aggregate
    assert aggregate is not None and aggregate.latency is not None

    ntpot = aggregate.latency.normalized_time_per_output_token
    assert ntpot is not None
    assert ntpot.mean == pytest.approx(0.2, abs=1e-6)
    # Not measurable without a token timeline; must be absent, not zero.
    assert aggregate.latency.time_to_first_token is None
    assert aggregate.latency.time_per_output_token is None

    assert aggregate.requests is not None and aggregate.requests.output_length is not None
    assert aggregate.requests.output_length.mean == pytest.approx(10.0)


def test_build_results_all_failed_stage_omits_throughput() -> None:
    """A stage where every request failed has no measured throughput, so the
    block must be absent (merge-by-absence), not fabricated zero rates: a
    composer cannot tell "the server managed 0 tok/s" from "nothing
    succeeded", and downstream averaging would fold the zeros in as real
    measurements."""
    now = time.time()
    metrics = [_streaming_metric(now + i * 0.1) for i in range(5)]
    for metric in metrics:
        metric.error = ErrorResponseInfo(error_type="timeout", error_msg="boom")

    results = build_results(metrics)
    assert results.request_performance is not None
    aggregate = results.request_performance.aggregate
    assert aggregate is not None and aggregate.requests is not None
    assert aggregate.requests.total == 5
    assert aggregate.requests.failures == 5
    assert aggregate.throughput is None


def test_build_results_zero_output_success_counts_in_output_length() -> None:
    """A successful request that produced zero output tokens (immediate EOS,
    stop sequence) is a real measurement of output length; it must land in
    the output_length distribution like it does in the native lifecycle
    report, while staying out of the per-token latency metrics, which divide
    by the count."""
    now = time.time()
    metrics = [
        _unary_metric(now, now + 1.0, output_tokens=0),
        _unary_metric(now, now + 2.0, output_tokens=10),
    ]

    results = build_results(metrics)
    assert results.request_performance is not None
    aggregate = results.request_performance.aggregate
    assert aggregate is not None and aggregate.requests is not None and aggregate.latency is not None

    output_length = aggregate.requests.output_length
    assert output_length is not None
    assert output_length.mean == pytest.approx(5.0)
    assert output_length.min == pytest.approx(0.0)
    assert output_length.max == pytest.approx(10.0)

    # NTPOT is only measurable where tokens were produced; the zero-output
    # request must be skipped, not folded in as a fabricated 0.0.
    ntpot = aggregate.latency.normalized_time_per_output_token
    assert ntpot is not None
    assert ntpot.mean == pytest.approx(0.2, abs=1e-6)


def test_build_results_missing_output_count_excluded_from_output_length() -> None:
    """No response_metrics means the output count is unknown, which is
    different from a measured zero: unknowns stay out of the distribution,
    mirroring the native report's filter."""
    now = time.time()
    known = _unary_metric(now, now + 1.0, output_tokens=10)
    unknown = _unary_metric(now, now + 1.0, output_tokens=0)
    unknown.info.response_metrics = None

    results = build_results([known, unknown])
    assert results.request_performance is not None
    aggregate = results.request_performance.aggregate
    assert aggregate is not None and aggregate.requests is not None
    assert aggregate.requests.total == 2

    output_length = aggregate.requests.output_length
    assert output_length is not None
    assert output_length.mean == pytest.approx(10.0)
    assert output_length.min == pytest.approx(10.0)


def test_build_results_respects_use_server_output_tokens() -> None:
    """With use_server_output_tokens=True, token counts must come from the
    server's usage.completion_tokens so the BR partial agrees with the native
    lifecycle reports of the same run (#577/#607)."""
    now = time.time()
    metric = _streaming_metric(now, output_tokens=11, itl=0.02, ttft=0.05)
    response_metrics = metric.info.response_metrics
    assert response_metrics is not None
    response_metrics.server_usage = {"completion_tokens": 21}
    window = metric.end_time - metric.start_time

    client = build_results([metric])
    server = build_results([metric], use_server_output_tokens=True)

    for results, tokens in ((client, 11), (server, 21)):
        assert results.request_performance is not None
        aggregate = results.request_performance.aggregate
        assert aggregate is not None and aggregate.latency is not None and aggregate.throughput is not None
        assert aggregate.latency.time_per_output_token is not None
        assert aggregate.latency.normalized_time_per_output_token is not None
        assert aggregate.throughput.output_token_rate is not None
        # Token timeline spans 10 gaps of 0.02s regardless of the count source.
        assert aggregate.latency.time_per_output_token.mean == pytest.approx(0.2 / (tokens - 1), abs=1e-6)
        assert aggregate.latency.normalized_time_per_output_token.mean == pytest.approx(window / tokens, abs=1e-6)
        assert aggregate.throughput.output_token_rate.mean == pytest.approx(tokens / window, abs=1e-6)


class _FragmentTokenizer(CustomTokenizer):
    """Counts whitespace-delimited words and records the add_special_tokens
    flag of every call."""

    def __init__(self) -> None:
        self.special_tokens_flags: List[bool] = []

    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        self.special_tokens_flags.append(add_special_tokens)
        return len(text.split())


def test_chunk_retokenization_counts_fragments_without_special_tokens() -> None:
    """The chunk re-tokenization fallback (no output_token_times) must count
    each chunk as a sequence fragment. Counting with special tokens prepends
    a BOS per chunk, inflating counts and deflating ITL (#564)."""
    start = 100.0
    chunks = [
        json.dumps({"choices": [{"text": "hello world"}]}),
        json.dumps({"choices": [{"text": "again"}]}),
    ]
    metric = RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=start - 0.001,
        start_time=start,
        end_time=start + 0.8,
        request_data="{}",
        response_data="ok",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=128)),
            response_metrics=StreamedResponseMetrics(
                response_chunks=chunks,
                chunk_times=[start + 0.5, start + 0.7],
                output_tokens=3,
                output_token_times=[],
            ),
        ),
        error=None,
    )
    tokenizer = _FragmentTokenizer()

    results = build_results([metric], tokenizer=tokenizer)

    assert tokenizer.special_tokens_flags == [False, False]
    assert results.request_performance is not None
    aggregate = results.request_performance.aggregate
    assert aggregate is not None and aggregate.latency is not None
    assert aggregate.latency.time_to_first_token is not None
    assert aggregate.latency.time_to_first_token.mean == pytest.approx(0.5, abs=1e-9)
    # Timeline [t0, t0, t1]: intra-chunk gap 0, inter-chunk gap 0.2.
    assert aggregate.latency.inter_token_latency is not None
    assert aggregate.latency.inter_token_latency.mean == pytest.approx(0.1, abs=1e-9)
    # TPOT spans the timeline over the client whole-message count (3 tokens).
    assert aggregate.latency.time_per_output_token is not None
    assert aggregate.latency.time_per_output_token.mean == pytest.approx(0.2 / 2, abs=1e-9)


# Fallback path (no output_token_times). Reasoning "a b" at +0.2, content "c" at
# +0.5: timeline [+0.2, +0.2, +0.5], so TTFT = 0.2 and ITL mean = 0.3 / 2 = 0.15.
# Reasoning only, "a b" at +0.2 and "c" at +0.4: TTFT = 0.2 rather than missing.
@pytest.mark.parametrize(
    "content, reasoning, expected_itl",
    [
        ([(0.5, "c")], [(0.2, "a b")], 0.15),
        ([], [(0.2, "a b"), (0.4, "c")], 0.1),
    ],
)
def test_chunk_retokenization_counts_reasoning_chunks(
    content: List[Tuple[float, str]], reasoning: List[Tuple[float, str]], expected_itl: float
) -> None:
    """The fallback re-tokenizes reasoning chunks the same way the native
    reports do (#559)."""
    start = 100.0
    metric = RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=start - 0.001,
        start_time=start,
        end_time=start + 0.8,
        request_data="{}",
        response_data="ok",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=128)),
            response_metrics=StreamedResponseMetrics(
                response_chunks=[json.dumps({"choices": [{"delta": {"content": text}}]}) for _, text in content],
                chunk_times=[start + t for t, _ in content],
                reasoning_chunks=[json.dumps({"choices": [{"delta": {"reasoning": text}}]}) for _, text in reasoning],
                reasoning_chunk_times=[start + t for t, _ in reasoning],
                output_tokens=3,
                output_token_times=[],
            ),
        ),
        error=None,
    )

    results = build_results([metric], tokenizer=_FragmentTokenizer())

    assert results.request_performance is not None
    aggregate = results.request_performance.aggregate
    assert aggregate is not None and aggregate.latency is not None
    assert aggregate.latency.time_to_first_token is not None
    assert aggregate.latency.time_to_first_token.mean == pytest.approx(0.2, abs=1e-9)
    assert aggregate.latency.inter_token_latency is not None
    assert aggregate.latency.inter_token_latency.mean == pytest.approx(expected_itl, abs=1e-9)


# Fallback path with raw chunks but no tokenizer: nothing to re-tokenize with,
# so no token times. TTFT is absent and the build does not fail.
def test_chunk_retokenization_needs_a_tokenizer() -> None:
    start = 100.0
    metric = RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=start - 0.001,
        start_time=start,
        end_time=start + 0.8,
        request_data="{}",
        response_data="ok",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=128)),
            response_metrics=StreamedResponseMetrics(
                response_chunks=[json.dumps({"choices": [{"delta": {"content": "a b"}}]})],
                chunk_times=[start + 0.5],
                output_tokens=2,
                output_token_times=[],
            ),
        ),
        error=None,
    )

    results = build_results([metric], tokenizer=None)

    assert results.request_performance is not None
    aggregate = results.request_performance.aggregate
    assert aggregate is not None and aggregate.latency is not None
    assert aggregate.latency.time_to_first_token is None


def _session_metric(
    session_id: str,
    start: float,
    duration: float = 1.0,
    num_events: int = 3,
    num_events_completed: int = 3,
    success: Optional[bool] = True,
    input_tokens: Optional[int] = 100,
    output_tokens: Optional[int] = 50,
) -> SessionLifecycleMetric:
    return SessionLifecycleMetric(
        session_id=session_id,
        stage_id=0,
        file_path=f"/traces/{session_id}.json",
        start_time=start,
        end_time=start + duration,
        duration_sec=duration,
        num_events=num_events,
        num_events_completed=num_events_completed,
        success=success,
        total_input_tokens=input_tokens,
        total_output_tokens=output_tokens,
    )


# Every non-session workload takes this path: no sessions means no
# session_performance block, so the emitted partial is unchanged.
def test_build_results_without_session_metrics_omits_session_performance() -> None:
    results = build_results([_streaming_metric(time.time())])

    assert results.request_performance is not None
    assert results.session_performance is None


def test_build_results_session_performance_aggregates_sessions() -> None:
    now = time.time()
    sessions = [
        _session_metric("s1", now, duration=2.0, num_events=4),
        _session_metric("s2", now + 2.0, duration=1.0, num_events=2, success=False),
    ]

    results = build_results([], session_metrics=sessions)

    assert results.request_performance is None
    assert results.session_performance is not None
    aggregate = results.session_performance.sessions
    assert aggregate is not None
    assert aggregate.total == 2
    assert aggregate.succeeded == 1
    assert aggregate.failed == 1
    assert aggregate.total_events == 6
    assert aggregate.total_events_completed == 6
    assert aggregate.session_duration is not None
    assert aggregate.session_duration.mean == pytest.approx(1.5, abs=1e-9)
    assert aggregate.events_per_session is not None
    assert aggregate.events_per_session.mean == pytest.approx(3.0, abs=1e-9)
    assert aggregate.input_tokens_per_session is not None
    assert aggregate.input_tokens_per_session.mean == pytest.approx(100.0, abs=1e-9)
    assert aggregate.output_tokens_per_session is not None
    assert aggregate.output_tokens_per_session.mean == pytest.approx(50.0, abs=1e-9)
    # Two sessions over the 3s window that spans them.
    assert aggregate.session_rate is not None
    assert aggregate.session_rate.units == Units.QUERY_PER_S
    assert aggregate.session_rate.mean == pytest.approx(2 / 3, abs=1e-9)


# Token totals come from the enrichment step: an unrecorded distribution must
# stay absent rather than be reported as a measured zero.
def test_build_results_session_token_totals_absent_when_not_recorded() -> None:
    now = time.time()
    sessions = [_session_metric("s1", now, input_tokens=None, output_tokens=None)]

    results = build_results([], session_metrics=sessions)

    assert results.session_performance is not None
    aggregate = results.session_performance.sessions
    assert aggregate is not None
    assert aggregate.total == 1
    assert aggregate.input_tokens_per_session is None
    assert aggregate.output_tokens_per_session is None


# A stage cut short by max_stage_duration leaves no metric behind for the
# sessions it stranded, so their count arrives beside the metrics, not in them.
# Landing them in `total` is what keeps a truncated stage from reading as a
# clean run: 1 completed + 3 stranded is not "1 session, 1 succeeded".
def test_build_results_counts_stranded_sessions_in_total() -> None:
    now = time.time()
    results = build_results([], session_metrics=[_session_metric("s1", now)], sessions_not_completed=3)

    assert results.session_performance is not None
    aggregate = results.session_performance.sessions
    assert aggregate is not None
    assert aggregate.total == 4
    assert aggregate.succeeded == 1
    assert aggregate.failed == 0
    # The distributions describe the sessions that reported, not the ones that
    # never got to run a request.
    assert aggregate.session_duration is not None
    assert aggregate.session_duration.mean == pytest.approx(1.0, abs=1e-9)
    assert aggregate.input_tokens_per_session is not None
    assert aggregate.input_tokens_per_session.mean == pytest.approx(100.0, abs=1e-9)


# A stage where every session stranded still reports its count, so the block is
# emitted for a non-zero total even with no session metrics to describe.
def test_build_results_reports_a_fully_stranded_stage() -> None:
    results = build_results([], sessions_not_completed=2)

    assert results.session_performance is not None
    aggregate = results.session_performance.sessions
    assert aggregate is not None
    assert aggregate.total == 2
    assert aggregate.succeeded == 0
    assert aggregate.failed == 0
    # Nothing to measure over: event totals are sums over an empty set (0, as in
    # the native report), everything distribution-shaped stays absent.
    assert aggregate.total_events == 0
    assert aggregate.total_events_completed == 0
    assert aggregate.session_rate is None
    assert aggregate.session_duration is None
    assert aggregate.events_per_session is None
