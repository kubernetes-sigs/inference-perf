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

import time
from typing import List

import pytest

from inference_perf.apis import (
    ErrorResponseInfo,
    InferenceInfo,
    RequestLifecycleMetric,
    StreamedResponseMetrics,
)
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.br.v0_2 import build_results
from inference_perf.reportgen.br.v0_2.schema import Units


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


def test_build_results_aggregate_units() -> None:
    """The vendored BR0.2 schema enforces unit compatibility per metric
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
