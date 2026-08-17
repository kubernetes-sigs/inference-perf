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
"""The metric sets under observability/metrics/sets: values, gating and naming.

Each hook is driven with hand-built RequestLifecycleMetrics whose expected
values are computable by hand, and latency values are checked against the
same derivation the report uses.
"""

import time
from typing import List, Optional

import pytest
from prometheus_client import Counter, Gauge, Histogram
from prometheus_client.exposition import generate_latest

from inference_perf.apis.base import (
    ErrorResponseInfo,
    InferenceInfo,
    RequestLifecycleMetric,
    StreamedResponseMetrics,
    UnaryResponseMetrics,
)
from inference_perf.config import APIConfig, Config, LoadConfig, StandardLoadStage
from inference_perf.observability.metrics import MetricsHub, RunContext, build_metrics
from inference_perf.observability.metrics.sets import ALL_SPECS
from inference_perf.observability.metrics.sets.core import output_tokens
from inference_perf.payloads import RequestMetrics, Text


def _streamed(
    stage_id: int = 0,
    start: float = 10.0,
    token_times: Optional[List[float]] = None,
    end: float = 12.0,
    client_output_tokens: int = 5,
    server_completion_tokens: Optional[int] = None,
    input_tokens: int = 11,
    error: Optional[str] = None,
) -> RequestLifecycleMetric:
    usage = {"completion_tokens": server_completion_tokens} if server_completion_tokens is not None else None
    return RequestLifecycleMetric(
        stage_id=stage_id,
        scheduled_time=start,
        start_time=start,
        end_time=end,
        request_data="r",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=input_tokens)),
            response_metrics=StreamedResponseMetrics(
                output_tokens=client_output_tokens,
                server_usage=usage,
                output_token_times=token_times if token_times is not None else [10.5, 11.0, 11.5, 12.0],
                chunk_times=token_times if token_times is not None else [10.5, 11.0, 11.5, 12.0],
            ),
        ),
        error=ErrorResponseInfo(error_type=error, error_msg="boom") if error else None,
    )


def _unary(stage_id: int = 0, client_output_tokens: int = 5) -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        stage_id=stage_id,
        scheduled_time=0.0,
        start_time=0.0,
        end_time=1.0,
        request_data="r",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=3)),
            response_metrics=UnaryResponseMetrics(output_tokens=client_output_tokens),
        ),
        error=None,
    )


def _streaming_hub(stages: int = 1) -> MetricsHub:
    config = Config(api=APIConfig(streaming=True), load=LoadConfig(stages=[StandardLoadStage(rate=1, duration=1)] * stages))
    return build_metrics(config)


def _sample(hub: MetricsHub, name: str, **labels: str) -> Optional[float]:
    return hub.registry.get_sample_value(name, labels or None)


# --- run and stage state -----------------------------------------------------


def test_stage_count_and_in_flight_come_from_run_context() -> None:
    hub = _streaming_hub(stages=3)
    assert _sample(hub, "inference_perf_stages") == 0.0
    in_flight = 4
    hub.on_run_start(
        RunContext(
            config=Config(load=LoadConfig(stages=[StandardLoadStage(rate=1, duration=1)] * 3)),
            in_flight_requests=lambda: in_flight,
        )
    )
    assert _sample(hub, "inference_perf_stages") == 3.0
    assert _sample(hub, "inference_perf_requests_in_flight") == 4.0
    in_flight = 0
    assert _sample(hub, "inference_perf_requests_in_flight") == 0.0


def test_stage_gauges_follow_transitions() -> None:
    hub = _streaming_hub()
    assert _sample(hub, "inference_perf_stage_running", stage="0") is None, "no series before the stage starts"

    before = time.time()
    hub.on_stage_start(0)
    assert _sample(hub, "inference_perf_stage_running", stage="0") == 1.0
    started = _sample(hub, "inference_perf_stage_start_timestamp_seconds", stage="0")
    assert started is not None and before <= started <= time.time()
    assert _sample(hub, "inference_perf_stage_end_timestamp_seconds", stage="0") is None

    hub.on_stage_end(0)
    assert _sample(hub, "inference_perf_stage_running", stage="0") == 0.0
    ended = _sample(hub, "inference_perf_stage_end_timestamp_seconds", stage="0")
    assert ended is not None and ended >= started

    hub.on_stage_start(1)
    assert _sample(hub, "inference_perf_stage_running", stage="1") == 1.0
    assert _sample(hub, "inference_perf_stage_running", stage="0") == 0.0


# --- request outcomes and tokens ------------------------------------------------


def test_errors_counted_by_class_and_only_for_failures() -> None:
    hub = _streaming_hub()
    hub.observe_request(_streamed(error="HTTP Error 503"))
    hub.observe_request(_streamed(error="HTTP Error 503"))
    hub.observe_request(_streamed(error="TimeoutError"))
    hub.observe_request(_streamed())

    errors = "inference_perf_request_errors_total"
    assert _sample(hub, errors, stage="0", error_type="HTTP Error 503") == 2.0
    assert _sample(hub, errors, stage="0", error_type="TimeoutError") == 1.0
    assert _sample(hub, "inference_perf_requests_total", stage="0", status="failure") == 3.0
    assert _sample(hub, "inference_perf_requests_total", stage="0", status="success") == 1.0
    assert 'error_type=""' not in generate_latest(hub.registry).decode(), "successes must not create an error series"


def test_output_tokens_prefer_server_usage_then_client_count() -> None:
    assert output_tokens(None) == 0
    assert output_tokens(UnaryResponseMetrics(output_tokens=5)) == 5
    assert output_tokens(UnaryResponseMetrics(output_tokens=5, server_usage={"completion_tokens": 7})) == 7
    assert output_tokens(UnaryResponseMetrics(output_tokens=5, server_usage={"prompt_tokens": 9})) == 5
    assert output_tokens(UnaryResponseMetrics(output_tokens=5, server_usage={"completion_tokens": 0})) == 5


def test_token_counters_sum_successful_requests_only() -> None:
    hub = _streaming_hub()
    hub.observe_request(_streamed(input_tokens=11, client_output_tokens=5))
    hub.observe_request(_streamed(input_tokens=13, client_output_tokens=5, server_completion_tokens=8))
    hub.observe_request(_streamed(input_tokens=100, client_output_tokens=100, error="TimeoutError"))
    hub.observe_request(_streamed(stage_id=1, input_tokens=2, client_output_tokens=3))

    assert _sample(hub, "inference_perf_prompt_tokens_total", stage="0") == 24.0
    assert _sample(hub, "inference_perf_output_tokens_total", stage="0") == 13.0
    assert _sample(hub, "inference_perf_prompt_tokens_total", stage="1") == 2.0
    assert _sample(hub, "inference_perf_output_tokens_total", stage="1") == 3.0


# --- latency ------------------------------------------------------------------


def test_latency_histograms_match_report_derivation() -> None:
    hub = _streaming_hub()
    # start=10.0, chunks at 10.5 .. 12.0, end=12.0, 5 output tokens (client) but the server says 4:
    # TTFT = 0.5, TPOT = (12.0 - 10.5) / (4 - 1) = 0.5, request latency = 2.0
    hub.observe_request(_streamed(start=10.0, token_times=[10.5, 11.0, 11.5, 12.0], end=12.0, server_completion_tokens=4))

    assert _sample(hub, "inference_perf_request_latency_seconds_count", stage="0") == 1.0
    assert _sample(hub, "inference_perf_request_latency_seconds_sum", stage="0") == pytest.approx(2.0)
    assert _sample(hub, "inference_perf_time_to_first_token_seconds_count", stage="0") == 1.0
    assert _sample(hub, "inference_perf_time_to_first_token_seconds_sum", stage="0") == pytest.approx(0.5)
    assert _sample(hub, "inference_perf_time_per_output_token_seconds_count", stage="0") == 1.0
    assert _sample(hub, "inference_perf_time_per_output_token_seconds_sum", stage="0") == pytest.approx(0.5)


def test_latency_histograms_skip_failures_and_degenerate_streams() -> None:
    hub = _streaming_hub()
    hub.observe_request(_streamed(error="TimeoutError"))  # failed: nothing observed
    hub.observe_request(_streamed(token_times=[10.5], client_output_tokens=1))  # one token: TTFT yes, TPOT no
    hub.observe_request(_streamed(token_times=[], client_output_tokens=0))  # no tokens: latency only

    assert _sample(hub, "inference_perf_request_latency_seconds_count", stage="0") == 2.0
    assert _sample(hub, "inference_perf_time_to_first_token_seconds_count", stage="0") == 1.0
    assert _sample(hub, "inference_perf_time_per_output_token_seconds_count", stage="0") is None


def test_ttft_and_tpot_absent_on_unary_runs() -> None:
    unary = build_metrics(Config(api=APIConfig(streaming=False)))
    unary.observe_request(_unary())
    body = generate_latest(unary.registry).decode()
    assert "inference_perf_request_latency_seconds" in body
    assert "inference_perf_time_to_first_token_seconds" not in body
    assert "inference_perf_time_per_output_token_seconds" not in body
    assert _sample(unary, "inference_perf_output_tokens_total", stage="0") == 5.0

    streaming = _streaming_hub()
    body = generate_latest(streaming.registry).decode()
    assert "inference_perf_time_to_first_token_seconds" in body
    assert "inference_perf_time_per_output_token_seconds" in body


# --- conventions (the checkable half of #628) -----------------------------------

ALLOWED_LABELS = {"stage", "status", "error_type"}


def test_all_specs_follow_naming_and_label_conventions() -> None:
    names = [spec.name for spec in ALL_SPECS]
    assert len(names) == len(set(names))
    for spec in ALL_SPECS:
        assert spec.name.startswith("inference_perf_"), spec.name
        assert not spec.name.endswith("_total"), f"{spec.name}: prometheus_client appends _total to counters"
        assert set(spec.labelnames) <= ALLOWED_LABELS, f"{spec.name}: labels must be bounded and from the allowlist"
        if spec.metric_type is Histogram:
            assert spec.name.endswith("_seconds"), f"{spec.name}: histograms here are durations in base units"
            assert spec.buckets is not None
        if spec.metric_type is Gauge and "timestamp" in spec.name:
            assert spec.name.endswith("_timestamp_seconds"), spec.name
        assert spec.metric_type in (Counter, Gauge, Histogram)
        assert spec.documentation.strip()
