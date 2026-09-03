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
from inference_perf.observability.metrics import MetricsHub, MetricStability, RunContext, StageContext, build_metrics
from inference_perf.observability.metrics.sets import ALL_SPECS
from inference_perf.observability.metrics.sets.core import output_tokens
from inference_perf.payloads import RequestMetrics, Text


# Builds one streamed request: starts at 10.0s, four chunks at 10.5/11.0/11.5/12.0s,
# ends at 12.0s, 11 prompt tokens and 5 client-counted output tokens. Pass
# server_completion_tokens to make the server report a different output count, or
# error= to turn it into a failure.
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


# Builds one successful unary request: 0s to 1s, 3 prompt tokens, 5 output tokens
# and no token timeline, which is what makes TTFT and TPOT inapplicable.
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


# A hub built from a streaming run of `stages` identical 1 req/s, 1s stages, so the
# streaming-only latency metrics are exported and stage labels 0..stages-1 exist.
def _streaming_hub(stages: int = 1) -> MetricsHub:
    config = Config(api=APIConfig(streaming=True), load=LoadConfig(stages=[StandardLoadStage(rate=1, duration=1)] * stages))
    return build_metrics(config)


# Reads one series out of the hub's registry by metric name and labels. Returns None
# when the series does not exist, which is how absence gets asserted.
def _sample(hub: MetricsHub, name: str, **labels: str) -> Optional[float]:
    return hub.registry.get_sample_value(name, labels or None)


# --- run and stage state -----------------------------------------------------


# A 3-stage run whose RunContext reports 4 requests in flight. Expects stages to
# read 0 before the run starts and 3 after, and in_flight to track the probe live:
# 4.0, then 0.0 once the probe returns 0 without any metric call.
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


# Starts stage 0, ends it, then starts stage 1. Expects no stage="0" series at all
# before it starts, then running=1 with a start timestamp, then running=0 with an
# end timestamp at or after it, and stage 0 staying at 0 once stage 1 is running.
def test_stage_gauges_follow_transitions() -> None:
    hub = _streaming_hub()
    assert _sample(hub, "inference_perf_stage_running", stage="0") is None, "no series before the stage starts"

    before = time.time()
    hub.on_stage_start(StageContext(stage_id=0))
    assert _sample(hub, "inference_perf_stage_running", stage="0") == 1.0
    started = _sample(hub, "inference_perf_stage_start_timestamp_seconds", stage="0")
    assert started is not None and before <= started <= time.time()
    assert _sample(hub, "inference_perf_stage_end_timestamp_seconds", stage="0") is None

    hub.on_stage_end(StageContext(stage_id=0))
    assert _sample(hub, "inference_perf_stage_running", stage="0") == 0.0
    ended = _sample(hub, "inference_perf_stage_end_timestamp_seconds", stage="0")
    assert ended is not None and ended >= started

    hub.on_stage_start(StageContext(stage_id=1))
    assert _sample(hub, "inference_perf_stage_running", stage="1") == 1.0
    assert _sample(hub, "inference_perf_stage_running", stage="0") == 0.0


# --- request outcomes and tokens ------------------------------------------------


# Four stage-0 requests: two HTTP 503 failures, one TimeoutError, one success.
# Expects errors_total to read 2 for "HTTP Error 503" and 1 for "TimeoutError",
# requests_total 3 failure / 1 success, and no empty error_type series.
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


# Feeds the resolver a missing response, a client count of 5, a server count of 7,
# a server payload with no completion_tokens, and a server count of 0. Expects
# 0, 5, 7, 5, 5: the server wins only when it reported a nonzero count.
def test_output_tokens_prefer_server_usage_then_client_count() -> None:
    assert output_tokens(None) == 0
    assert output_tokens(UnaryResponseMetrics(output_tokens=5)) == 5
    assert output_tokens(UnaryResponseMetrics(output_tokens=5, server_usage={"completion_tokens": 7})) == 7
    assert output_tokens(UnaryResponseMetrics(output_tokens=5, server_usage={"prompt_tokens": 9})) == 5
    assert output_tokens(UnaryResponseMetrics(output_tokens=5, server_usage={"completion_tokens": 0})) == 5


# Stage 0 gets 11+5 tokens, then 13 prompt with the server reporting 8 output, then
# a 100/100 failure; stage 1 gets 2+3. Expects prompt 24 and output 13 for stage 0
# (the failure counted nowhere, the server's 8 preferred) and 2 and 3 for stage 1.
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


# One streamed request, start 10.0, chunks 10.5..12.0, end 12.0, server says 4
# output tokens. Expects one observation in each histogram: latency 2.0, TTFT 0.5,
# TPOT (12.0 - 10.5) / (4 - 1) = 0.5, the same arithmetic the report uses.
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


# Three requests: a failure, a one-token stream, and a zero-token stream. Expects
# latency observed twice (failures never), TTFT once (needs a first chunk) and TPOT
# never, since a single token gives no inter-token interval.
def test_latency_histograms_skip_failures_and_degenerate_streams() -> None:
    hub = _streaming_hub()
    hub.observe_request(_streamed(error="TimeoutError"))  # failed: nothing observed
    hub.observe_request(_streamed(token_times=[10.5], client_output_tokens=1))  # one token: TTFT yes, TPOT no
    hub.observe_request(_streamed(token_times=[], client_output_tokens=0))  # no tokens: latency only

    assert _sample(hub, "inference_perf_request_latency_seconds_count", stage="0") == 2.0
    assert _sample(hub, "inference_perf_time_to_first_token_seconds_count", stage="0") == 1.0
    assert _sample(hub, "inference_perf_time_per_output_token_seconds_count", stage="0") is None


# Builds one hub with streaming off and one with it on. Expects the unary hub to
# export request latency and output tokens (5.0) but no TTFT or TPOT series at all,
# and the streaming hub to export both.
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


# Walks every spec in ALL_SPECS. Expects unique inference_perf_-prefixed names with
# no hand-written _total suffix, labels drawn only from the allowlist, histograms
# named _seconds with explicit buckets, timestamps named _timestamp_seconds, and a
# non-empty description on each.
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


# Reads the stability level off every spec in ALL_SPECS. Expects all of them to be
# ALPHA: the set ships changeable, so promoting one to BETA or STABLE is a release
# decision that has to come here and say so.
def test_no_metric_is_promoted_yet() -> None:
    for spec in ALL_SPECS:
        assert spec.stability is MetricStability.ALPHA, (
            f"{spec.name}: promotion is a release decision, and nothing is promoted before v1.0.0"
        )
