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
from prometheus_client import Counter, Gauge, Histogram  # noqa: TID251 (throwaway metrics for these tests)
from prometheus_client.exposition import generate_latest

from inference_perf.apis.base import (
    ErrorResponseInfo,
    InferenceInfo,
    RequestLifecycleMetric,
    StreamedResponseMetrics,
    UnaryResponseMetrics,
)
from inference_perf.config import (
    APIConfig,
    Config,
    LoadConfig,
    ReportConfig,
    RequestLifecycleMetricsReportConfig,
    StageGenType,
    StandardLoadStage,
    SweepConfig,
)
from inference_perf.observability.metrics import (
    MetricSpec,
    MetricsHub,
    MetricStability,
    RunContext,
    StageContext,
    build_metrics,
)
from inference_perf.observability.metrics.sets import ALL_SPECS
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.utils.numeric.expression import SERIES, SIGNALS


# Builds one streamed request: scheduled and started at 10.0s, four chunks at
# 10.5/11.0/11.5/12.0s, ends at 12.0s, 11 prompt tokens and 5 client-counted output
# tokens. Pass scheduled= to send it late, server_completion_tokens to make the server
# report a different output count, or error= to turn it into a failure.
def _streamed(
    stage_id: int = 0,
    scheduled: Optional[float] = None,
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
        scheduled_time=scheduled if scheduled is not None else start,
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
# use_server_output_tokens sets the report flag the token-based metrics follow.
def _streaming_hub(stages: int = 1, use_server_output_tokens: bool = False) -> MetricsHub:
    config = Config(
        api=APIConfig(streaming=True),
        load=LoadConfig(stages=[StandardLoadStage(rate=1, duration=1)] * stages),
        report=ReportConfig(
            request_lifecycle=RequestLifecycleMetricsReportConfig(use_server_output_tokens=use_server_output_tokens)
        ),
    )
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


# A sweep run: the config has no stages, and the load generator's stage count goes
# from 0 to 4 once its saturation probe generates them. Expects stages to read 0
# at run start and 4 afterwards, from the probe rather than from the config.
def test_stage_count_follows_stages_generated_after_run_start() -> None:
    hub = _streaming_hub()
    generated: List[StandardLoadStage] = []
    hub.on_run_start(
        RunContext(
            config=Config(load=LoadConfig(sweep=SweepConfig(type=StageGenType.LINEAR))),
            stage_count=lambda: len(generated),
        )
    )
    assert _sample(hub, "inference_perf_stages") == 0.0
    generated.extend([StandardLoadStage(rate=1, duration=1)] * 4)
    assert _sample(hub, "inference_perf_stages") == 4.0


# Ends stage 0 having lost two workers (a RuntimeError crash and a SIGKILL) and
# stage 1 having lost none. Expects one series per cause on stage 0, each at 1.0,
# and no series at all for stage 1: a stage that kept its workers must not create
# a zero sample that looks like a measured absence of failure.
def test_workers_lost_counts_one_series_per_cause() -> None:
    hub = _streaming_hub()
    assert _sample(hub, "inference_perf_workers_lost_total", stage="0", cause="RuntimeError") is None

    hub.on_stage_end(StageContext(stage_id=0, workers_lost=lambda: ("RuntimeError", "SIGKILL")))
    assert _sample(hub, "inference_perf_workers_lost_total", stage="0", cause="RuntimeError") == 1.0
    assert _sample(hub, "inference_perf_workers_lost_total", stage="0", cause="SIGKILL") == 1.0

    hub.on_stage_end(StageContext(stage_id=1))
    assert _sample(hub, "inference_perf_workers_lost_total", stage="1", cause="RuntimeError") is None


# Ends one stage that lost two workers to the same cause. Expects the two deaths
# to add up on a single series rather than overwrite each other.
def test_workers_lost_accumulates_repeats_of_one_cause() -> None:
    hub = _streaming_hub()
    hub.on_stage_end(StageContext(stage_id=0, workers_lost=lambda: ("RuntimeError", "RuntimeError")))
    assert _sample(hub, "inference_perf_workers_lost_total", stage="0", cause="RuntimeError") == 2.0


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


# One success the client counts as 5 output tokens and the server as 8, then one with
# no server count. Expects the counter to add 5 + 5 = 10 with use_server_output_tokens
# off and 8 + 5 = 13 with it on: the count the report's output_tokens_per_sec uses.
@pytest.mark.parametrize(("use_server_output_tokens", "expected"), [(False, 10.0), (True, 13.0)])
def test_output_token_counter_follows_use_server_output_tokens(use_server_output_tokens: bool, expected: float) -> None:
    hub = _streaming_hub(use_server_output_tokens=use_server_output_tokens)
    hub.observe_request(_streamed(client_output_tokens=5, server_completion_tokens=8))
    hub.observe_request(_streamed(client_output_tokens=5))

    assert _sample(hub, "inference_perf_output_tokens_total", stage="0") == expected


# Stage 0 gets 11+5 tokens, then 13 prompt with the server reporting 8 output, then
# a 100/100 failure; stage 1 gets 2+3. Expects prompt 24 and output 10 for stage 0
# (the failure counted nowhere, the client's 5 counted as the report does by default)
# and 2 and 3 for stage 1.
def test_token_counters_sum_successful_requests_only() -> None:
    hub = _streaming_hub()
    hub.observe_request(_streamed(input_tokens=11, client_output_tokens=5))
    hub.observe_request(_streamed(input_tokens=13, client_output_tokens=5, server_completion_tokens=8))
    hub.observe_request(_streamed(input_tokens=100, client_output_tokens=100, error="TimeoutError"))
    hub.observe_request(_streamed(stage_id=1, input_tokens=2, client_output_tokens=3))

    assert _sample(hub, "inference_perf_prompt_tokens_total", stage="0") == 24.0
    assert _sample(hub, "inference_perf_output_tokens_total", stage="0") == 10.0
    assert _sample(hub, "inference_perf_prompt_tokens_total", stage="1") == 2.0
    assert _sample(hub, "inference_perf_output_tokens_total", stage="1") == 3.0


# --- latency ------------------------------------------------------------------


# One streamed request, start 10.0, chunks 10.5..12.0, end 12.0, the client counts 5
# output tokens and the server 4. Expects latency 2.0, TTFT 0.5 and TPOT
# (12.0 - 10.5) / (5 - 1) = 0.375 with use_server_output_tokens off, or
# (12.0 - 10.5) / (4 - 1) = 0.5 with it on: the arithmetic and count the report use.
@pytest.mark.parametrize(("use_server_output_tokens", "tpot"), [(False, 0.375), (True, 0.5)])
def test_latency_histograms_match_report_derivation(use_server_output_tokens: bool, tpot: float) -> None:
    hub = _streaming_hub(use_server_output_tokens=use_server_output_tokens)
    hub.observe_request(_streamed(start=10.0, token_times=[10.5, 11.0, 11.5, 12.0], end=12.0, server_completion_tokens=4))

    assert _sample(hub, "inference_perf_request_latency_seconds_count", stage="0") == 1.0
    assert _sample(hub, "inference_perf_request_latency_seconds_sum", stage="0") == pytest.approx(2.0)
    assert _sample(hub, "inference_perf_time_to_first_token_seconds_count", stage="0") == 1.0
    assert _sample(hub, "inference_perf_time_to_first_token_seconds_sum", stage="0") == pytest.approx(0.5)
    assert _sample(hub, "inference_perf_time_per_output_token_seconds_count", stage="0") == 1.0
    assert _sample(hub, "inference_perf_time_per_output_token_seconds_sum", stage="0") == pytest.approx(tpot)


# A success scheduled at 9.9 and sent at 10.0, ending at 12.0 with 5 client tokens, and a
# failure scheduled at 10.0 and sent at 10.25. Expects NTPOT observed once, 2.0 / 5 = 0.4,
# and schedule delay observed for both, summing 0.1 + 0.25 = 0.35.
def test_ntpot_and_schedule_delay_histograms() -> None:
    hub = _streaming_hub()
    hub.observe_request(_streamed(scheduled=9.9, start=10.0, end=12.0, client_output_tokens=5))
    hub.observe_request(_streamed(scheduled=10.0, start=10.25, error="TimeoutError"))

    assert _sample(hub, "inference_perf_normalized_time_per_output_token_seconds_count", stage="0") == 1.0
    assert _sample(hub, "inference_perf_normalized_time_per_output_token_seconds_sum", stage="0") == pytest.approx(0.4)
    assert _sample(hub, "inference_perf_schedule_delay_seconds_count", stage="0") == 2.0
    assert _sample(hub, "inference_perf_schedule_delay_seconds_sum", stage="0") == pytest.approx(0.35)


# A request that finished 2s ago on the perf_counter clock, then a failure that finished
# 1s ago. Expects the stage's last completion gauge to read the Unix time of the later
# one, time.time() - 1.0, within 50ms.
def test_last_completion_timestamp_is_the_unix_time_the_last_request_finished() -> None:
    hub = _streaming_hub()
    now = time.perf_counter()
    hub.observe_request(_streamed(start=now - 3.0, token_times=[now - 2.5], end=now - 2.0))
    hub.observe_request(_streamed(start=now - 2.0, end=now - 1.0, error="TimeoutError"))

    assert _sample(hub, "inference_perf_last_completion_timestamp_seconds", stage="0") == pytest.approx(
        time.time() - 1.0, abs=0.05
    )


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


# --- expression terms ---------------------------------------------------------


# Walks ALL_SPECS for the series and signal each metric declares. Expects every name in
# SERIES and SIGNALS backed by exactly one metric, e.g. ttft by
# inference_perf_time_to_first_token_seconds, so each term reads one runtime metric.
@pytest.mark.parametrize(("attribute", "names"), [("series", SERIES), ("signal", SIGNALS)])
def test_every_expression_term_has_one_runtime_metric(attribute: str, names: dict[str, object]) -> None:
    backed = [getattr(spec, attribute) for spec in ALL_SPECS if getattr(spec, attribute) is not None]
    assert sorted(backed) == sorted(names), f"{attribute}: backed {sorted(backed)}, terms {sorted(names)}"


# A spec declaring series="ttfb", which is not a series. Expects a ValueError naming it.
def test_spec_with_an_unknown_series_is_rejected() -> None:
    with pytest.raises(ValueError, match="unknown series 'ttfb'"):
        MetricSpec(name="inference_perf_x_seconds", documentation="x", metric_type=Histogram, series="ttfb")


# --- conventions (the checkable half of #628) -----------------------------------

# "cause" is bounded the same way "error_type" is: an exception class name, a
# signal name, or an exit code, never a message or an id.
ALLOWED_LABELS = {"stage", "status", "error_type", "cause"}


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
