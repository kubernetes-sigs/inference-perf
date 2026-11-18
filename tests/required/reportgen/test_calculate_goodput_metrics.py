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
from unittest.mock import Mock
from inference_perf.apis import RequestLifecycleMetric
from inference_perf.config import GoodputConfig
from inference_perf.reportgen.base import calculate_goodput_metrics


def test_calculate_goodput_metrics_empty_metrics() -> None:
    """Test with empty metrics list returns None."""
    config = GoodputConfig(constraints={"ttft": 0.5})
    result = calculate_goodput_metrics([], config, [], [], [], [], [])
    assert result is None


def test_calculate_goodput_metrics_no_config() -> None:
    """Test with no goodput config returns None."""
    metric = Mock(spec=RequestLifecycleMetric)
    metric.ttft_slo_sec = None
    metric.tpot_slo_sec = None
    result = calculate_goodput_metrics([metric], None, [0.3], [None], [0.1], [1.0], [None])
    assert result is None


def test_calculate_goodput_metrics_no_constraints() -> None:
    """Test with empty constraints returns None."""
    config = GoodputConfig(constraints={})
    metric = Mock(spec=RequestLifecycleMetric)
    metric.ttft_slo_sec = None
    metric.tpot_slo_sec = None
    result = calculate_goodput_metrics([metric], config, [0.3], [None], [0.1], [1.0], [None])
    assert result is None


def test_calculate_goodput_metrics_ttft_constraint() -> None:
    """Test goodput calculation with TTFT constraint."""
    config = GoodputConfig(constraints={"ttft": 0.5})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 2.0
    metric1.info = Mock()
    metric1.info.request_metrics.text.input_tokens = 0
    metric1.info.response_metrics.output_tokens = 10
    metric1.ttft_slo_sec = None
    metric1.tpot_slo_sec = None

    metric2 = Mock(spec=RequestLifecycleMetric)
    metric2.start_time = 2.0
    metric2.end_time = 4.0
    metric2.info = Mock()
    metric2.info.request_metrics.text.input_tokens = 0
    metric2.info.response_metrics.output_tokens = 20
    metric2.ttft_slo_sec = None
    metric2.tpot_slo_sec = None

    metrics = [metric1, metric2]
    ttft_values: list[float | None] = [0.3, 0.6]  # metric1 meets, metric2 fails

    result = calculate_goodput_metrics(metrics, config, ttft_values, [None, None], [0.1, 0.1], [2.0, 2.0], [None, None])  # type: ignore[arg-type]

    assert result is not None
    assert result["goodput_percentage"] == 50.0
    assert result["good_requests"] == 1
    assert result["total_requests"] == 2
    # Total benchmark time = 4.0 - 0.0 = 4.0
    assert result["request_goodput"] == 1 / 4.0
    # Good tokens = 10 (from metric1)
    assert result["token_goodput"] == 10 / 4.0


def test_calculate_goodput_metrics_total_tokens() -> None:
    """Test that token goodput uses total tokens (input + output)."""
    config = GoodputConfig(constraints={"ttft": 0.5})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 2.0
    metric1.info = Mock()
    metric1.info.request_metrics.text.input_tokens = 5
    metric1.info.response_metrics.output_tokens = 10
    metric1.ttft_slo_sec = None
    metric1.tpot_slo_sec = None

    metrics = [metric1]
    ttft_values: list[float | None] = [0.3]  # Meets constraint

    result = calculate_goodput_metrics(metrics, config, ttft_values, [None], [0.1], [2.0], [None])  # type: ignore[arg-type]

    assert result is not None
    assert result["goodput_percentage"] == 100.0
    assert result["good_requests"] == 1
    # Total tokens = 5 + 10 = 15
    # Total benchmark time = 2.0 - 0.0 = 2.0
    assert result["token_goodput"] == 15 / 2.0


def test_calculate_goodput_metrics_multiple_constraints() -> None:
    """Test goodput calculation with multiple constraints."""
    config = GoodputConfig(constraints={"ttft": 0.5, "tpot": 0.1})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 2.0
    metric1.info = Mock()
    metric1.info.request_metrics.text.input_tokens = 0
    metric1.info.response_metrics.output_tokens = 10
    metric1.ttft_slo_sec = None
    metric1.tpot_slo_sec = None

    metric2 = Mock(spec=RequestLifecycleMetric)
    metric2.start_time = 2.0
    metric2.end_time = 4.0
    metric2.info = Mock()
    metric2.info.request_metrics.text.input_tokens = 0
    metric2.info.response_metrics.output_tokens = 20
    metric2.ttft_slo_sec = None
    metric2.tpot_slo_sec = None

    metrics = [metric1, metric2]
    ttft_values: list[float | None] = [0.3, 0.3]  # Both meet TTFT
    tpot_values: list[float | None] = [0.08, 0.12]  # metric1 meets, metric2 fails TPOT

    result = calculate_goodput_metrics(metrics, config, ttft_values, tpot_values, [0.1, 0.1], [2.0, 2.0], [None, None])  # type: ignore[arg-type]

    assert result is not None
    assert result["goodput_percentage"] == 50.0
    assert result["good_requests"] == 1


def test_calculate_goodput_metrics_itl_constraint() -> None:
    """Test goodput calculation with ITL constraint."""
    config = GoodputConfig(constraints={"itl": 0.05})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 2.0
    metric1.info = Mock()
    metric1.info.request_metrics.text.input_tokens = 0
    metric1.info.response_metrics.output_tokens = 10
    metric1.ttft_slo_sec = None
    metric1.tpot_slo_sec = None

    metrics = [metric1]
    itl_values: list[float | None] = [0.04]  # Meets ITL

    result = calculate_goodput_metrics(metrics, config, [None], [None], [0.1], [2.0], itl_values)  # type: ignore[arg-type]

    assert result is not None
    assert result["goodput_percentage"] == 100.0


def test_calculate_goodput_metrics_ntpot_constraint() -> None:
    """Test goodput calculation with NTPOT constraint."""
    config = GoodputConfig(constraints={"ntpot": 0.1})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 2.0
    metric1.info = Mock()
    metric1.info.request_metrics.text.input_tokens = 0
    metric1.info.response_metrics.output_tokens = 10
    metric1.ttft_slo_sec = None
    metric1.tpot_slo_sec = None

    metrics = [metric1]
    ntpot_values: list[float | None] = [0.05]  # Meets NTPOT

    result = calculate_goodput_metrics(metrics, config, [None], [None], ntpot_values, [2.0], [None])  # type: ignore[arg-type]

    assert result is not None
    assert result["goodput_percentage"] == 100.0


def test_calculate_goodput_metrics_request_latency_constraint() -> None:
    """Test goodput calculation with request_latency constraint."""
    config = GoodputConfig(constraints={"request_latency": 1.0})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 0.5
    metric1.info = Mock()
    metric1.info.request_metrics.text.input_tokens = 0
    metric1.info.response_metrics.output_tokens = 10
    metric1.ttft_slo_sec = None
    metric1.tpot_slo_sec = None

    metrics = [metric1]
    request_latency_values: list[float | None] = [0.5]  # Meets latency

    result = calculate_goodput_metrics(metrics, config, [None], [None], [0.1], request_latency_values, [None])  # type: ignore[arg-type]

    assert result is not None
    assert result["goodput_percentage"] == 100.0


def test_calculate_goodput_metrics_per_request_override() -> None:
    """Test that per-request SLOs override global constraints."""
    config = GoodputConfig(constraints={"ttft": 0.5})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 2.0
    metric1.info = Mock()
    metric1.info.request_metrics.text.input_tokens = 0
    metric1.info.response_metrics.output_tokens = 10
    metric1.ttft_slo_sec = 0.2  # Stricter than global 0.5
    metric1.tpot_slo_sec = None

    metrics = [metric1]
    ttft_values: list[float | None] = [0.3]  # Fails stricter per-request SLO, would meet global

    result = calculate_goodput_metrics(metrics, config, ttft_values, [None], [0.1], [2.0], [None])  # type: ignore[arg-type]

    assert result is not None
    assert result["goodput_percentage"] == 0.0
    assert result["good_requests"] == 0

    # Now test that it meets if it is within per-request SLO
    ttft_values = [0.1]
    result = calculate_goodput_metrics(metrics, config, ttft_values, [None], [0.1], [2.0], [None])  # type: ignore[arg-type]
    assert result is not None
    assert result["goodput_percentage"] == 100.0
    assert result["good_requests"] == 1


def test_calculate_goodput_metrics_individual_attainment() -> None:
    """Test that individual attainment percentages are calculated."""
    config = GoodputConfig(constraints={"ttft": 0.5, "tpot": 0.1})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 2.0
    metric1.info = Mock()
    metric1.info.request_metrics.text.input_tokens = 0
    metric1.info.response_metrics.output_tokens = 10
    metric1.ttft_slo_sec = None
    metric1.tpot_slo_sec = None

    metric2 = Mock(spec=RequestLifecycleMetric)
    metric2.start_time = 2.0
    metric2.end_time = 4.0
    metric2.info = Mock()
    metric2.info.request_metrics.text.input_tokens = 0
    metric2.info.response_metrics.output_tokens = 20
    metric2.ttft_slo_sec = None
    metric2.tpot_slo_sec = None

    metrics = [metric1, metric2]
    ttft_values: list[float | None] = [0.3, 0.6]  # metric1 meets, metric2 fails TTFT
    tpot_values: list[float | None] = [0.08, 0.05]  # Both meet TPOT

    result = calculate_goodput_metrics(metrics, config, ttft_values, tpot_values, [0.1, 0.1], [2.0, 2.0], [None, None])  # type: ignore[arg-type]

    assert result is not None
    assert result["ttft_attainment_percentage"] == 50.0
    assert result["tpot_attainment_percentage"] == 100.0


def test_calculate_goodput_metrics_ttft_none_fails() -> None:
    """Test that if TTFT is None, it fails the constraint."""
    config = GoodputConfig(constraints={"ttft": 0.5})

    metric1 = Mock(spec=RequestLifecycleMetric)
    metric1.start_time = 0.0
    metric1.end_time = 2.0
    metric1.info = Mock()
    metric1.info.request_metrics.text.input_tokens = 0
    metric1.info.response_metrics.output_tokens = 10
    metric1.ttft_slo_sec = None
    metric1.tpot_slo_sec = None

    metrics = [metric1]
    ttft_values: list[float | None] = [None]  # Value is None

    result = calculate_goodput_metrics(metrics, config, ttft_values, [None], [0.1], [2.0], [None])  # type: ignore[arg-type]

    assert result is not None
    assert result["goodput_percentage"] == 0.0
    assert result["good_requests"] == 0


# 1 on-time success (40 out tok), 1 late success, 1 failed 504 (t=2..10), and 1 teardown-cancelled request (t=3..120).
# Expects submitted_requests=4, submitted_attainment_percentage=25%, window=10s (cancelled excluded from window per #1081).
def test_calculate_goodput_metrics_per_request_deadline_failures_and_cancelled() -> None:
    m_on_time = Mock(spec=RequestLifecycleMetric)
    m_on_time.start_time = 0.0
    m_on_time.end_time = 4.0
    m_on_time.info = Mock()
    m_on_time.info.request_metrics.text.input_tokens = 20
    m_on_time.info.response_metrics.output_tokens = 40
    m_on_time.ttft_slo_sec = None
    m_on_time.tpot_slo_sec = None
    m_on_time.request_latency_slo_sec = 5.0

    m_late = Mock(spec=RequestLifecycleMetric)
    m_late.start_time = 1.0
    m_late.end_time = 8.0
    m_late.info = Mock()
    m_late.info.request_metrics.text.input_tokens = 20
    m_late.info.response_metrics.output_tokens = 50
    m_late.ttft_slo_sec = None
    m_late.tpot_slo_sec = None
    m_late.request_latency_slo_sec = 5.0

    m_failed_504 = Mock(spec=RequestLifecycleMetric)
    m_failed_504.start_time = 2.0
    m_failed_504.end_time = 10.0
    m_failed_504.ttft_slo_sec = None
    m_failed_504.tpot_slo_sec = None
    m_failed_504.request_latency_slo_sec = 5.0

    m_cancelled = Mock(spec=RequestLifecycleMetric)
    m_cancelled.start_time = 3.0
    m_cancelled.end_time = 120.0
    m_cancelled.ttft_slo_sec = None
    m_cancelled.tpot_slo_sec = None
    m_cancelled.request_latency_slo_sec = 5.0

    result = calculate_goodput_metrics(
        [m_on_time, m_late],
        None,
        [None, None],
        [None, None],
        [0.1, 0.1],
        [4.0, 7.0],
        [None, None],
        failed_metrics=[m_failed_504],
        cancelled_metrics=[m_cancelled],
    )

    assert result is not None
    assert result["good_requests"] == 1
    assert result["total_requests"] == 2
    assert result["submitted_requests"] == 4
    assert result["goodput_percentage"] == 50.0
    assert result["submitted_attainment_percentage"] == 25.0
    assert result["good_output_tokens"] == 40
    # Answered window spans [0.0, 10.0] = 10.0s (teardown-cancelled end_time=120s does not stretch window)
    assert result["request_goodput"] == 1 / 10.0
    assert result["output_token_goodput"] == 40 / 10.0
    assert result["token_goodput"] == 60 / 10.0
    assert result["request_latency_attainment_percentage"] == 50.0


# 1 failed 504 request with request_latency constraint=5.0s and zero successes.
# Expects goodput_metrics emitted with 0% attainment, 0 good_output_tokens, and 0.0 rates.
def test_calculate_goodput_metrics_all_failed_requests() -> None:
    config = GoodputConfig(constraints={"request_latency": 5.0})
    m_fail = Mock(spec=RequestLifecycleMetric)
    m_fail.start_time = 0.0
    m_fail.end_time = 5.0
    m_fail.ttft_slo_sec = None
    m_fail.tpot_slo_sec = None
    m_fail.request_latency_slo_sec = None

    result = calculate_goodput_metrics(
        [],
        config,
        [],
        [],
        [],
        [],
        [],
        failed_metrics=[m_fail],
    )
    assert result is not None
    assert result["good_requests"] == 0
    assert result["total_requests"] == 0
    assert result["submitted_requests"] == 1
    assert result["goodput_percentage"] == 0.0
    assert result["submitted_attainment_percentage"] == 0.0
    assert result["good_output_tokens"] == 0
    assert result["request_goodput"] == 0.0
    assert result["output_token_goodput"] == 0.0


# Strict api.slo_unit parsing: slo_unit='s' parses x-llm-d-request-timeout-seconds as seconds,
# default slo_unit ('ms') parses header values as milliseconds, and per-request headers override globals.
def test_extract_request_slo_thresholds_uses_api_slo_unit() -> None:
    from inference_perf.client.modelserver.openai_client import extract_request_slo_thresholds
    from inference_perf.config import APIConfig, APIType

    cfg_sec = APIConfig(
        type=APIType.Completion,
        slo_unit="s",
        slo_request_latency_header="x-llm-d-request-timeout-seconds",
        headers={"x-llm-d-request-timeout-seconds": "180"},
    )
    ttft, tpot, req_lat = extract_request_slo_thresholds(cfg_sec)
    assert ttft is None
    assert tpot is None
    assert req_lat == 180.0

    # Default slo_unit is 'ms' even if the header name ends with '-seconds'
    cfg_default_ms = APIConfig(
        type=APIType.Completion,
        slo_request_latency_header="x-llm-d-request-timeout-seconds",
        headers={"x-llm-d-request-timeout-seconds": "2500"},
    )
    _, _, req_lat_default_ms = extract_request_slo_thresholds(cfg_default_ms)
    assert req_lat_default_ms == 2.5

    # Per-request header overrides global header
    cfg_ms = APIConfig(
        type=APIType.Completion,
        slo_unit="ms",
        headers={"x-slo-request-latency-ms": "2500", "x-slo-ttft-ms": "200", "x-slo-tpot-ms": "50"},
    )
    ttft_ms, tpot_ms, req_lat_override = extract_request_slo_thresholds(cfg_ms, {"x-slo-request-latency-ms": "1200"})
    assert ttft_ms == 0.2
    assert tpot_ms == 0.05
    assert req_lat_override == 1.2

    # Invalid numeric values and 'default' sentinel are ignored without raising
    cfg_bad = APIConfig(
        type=APIType.Completion,
        slo_unit="s",
        headers={"x-slo-ttft-s": "bad", "x-slo-tpot-s": "not-a-num", "x-slo-request-latency-s": "oops"},
    )
    assert extract_request_slo_thresholds(cfg_bad) == (None, None, None)

    cfg_default = APIConfig(
        type=APIType.Completion,
        slo_unit="s",
        headers={"x-slo-ttft-s": "default", "x-slo-tpot-s": "default", "x-slo-request-latency-s": "default"},
    )
    assert extract_request_slo_thresholds(cfg_default) == (None, None, None)


# Full summarize_requests + build_per_request_lifecycle_entry round-trip with per-request deadline SLO,
# a failed 504 request, and a stage-teardown cancelled request. Expects benchmark_time_seconds alignment
# and computed_metrics.request_latency_slo_sec populated.
def test_summarize_requests_and_per_request_entry_with_deadline_slo() -> None:
    from inference_perf.apis import ErrorResponseInfo, InferenceInfo, ResponseMetrics
    from inference_perf.apis.base import stage_teardown_cancelled_error
    from inference_perf.config import PerRequestFieldsConfig
    from inference_perf.payloads import RequestMetrics, Text
    from inference_perf.reportgen.base import build_per_request_lifecycle_entry, summarize_requests

    ok_metric = RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=0.0,
        start_time=0.0,
        end_time=2.0,
        request_data="{}",
        response_data="{}",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=10)),
            response_metrics=ResponseMetrics(output_tokens=20),
        ),
        error=None,
        request_latency_slo_sec=5.0,
    )
    fail_metric = RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=1.0,
        start_time=1.0,
        end_time=10.0,
        request_data="{}",
        response_data="{}",
        info=InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=10))),
        error=ErrorResponseInfo(error_type="HTTP Error 504", error_msg="DEADLINE_EXCEEDED"),
        request_latency_slo_sec=5.0,
    )
    cancel_metric = RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=2.0,
        start_time=2.0,
        end_time=120.0,
        request_data="{}",
        response_data="",
        info=InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=10))),
        error=stage_teardown_cancelled_error(),
        request_latency_slo_sec=5.0,
    )

    summary = summarize_requests([ok_metric, fail_metric, cancel_metric], percentiles=[50.0, 90.0])
    assert summary.benchmark_time_seconds == 10.0
    gp = summary.successes["goodput_metrics"]
    assert gp["good_requests"] == 1
    assert gp["total_requests"] == 1
    assert gp["submitted_requests"] == 3
    assert gp["goodput_percentage"] == 100.0
    assert abs(gp["submitted_attainment_percentage"] - (100.0 / 3.0)) < 1e-6
    assert gp["good_output_tokens"] == 20
    assert gp["request_goodput"] == summary.successes["throughput"]["requests_per_sec"]
    assert gp["output_token_goodput"] == summary.successes["throughput"]["output_tokens_per_sec"]
    assert gp["token_goodput"] == summary.successes["throughput"]["total_tokens_per_sec"]

    per_req = build_per_request_lifecycle_entry(ok_metric, PerRequestFieldsConfig(computed_metrics=True))
    assert per_req["computed_metrics"]["request_latency_slo_sec"] == 5.0


# Validator checks for submitted_requests, submitted_attainment_percentage, and rate upper bounds.
# Expects zero findings on consistent data and an error finding when any invariant is violated.
def test_goodput_findings_validates_submitted_and_rate_invariants() -> None:
    from inference_perf.reportgen.summary.validation import goodput_findings

    valid: dict[str, object] = {
        "load_summary": {"count": 10},
        "successes": {
            "count": 8,
            "throughput": {
                "requests_per_sec": 4.0,
                "output_tokens_per_sec": 80.0,
                "total_tokens_per_sec": 120.0,
            },
            "goodput_metrics": {
                "good_requests": 6,
                "total_requests": 8,
                "submitted_requests": 10,
                "goodput_percentage": 75.0,
                "submitted_attainment_percentage": 60.0,
                "request_goodput": 3.0,
                "output_token_goodput": 60.0,
                "token_goodput": 90.0,
            },
        },
        "failures": {"count": 2},
    }
    assert goodput_findings("summary.json", valid, "summary.goodput") == []

    successes_dict = valid["successes"]
    assert isinstance(successes_dict, dict)
    gp_dict = successes_dict["goodput_metrics"]
    assert isinstance(gp_dict, dict)

    bad_submitted = {
        **valid,
        "successes": {**successes_dict, "goodput_metrics": {**gp_dict, "submitted_requests": 9}},
    }
    assert any("submitted_requests" in f.message for f in goodput_findings("summary.json", bad_submitted, "summary.goodput"))

    bad_submitted_lt_total = {
        **valid,
        "successes": {**successes_dict, "goodput_metrics": {**gp_dict, "submitted_requests": 5}},
    }
    assert any(
        "submitted_requests (5) < total_requests (8)" in f.message
        for f in goodput_findings("summary.json", bad_submitted_lt_total, "summary.goodput")
    )

    bad_att = {
        **valid,
        "successes": {**successes_dict, "goodput_metrics": {**gp_dict, "submitted_attainment_percentage": 80.0}},
    }
    assert any(
        "submitted_attainment_percentage" in f.message for f in goodput_findings("summary.json", bad_att, "summary.goodput")
    )

    bad_rate = {
        **valid,
        "successes": {**successes_dict, "goodput_metrics": {**gp_dict, "request_goodput": 5.0}},
    }
    assert any("request_goodput" in f.message for f in goodput_findings("summary.json", bad_rate, "summary.goodput"))
