import pytest

from inference_perf.apis import ErrorResponseInfo, InferenceInfo, RequestLifecycleMetric, StreamedResponseMetrics
from inference_perf.config import PerRequestFieldsConfig
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.base import build_per_request_lifecycle_entry


class _FakeTokenizer:
    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        return len(text.split())


def _metric() -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=0.0,
        start_time=1.0,
        end_time=2.0,
        request_data="raw request",
        response_data="raw response",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=5)),
            response_metrics=StreamedResponseMetrics(
                output_tokens=2,
                response_chunks=['{"choices": [{"text": "hello"}]}'],
                chunk_times=[1.5],
                output_token_times=[1.5],
            ),
        ),
        error=None,
    )


def test_per_request_fields_default_preserves_full_report_shape() -> None:
    entry = build_per_request_lifecycle_entry(_metric(), PerRequestFieldsConfig())

    assert entry["start_time"] == 1.0
    assert entry["end_time"] == 2.0
    assert entry["request"] == "raw request"
    assert entry["response"] == "raw response"
    assert entry["info"]["response_metrics"]["response_chunks"] == ['{"choices": [{"text": "hello"}]}']
    assert entry["error"] is None


def test_per_request_fields_can_omit_top_level_raw_fields() -> None:
    entry = build_per_request_lifecycle_entry(
        _metric(),
        PerRequestFieldsConfig(request=False, response=False, info=False),
    )

    assert "request" not in entry
    assert "response" not in entry
    assert "info" not in entry
    assert entry["start_time"] == 1.0
    assert entry["end_time"] == 2.0
    assert entry["error"] is None


def test_per_request_fields_can_omit_response_chunks_without_mutating_metric() -> None:
    metric = _metric()
    entry = build_per_request_lifecycle_entry(metric, PerRequestFieldsConfig(response_chunks=False))

    response_metrics = entry["info"]["response_metrics"]
    assert "response_chunks" not in response_metrics
    assert response_metrics["chunk_times"] == [1.5]
    assert response_metrics["output_token_times"] == [1.5]
    assert isinstance(metric.info.response_metrics, StreamedResponseMetrics)
    assert metric.info.response_metrics.response_chunks == ['{"choices": [{"text": "hello"}]}']


def test_per_request_fields_computed_metrics_disabled_by_default() -> None:
    entry = build_per_request_lifecycle_entry(_metric(), PerRequestFieldsConfig())

    assert "computed_metrics" not in entry


def test_per_request_fields_computed_metrics_can_be_enabled() -> None:
    entry = build_per_request_lifecycle_entry(_metric(), PerRequestFieldsConfig(computed_metrics=True))

    assert entry["computed_metrics"] == {
        "request_latency": 1.0,
        "normalized_time_per_output_token": 0.5,
        "time_to_first_token": None,
        "time_per_output_token": None,
        "inter_token_latency": None,
        "inter_token_latencies": [],
        "input_tokens": 5,
        "output_tokens": 2,
        "ttft_slo_sec": None,
        "tpot_slo_sec": None,
    }


def test_per_request_fields_disabled_computed_metrics_does_not_mutate_token_metrics() -> None:
    metric = _metric()

    build_per_request_lifecycle_entry(metric, PerRequestFieldsConfig(computed_metrics=False), tokenizer=_FakeTokenizer())

    assert isinstance(metric.info.response_metrics, StreamedResponseMetrics)
    assert metric.info.response_metrics.output_tokens == 2
    assert metric.info.response_metrics.output_token_times == [1.5]


def test_per_request_fields_computed_metrics_omitted_for_failed_requests() -> None:
    metric = _metric()
    metric.error = ErrorResponseInfo(error_type="TimeoutError", error_msg="boom")

    entry = build_per_request_lifecycle_entry(metric, PerRequestFieldsConfig(computed_metrics=True))

    assert "computed_metrics" not in entry


def test_per_request_fields_computed_metrics_uses_tokenizer_corrected_chunks() -> None:
    metric = RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=0.0,
        start_time=1.0,
        end_time=2.0,
        request_data="raw request",
        response_data="raw response",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=5)),
            response_metrics=StreamedResponseMetrics(
                output_tokens=3,
                response_chunks=[
                    '{"choices": [{"text": "hello"}]}',
                    '{"choices": [{"text": " world foo"}]}',
                ],
                chunk_times=[1.2, 1.6],
                output_token_times=[],
            ),
        ),
        error=None,
    )

    entry = build_per_request_lifecycle_entry(
        metric, PerRequestFieldsConfig(computed_metrics=True), tokenizer=_FakeTokenizer()
    )

    assert entry["info"]["response_metrics"]["output_token_times"] == pytest.approx([1.2, 1.6, 1.6])
    assert entry["computed_metrics"]["time_to_first_token"] == pytest.approx(0.2)
    assert entry["computed_metrics"]["time_per_output_token"] == pytest.approx(0.2)
    assert entry["computed_metrics"]["inter_token_latency"] == pytest.approx(0.2)
    assert entry["computed_metrics"]["inter_token_latencies"] == pytest.approx([0.4, 0.0])
    assert entry["computed_metrics"]["input_tokens"] == 5
    # The correction sets output_token_times only; output_tokens keeps the API layer's
    # whole-message count (see #564).
    assert entry["computed_metrics"]["output_tokens"] == 3
