from inference_perf.apis import InferenceInfo, RequestLifecycleMetric, StreamedResponseMetrics
from inference_perf.config import PerRequestFieldsConfig
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.base import build_per_request_lifecycle_entry


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
