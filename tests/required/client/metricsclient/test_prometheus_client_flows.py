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
"""Prometheus client flows against canned HTTP responses (#660).

Covers the paths between a declared metric and a report number: the HTTP
request/response handling in execute_query (success, empty, non-200,
error body, non-numeric value, connection failure) and the window/eval-time
arithmetic in collect_metrics_summary and collect_metrics_for_stage. No
live Prometheus; every response is canned through a mocked session, so a
silently swallowed error or a wrong query window fails here in plain CI.
"""

from typing import Any, Dict, List, Optional, Tuple, cast
from unittest.mock import MagicMock, patch

import pytest
from pydantic import HttpUrl

from inference_perf.client.modelserver.metrics import CounterMetric, GaugeMetric, HistogramMetric
from inference_perf.client.modelserver.openai_client import OpenAIMetrics
from inference_perf.client.server_metrics.base import PerfRuntimeParameters, StageRuntimeInfo, StageStatus
from inference_perf.client.server_metrics.prometheus_client.base import PrometheusMetricsClient
from inference_perf.config import PrometheusClientConfig

REQUESTS_GET = "inference_perf.client.server_metrics.prometheus_client.base.requests.get"
TIME_TIME = "inference_perf.client.server_metrics.prometheus_client.base.time.time"
TIME_SLEEP = "inference_perf.client.server_metrics.prometheus_client.base.time.sleep"


def make_client(scrape_interval: int = 15) -> PrometheusMetricsClient:
    return PrometheusMetricsClient(PrometheusClientConfig(url=HttpUrl("http://prom:9090"), scrape_interval=scrape_interval))


def prom_response(body: Dict[str, Any], status_error: Optional[Exception] = None) -> MagicMock:
    response = MagicMock()
    response.json.return_value = body
    if status_error is not None:
        response.raise_for_status.side_effect = status_error
    else:
        response.raise_for_status.return_value = None
    return response


def vector_body(value: str) -> Dict[str, Any]:
    return {"status": "success", "data": {"resultType": "vector", "result": [{"metric": {}, "value": [1700000000.0, value]}]}}


def metrics_metadata() -> OpenAIMetrics:
    """The six fields ModelServerMetrics requires, with recognizable fake names."""
    return OpenAIMetrics(
        filters=["model_name='m'"],
        prompt_tokens=CounterMetric("fake:pt"),
        output_tokens=CounterMetric("fake:ot"),
        requests=CounterMetric("fake:req"),
        request_latency=HistogramMetric("fake:lat"),
        queue_length=GaugeMetric("fake:q"),
        time_per_output_token=HistogramMetric("fake:tpot"),
    )


def record_queries(client: PrometheusMetricsClient) -> List[Tuple[str, str]]:
    """Replace execute_query with a recorder returning 1.0 for every query."""
    recorded: List[Tuple[str, str]] = []

    def fake_execute(query: str, eval_time: str) -> float:
        recorded.append((query, eval_time))
        return 1.0

    cast(Any, client).execute_query = fake_execute
    return recorded


# --- __init__ ---


def test_init_builds_query_url_and_scrape_interval() -> None:
    client = make_client(scrape_interval=25)
    assert client.query_url == "http://prom:9090/api/v1/query"
    assert client.scrape_interval == 25


def test_init_rejects_missing_url() -> None:
    with pytest.raises(Exception, match="prometheus url missing"):
        PrometheusMetricsClient(PrometheusClientConfig(google_managed=True))


def test_init_rejects_missing_config() -> None:
    with pytest.raises(Exception, match="prometheus config missing"):
        PrometheusMetricsClient(cast(PrometheusClientConfig, None))


def test_wait_sleeps_for_scrape_interval_plus_buffer() -> None:
    with patch(TIME_SLEEP) as mock_sleep:
        make_client(scrape_interval=15).wait()
    mock_sleep.assert_called_once_with(17)


# --- execute_query: canned HTTP responses ---


def test_execute_query_success_returns_rounded_float() -> None:
    client = make_client()
    with patch(REQUESTS_GET, return_value=prom_response(vector_body("0.12345678"))) as mock_get:
        assert client.execute_query("up", "1700000000.0") == 0.123457
    mock_get.assert_called_once_with(
        "http://prom:9090/api/v1/query", headers={}, params={"query": "up", "time": "1700000000.0"}
    )


def test_execute_query_empty_result_returns_zero() -> None:
    body: Dict[str, Any] = {"status": "success", "data": {"resultType": "vector", "result": []}}
    with patch(REQUESTS_GET, return_value=prom_response(body)):
        assert make_client().execute_query("up", "0") == 0.0


def test_execute_query_http_error_returns_zero() -> None:
    response = prom_response({}, status_error=Exception("500 Server Error"))
    with patch(REQUESTS_GET, return_value=response):
        assert make_client().execute_query("up", "0") == 0.0


def test_execute_query_connection_failure_returns_zero() -> None:
    with patch(REQUESTS_GET, side_effect=ConnectionError("refused")):
        assert make_client().execute_query("up", "0") == 0.0


def test_execute_query_error_status_body_returns_zero() -> None:
    body = {"status": "error", "errorType": "bad_data", "error": "parse error"}
    with patch(REQUESTS_GET, return_value=prom_response(body)):
        assert make_client().execute_query("up{", "0") == 0.0


def test_execute_query_non_numeric_value_returns_zero() -> None:
    with patch(REQUESTS_GET, return_value=prom_response(vector_body("NaN-ish"))):
        assert make_client().execute_query("up", "0") == 0.0


# --- collect_metrics_summary / collect_metrics_for_stage: window arithmetic ---


def test_collect_metrics_summary_windows_run_duration() -> None:
    client = make_client()
    recorded = record_queries(client)
    runtime = PerfRuntimeParameters(start_time=940.0, duration=60.0, model_server_metrics=metrics_metadata(), stages={})

    with patch(TIME_TIME, return_value=1000.0):
        result = client.collect_metrics_summary(runtime)

    assert result is not None
    # 23 queries: 3 per counter (x3), 5 per histogram (x2), 4 for the gauge.
    assert len(recorded) == 23
    # Window is now - start_time; eval time is now.
    assert recorded[0] == ("sum(increase(fake:pt{model_name='m'}[60s]))", "1000.0")
    assert all(eval_time == "1000.0" for _, eval_time in recorded)
    assert all("[60s]" in query for query, _ in recorded)
    # Every field is populated from the canned 1.0 results.
    assert result.requests.total == 1.0
    assert result.queue_length.p99 == 1.0
    assert result.request_latency.per_second == 1.0


def test_collect_metrics_summary_without_runtime_parameters_returns_none() -> None:
    assert make_client().collect_metrics_summary(cast(PerfRuntimeParameters, None)) is None


def test_collect_metrics_for_stage_windows_stage_duration() -> None:
    client = make_client(scrape_interval=15)
    recorded = record_queries(client)
    stage = StageRuntimeInfo(stage_id=0, rate=1.0, start_time=100.0, end_time=160.0, status=StageStatus.COMPLETED)
    runtime = PerfRuntimeParameters(
        start_time=100.0, duration=60.0, model_server_metrics=metrics_metadata(), stages={0: stage}
    )

    result = client.collect_metrics_for_stage(runtime, 0)

    assert result is not None
    # Eval time is stage end + scrape interval + buffer; window reaches back to stage start.
    # 160 + 15 + 2 = 177, 177 - 100 = 77.
    assert recorded[0] == ("sum(increase(fake:pt{model_name='m'}[77s]))", "177.0")
    assert all(eval_time == "177.0" for _, eval_time in recorded)
    assert all("[77s]" in query for query, _ in recorded)


def test_collect_metrics_for_stage_with_unknown_stage_returns_none() -> None:
    client = make_client()
    runtime = PerfRuntimeParameters(start_time=100.0, duration=60.0, model_server_metrics=metrics_metadata(), stages={})
    assert client.collect_metrics_for_stage(runtime, 7) is None


def test_collect_metrics_for_stage_without_runtime_parameters_returns_none() -> None:
    assert make_client().collect_metrics_for_stage(cast(PerfRuntimeParameters, None), 0) is None
