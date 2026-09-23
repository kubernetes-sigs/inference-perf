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
from typing import Any, Dict, Iterator, List, Optional, Tuple
import pytest
from pydantic import ValidationError
from unittest.mock import Mock, patch
from inference_perf.client.server_metrics.prometheus_client.base import PrometheusMetricsClient
from inference_perf.client.server_metrics.base import ModelServerMetrics
from inference_perf.config import PrometheusClientConfig
from inference_perf.client.modelserver.metrics import (
    BaseMetrics,
    CounterMetric,
    CounterResult,
    GaugeMetric,
    HistogramMetric,
    Metric,
)


def _required_metrics() -> Dict[str, Metric[Any]]:
    """The common ModelServerMetrics fields; every real client's metadata declares them."""
    return {
        "prompt_tokens": CounterMetric("fake:pt"),
        "output_tokens": CounterMetric("fake:ot"),
        "requests": CounterMetric("fake:req"),
        "request_latency": HistogramMetric("fake:lat"),
        "queue_length": GaugeMetric("fake:q"),
        "time_per_output_token": HistogramMetric("fake:tpot"),
    }


def test_get_model_server_metrics_base_metrics() -> None:
    """Test get_model_server_metrics with a BaseMetrics subclass."""
    config = PrometheusClientConfig(url="http://localhost:9090")
    client = PrometheusMetricsClient(config)

    class FakeBaseMetrics(BaseMetrics):
        def _iter_metrics(self) -> Iterator[Tuple[str, Metric[Any]]]:
            yield from _required_metrics().items()
            yield "inter_token_latency", HistogramMetric("fake:itl")

    def mock_execute(query: str, eval_time: str) -> float:
        if "fake:itl" in query and "_sum" in query and "/" in query:
            return 1.23
        if "fake:tpot" in query and "_sum" in query and "/" in query:
            return 4.56
        return 0.0

    with patch.object(PrometheusMetricsClient, "execute_query", side_effect=mock_execute):
        result = client.get_model_server_metrics(FakeBaseMetrics(), query_duration=30, query_eval_time=100)

    assert isinstance(result, ModelServerMetrics)
    assert result.inter_token_latency.avg == 1.23
    assert result.time_per_output_token.avg == 4.56


def test_get_model_server_metrics_uses_custom_metrics_by_default() -> None:
    """A bare BaseMetrics should query its custom_metrics via the default get_all_metrics."""
    config = PrometheusClientConfig(url="http://localhost:9090")
    client = PrometheusMetricsClient(config)

    metadata = BaseMetrics(custom_metrics={**_required_metrics(), "inter_token_latency": HistogramMetric("fake:itl")})

    with patch.object(PrometheusMetricsClient, "execute_query", return_value=1.23):
        result = client.get_model_server_metrics(metadata, query_duration=30, query_eval_time=100)

    assert isinstance(result, ModelServerMetrics)
    assert result.inter_token_latency.avg == 1.23


@pytest.mark.parametrize(
    ("field", "metric"),
    [
        ("queue_length", GaugeMetric("sglang:num_queue_reqs")),
        ("prompt_tokens", CounterMetric("sglang:prompt_tokens_total")),
        ("request_latency", HistogramMetric("sglang:e2e_request_latency_seconds")),
    ],
)
def test_get_model_server_metrics_retries_underscored_metric_names(field: str, metric: Metric[Any]) -> None:
    """Prometheus can normalize colons in gauge, counter, and histogram names on scrape."""
    config = PrometheusClientConfig(url="http://localhost:9090")
    client = PrometheusMetricsClient(config)
    metadata = BaseMetrics(custom_metrics={field: metric})
    seen_queries: List[str] = []

    def mock_execute(query: str, eval_time: str) -> Any:
        seen_queries.append(query)
        if "sglang:" in query:
            return None  # Successful Prometheus query with no matching series.
        return float(len(seen_queries))

    with patch.object(PrometheusMetricsClient, "execute_query", side_effect=mock_execute):
        result = client.get_model_server_metrics(metadata, query_duration=30, query_eval_time=100)

    assert result is not None
    expected_queries = metric.get_queries(30, metadata.filters)
    assert seen_queries == [
        query_variant for query in expected_queries for query_variant in (query, query.replace("sglang:", "sglang_"))
    ]
    assert all(value > 0 for value in getattr(result, field).model_dump().values())


def test_execute_query_distinguishes_empty_result_from_zero() -> None:
    config = PrometheusClientConfig(url="http://localhost:9090")
    client = PrometheusMetricsClient(config)
    empty_response = Mock()
    empty_response.json.return_value = {"status": "success", "data": {"result": []}}
    zero_response = Mock()
    zero_response.json.return_value = {
        "status": "success",
        "data": {"result": [{"metric": {}, "value": [100, "0"]}]},
    }

    with patch(
        "inference_perf.client.server_metrics.prometheus_client.base.requests.get",
        side_effect=[empty_response, zero_response],
    ):
        assert client.execute_query("sglang:num_queue_reqs{}", "100") is None
        assert client.execute_query("sglang:num_queue_reqs{}", "100") == 0.0


def test_get_model_server_metrics_does_not_retry_a_zero_value() -> None:
    config = PrometheusClientConfig(url="http://localhost:9090")
    client = PrometheusMetricsClient(config)
    metric = GaugeMetric("sglang:num_queue_reqs")
    metadata = BaseMetrics(custom_metrics={"queue_length": metric})
    seen_queries: List[str] = []

    def mock_execute(query: str, eval_time: str) -> float:
        seen_queries.append(query)
        return 0.0 if "sglang:" in query else 12.0

    with patch.object(PrometheusMetricsClient, "execute_query", side_effect=mock_execute):
        result = client.get_model_server_metrics(metadata, query_duration=30, query_eval_time=100)

    assert seen_queries == metric.get_queries(30, metadata.filters)
    assert result is not None
    assert result.queue_length.model_dump() == {"avg": 0.0, "median": 0.0, "p90": 0.0, "p99": 0.0}


def test_underscored_metric_query_preserves_label_values() -> None:
    metric = GaugeMetric("sglang:num_queue_reqs")
    filters = "model_name='sglang:num_queue_reqs'"
    metadata = BaseMetrics(filters=[filters], custom_metrics={"queue_length": metric})
    query = metric.get_queries(30, metadata.filters)[0]
    seen_queries: List[str] = []

    def mock_execute(query: str, eval_time: str) -> Optional[float]:
        seen_queries.append(query)
        if "(sglang:" in query or ", sglang:" in query:
            return None
        return 1.0

    with patch.object(PrometheusMetricsClient, "execute_query", side_effect=mock_execute):
        result = PrometheusMetricsClient(PrometheusClientConfig(url="http://localhost:9090")).get_model_server_metrics(
            metadata, query_duration=30, query_eval_time=100
        )

    assert seen_queries == [
        query,
        query.replace("sglang:num_queue_reqs{", "sglang_num_queue_reqs{"),
        *[
            query_variant
            for query in metric.get_queries(30, metadata.filters)[1:]
            for query_variant in (query, query.replace("sglang:num_queue_reqs{", "sglang_num_queue_reqs{"))
        ],
    ]
    assert result is not None


def test_get_model_server_metrics_empty_metadata_returns_defaults() -> None:
    """A client that declares no metrics (the mock model server) yields an all-zeros result, not a crash.

    Regression test: report generation runs after the load has been sent, so a ValidationError
    here would have cost the user every report of a completed run (mock server + Prometheus
    metrics client is a legal config combination).
    """
    config = PrometheusClientConfig(url="http://localhost:9090")
    client = PrometheusMetricsClient(config)

    with patch.object(PrometheusMetricsClient, "execute_query") as execute:
        result = client.get_model_server_metrics(BaseMetrics(), query_duration=30, query_eval_time=100)

    execute.assert_not_called()
    assert result == ModelServerMetrics()


def test_get_model_server_metrics_rejects_unknown_field() -> None:
    """A declaration whose key is not a ModelServerMetrics field fails before any query runs.

    model_validate ignores extra keys, so without the collection-time guard the metric's
    queries would run and the results be silently dropped (a zero column, no error).
    """
    config = PrometheusClientConfig(url="http://localhost:9090")
    client = PrometheusMetricsClient(config)

    metadata = BaseMetrics(custom_metrics={**_required_metrics(), "not_a_real_field": CounterMetric("fake:x")})

    with patch.object(PrometheusMetricsClient, "execute_query") as execute:
        with pytest.raises(ValueError, match="not_a_real_field"):
            client.get_model_server_metrics(metadata, query_duration=30, query_eval_time=100)

    execute.assert_not_called()


def test_get_model_server_metrics_rejects_wrong_result_type() -> None:
    """A metric whose result type does not match its target field fails at construction.

    This is the strictness the typed assembly buys over the old setattr: the result
    is validated against the field's declared type instead of being written blindly.
    """
    config = PrometheusClientConfig(url="http://localhost:9090")
    client = PrometheusMetricsClient(config)

    class WrongTypeMetric(Metric[CounterResult]):
        metric_name = "fake:wrong"

        def get_queries(self, duration: float, filters: str) -> List[str]:
            return ["q"]

        def parse(self, results: List[float]) -> CounterResult:
            return CounterResult(total=1.0)

    # request_latency is declared HistogramResult; supplying a CounterResult there is the only
    # validation failure (the other required fields are present and correctly typed).
    metadata = BaseMetrics(custom_metrics={**_required_metrics(), "request_latency": WrongTypeMetric()})

    with patch.object(PrometheusMetricsClient, "execute_query", return_value=1.0):
        with pytest.raises(ValidationError):
            client.get_model_server_metrics(metadata, query_duration=30, query_eval_time=100)
