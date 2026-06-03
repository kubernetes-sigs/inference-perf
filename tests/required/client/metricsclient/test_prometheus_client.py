from typing import Any, Dict, Iterator, List, Tuple
import pytest
from pydantic import ValidationError
from unittest.mock import patch
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
    """The fields ModelServerMetrics requires; every real client's metadata declares them."""
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
