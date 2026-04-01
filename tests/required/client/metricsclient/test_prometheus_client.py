from typing import Any, List
from unittest.mock import patch
from inference_perf.client.server_metrics.prometheus_client.base import PrometheusMetricsClient
from inference_perf.client.server_metrics.base import ModelServerMetrics
from inference_perf.config import PrometheusClientConfig
from inference_perf.client.modelserver.base import BaseMetrics, HistogramMetric, Metric


def test_get_model_server_metrics_base_metrics() -> None:
    """Test get_model_server_metrics with a BaseMetrics subclass."""
    config = PrometheusClientConfig(url="http://localhost:9090")
    client = PrometheusMetricsClient(config)

    class FakeBaseMetrics(BaseMetrics):
        def get_all_metrics(self) -> List[Metric[Any]]:
            return [
                HistogramMetric("inter_token_latency", "fake:itl", []),
                HistogramMetric("time_per_output_token", "fake:tpot", []),
            ]

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

    metadata = BaseMetrics(custom_metrics=[HistogramMetric("inter_token_latency", "fake:itl", [])])

    with patch.object(PrometheusMetricsClient, "execute_query", return_value=1.23):
        result = client.get_model_server_metrics(metadata, query_duration=30, query_eval_time=100)

    assert isinstance(result, ModelServerMetrics)
    assert result.inter_token_latency.avg == 1.23
