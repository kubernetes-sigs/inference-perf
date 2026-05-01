from unittest.mock import patch
from inference_perf.client.metricsclient.prometheus_client.base import PrometheusMetricsClient
from inference_perf.client.metricsclient.base import ModelServerMetrics
from inference_perf.config import PrometheusClientConfig
from inference_perf.client.modelserver.base import BaseMetrics, Metric


# Mock a metric that implements get_queries
class MockMetric(Metric):
    def __init__(self, target_attr: str, query_template: str) -> None:
        self.target_attr = target_attr
        self.query_template = query_template

    def get_queries(self, duration: float) -> list[tuple[str, str]]:
        return [(self.target_attr, self.query_template)]


def test_get_model_server_metrics_list() -> None:
    """Test get_model_server_metrics with a list of metrics."""
    config = PrometheusClientConfig(url="http://localhost:9090")
    client = PrometheusMetricsClient(config)

    # Create dummy metrics
    metric1 = MockMetric("avg_inter_token_latency", "query1")
    metric2 = MockMetric("avg_time_per_output_token", "query2")
    metrics_list = [metric1, metric2]

    # Mock execute_query to return specific values
    def mock_execute(query: str, eval_time: float) -> float:
        if query == "query1":
            return 1.23
        elif query == "query2":
            return 4.56
        return 0.0

    with patch.object(PrometheusMetricsClient, "execute_query", side_effect=mock_execute):
        result = client.get_model_server_metrics(metrics_list, query_duration=30, query_eval_time=100)

    assert isinstance(result, ModelServerMetrics)
    assert result.avg_inter_token_latency == 1.23
    assert result.avg_time_per_output_token == 4.56


def test_get_model_server_metrics_base_metrics() -> None:
    """Test get_model_server_metrics with a BaseMetrics object."""
    config = PrometheusClientConfig(url="http://localhost:9090")
    client = PrometheusMetricsClient(config)

    # Let's use a real class or a plain object instead of MagicMock for simplicity with hasattr.
    class FakeBaseMetrics(BaseMetrics):
        def get_all_metrics(self) -> list[Metric]:
            return [MockMetric("avg_inter_token_latency", "query1")]

    with patch.object(PrometheusMetricsClient, "execute_query", return_value=1.23):
        result = client.get_model_server_metrics(FakeBaseMetrics(), query_duration=30, query_eval_time=100)

    assert isinstance(result, ModelServerMetrics)
    assert result.avg_inter_token_latency == 1.23


def test_get_model_server_metrics_unknown_type() -> None:
    """Test get_model_server_metrics with an unknown type."""
    config = PrometheusClientConfig(url="http://localhost:9090")
    client = PrometheusMetricsClient(config)

    result = client.get_model_server_metrics("not a list or BaseMetrics", query_duration=30, query_eval_time=100)

    assert isinstance(result, ModelServerMetrics)
    # Fields should be default (likely 0.0 or None)
    assert result.avg_inter_token_latency == 0.0  # Assuming default is 0.0
