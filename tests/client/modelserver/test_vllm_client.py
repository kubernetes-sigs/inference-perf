from unittest.mock import MagicMock, patch
from inference_perf.client.modelserver.vllm_client import vLLMModelServerClient
from inference_perf.client.modelserver.openai_client import OpenAIMetrics
from inference_perf.config import APIType


def test_vllm_get_prometheus_metric_metadata() -> None:
    """Test get_prometheus_metric_metadata returns vLLM-specific metrics."""
    mock_collector = MagicMock()
    mock_api_config = MagicMock()
    mock_api_config.type = APIType.Completion

    with patch("inference_perf.client.modelserver.openai_client.CustomTokenizer"):
        client = vLLMModelServerClient(
            metrics_collector=mock_collector,
            api_config=mock_api_config,
            uri="http://localhost:8000",
            model_name="test-model",
            tokenizer_config=None,
            max_tcp_connections=10,
            additional_filters=[],
        )

    metrics = client.get_prometheus_metric_metadata()

    assert isinstance(metrics, OpenAIMetrics)

    # Verify some vLLM specific metrics are present in custom_metrics
    custom_metric_names = [m.metric_name for m in metrics.custom_metrics]
    assert "vllm:num_requests_running" in custom_metric_names
    assert "vllm:time_to_first_token_seconds" in custom_metric_names
    assert "vllm:kv_cache_usage_perc" in custom_metric_names
