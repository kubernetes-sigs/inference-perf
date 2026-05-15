from unittest.mock import MagicMock, patch
from inference_perf.client.modelserver.tgi_client import TGImodelServerClient
from inference_perf.client.modelserver.openai_client import OpenAIMetrics
from inference_perf.config import APIType


def test_tgi_get_prometheus_metric_metadata() -> None:
    """Test get_prometheus_metric_metadata returns TGI-specific metrics."""
    mock_collector = MagicMock()
    mock_api_config = MagicMock()
    mock_api_config.type = APIType.Completion

    with patch("inference_perf.client.modelserver.openai_client.CustomTokenizer"):
        client = TGImodelServerClient(
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

    # Verify TGI specific metrics (they use tgi_ prefix)
    assert metrics.prompt_tokens.metric_name == "tgi_request_input_length"
    assert metrics.output_tokens.metric_name == "tgi_request_generated_tokens"
    assert metrics.queue_length.metric_name == "tgi_queue_size"

    # Verify custom_metrics is empty for TGI (based on implementation)
    assert len(metrics.custom_metrics) == 0
