from unittest.mock import MagicMock, patch
from inference_perf.client.modelserver.openai_client import openAIModelServerClient
from inference_perf.config import APIConfig, APIType


@patch("inference_perf.client.modelserver.openai_client.CustomTokenizer")
def test_openai_client_prometheus_metadata(mock_tokenizer):
    # Setup standard mock configs
    mock_collector = MagicMock()
    mock_api_config = APIConfig(type=APIType.Chat, streaming=False)

    # Initialize the client specifically to test the abstract method override
    client = openAIModelServerClient(
        metrics_collector=mock_collector,
        api_config=mock_api_config,
        uri="http://0.0.0.0:8000",
        model_name="mock-model",
        tokenizer_config=None,
        max_tcp_connections=10,
        additional_filters=[],
    )

    # Verify the empty MetricMetadata is safely returned
    metadata = client.get_prometheus_metric_metadata()
    assert isinstance(metadata, dict)
    assert len(metadata) == 0


@patch("inference_perf.client.modelserver.openai_client.CustomTokenizer")
def test_openai_client_supported_apis(mock_tokenizer):
    mock_collector = MagicMock()
    mock_api_config = APIConfig(type=APIType.Chat, streaming=False)
    client = openAIModelServerClient(
        metrics_collector=mock_collector,
        api_config=mock_api_config,
        uri="http://0.0.0.0:8000",
        model_name="mock-model",
        tokenizer_config=None,
        max_tcp_connections=10,
        additional_filters=[],
    )
    apis = client.get_supported_apis()
    assert APIType.Completion in apis
    assert APIType.Chat in apis


@patch("inference_perf.client.modelserver.openai_client.CustomTokenizer")
@patch("inference_perf.client.modelserver.openai_client.requests.get")
def test_openai_client_get_supported_models(mock_get, mock_tokenizer):
    mock_response = MagicMock()
    mock_response.json.return_value = {"data": [{"id": "model-1"}, {"id": "model-2"}]}
    mock_get.return_value = mock_response

    mock_collector = MagicMock()
    mock_api_config = APIConfig(type=APIType.Chat, streaming=False)
    client = openAIModelServerClient(
        metrics_collector=mock_collector,
        api_config=mock_api_config,
        uri="http://0.0.0.0:8000",
        model_name="mock-model",
        tokenizer_config=None,
        max_tcp_connections=10,
        additional_filters=[],
    )
    models = client.get_supported_models()
    assert len(models) == 2
    assert models[0]["id"] == "model-1"
    mock_get.assert_called_once_with("http://0.0.0.0:8000/v1/models")


@patch("inference_perf.client.modelserver.openai_client.CustomTokenizer")
@patch("inference_perf.client.modelserver.openai_client.requests.get")
def test_openai_client_get_supported_models_error(mock_get, mock_tokenizer):
    mock_get.side_effect = Exception("Connection Error")

    mock_collector = MagicMock()
    mock_api_config = APIConfig(type=APIType.Chat, streaming=False)
    client = openAIModelServerClient(
        metrics_collector=mock_collector,
        api_config=mock_api_config,
        uri="http://0.0.0.0:8000",
        model_name="mock-model",
        tokenizer_config=None,
        max_tcp_connections=10,
        additional_filters=[],
    )
    models = client.get_supported_models()
    assert models == []
