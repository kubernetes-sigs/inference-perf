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
import pytest
import asyncio
from typing import List
import aiohttp
import json
import ssl
from unittest.mock import AsyncMock, MagicMock, patch
from inference_perf.client.modelserver.openai_client import openAIModelServerClient, openAIModelServerClientSession
from inference_perf.apis.base import InferenceInfo, ErrorResponseInfo
from inference_perf.config import APIConfig, ResponseFormatType, APIType


# Dummy concrete class for testing the abstract openAIModelServerClient
class DummyOpenAIClient(openAIModelServerClient):
    def get_prometheus_metric_metadata(self) -> MagicMock:
        return MagicMock()

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.CHAT, APIType.COMPLETION]


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.uri = "http://test-uri"
    client.api_config = MagicMock()
    client.api_config.headers = {}
    client.api_config.response_format = None
    client.tokenizer = MagicMock()
    client.metrics_collector = MagicMock()
    client.cert_path = None
    client.key_path = None
    client.api_key = None
    client.timeout = None
    client.max_tcp_connections = 100
    return client


@pytest.fixture
def mock_data() -> MagicMock:
    data = MagicMock()
    data.get_route.return_value = "/test"
    data.process_failure = AsyncMock(return_value=InferenceInfo())
    data.process_response = AsyncMock(return_value=InferenceInfo())
    data.to_payload = AsyncMock(return_value={"mock": "data"})
    return data


@pytest.mark.asyncio
async def test_process_request_timeout(mock_client: MagicMock, mock_data: MagicMock) -> None:
    session = openAIModelServerClientSession(mock_client)
    session.session = MagicMock()

    # Mock the post request context manager to raise a TimeoutError
    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError("Test timeout"))
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
    session.session.post.return_value = mock_post_ctx

    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)

    # Verify the metric was recorded with the correct ErrorResponseInfo
    mock_client.metrics_collector.record_metric.assert_called_once()
    metric = mock_client.metrics_collector.record_metric.call_args[0][0]
    assert isinstance(metric.error, ErrorResponseInfo)
    assert metric.error.error_type == "TimeoutError"


@pytest.mark.asyncio
async def test_process_request_client_error(mock_client: MagicMock, mock_data: MagicMock) -> None:
    session = openAIModelServerClientSession(mock_client)
    session.session = MagicMock()

    # Mock the post request context manager to raise a ClientError
    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(side_effect=aiohttp.ClientError("Test client error"))
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
    session.session.post.return_value = mock_post_ctx

    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)

    # Verify the metric was recorded with the correct ErrorResponseInfo
    mock_client.metrics_collector.record_metric.assert_called_once()
    metric = mock_client.metrics_collector.record_metric.call_args[0][0]
    assert isinstance(metric.error, ErrorResponseInfo)
    assert metric.error.error_type == "ClientError"


@pytest.mark.asyncio
async def test_process_request_general_exception(mock_client: MagicMock, mock_data: MagicMock) -> None:
    session = openAIModelServerClientSession(mock_client)
    session.session = MagicMock()

    # Mock the post request context manager to raise a generic Exception
    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(side_effect=ValueError("Test general error"))
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
    session.session.post.return_value = mock_post_ctx

    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)

    # Verify the metric was recorded with the correct ErrorResponseInfo
    mock_client.metrics_collector.record_metric.assert_called_once()
    metric = mock_client.metrics_collector.record_metric.call_args[0][0]
    assert isinstance(metric.error, ErrorResponseInfo)
    assert metric.error.error_type == "ValueError"


# Tests for openAIModelServerClient.__init__ using DummyOpenAIClient


@patch("inference_perf.client.modelserver.openai_client.CustomTokenizer")
def test_openai_client_init_with_model_name(mock_custom_tokenizer: MagicMock) -> None:
    metrics_collector = MagicMock()
    api_config = APIConfig(type=APIType.CHAT)

    client = DummyOpenAIClient(
        metrics_collector=metrics_collector,
        api_config=api_config,
        uri="http://test-uri",
        model_name="test-model",
        tokenizer_config=None,
        max_tcp_connections=100,
        additional_filters=[],
    )

    assert client.model_name == "test-model"
    assert client.uri == "http://test-uri"
    mock_custom_tokenizer.assert_called_once()


@patch("inference_perf.client.modelserver.openai_client.CustomTokenizer")
@patch("inference_perf.client.modelserver.openai_client.requests.get")
def test_openai_client_init_resolve_model(mock_get: MagicMock, mock_custom_tokenizer: MagicMock) -> None:
    metrics_collector = MagicMock()
    api_config = APIConfig(type=APIType.CHAT)

    # Mock successful response from /v1/models
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": [{"id": "resolved-model"}]}
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    client = DummyOpenAIClient(
        metrics_collector=metrics_collector,
        api_config=api_config,
        uri="http://test-uri",
        model_name=None,  # Request resolution
        tokenizer_config=None,
        max_tcp_connections=100,
        additional_filters=[],
    )

    assert client.model_name == "resolved-model"
    mock_get.assert_called_once_with("http://test-uri/v1/models")


@patch("inference_perf.client.modelserver.openai_client.CustomTokenizer")
@patch("inference_perf.client.modelserver.openai_client.requests.get")
def test_openai_client_init_resolve_model_empty(mock_get: MagicMock, mock_custom_tokenizer: MagicMock) -> None:
    metrics_collector = MagicMock()
    api_config = APIConfig(type=APIType.CHAT)

    # Mock empty response from /v1/models
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": []}
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    with pytest.raises(Exception) as excinfo:
        DummyOpenAIClient(
            metrics_collector=metrics_collector,
            api_config=api_config,
            uri="http://test-uri",
            model_name=None,
            tokenizer_config=None,
            max_tcp_connections=100,
            additional_filters=[],
        )
    assert "no model_name could be found" in str(excinfo.value)


# NEW TESTS FOR COVERAGE GAPS


@patch("inference_perf.client.modelserver.openai_client.CustomTokenizer")
@patch("inference_perf.client.modelserver.openai_client.requests.get")
def test_openai_client_init_resolve_model_multiple(mock_get: MagicMock, mock_custom_tokenizer: MagicMock) -> None:
    metrics_collector = MagicMock()
    api_config = APIConfig(type=APIType.CHAT)

    # Mock response with 2 models
    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": [{"id": "model1"}, {"id": "model2"}]}
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    with patch("inference_perf.client.modelserver.openai_client.logger.warning") as mock_warning:
        client = DummyOpenAIClient(
            metrics_collector=metrics_collector,
            api_config=api_config,
            uri="http://test-uri",
            model_name=None,
            tokenizer_config=None,
            max_tcp_connections=100,
            additional_filters=[],
        )
        assert client.model_name == "model1"
        mock_warning.assert_called_once()


@patch("inference_perf.client.modelserver.openai_client.CustomTokenizer")
@patch("inference_perf.client.modelserver.openai_client.requests.get")
def test_openai_client_init_with_lora_success(mock_get: MagicMock, mock_custom_tokenizer: MagicMock) -> None:
    metrics_collector = MagicMock()
    api_config = APIConfig(type=APIType.CHAT)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": [{"id": "base-model"}, {"id": "lora-1"}]}
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    lora_item = MagicMock()
    lora_item.name = "lora-1"

    client = DummyOpenAIClient(
        metrics_collector=metrics_collector,
        api_config=api_config,
        uri="http://test-uri",
        model_name="base-model",
        tokenizer_config=None,
        max_tcp_connections=100,
        additional_filters=[],
        lora_config=[lora_item],
    )
    assert client.model_name == "base-model"


@patch("inference_perf.client.modelserver.openai_client.CustomTokenizer")
@patch("inference_perf.client.modelserver.openai_client.requests.get")
def test_openai_client_init_with_lora_failure(mock_get: MagicMock, mock_custom_tokenizer: MagicMock) -> None:
    metrics_collector = MagicMock()
    api_config = APIConfig(type=APIType.CHAT)

    mock_resp = MagicMock()
    mock_resp.json.return_value = {"data": [{"id": "base-model"}]}
    mock_resp.raise_for_status = MagicMock()
    mock_get.return_value = mock_resp

    lora_item = MagicMock()
    lora_item.name = "lora-missing"

    with pytest.raises(ValueError) as excinfo:
        DummyOpenAIClient(
            metrics_collector=metrics_collector,
            api_config=api_config,
            uri="http://test-uri",
            model_name="base-model",
            tokenizer_config=None,
            max_tcp_connections=100,
            additional_filters=[],
            lora_config=[lora_item],
        )
    assert "not found in model server's available models" in str(excinfo.value)


@pytest.mark.asyncio
async def test_client_process_request(mock_data: MagicMock) -> None:
    metrics_collector = MagicMock()
    api_config = APIConfig(type=APIType.CHAT)

    with patch("inference_perf.client.modelserver.openai_client.CustomTokenizer"):
        client = DummyOpenAIClient(
            metrics_collector=metrics_collector,
            api_config=api_config,
            uri="http://test-uri",
            model_name="test-model",
            tokenizer_config=None,
            max_tcp_connections=100,
            additional_filters=[],
        )

        # Mock the session
        mock_session = AsyncMock()
        client._session = mock_session

        await client.process_request(mock_data, stage_id=1, scheduled_time=0.0)

        mock_session.process_request.assert_called_once_with(mock_data, 1, 0.0, None)


@pytest.mark.asyncio
async def test_client_close() -> None:
    metrics_collector = MagicMock()
    api_config = APIConfig(type=APIType.CHAT)

    with patch("inference_perf.client.modelserver.openai_client.CustomTokenizer"):
        client = DummyOpenAIClient(
            metrics_collector=metrics_collector,
            api_config=api_config,
            uri="http://test-uri",
            model_name="test-model",
            tokenizer_config=None,
            max_tcp_connections=100,
            additional_filters=[],
        )

        mock_session = AsyncMock()
        client._session = mock_session

        await client.close()

        mock_session.close.assert_called_once()
        assert client._session is None


@patch("inference_perf.client.modelserver.openai_client.requests.get")
def test_get_supported_models_exception(mock_get: MagicMock) -> None:
    metrics_collector = MagicMock()
    api_config = APIConfig(type=APIType.CHAT)

    mock_get.side_effect = Exception("Network error")

    with patch("inference_perf.client.modelserver.openai_client.CustomTokenizer"):
        client = DummyOpenAIClient(
            metrics_collector=metrics_collector,
            api_config=api_config,
            uri="http://test-uri",
            model_name="test-model",
            tokenizer_config=None,
            max_tcp_connections=100,
            additional_filters=[],
        )

        models = client.get_supported_models()
        assert models == []


# Tests for openAIModelServerClientSession.process_request


@pytest.mark.asyncio
async def test_process_request_success(mock_client: MagicMock, mock_data: MagicMock) -> None:
    session = openAIModelServerClientSession(mock_client)
    session.session = MagicMock()

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.text = AsyncMock(return_value="{}")

    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
    session.session.post.return_value = mock_post_ctx

    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)

    mock_data.process_response.assert_called_once()


@pytest.mark.asyncio
async def test_process_request_payload_construction(mock_client: MagicMock, mock_data: MagicMock) -> None:
    session = openAIModelServerClientSession(mock_client)
    session.session = MagicMock()

    mock_client.api_config.headers = {"X-Custom-Header": "value"}
    mock_client.api_config.response_format = MagicMock()
    mock_client.api_config.response_format.type = ResponseFormatType.JSON_OBJECT

    with patch("inference_perf.client.modelserver.openai_client.to_api_format", return_value={"type": "json_object"}):
        mock_resp = MagicMock()
        mock_resp.status = 200
        mock_resp.text = AsyncMock(return_value="{}")

        mock_post_ctx = MagicMock()
        mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
        mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
        session.session.post.return_value = mock_post_ctx

        await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)

        session.session.post.assert_called_once()
        args, kwargs = session.session.post.call_args

        headers = kwargs["headers"]
        assert headers["X-Custom-Header"] == "value"
        assert headers["Content-Type"] == "application/json"

        sent_data = json.loads(kwargs["data"])
        assert sent_data["response_format"] == {"type": "json_object"}


@pytest.mark.asyncio
async def test_session_init_with_ssl(mock_client: MagicMock) -> None:
    mock_client.cert_path = "/path/to/cert"
    mock_client.key_path = "/path/to/key"

    with patch("ssl.create_default_context") as mock_create_context:
        # Use spec=ssl.SSLContext to satisfy isinstance checks in aiohttp
        mock_context = MagicMock(spec=ssl.SSLContext)
        mock_create_context.return_value = mock_context

        session = openAIModelServerClientSession(mock_client)

        mock_create_context.assert_called_once_with(ssl.Purpose.SERVER_AUTH)
        mock_context.load_cert_chain.assert_called_once_with(certfile="/path/to/cert", keyfile="/path/to/key")
        assert session.session is not None
        await session.close()


@pytest.mark.asyncio
async def test_process_request_with_slo_headers(mock_client: MagicMock, mock_data: MagicMock) -> None:
    session = openAIModelServerClientSession(mock_client)
    session.session = MagicMock()

    mock_resp = MagicMock()
    mock_resp.status = 200
    mock_resp.text = AsyncMock(return_value="{}")

    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_resp)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
    session.session.post.return_value = mock_post_ctx

    # Use real InferenceInfo with truthy output_token_times
    response_info = InferenceInfo(output_token_times=[1.0])
    mock_data.process_response.return_value = response_info

    # Set up client config with headers
    mock_client.api_config.headers = {"x-slo-ttft-ms": "100", "x-slo-tpot-ms": "10"}
    mock_client.api_config.slo_unit = "ms"
    # Explicitly set header attributes to None to avoid MagicMock returning new mocks
    mock_client.api_config.slo_ttft_header = None
    mock_client.api_config.slo_tpot_header = None

    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)

    # Verify that the metric passed to record_metric had the slo fields set
    mock_client.metrics_collector.record_metric.assert_called_once()
    metric = mock_client.metrics_collector.record_metric.call_args[0][0]

    assert metric.ttft_slo_sec == 0.1  # 100 * 0.001
    assert metric.tpot_slo_sec == 0.01  # 10 * 0.001
