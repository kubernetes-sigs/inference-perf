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
import aiohttp
from unittest.mock import AsyncMock, MagicMock
from inference_perf.client.modelserver.openai_client import openAIModelServerClientSession, ErrorResponseInfo, InferenceInfo


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
