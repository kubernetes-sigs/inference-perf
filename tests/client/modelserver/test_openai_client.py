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
from inference_perf.client.modelserver.openai_client import openAIModelServerClientSession, OpenAIMetrics
from inference_perf.client.modelserver.base import Metric
from inference_perf.apis import ErrorResponseInfo, InferenceInfo


@pytest.fixture
def mock_client() -> MagicMock:
    client = MagicMock()
    client.uri = "http://test-uri"
    client.api_config = MagicMock()
    client.api_config.headers = {}
    client.api_config.response_format = None
    client.api_config.streaming = False
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


def test_openai_metrics_get_all_metrics() -> None:
    """Test get_all_metrics deduplicates and collects from lists."""

    class FakeMetric(Metric):
        def __init__(self, name: str) -> None:
            self.name = name

        def get_queries(self, duration: float) -> list[tuple[str, str]]:
            return []

    m1 = FakeMetric("m1")
    m2 = FakeMetric("m2")
    m3 = FakeMetric("m3")
    m4 = FakeMetric("m4")

    metrics = OpenAIMetrics(
        prompt_tokens=m1,
        output_tokens=m2,
        requests=m3,
        request_latency=m1,  # Duplicate
        queue_length=m2,  # Duplicate
        time_per_output_token=m3,  # Duplicate
        custom_metrics=[m1, m4],
    )

    all_metrics = metrics.get_all_metrics()

    # Should contain m1, m2, m3, m4 (deduplicated)
    assert len(all_metrics) == 4
    assert m1 in all_metrics
    assert m2 in all_metrics
    assert m3 in all_metrics
    assert m4 in all_metrics


@pytest.mark.asyncio
async def test_process_request_success(mock_client: MagicMock, mock_data: MagicMock) -> None:
    """Test process_request with HTTP 200 success."""
    session = openAIModelServerClientSession(mock_client)
    session.session = MagicMock()

    # Mock the response
    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.text = AsyncMock(return_value="success_response_text")

    # Mock data.process_response
    expected_info = InferenceInfo()
    mock_data.process_response.return_value = expected_info

    # Mock the post context
    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
    session.session.post.return_value = mock_post_ctx

    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)

    # Verify process_response was called
    mock_data.process_response.assert_called_once_with(
        response=mock_response,
        config=mock_client.api_config,
        tokenizer=mock_client.tokenizer,
        lora_adapter=None,
    )

    # Verify metric was recorded
    mock_client.metrics_collector.record_metric.assert_called_once()
    metric = mock_client.metrics_collector.record_metric.call_args[0][0]
    assert metric.info == expected_info
    assert metric.response_data == "success_response_text"
    assert metric.error is None
