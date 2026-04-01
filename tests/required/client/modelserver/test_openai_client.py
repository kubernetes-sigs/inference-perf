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
from inference_perf.client.modelserver.base import Metric, CounterResult
from inference_perf.apis import ErrorResponseInfo, InferenceInfo
from inference_perf.payloads import RequestMetrics, Text


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
    data.process_failure = AsyncMock(return_value=InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=0))))
    data.process_response = AsyncMock(return_value=InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=0))))
    data.to_request_body = AsyncMock(return_value={"mock": "data"})
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


@pytest.mark.asyncio
async def test_otel_records_output_from_sse_response(mock_client: MagicMock, mock_data: MagicMock) -> None:
    """OTEL metrics should correctly parse SSE streaming response content."""
    from contextlib import contextmanager
    from inference_perf.config import APIType

    mock_client.api_config.streaming = True
    mock_client.api_config.type = APIType.Chat
    mock_client.api_config.response_format = None
    mock_client.api_key = None
    mock_client.model_name = "test-model"
    mock_client.max_completion_tokens = 30
    mock_client.ignore_eos = True

    # Enable OTEL with a mock span
    mock_span = MagicMock()
    mock_client.otel.enabled = True

    @contextmanager
    def fake_trace(**kwargs):  # type: ignore[no-untyped-def]
        yield mock_span

    mock_client.otel.trace_llm_request = fake_trace

    # Real SSE response content from vLLM (last chunk has both content and finish_reason)
    sse_content = (
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1779111798,"model":"test-model","choices":[{"index":0,"delta":{"role":"assistant","content":""},"logprobs":null,"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1779111798,"model":"test-model","choices":[{"index":0,"delta":{"content":"Hello"},"logprobs":null,"finish_reason":null}]}\n\n'
        'data: {"id":"chatcmpl-1","object":"chat.completion.chunk","created":1779111798,"model":"test-model","choices":[{"index":0,"delta":{"content":" world"},"logprobs":null,"finish_reason":"length"}]}\n\n'
        "data: [DONE]\n\n"
    )

    mock_data.session_id = None
    mock_data.otel_context = None
    mock_data.messages = None
    mock_data.prompt = None
    mock_data.process_response = AsyncMock(
        return_value=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=5)),
            extra_info={"raw_response": sse_content},
        )
    )

    # Mock the HTTP response
    mock_response = MagicMock()
    mock_response.status = 200
    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)

    session = openAIModelServerClientSession(mock_client)
    session.session = MagicMock()
    session.session.post.return_value = mock_post_ctx

    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)

    # Verify OTEL recorded the full concatenated output from all SSE chunks
    mock_client.otel.record_response_metrics.assert_called_once()
    call_kwargs = mock_client.otel.record_response_metrics.call_args[1]
    response_info = call_kwargs["response_info"]
    assert response_info["output_text"] == "Hello world"


@pytest.mark.asyncio
async def test_session_id_header_injected_when_both_set(mock_client: MagicMock, mock_data: MagicMock) -> None:
    mock_data.session_id = "trace0_test_session"
    mock_client.api_config.session_id_header_key = "x-session-id"

    session = openAIModelServerClientSession(mock_client)
    session.session = MagicMock()

    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError("force exit"))
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
    session.session.post.return_value = mock_post_ctx

    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)

    headers_passed = session.session.post.call_args.kwargs["headers"]
    assert "x-session-id" in headers_passed
    assert headers_passed["x-session-id"] == "trace0_test_session"


@pytest.mark.asyncio
async def test_session_id_header_not_injected_when_session_id_is_none(mock_client: MagicMock, mock_data: MagicMock) -> None:
    mock_data.session_id = None
    mock_client.api_config.session_id_header_key = "x-session-id"

    session = openAIModelServerClientSession(mock_client)
    session.session = MagicMock()

    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError("force exit"))
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
    session.session.post.return_value = mock_post_ctx

    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)

    headers_passed = session.session.post.call_args.kwargs["headers"]
    assert "x-session-id" not in headers_passed


@pytest.mark.asyncio
async def test_session_id_header_not_injected_when_header_key_is_none(mock_client: MagicMock, mock_data: MagicMock) -> None:
    mock_data.session_id = "trace0_test_session"
    mock_client.api_config.session_id_header_key = None

    session = openAIModelServerClientSession(mock_client)
    session.session = MagicMock()

    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError("force exit"))
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
    session.session.post.return_value = mock_post_ctx

    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)

    headers_passed = session.session.post.call_args.kwargs["headers"]
    assert "x-session-id" not in headers_passed


def test_openai_metrics_get_all_metrics() -> None:
    """Test get_all_metrics deduplicates and collects from lists."""

    class FakeMetric(Metric[CounterResult]):
        def __init__(self, name: str) -> None:
            self.name = name
            self.metric_name = name
            self.target_field = name

        def get_queries(self, duration: float) -> list[str]:
            return []

        def parse(self, results: list[float]) -> CounterResult:
            return CounterResult()

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
    expected_info = InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=0)))
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
