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
import json
import logging
import random
import pytest
import asyncio
import aiohttp
from typing import cast
from unittest.mock import AsyncMock, MagicMock
from inference_perf.client.modelserver.openai_client import openAIModelServerClientSession, OpenAIMetrics
from inference_perf.client.modelserver.metrics import Metric, CounterResult
from inference_perf.apis import AnthropicMessagesAPIData, ChatMessage, ErrorResponseInfo, InferenceInfo
from inference_perf.apis.anthropic_messages import ANTHROPIC_VERSION
from inference_perf.config import APIType
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
    # Real ints, not MagicMocks: the retry loop does arithmetic on these.
    # Defaults mirror LoadConfig, i.e. retries off.
    client.request_retries = 0
    client.request_retry_backoff_sec = 0.5
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
async def test_otel_assembles_streaming_tool_call_deltas(mock_client: MagicMock, mock_data: MagicMock) -> None:
    """Streaming tool_call deltas are merged by index into one assistant message.

    A streaming tool call arrives as many `function.arguments` fragments. Recording the
    raw deltas would make gen_ai.output.message a list of fragments -- a different JSON
    shape than the non-streaming branch, which records the server-assembled message dict.
    The fragments below are shaped like real vLLM 0.27.1 output: two parallel tool calls,
    each with its `arguments` split across chunks, and the id/name only on the first chunk.
    """
    from contextlib import contextmanager

    mock_client.api_config.streaming = True
    mock_client.api_config.type = APIType.Chat
    mock_client.api_config.response_format = None
    mock_client.api_key = None
    mock_client.model_name = "test-model"
    mock_client.max_completion_tokens = 30
    mock_client.ignore_eos = True

    mock_span = MagicMock()
    mock_client.otel.enabled = True

    @contextmanager
    def fake_trace(**kwargs):  # type: ignore[no-untyped-def]
        yield mock_span

    mock_client.otel.trace_llm_request = fake_trace

    def chunk(tool_calls: list[dict[str, object]]) -> str:
        body = {"id": "chatcmpl-1", "choices": [{"index": 0, "delta": {"tool_calls": tool_calls}, "finish_reason": None}]}
        return f"data: {json.dumps(body)}\n\n"

    # Two parallel tool calls; id/name arrive only on the first chunk of each, and the
    # arguments JSON is split across chunks exactly as vLLM emits it.
    sse_content = (
        chunk([{"index": 0, "id": "call_a", "type": "function", "function": {"name": "Glob"}}])
        + chunk([{"index": 0, "function": {"arguments": '{"pattern": "'}}])
        + chunk([{"index": 0, "function": {"arguments": "**/*."}}])
        + chunk([{"index": 0, "function": {"arguments": 'sv"}'}}])
        + chunk([{"index": 1, "id": "call_b", "type": "function", "function": {"name": "Bash"}}])
        + chunk([{"index": 1, "function": {"arguments": '{"command": "ls"}'}}])
        + "data: [DONE]\n\n"
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

    mock_response = MagicMock()
    mock_response.status = 200
    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)

    session = openAIModelServerClientSession(mock_client)
    session.session = MagicMock()
    session.session.post.return_value = mock_post_ctx

    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)

    response_info = mock_client.otel.record_response_metrics.call_args[1]["response_info"]
    assembled = json.loads(response_info["output_message"])

    # Same shape as the non-streaming branch: a dict, not a list of raw deltas.
    assert isinstance(assembled, dict)
    assert assembled["role"] == "assistant"
    tool_calls = assembled["tool_calls"]
    assert len(tool_calls) == 2

    # Fragments are concatenated per index, so each arguments string is valid JSON.
    assert tool_calls[0]["id"] == "call_a"
    assert tool_calls[0]["function"]["name"] == "Glob"
    assert json.loads(tool_calls[0]["function"]["arguments"]) == {"pattern": "**/*.sv"}
    assert tool_calls[1]["id"] == "call_b"
    assert tool_calls[1]["function"]["name"] == "Bash"
    assert json.loads(tool_calls[1]["function"]["arguments"]) == {"command": "ls"}


@pytest.mark.asyncio
async def test_anthropic_messages_request_uses_messages_route_and_headers(mock_client: MagicMock) -> None:
    mock_client.api_config.type = APIType.AnthropicMessages
    mock_client.api_config.streaming = False
    mock_client.api_config.session_id_header_key = None
    mock_client.api_key = "test-key"
    mock_client.model_name = "claude-sonnet"
    mock_client.max_completion_tokens = 128
    mock_client.ignore_eos = True

    data = AnthropicMessagesAPIData(messages=[ChatMessage(role="user", content="hello")])
    session = openAIModelServerClientSession(mock_client)
    session.session = MagicMock()

    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError("force exit"))
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
    session.session.post.return_value = mock_post_ctx

    await session.process_request(data, stage_id=1, scheduled_time=0.0)

    assert session.session.post.call_args.args[0] == "http://test-uri/v1/messages"
    headers_passed = session.session.post.call_args.kwargs["headers"]
    assert headers_passed["x-api-key"] == "test-key"
    assert headers_passed["anthropic-version"] == ANTHROPIC_VERSION
    assert "Authorization" not in headers_passed
    assert '"ignore_eos"' not in session.session.post.call_args.kwargs["data"]


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
async def test_user_session_id_used_when_session_id_is_none(mock_client: MagicMock, mock_data: MagicMock) -> None:
    mock_data.session_id = None
    mock_data.user_session_id = "conv_0"
    mock_client.api_config.session_id_header_key = "x-session-id"

    session = openAIModelServerClientSession(mock_client)
    session.session = MagicMock()

    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError("force exit"))
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
    session.session.post.return_value = mock_post_ctx

    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)

    headers_passed = session.session.post.call_args.kwargs["headers"]
    assert headers_passed.get("x-session-id") == "conv_0"


@pytest.mark.asyncio
async def test_session_id_takes_precedence_over_user_session_id(mock_client: MagicMock, mock_data: MagicMock) -> None:
    mock_data.session_id = "trace0_test_session"
    mock_data.user_session_id = "conv_0"
    mock_client.api_config.session_id_header_key = "x-session-id"

    session = openAIModelServerClientSession(mock_client)
    session.session = MagicMock()

    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(side_effect=asyncio.TimeoutError("force exit"))
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
    session.session.post.return_value = mock_post_ctx

    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)

    headers_passed = session.session.post.call_args.kwargs["headers"]
    assert headers_passed.get("x-session-id") == "trace0_test_session"


@pytest.mark.asyncio
async def test_session_id_header_not_injected_when_neither_id_is_set(mock_client: MagicMock, mock_data: MagicMock) -> None:
    mock_data.session_id = None
    mock_data.user_session_id = None
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
    mock_data.user_session_id = "conv_0"
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


def test_openai_metrics_iteration_yields_each_field_once() -> None:
    """Iterating OpenAIMetrics yields (target_field, metric) pairs; named fields take precedence
    over a custom_metrics entry that reuses a named field's key."""

    class FakeMetric(Metric[CounterResult]):
        def __init__(self, name: str) -> None:
            self.metric_name = name

        def get_queries(self, duration: float, filters: str) -> list[str]:
            return []

        def parse(self, results: list[float]) -> CounterResult:
            return CounterResult()

    metrics = OpenAIMetrics(
        filters=[],
        prompt_tokens=FakeMetric("pt"),
        output_tokens=FakeMetric("ot"),
        requests=FakeMetric("req"),
        request_latency=FakeMetric("lat"),
        queue_length=FakeMetric("q"),
        time_per_output_token=FakeMetric("tpot"),
        custom_metrics={
            "kv_cache_usage": FakeMetric("kv"),
            "prompt_tokens": FakeMetric("custom-pt"),  # collides with the named field
        },
    )

    fields = [field for field, _ in metrics]
    by_field = dict(metrics)

    # Each field appears once, named field wins over the colliding custom entry.
    assert len(fields) == len(set(fields))
    assert set(fields) == {
        "prompt_tokens",
        "output_tokens",
        "requests",
        "request_latency",
        "queue_length",
        "time_per_output_token",
        "kv_cache_usage",
    }
    assert by_field["prompt_tokens"].metric_name == "pt"


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


def _session_token_test_setup(
    mock_client: MagicMock, mock_data: MagicMock, response_token: str
) -> tuple[openAIModelServerClientSession, MagicMock]:
    """Configure client/data mocks for session token tests and return a client session
    whose mocked POST responds with the given session token header, along with the
    mocked HTTP session for inspecting sent requests."""
    mock_client.otel.enabled = False
    mock_client.api_config.streaming = False
    mock_client.api_config.session_id_header_key = None
    mock_client.api_config.session_token_header_key = "x-session-token"
    mock_data.session_id = "trace0_session0"
    mock_data.headers = None

    session = openAIModelServerClientSession(mock_client)
    mock_http_session = MagicMock()
    session.session = mock_http_session

    mock_response = MagicMock()
    mock_response.status = 200
    mock_response.text = AsyncMock(return_value="{}")
    mock_response.headers = {"x-session-token": response_token}
    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=mock_response)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
    mock_http_session.post.return_value = mock_post_ctx
    return session, mock_http_session


@pytest.mark.asyncio
async def test_session_token_captured_and_replayed(mock_client: MagicMock, mock_data: MagicMock) -> None:
    session, mock_http_session = _session_token_test_setup(mock_client, mock_data, response_token="encoded-pod-a")

    # First request of the session: no token captured yet, header must be absent.
    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)
    first_headers = mock_http_session.post.call_args.kwargs["headers"]
    assert "x-session-token" not in first_headers

    # Second request of the same session replays the token from the first response.
    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)
    second_headers = mock_http_session.post.call_args.kwargs["headers"]
    assert second_headers["x-session-token"] == "encoded-pod-a"


@pytest.mark.asyncio
async def test_session_token_not_shared_across_sessions(mock_client: MagicMock, mock_data: MagicMock) -> None:
    session, mock_http_session = _session_token_test_setup(mock_client, mock_data, response_token="encoded-pod-a")

    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)

    # A different session must not inherit the first session's token.
    mock_data.session_id = "trace0_session1"
    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)
    headers_passed = mock_http_session.post.call_args.kwargs["headers"]
    assert "x-session-token" not in headers_passed


@pytest.mark.asyncio
async def test_session_token_not_captured_when_header_key_is_none(mock_client: MagicMock, mock_data: MagicMock) -> None:
    session, mock_http_session = _session_token_test_setup(mock_client, mock_data, response_token="encoded-pod-a")
    mock_client.api_config.session_token_header_key = None

    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)
    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)

    headers_passed = mock_http_session.post.call_args.kwargs["headers"]
    assert "x-session-token" not in headers_passed


@pytest.mark.asyncio
async def test_session_token_replayed_for_user_session_id_workloads(mock_client: MagicMock, mock_data: MagicMock) -> None:
    """conversation_replay and shared_prefix carry session identity as user_session_id
    rather than the loadgen-stamped session_id used by trace replay."""
    session, mock_http_session = _session_token_test_setup(mock_client, mock_data, response_token="encoded-pod-a")
    mock_data.session_id = None
    mock_data.user_session_id = "conv_0"

    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)
    assert "x-session-token" not in mock_http_session.post.call_args.kwargs["headers"]

    await session.process_request(mock_data, stage_id=1, scheduled_time=0.0)
    assert mock_http_session.post.call_args.kwargs["headers"]["x-session-token"] == "encoded-pod-a"


# --- Bounded retry for transport faults raised before response headers (#777) ---
#
# A replay session is a graph: one dropped connection fails the whole session and
# cancels every event downstream of it. These tests pin the boundary that makes the
# retry measurement-safe -- only faults raised before a response was established are
# retried, because anything later has already produced a TTFT. Note the boundary is the
# client's view: no TTFT exists to contaminate, but the server may already have received
# the failed attempt, which is why the retry is bounded and backed off.


def _retrying_session(mock_client: MagicMock, retries: int, backoff: float = 0.0) -> openAIModelServerClientSession:
    """A session whose client is configured for `retries` extra attempts."""
    mock_client.request_retries = retries
    mock_client.request_retry_backoff_sec = backoff
    session = openAIModelServerClientSession(mock_client)
    session.session = MagicMock()
    return session


def _post(session: openAIModelServerClientSession) -> MagicMock:
    """The mocked ``post`` on a session built by ``_retrying_session``.

    ``session.session`` is typed as a real ``ClientSession``, so reach the mock's
    ``side_effect``/``call_count`` through a cast rather than off the typed attribute.
    """
    return cast(MagicMock, cast(MagicMock, session.session).post)


def _failing_ctx(exc: BaseException) -> MagicMock:
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(side_effect=exc)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


def _ok_ctx() -> MagicMock:
    response = MagicMock()
    response.status = 200
    response.json = AsyncMock(return_value={"choices": [{"text": "ok"}]})
    response.text = AsyncMock(return_value='{"choices": [{"text": "ok"}]}')
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=response)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


def _error_ctx(status: int = 500) -> MagicMock:
    """A response that completes without raising but carries a failing HTTP status."""
    response = MagicMock()
    response.status = status
    response.json = AsyncMock(return_value={"error": "boom"})
    response.text = AsyncMock(return_value='{"error": "boom"}')
    ctx = MagicMock()
    ctx.__aenter__ = AsyncMock(return_value=response)
    ctx.__aexit__ = AsyncMock(return_value=None)
    return ctx


@pytest.mark.asyncio
async def test_retry_recovers_from_server_disconnect(mock_client: MagicMock, mock_data: MagicMock) -> None:
    """A dropped connection is re-POSTed and the retry's success is what gets reported."""
    session = _retrying_session(mock_client, retries=2)
    # side_effect, not return_value: each attempt needs its own context manager.
    _post(session).side_effect = [_failing_ctx(aiohttp.ServerDisconnectedError()), _ok_ctx()]

    await session.process_request(mock_data, stage_id=0, scheduled_time=0.0)

    assert _post(session).call_count == 2
    # Exactly one lifecycle metric per logical request, however many attempts it took.
    mock_client.metrics_collector.record_metric.assert_called_once()
    metric = mock_client.metrics_collector.record_metric.call_args[0][0]
    assert metric.error is None
    assert metric.info.retries_attempted == 1
    assert metric.info.retries_recovered is True


@pytest.mark.asyncio
async def test_retries_exhausted_reports_the_transport_error(mock_client: MagicMock, mock_data: MagicMock) -> None:
    """When every attempt drops, the request fails exactly as it does today -- but still
    reports the attempts it burned, so an exhausted retry is not invisible."""
    session = _retrying_session(mock_client, retries=2)
    _post(session).side_effect = [_failing_ctx(aiohttp.ServerDisconnectedError()) for _ in range(3)]

    await session.process_request(mock_data, stage_id=0, scheduled_time=0.0)

    assert _post(session).call_count == 3  # 1 + request_retries
    metric = mock_client.metrics_collector.record_metric.call_args[0][0]
    assert metric.error is not None
    assert metric.error.error_type == "ServerDisconnectedError"
    assert metric.info.retries_attempted == 2
    assert metric.info.retries_recovered is False


@pytest.mark.asyncio
async def test_no_retry_by_default(mock_client: MagicMock, mock_data: MagicMock) -> None:
    """request_retries defaults to 0, so existing runs' numbers are unchanged."""
    session = _retrying_session(mock_client, retries=0)
    _post(session).side_effect = [_failing_ctx(aiohttp.ServerDisconnectedError()), _ok_ctx()]

    await session.process_request(mock_data, stage_id=0, scheduled_time=0.0)

    assert _post(session).call_count == 1
    metric = mock_client.metrics_collector.record_metric.call_args[0][0]
    assert metric.error is not None
    assert metric.info.retries_attempted == 0


@pytest.mark.asyncio
async def test_timeout_is_never_retried(mock_client: MagicMock, mock_data: MagicMock) -> None:
    """ServerTimeoutError inherits from BOTH ClientConnectionError and
    asyncio.TimeoutError. Retrying it would multiply request_timeout by the attempt
    count -- with request_timeout: 900 that turns one stalled event into 45 minutes."""
    session = _retrying_session(mock_client, retries=2)
    _post(session).side_effect = [_failing_ctx(aiohttp.ServerTimeoutError()), _ok_ctx()]

    await session.process_request(mock_data, stage_id=0, scheduled_time=0.0)

    assert _post(session).call_count == 1
    metric = mock_client.metrics_collector.record_metric.call_args[0][0]
    assert metric.error is not None
    assert metric.info.retries_attempted == 0


@pytest.mark.asyncio
async def test_client_os_error_is_retried(mock_client: MagicMock, mock_data: MagicMock) -> None:
    """ClientOSError (a connection reset from the pool) was the second-largest error
    class in the reference run, so it must fall inside the predicate."""
    session = _retrying_session(mock_client, retries=1)
    _post(session).side_effect = [_failing_ctx(aiohttp.ClientOSError(104, "Connection reset by peer")), _ok_ctx()]

    await session.process_request(mock_data, stage_id=0, scheduled_time=0.0)

    assert _post(session).call_count == 2
    metric = mock_client.metrics_collector.record_metric.call_args[0][0]
    assert metric.error is None
    assert metric.info.retries_recovered is True


@pytest.mark.asyncio
async def test_reported_latency_includes_failed_attempts_and_backoff(
    mock_client: MagicMock, mock_data: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """start_time must NOT be re-stamped per attempt.

    `start_time` is the logical request's dispatch time, not a latency origin. Re-stamping
    it on each attempt would make a retried request report only its final attempt, hiding
    time the workload genuinely spent waiting -- and, worse, it would charge the retry's
    backoff to *scheduling* instead (see the schedule-delay test below). The serving-side
    view is reported separately as `info.retry_wasted_sec`.
    """
    session = _retrying_session(mock_client, retries=1, backoff=10.0)
    _post(session).side_effect = [_failing_ctx(aiohttp.ServerDisconnectedError()), _ok_ctx()]

    # Attempt 1 spans 0->5s, backoff runs to 100s, attempt 2 spans 100->101s.
    ticks = iter([0.0, 5.0, 100.0, 101.0, 101.0, 101.0])
    monkeypatch.setattr("inference_perf.client.modelserver.openai_client.time.perf_counter", lambda: next(ticks))
    # Don't actually sleep out the backoff.
    monkeypatch.setattr("inference_perf.client.modelserver.openai_client.sleep", AsyncMock())

    await session.process_request(mock_data, stage_id=0, scheduled_time=0.0)

    metric = mock_client.metrics_collector.record_metric.call_args[0][0]
    # The whole 101s: dispatch -> failed attempt -> backoff -> answering attempt.
    assert metric.start_time == pytest.approx(0.0)
    assert metric.end_time - metric.start_time == pytest.approx(101.0)
    # ...of which 100s was waste: the 5s failed attempt plus the 95s of backoff after it.
    # Reported alongside the full latency above rather than carved out of it.
    assert metric.info.retry_wasted_sec == pytest.approx(100.0)


@pytest.mark.asyncio
async def test_retry_backoff_is_not_charged_to_schedule_delay(
    mock_client: MagicMock, mock_data: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A retry must not inflate `schedule_delay`.

    The report derives schedule_delay (`start_time - scheduled_time`), send_duration and
    achieved_rate from `start_time`, so re-stamping it per attempt would report a request
    dispatched on time as having waited out its own retry backoff in the queue -- an easy
    regression to miss, because every latency assertion still passes.
    """
    session = _retrying_session(mock_client, retries=1, backoff=10.0)
    _post(session).side_effect = [_failing_ctx(aiohttp.ServerDisconnectedError()), _ok_ctx()]

    ticks = iter([0.0, 5.0, 100.0, 101.0, 101.0, 101.0])
    monkeypatch.setattr("inference_perf.client.modelserver.openai_client.time.perf_counter", lambda: next(ticks))
    monkeypatch.setattr("inference_perf.client.modelserver.openai_client.sleep", AsyncMock())

    # Dispatched exactly when it was scheduled; the 100s of failure + backoff that follow
    # are the request's latency, not queue delay.
    await session.process_request(mock_data, stage_id=0, scheduled_time=0.0)

    metric = mock_client.metrics_collector.record_metric.call_args[0][0]
    assert metric.start_time - metric.scheduled_time == pytest.approx(0.0)


@pytest.mark.asyncio
async def test_retry_wasted_sec_absent_without_retry(
    mock_client: MagicMock, mock_data: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A request that never retried wasted nothing, and serializes without the key -- so a
    default-config run's per-request JSON is unchanged."""
    session = _retrying_session(mock_client, retries=2)
    _post(session).side_effect = [_ok_ctx()]

    await session.process_request(mock_data, stage_id=0, scheduled_time=0.0)

    metric = mock_client.metrics_collector.record_metric.call_args[0][0]
    assert metric.info.retry_wasted_sec is None
    # Serialized shape checked on a clean InferenceInfo: the fixture's info carries
    # MagicMock labels/graph_event_id, which make model_dump warn about unrelated fields.
    assert "retry_wasted_sec" not in InferenceInfo(request_metrics=RequestMetrics(text=Text())).model_dump()


@pytest.mark.asyncio
async def test_retry_reduces_failures_without_inflating_request_count(
    mock_client: MagicMock, mock_data: MagicMock, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Statistical guard: at a fixed pre-header fault rate, raising request_retries
    must drive failures down while the number of reported requests stays constant.

    The per-request tests above pin the mechanics of one retry. This one pins the two
    properties a run's numbers depend on, which no single-request test can observe:

    1. Failures fall roughly geometrically -- each extra attempt faults independently.
    2. Exactly one RequestLifecycleMetric is recorded per logical request in every arm.
       If retries ever recorded a metric per attempt, throughput and every latency
       percentile would be silently wrong, and no assertion about a single request
       would catch it.

    Deliberately loose bounds: the fault sequence is seeded so the arms are comparable,
    but this asserts the *shape* of the improvement, not exact counts, so the test does
    not become a tripwire on an unrelated change to attempt ordering.
    """
    monkeypatch.setattr("inference_perf.client.modelserver.openai_client.sleep", AsyncMock())
    n_requests = 300
    fault_rate = 0.2

    async def run_arm(retries: int) -> tuple[int, int, int]:
        session = _retrying_session(mock_client, retries=retries)
        mock_client.metrics_collector.record_metric.reset_mock()
        rng = random.Random(20260907)  # same fault sequence in every arm

        def post(*args: object, **kwargs: object) -> MagicMock:
            if rng.random() < fault_rate:
                return _failing_ctx(aiohttp.ServerDisconnectedError())
            return _ok_ctx()

        _post(session).side_effect = post
        # A fresh InferenceInfo per call: the shared `mock_data` fixture returns one
        # instance via return_value, so every request would otherwise write
        # retries_recovered onto the same object and only the last write would survive.
        mock_data.process_response.side_effect = lambda *a, **k: InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=0))
        )
        mock_data.process_failure.side_effect = lambda *a, **k: InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=0))
        )
        for _ in range(n_requests):
            await session.process_request(mock_data, stage_id=0, scheduled_time=0.0)

        metrics = [call[0][0] for call in mock_client.metrics_collector.record_metric.call_args_list]
        failed = sum(1 for m in metrics if m.error is not None)
        recovered = sum(1 for m in metrics if m.info is not None and m.info.retries_recovered)
        return len(metrics), failed, recovered

    recorded_0, failed_0, recovered_0 = await run_arm(0)
    recorded_2, failed_2, recovered_2 = await run_arm(2)

    # One metric per logical request, no matter how many network attempts it took.
    assert recorded_0 == n_requests
    assert recorded_2 == n_requests

    # Retries off: nothing is recovered and the fault rate shows up as failures.
    assert recovered_0 == 0
    assert failed_0 > 0

    # Two extra attempts cut a 20% fault rate to roughly 20%^3, so most failures go away.
    assert failed_2 < failed_0 / 4
    assert recovered_2 > 0


@pytest.mark.asyncio
async def test_recovered_retry_does_not_log_an_error(
    mock_client: MagicMock, mock_data: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """A fault that is about to be retried logs a WARNING, not an ERROR with a traceback.

    The except blocks run before the retry decision is known, so logging there would
    stamp ERROR + a full stack trace on every fault the mechanism then silently fixed --
    making a working retry look like a failing run to anyone reading logs or alerting
    on ERROR. The log is deferred until we know we are giving up.
    """
    session = _retrying_session(mock_client, retries=2)
    _post(session).side_effect = [_failing_ctx(aiohttp.ServerDisconnectedError()), _ok_ctx()]

    with caplog.at_level(logging.DEBUG, logger="inference_perf.client.modelserver.openai_client"):
        await session.process_request(mock_data, stage_id=0, scheduled_time=0.0)

    client_logger = "inference_perf.client.modelserver.openai_client"
    errors = [r for r in caplog.records if r.levelno == logging.ERROR and r.name == client_logger]
    assert errors == []
    assert any("Retrying request after ServerDisconnectedError" in r.message for r in caplog.records)


@pytest.mark.asyncio
async def test_exhausted_retry_still_logs_an_error(
    mock_client: MagicMock, mock_data: MagicMock, caplog: pytest.LogCaptureFixture
) -> None:
    """Deferring the log must not swallow it: a request that really fails still logs
    ERROR with the exception attached, exactly as it did before retries existed."""
    session = _retrying_session(mock_client, retries=1)
    _post(session).side_effect = [_failing_ctx(aiohttp.ServerDisconnectedError()) for _ in range(2)]

    with caplog.at_level(logging.DEBUG, logger="inference_perf.client.modelserver.openai_client"):
        await session.process_request(mock_data, stage_id=0, scheduled_time=0.0)

    client_logger = "inference_perf.client.modelserver.openai_client"
    errors = [r for r in caplog.records if r.levelno == logging.ERROR and r.name == client_logger]
    assert len(errors) == 1
    assert errors[0].exc_info is not None  # the traceback is still attached


@pytest.mark.asyncio
async def test_retry_ending_in_http_error_is_not_recovered(mock_client: MagicMock, mock_data: MagicMock) -> None:
    """A retry whose final attempt returns a failing status has not recovered anything.

    The transport fault is gone, but the request still failed, so counting it as
    recovered would overstate how many requests the retry actually rescued.
    """
    session = _retrying_session(mock_client, retries=2)
    _post(session).side_effect = [_failing_ctx(aiohttp.ServerDisconnectedError()), _error_ctx(500)]

    await session.process_request(mock_data, stage_id=0, scheduled_time=0.0)

    assert _post(session).call_count == 2
    metric = mock_client.metrics_collector.record_metric.call_args.args[0]
    assert metric.error is not None
    assert metric.info.retries_attempted == 1
    assert metric.info.retries_recovered is False
