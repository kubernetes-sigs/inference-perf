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
"""Shared test suite for the OpenAI-compatible client subclasses (#517).

Each test_<backend>_client.py module in this directory carries a transcription
of that server's wire format in a ``Backend`` and runs this suite against it
(vLLM's stop_reason/prompt_logprobs, SGLang's matched_stop, TGI's
system_fingerprint and flat error body). Parsing must produce identical
metrics across backends for the same text, so a backend-specific parsing or
serialization regression fails here in plain CI instead of only surfacing
against a live server. No GPU or network required.
"""

import json
from dataclasses import dataclass
from typing import Any, AsyncGenerator, ClassVar, Dict, List, Optional, Type, cast
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from aiohttp import ClientResponse

from inference_perf.apis import ChatCompletionAPIData, ChatMessage, CompletionAPIData, StreamedResponseMetrics
from inference_perf.client.modelserver.openai_client import openAIModelServerClient, openAIModelServerClientSession
from inference_perf.config import APIConfig, APIType
from inference_perf.utils.custom_tokenizer import CustomTokenizer

MODEL = "meta-llama/Llama-3.1-8B-Instruct"
BASE_URI = "http://localhost:8000"
COMPLETION_PROMPT = "The capital of France is"  # 5 whitespace tokens
CHAT_PROMPT = "What is the capital of France?"  # 6 whitespace tokens
# All backends' payloads wrap the same generated text, so parsed token counts
# must agree across backends even though the envelopes differ.
COMPLETION_TEXT = " Paris, the capital of France."  # 5 whitespace tokens
CHAT_TEXT = "The capital of France is Paris."  # 6 whitespace tokens


def sse_event(chunk: Dict[str, Any]) -> str:
    return f"data: {json.dumps(chunk, separators=(',', ':'))}\n\n"


def sse_stream(chunks: List[Dict[str, Any]]) -> str:
    return "".join(sse_event(chunk) for chunk in chunks) + "data: [DONE]\n\n"


@dataclass
class Backend:
    """One backend's wire format plus the client-facing values the suite asserts."""

    client_cls: Type[openAIModelServerClient]
    completion_response: Dict[str, Any]
    chat_response: Dict[str, Any]
    completion_stream: List[Dict[str, Any]]
    chat_stream: List[Dict[str, Any]]
    error_status: int
    error_response: Dict[str, Any]
    models_response: Dict[str, Any]
    expected_metric_filters: List[str]
    queue_metric_name: str


class StubTokenizer:
    """Deterministic whitespace tokenizer so tests need no HF downloads."""

    def __init__(self, config: Any = None) -> None:
        pass

    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        return len(text.split())


def stub_tokenizer() -> CustomTokenizer:
    return cast(CustomTokenizer, StubTokenizer())


class FakeUnaryResponse:
    """Minimal aiohttp ClientResponse stand-in for a fully buffered body."""

    def __init__(self, body: str, status: int = 200) -> None:
        self.status = status
        self.headers = {"Content-Type": "application/json"}
        self._body = body

    async def text(self) -> str:
        return self._body

    async def json(self) -> Any:
        return json.loads(self._body)


class FakeStreamingResponse:
    """Minimal aiohttp ClientResponse stand-in that yields preset SSE bytes.

    When ``interrupt_with`` is set, the exception is raised after the preset
    chunks are exhausted, emulating a connection severed mid-stream.
    """

    def __init__(self, chunks: List[bytes], interrupt_with: Optional[Exception] = None) -> None:
        self.status = 200
        self.headers = {"Content-Type": "text/event-stream"}
        self._chunks = chunks
        self._interrupt_with = interrupt_with
        self.content = self._make_content()

    def _make_content(self) -> MagicMock:
        content = MagicMock()

        async def iter_any() -> AsyncGenerator[bytes, None]:
            for chunk in self._chunks:
                yield chunk
            if self._interrupt_with is not None:
                raise self._interrupt_with

        content.iter_any = iter_any
        return content


def make_client(backend: Backend, api_config: APIConfig, api_key: Optional[str] = None) -> openAIModelServerClient:
    with patch("inference_perf.client.modelserver.openai_client.CustomTokenizer", StubTokenizer):
        return backend.client_cls(
            metrics_collector=MagicMock(),
            api_config=api_config,
            uri=BASE_URI,
            model_name=MODEL,
            tokenizer_config=None,
            max_tcp_connections=4,
            additional_filters=[],
            api_key=api_key,
        )


async def make_session(client: openAIModelServerClient, response: Any) -> openAIModelServerClientSession:
    """Build a client session whose HTTP layer replays a canned response."""
    session = openAIModelServerClientSession(client)
    await session.session.close()
    mock_post_ctx = MagicMock()
    mock_post_ctx.__aenter__ = AsyncMock(return_value=response)
    mock_post_ctx.__aexit__ = AsyncMock(return_value=None)
    session.session = MagicMock()
    session.session.post.return_value = mock_post_ctx
    return session


def recorded_metric(client: openAIModelServerClient) -> Any:
    collector = cast(MagicMock, client.metrics_collector)
    collector.record_metric.assert_called_once()
    return collector.record_metric.call_args[0][0]


class BackendClientSuite:
    """Test methods shared by all backend client subclasses.

    Subclass in a test module as ``class Test<Backend>Client(BackendClientSuite)``
    with ``backend`` set to that module's wire-format transcription.
    """

    backend: ClassVar[Backend]

    # --- Response parsing: success paths ---

    @pytest.mark.asyncio
    async def test_non_streaming_completion_parsing(self) -> None:
        data = CompletionAPIData(prompt=COMPLETION_PROMPT)
        config = APIConfig(type=APIType.Completion, streaming=False)
        response = FakeUnaryResponse(json.dumps(self.backend.completion_response))

        info = await data.process_response(cast(ClientResponse, response), config, stub_tokenizer())

        assert info.request_metrics.text.input_tokens == 5
        assert info.response_metrics is not None
        assert info.response_metrics.output_tokens == 5
        assert data.model_response == COMPLETION_TEXT
        usage = info.response_metrics.server_usage
        assert usage is not None
        assert usage["prompt_tokens"] == 7
        assert usage["completion_tokens"] == 8

    @pytest.mark.asyncio
    async def test_non_streaming_chat_parsing(self) -> None:
        data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content=CHAT_PROMPT)])
        config = APIConfig(type=APIType.Chat, streaming=False)
        response = FakeUnaryResponse(json.dumps(self.backend.chat_response))

        info = await data.process_response(cast(ClientResponse, response), config, stub_tokenizer())

        assert info.request_metrics.text.input_tokens == 6
        assert info.response_metrics is not None
        assert info.response_metrics.output_tokens == 6
        usage = info.response_metrics.server_usage
        assert usage is not None
        assert usage["prompt_tokens"] == 12
        assert usage["completion_tokens"] == 7

    @pytest.mark.asyncio
    async def test_streaming_completion_parsing(self) -> None:
        sse = sse_stream(self.backend.completion_stream)
        data = CompletionAPIData(prompt=COMPLETION_PROMPT)
        config = APIConfig(type=APIType.Completion, streaming=True)
        response = FakeStreamingResponse([sse.encode()])

        info = await data.process_response(cast(ClientResponse, response), config, stub_tokenizer())

        metrics = info.response_metrics
        assert isinstance(metrics, StreamedResponseMetrics)
        # Three content-bearing chunks; the trailing usage chunk and [DONE] must
        # not produce timestamps.
        assert len(metrics.chunk_times) == 3
        assert len(metrics.response_chunks) == 3
        assert metrics.output_tokens == 5
        assert data.model_response == COMPLETION_TEXT
        usage = metrics.server_usage
        assert usage is not None
        assert usage["completion_tokens"] == 8
        # The raw stream is preserved verbatim for the per-request report.
        assert info.extra_info["raw_response"] == sse

    @pytest.mark.asyncio
    async def test_streaming_chat_parsing(self) -> None:
        sse = sse_stream(self.backend.chat_stream)
        data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content=CHAT_PROMPT)])
        config = APIConfig(type=APIType.Chat, streaming=True)
        response = FakeStreamingResponse([sse.encode()])

        info = await data.process_response(cast(ClientResponse, response), config, stub_tokenizer())

        metrics = info.response_metrics
        assert isinstance(metrics, StreamedResponseMetrics)
        # Three content-bearing deltas; the role-only first chunk, the trailing
        # usage chunk, and [DONE] must not produce timestamps.
        assert len(metrics.chunk_times) == 3
        assert len(metrics.response_chunks) == 3
        assert metrics.output_tokens == 6
        usage = metrics.server_usage
        assert usage is not None
        assert usage["completion_tokens"] == 7
        assert info.extra_info["raw_response"] == sse

    # --- Request payload serialization ---

    @pytest.mark.asyncio
    async def test_streaming_completion_request_serialization(self) -> None:
        client = make_client(self.backend, APIConfig(type=APIType.Completion, streaming=True), api_key="test-key")
        response = FakeStreamingResponse([sse_stream(self.backend.completion_stream).encode()])
        session = await make_session(client, response)

        await session.process_request(CompletionAPIData(prompt=COMPLETION_PROMPT), stage_id=0, scheduled_time=0.0)

        call = cast(MagicMock, session.session).post.call_args
        assert call.args[0] == f"{BASE_URI}/v1/completions"
        headers = call.kwargs["headers"]
        assert headers["Content-Type"] == "application/json"
        assert headers["Authorization"] == "Bearer test-key"
        assert json.loads(call.kwargs["data"]) == {
            "model": MODEL,
            "prompt": COMPLETION_PROMPT,
            "max_tokens": 30,
            "ignore_eos": True,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        metric = recorded_metric(client)
        assert metric.error is None
        assert metric.info.response_metrics.output_tokens == 5

    @pytest.mark.asyncio
    async def test_non_streaming_chat_request_serialization(self) -> None:
        client = make_client(self.backend, APIConfig(type=APIType.Chat, streaming=False))
        response = FakeUnaryResponse(json.dumps(self.backend.chat_response))
        session = await make_session(client, response)

        data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content=CHAT_PROMPT)])
        await session.process_request(data, stage_id=0, scheduled_time=0.0)

        call = cast(MagicMock, session.session).post.call_args
        assert call.args[0] == f"{BASE_URI}/v1/chat/completions"
        assert "Authorization" not in call.kwargs["headers"]
        assert json.loads(call.kwargs["data"]) == {
            "model": MODEL,
            "messages": [{"role": "user", "content": CHAT_PROMPT}],
            "max_tokens": 30,
            "ignore_eos": True,
            "stream": False,
        }
        metric = recorded_metric(client)
        assert metric.error is None
        assert metric.info.response_metrics.output_tokens == 6

    # --- Response parsing: error paths ---

    @pytest.mark.asyncio
    async def test_http_error_recorded_with_backend_error_body(self) -> None:
        """Backend-native error bodies (status and shape differ per backend) must be
        preserved verbatim on the recorded metric so failures stay diagnosable."""
        error_body = json.dumps(self.backend.error_response)
        client = make_client(self.backend, APIConfig(type=APIType.Completion, streaming=False))
        session = await make_session(client, FakeUnaryResponse(error_body, status=self.backend.error_status))

        await session.process_request(CompletionAPIData(prompt=COMPLETION_PROMPT), stage_id=0, scheduled_time=0.0)

        metric = recorded_metric(client)
        assert metric.error is not None
        assert metric.error.error_type == f"HTTP Error {self.backend.error_status}"
        assert metric.error.error_msg == error_body
        assert metric.response_data == error_body

    @pytest.mark.asyncio
    async def test_malformed_200_recorded_as_failure(self) -> None:
        """An HTTP 200 whose body is not valid JSON (e.g. a proxy error page) must be
        counted as a failure with the body preserved, not crash the worker."""
        body = "<html><body>502 Bad Gateway</body></html>"
        client = make_client(self.backend, APIConfig(type=APIType.Completion, streaming=False))
        session = await make_session(client, FakeUnaryResponse(body, status=200))

        await session.process_request(CompletionAPIData(prompt=COMPLETION_PROMPT), stage_id=0, scheduled_time=0.0)

        metric = recorded_metric(client)
        assert metric.error is not None
        assert metric.error.error_type == "JSONDecodeError"
        assert metric.response_data == body

    @pytest.mark.asyncio
    async def test_interrupted_stream_preserves_partial_body(self) -> None:
        """A stream severed mid-response must record the transport error and keep the
        bytes received so far (regression coverage for #531 per backend)."""
        # Replay only the first two SSE events, then sever the connection.
        partial = "".join(sse_event(chunk) for chunk in self.backend.completion_stream[:2])
        response = FakeStreamingResponse([partial.encode()], interrupt_with=ConnectionResetError("Connection reset by peer"))
        client = make_client(self.backend, APIConfig(type=APIType.Completion, streaming=True))
        session = await make_session(client, response)

        await session.process_request(CompletionAPIData(prompt=COMPLETION_PROMPT), stage_id=0, scheduled_time=0.0)

        metric = recorded_metric(client)
        assert metric.error is not None
        assert metric.error.error_type == "ConnectionResetError"
        assert "Paris," in metric.response_data

    # --- Client construction against recorded /v1/models responses ---

    def test_model_name_inferred_from_models_endpoint(self) -> None:
        mock_get = MagicMock()
        mock_get.return_value.json.return_value = self.backend.models_response
        with (
            patch("inference_perf.client.modelserver.openai_client.CustomTokenizer", StubTokenizer),
            patch("inference_perf.client.modelserver.openai_client.requests.get", mock_get),
        ):
            client = self.backend.client_cls(
                metrics_collector=MagicMock(),
                api_config=APIConfig(type=APIType.Completion, streaming=False),
                uri=BASE_URI,
                model_name=None,
                tokenizer_config=None,
                max_tcp_connections=4,
                additional_filters=[],
            )
        assert client.model_name == MODEL
        mock_get.assert_called_once_with(f"{BASE_URI}/v1/models")

    def test_supported_apis_and_metric_metadata(self) -> None:
        client = make_client(self.backend, APIConfig(type=APIType.Completion, streaming=False))
        assert client.get_supported_apis() == [APIType.Completion, APIType.Chat]
        # metric_filters is defined per-subclass, not on the shared base client.
        assert cast(Any, client).metric_filters == self.backend.expected_metric_filters
        metadata = client.get_prometheus_metric_metadata()
        queue_metric = metadata["avg_queue_length"]
        assert queue_metric is not None
        assert queue_metric.name == self.backend.queue_metric_name
