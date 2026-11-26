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
"""The server's input token count for requests that got no usage back (#979).

A successful request records the server's ``usage.prompt_tokens``. A failed or cancelled
one gets no usage, so it recorded 0 input tokens and the report's ``failures.prompt_tokens``
summed zeros. After the last stage the runner asks the profile's token-count route (vLLM:
``/tokenize``) for each of them. The fake server is a stdlib HTTP server on its own thread,
so the client's aiohttp session talks to it over real HTTP from any event loop.
"""

import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import TracebackType
from typing import Dict, List, Optional, Type
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from backend_client_suite import MODEL, StubTokenizer
from inference_perf.apis import (
    ChatCompletionAPIData,
    ChatMessage,
    CompletionAPIData,
    ErrorResponseInfo,
    InferenceAPIData,
    InferenceInfo,
    RequestLifecycleMetric,
)
from inference_perf.apis.base import stage_teardown_cancelled_error
from inference_perf.client.modelserver.mock_client import MockModelServerClient
from inference_perf.client.modelserver.openai_client import openAIModelServerClient
from inference_perf.client.modelserver.sglang_client import SGlangModelServerClient
from inference_perf.client.modelserver.tgi_client import TGImodelServerClient
from inference_perf.client.modelserver.vllm_client import vLLMModelServerClient
from inference_perf.client.modelserver import openai_client as openai_client_module
from inference_perf.config import APIConfig, APIType
from inference_perf.config.reportgen.config import ReportConfig
from inference_perf.main import InferencePerfRunner
from inference_perf.metrics.request_collector.local import LocalRequestMetricCollector
from inference_perf.payloads import RequestMetrics, Text

TOOLS = [{"type": "function", "function": {"name": "lookup", "description": "Look up a city", "parameters": {}}}]


# Answers every inference route with 503 and POST /tokenize with vLLM's TokenizeResponse.
# The count is the prompt's words + 1 (a BOS), or the message words + 10 (a chat template),
# so it never equals the client's whitespace count. `tokenize_statuses` scripts the status
# of each /tokenize call in turn (200 once the list runs out); `tokenize_body` replaces the
# 200 body; `tokenize_delay` holds each /tokenize answer back that many seconds. Every
# /tokenize body received is kept in `tokenize_requests`.
class FakeVLLMServer:
    # No /tokenize calls scripted: every one answers 200 with the computed count.
    def __init__(
        self,
        tokenize_statuses: Optional[List[int]] = None,
        tokenize_body: Optional[object] = None,
        tokenize_delay: float = 0.0,
    ):
        self.tokenize_statuses = list(tokenize_statuses or [])
        self.tokenize_delay = tokenize_delay
        self.tokenize_body = tokenize_body
        self.tokenize_requests: List[Dict[str, object]] = []
        fake = self

        class Handler(BaseHTTPRequestHandler):
            # POST /tokenize -> scripted status and count; any other POST -> 503.
            def do_POST(self) -> None:
                body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
                if self.path != "/tokenize":
                    self._reply(503, {"error": "overloaded"})
                    return
                fake.tokenize_requests.append(body)
                time.sleep(fake.tokenize_delay)
                status = fake.tokenize_statuses.pop(0) if fake.tokenize_statuses else 200
                self._reply(status, fake.tokenize_body if fake.tokenize_body is not None else _tokenize_response(body))

            # (200, {"count": 5}) -> a JSON response with that status; a str body is sent as is.
            def _reply(self, status: int, body: object) -> None:
                data = (body if isinstance(body, str) else json.dumps(body)).encode()
                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            # Silences the per-request stderr line.
            def log_message(self, format: str, *args: object) -> None:
                pass

        self._server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        self.url = f"http://127.0.0.1:{self._server.server_address[1]}"

    # Serves on a daemon thread until the block exits.
    def __enter__(self) -> "FakeVLLMServer":
        threading.Thread(target=self._server.serve_forever, daemon=True).start()
        return self

    # Stops serving and closes the socket.
    def __exit__(
        self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]
    ) -> None:
        self._server.shutdown()
        self._server.server_close()


# {"prompt": "a b c"} -> count 4; {"messages": [{"content": "a b"}]} -> count 12.
def _tokenize_response(body: Dict[str, object]) -> Dict[str, object]:
    prompt = body.get("prompt")
    messages = body.get("messages")
    if isinstance(prompt, str):
        count = len(prompt.split()) + 1
    elif isinstance(messages, list):
        count = sum(len(str(message["content"]).split()) for message in messages) + 10
    else:
        count = 0
    return {"count": count, "max_model_len": 2048, "tokens": list(range(count))}


# A port nothing listens on: bind an ephemeral port, then release it.
def _closed_port_url() -> str:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        port = sock.getsockname()[1]
    return f"http://127.0.0.1:{port}"


# vLLM profile for `api_type` at `url`, model MODEL, with the Hub tokenizer and the readiness wait stubbed.
def _client(
    url: str,
    collector: LocalRequestMetricCollector,
    api_type: APIType = APIType.Completion,
    client_cls: Type[openAIModelServerClient] = vLLMModelServerClient,
) -> openAIModelServerClient:
    with (
        patch("inference_perf.client.modelserver.openai_client.CustomTokenizer", StubTokenizer),
        patch.object(openAIModelServerClient, "wait_until_ready"),
    ):
        return client_cls(
            metrics_collector=collector,
            api_config=APIConfig(type=api_type, streaming=False),
            uri=url,
            model_name=MODEL,
            tokenizer_config=None,
            max_tcp_connections=1,
            additional_filters=[],
        )


# Sends each of `requests` once through a fresh session of `client`.
async def _send(client: openAIModelServerClient, *requests: InferenceAPIData) -> None:
    session = client.new_session()
    try:
        for data in requests:
            await session.process_request(data, stage_id=0, scheduled_time=time.perf_counter())
    finally:
        await session.close()


# A recorded request with `body` as its payload, `input_tokens` resolved and `error` set
# (None for a success) -> the metric the collector would hold for it.
def _metric(body: Dict[str, object], error: Optional[ErrorResponseInfo], input_tokens: int = 0) -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        stage_id=0,
        request_data=json.dumps(body),
        response_data="",
        info=InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=input_tokens))),
        error=error,
        start_time=0.0,
        end_time=1.0,
        scheduled_time=0.0,
    )


HTTP_503 = ErrorResponseInfo(error_type="HTTP Error 503", error_msg="overloaded")
PROMPT_BODY: Dict[str, object] = {"model": MODEL, "prompt": "the quick brown fox", "max_tokens": 4}


# vLLM counts on /tokenize; SGLang and TGI name no route yet (#991, #997), so they count nothing.
def test_token_count_route_per_profile() -> None:
    assert vLLMModelServerClient.token_count_route == "/tokenize"
    assert SGlangModelServerClient.token_count_route is None
    assert TGImodelServerClient.token_count_route is None


# One completion "the quick brown fox" answered 503 -> 0 input tokens while the load runs,
# then the server's 5 once the runner generates the reports, before the reports read it.
def test_runner_records_server_count_for_failed_request() -> None:
    collector = LocalRequestMetricCollector()
    with FakeVLLMServer() as server:
        client = _client(server.url, collector)

        # The whole run: one completion request.
        async def run(c: openAIModelServerClient) -> None:
            await _send(c, CompletionAPIData(prompt="the quick brown fox", max_tokens=4))

        loadgen = MagicMock()
        loadgen.run = AsyncMock(side_effect=run)
        reportgen = MagicMock()
        reportgen.get_metrics_collector.return_value = collector
        reportgen.generate_reports = AsyncMock(return_value=[])
        runner = InferencePerfRunner(client, loadgen, reportgen, [])
        runner.run()
        assert collector.get_metrics()[0].info.request_metrics.text.input_tokens == 0
        runner.generate_reports(ReportConfig(), MagicMock())

    [metric] = collector.get_metrics()
    assert metric.error is not None and metric.error.error_type == "HTTP Error 503"
    assert metric.info.request_metrics.text.input_tokens == 5
    reportgen.generate_reports.assert_awaited_once()


# Completion with add_special_tokens=False -> /tokenize gets model, prompt and that flag, no sampling fields.
async def test_completion_body_carries_only_the_input() -> None:
    collector = LocalRequestMetricCollector()
    with FakeVLLMServer() as server:
        client = _client(server.url, collector)
        await _send(client, CompletionAPIData(prompt="the quick brown fox", max_tokens=4, add_special_tokens=False))
        await client.count_missing_input_tokens(collector.get_metrics())

    assert server.tokenize_requests == [{"model": MODEL, "prompt": "the quick brown fox", "add_special_tokens": False}]
    assert collector.get_metrics()[0].info.request_metrics.text.input_tokens == 5


# Chat "what is the capital of France" with one tool -> /tokenize gets model, messages and
# tools, and the request records the server's 6 + 10 = 16.
async def test_chat_body_carries_messages_and_tools() -> None:
    collector = LocalRequestMetricCollector()
    with FakeVLLMServer() as server:
        client = _client(server.url, collector, APIType.Chat)
        data = ChatCompletionAPIData(
            messages=[ChatMessage(role="user", content="what is the capital of France")], max_tokens=4, tool_definitions=TOOLS
        )
        await _send(client, data)
        await client.count_missing_input_tokens(collector.get_metrics())

    [body] = server.tokenize_requests
    assert sorted(body) == ["messages", "model", "tools"]
    assert body["messages"] == [{"role": "user", "content": "what is the capital of France"}]
    assert collector.get_metrics()[0].info.request_metrics.text.input_tokens == 16


# A request cancelled at stage teardown got no usage either -> it is counted too (5).
async def test_cancelled_request_is_counted() -> None:
    metric = _metric(PROMPT_BODY, stage_teardown_cancelled_error())
    with FakeVLLMServer() as server:
        await _client(server.url, LocalRequestMetricCollector()).count_missing_input_tokens([metric])
    assert metric.info.request_metrics.text.input_tokens == 5


# A success, and a failure that already carries a count (12) -> no /tokenize call, both unchanged.
@pytest.mark.parametrize("error, input_tokens", [(None, 0), (HTTP_503, 12)], ids=["success", "failure-with-count"])
async def test_requests_with_a_count_are_left_alone(error: Optional[ErrorResponseInfo], input_tokens: int) -> None:
    metric = _metric(PROMPT_BODY, error, input_tokens)
    with FakeVLLMServer() as server:
        await _client(server.url, LocalRequestMetricCollector()).count_missing_input_tokens([metric])
    assert server.tokenize_requests == []
    assert metric.info.request_metrics.text.input_tokens == input_tokens


# SGLang names no token-count route -> no /tokenize call, the failure keeps 0.
async def test_profile_without_route_counts_nothing() -> None:
    metric = _metric(PROMPT_BODY, HTTP_503)
    with FakeVLLMServer() as server:
        client = _client(server.url, LocalRequestMetricCollector(), client_cls=SGlangModelServerClient)
        await client.count_missing_input_tokens([metric])
    assert server.tokenize_requests == []
    assert metric.info.request_metrics.text.input_tokens == 0


# The mock client has no server to ask -> the base class leaves the failure at 0.
async def test_mock_client_counts_nothing() -> None:
    metric = _metric(PROMPT_BODY, HTTP_503)
    client = MockModelServerClient(LocalRequestMetricCollector(), APIConfig(type=APIType.Completion, streaming=False))
    await client.count_missing_input_tokens([metric])
    assert metric.info.request_metrics.text.input_tokens == 0


# Embeddings and Anthropic Messages bodies are not /tokenize's -> no call for either.
@pytest.mark.parametrize("api_type", [APIType.Embeddings, APIType.AnthropicMessages])
async def test_other_api_types_count_nothing(api_type: APIType) -> None:
    metric = _metric({"model": MODEL, "input": "the quick brown fox"}, HTTP_503)
    with FakeVLLMServer() as server:
        await _client(server.url, LocalRequestMetricCollector(), api_type).count_missing_input_tokens([metric])
    assert server.tokenize_requests == []
    assert metric.info.request_metrics.text.input_tokens == 0


# /tokenize answers 404 (a proxy that does not route it) -> one call for two failures, both keep 0.
async def test_404_stops_the_pass() -> None:
    metrics = [_metric(PROMPT_BODY, HTTP_503), _metric(PROMPT_BODY, HTTP_503)]
    with FakeVLLMServer(tokenize_statuses=[404]) as server:
        await _client(server.url, LocalRequestMetricCollector()).count_missing_input_tokens(metrics)
    assert len(server.tokenize_requests) == 1
    assert [m.info.request_metrics.text.input_tokens for m in metrics] == [0, 0]


# /tokenize answers 400 to the first request -> that one keeps 0, the second still gets 5.
async def test_other_status_skips_only_that_request() -> None:
    metrics = [_metric(PROMPT_BODY, HTTP_503), _metric(PROMPT_BODY, HTTP_503)]
    with FakeVLLMServer(tokenize_statuses=[400]) as server:
        await _client(server.url, LocalRequestMetricCollector()).count_missing_input_tokens(metrics)
    assert [m.info.request_metrics.text.input_tokens for m in metrics] == [0, 5]


# The server is gone (connection refused) -> the pass returns without raising, the failure keeps 0.
async def test_unreachable_server_leaves_counts_alone() -> None:
    metric = _metric(PROMPT_BODY, HTTP_503)
    await _client(_closed_port_url(), LocalRequestMetricCollector()).count_missing_input_tokens([metric])
    assert metric.info.request_metrics.text.input_tokens == 0


# /tokenize answers after 1 s with a 0.1 s cap -> one call for two failures, both keep 0.
async def test_timeout_stops_the_pass() -> None:
    metrics = [_metric(PROMPT_BODY, HTTP_503), _metric(PROMPT_BODY, HTTP_503)]
    with (
        FakeVLLMServer(tokenize_delay=1.0) as server,
        patch.object(openai_client_module, "_TOKEN_COUNT_TIMEOUT_SEC", 0.1),
    ):
        await _client(server.url, LocalRequestMetricCollector()).count_missing_input_tokens(metrics)
    assert len(server.tokenize_requests) == 1
    assert [m.info.request_metrics.text.input_tokens for m in metrics] == [0, 0]


# A 200 whose count is missing, a string, a bool or negative, or that is not JSON -> the failure keeps 0.
@pytest.mark.parametrize(
    "body",
    [{"tokens": [1, 2]}, {"count": "5"}, {"count": True}, {"count": -1}, "<html>ok</html>"],
    ids=["missing", "string", "bool", "negative", "not-json"],
)
async def test_body_without_integer_count_is_ignored(body: object) -> None:
    metric = _metric(PROMPT_BODY, HTTP_503)
    with FakeVLLMServer(tokenize_body=body) as server:
        await _client(server.url, LocalRequestMetricCollector()).count_missing_input_tokens([metric])
    assert len(server.tokenize_requests) == 1
    assert metric.info.request_metrics.text.input_tokens == 0


# A recorded body that is not a JSON object -> skipped without a call.
@pytest.mark.parametrize("request_data", ["not json", "[1, 2]"], ids=["not-json", "not-object"])
async def test_unparsable_request_body_is_skipped(request_data: str) -> None:
    metric = _metric(PROMPT_BODY, HTTP_503)
    metric.request_data = request_data
    with FakeVLLMServer() as server:
        await _client(server.url, LocalRequestMetricCollector()).count_missing_input_tokens([metric])
    assert server.tokenize_requests == []
    assert metric.info.request_metrics.text.input_tokens == 0
