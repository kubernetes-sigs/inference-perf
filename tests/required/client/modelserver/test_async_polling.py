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
"""Tests for asynchronous HTTP 202 Accepted enqueue + GET status polling."""

from __future__ import annotations

import asyncio
import json
from types import TracebackType
from typing import Dict, List
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest

from inference_perf.apis import CompletionAPIData, RequestLifecycleMetric
from inference_perf.apis.response_errors import InvalidResponseError
from inference_perf.client.modelserver.openai_client import (
    _extract_async_request_id,
    openAIModelServerClientSession,
)
from inference_perf.config import APIConfig, APIType, AsyncPollingConfig
from inference_perf.config.redaction import REDACTED, redact


class _MockHttpContext:
    # Build an async context manager returning a mock aiohttp response for the given status, body, and headers.
    def __init__(self, status: int, body: str, headers: Dict[str, str] | None = None, delay_sec: float = 0.0) -> None:
        self.delay_sec = delay_sec
        self.resp = MagicMock()
        self.resp.status = status
        self.resp.headers = headers or {}
        self.resp.text = AsyncMock(return_value=body)
        self.resp.read = AsyncMock(return_value=body.encode("utf-8"))
        self.resp.json = AsyncMock(side_effect=lambda: json.loads(body))

    # Return the mock response when entering `async with`, optionally sleeping first.
    async def __aenter__(self) -> MagicMock:
        if self.delay_sec > 0:
            await asyncio.sleep(self.delay_sec)
        return self.resp

    # Exit `async with` without suppressing exceptions.
    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        return None


# Construct a mock OpenAI model server client with the given APIConfig and timeout.
def _make_mock_client(api_config: APIConfig, timeout: float | None = 10.0) -> MagicMock:
    client = MagicMock()
    client.uri = "http://gateway:8081"
    client.model_name = "Qwen/Qwen3-0.6B"
    client.max_completion_tokens = 32
    client.ignore_eos = True
    client.api_key = "secret-key"
    client.api_config = api_config
    client.timeout = timeout
    client.cert_path = None
    client.key_path = None
    client.request_retries = 2
    client.request_retry_backoff_sec = 0.01
    client.otel = MagicMock()
    client.otel.enabled = False
    client.otel.trace_llm_request.return_value.__enter__.return_value = None
    client.otel.trace_llm_request.return_value.__exit__.return_value = None
    client.tokenizer = MagicMock()
    client.tokenizer.count_tokens.return_value = 8
    client.metrics_collector = MagicMock()
    return client


# Create a session with a mocked aiohttp.ClientSession, closing the initial real session cleanly.
async def _make_session(client: MagicMock) -> tuple[openAIModelServerClientSession, MagicMock]:
    session = openAIModelServerClientSession(client)
    await session.close()
    mock_http = MagicMock()
    session.session = mock_http
    return session, mock_http


# Reject invalid poll_url_template or streaming+async_polling, accept absolute URLs, and redact sensitive poll headers.
def test_async_polling_config_validation_and_redaction() -> None:
    with pytest.raises(ValueError, match="must contain '\\{id\\}'"):
        AsyncPollingConfig(poll_url_template="/v1/requests/static")

    with pytest.raises(ValueError, match="poll_url_template must start with"):
        AsyncPollingConfig(poll_url_template="v1/requests/{id}")

    cfg_http = AsyncPollingConfig(poll_url_template="http://status.local/jobs/{id}")
    assert cfg_http.poll_url_template == "http://status.local/jobs/{id}"

    cfg_https = AsyncPollingConfig(poll_url_template="https://status.local/jobs/{id}")
    assert cfg_https.poll_url_template == "https://status.local/jobs/{id}"

    with pytest.raises(ValueError, match="streaming is not supported when async_polling is configured"):
        APIConfig(
            type=APIType.Completion,
            streaming=True,
            async_polling=AsyncPollingConfig(),
        )

    raw = {
        "type": "completion",
        "headers": {"x-llm-d-tenant": "batch", "Authorization": "Bearer top-secret"},
        "async_polling": {
            "poll_url_template": "/v1/requests/{id}",
            "headers": {"X-Api-Key": "poll-secret", "x-llm-d-tenant": "batch"},
        },
    }
    masked = redact(raw, APIConfig)
    assert masked["headers"]["Authorization"] == REDACTED
    assert masked["headers"]["x-llm-d-tenant"] == "batch"
    assert masked["async_polling"]["headers"]["X-Api-Key"] == REDACTED
    assert masked["async_polling"]["headers"]["x-llm-d-tenant"] == "batch"


# Extract request IDs from nested JSON fields, integer IDs, malformed JSON + header fallback, and reject booleans.
def test_extract_async_request_id_variants() -> None:
    nested_cfg = AsyncPollingConfig(id_json_field="data.job_id", response_id_header="x-request-id")
    assert _extract_async_request_id(json.dumps({"data": {"job_id": "req-7"}}), {}, nested_cfg) == "req-7"
    assert _extract_async_request_id(json.dumps({"data": {"job_id": 123}}), {}, nested_cfg) == "123"
    # Boolean values must not be coerced via isinstance(True, int).
    assert _extract_async_request_id(json.dumps({"data": {"job_id": True}}), {}, nested_cfg) is None
    # Malformed JSON falls back to case-insensitive response_id_header.
    assert _extract_async_request_id("not-json", {"X-Request-Id": "hdr-9"}, nested_cfg) == "hdr-9"
    # Missing from both body and header returns None.
    assert _extract_async_request_id("{}", {}, nested_cfg) is None


# Poll a 202 enqueue response until 200, preserving tenant/auth headers and dropping excluded/content-type headers.
@pytest.mark.asyncio
async def test_enqueue_202_polls_until_200_and_preserves_tenant_header() -> None:
    api_cfg = APIConfig(
        type=APIType.Completion,
        streaming=False,
        headers={
            "x-llm-d-async-mode": "enqueue",
            "x-llm-d-tenant": "premium",
            "x-llm-d-request-timeout-seconds": "45",
        },
        async_polling=AsyncPollingConfig(
            poll_url_template="/v1/requests/{id}",
            poll_interval_sec=0.005,
        ),
    )
    client = _make_mock_client(api_cfg)
    session, mock_http = await _make_session(client)

    mock_http.post.return_value = _MockHttpContext(
        202,
        json.dumps({"id": "job/4217", "status": "pending"}),
    )

    completion_payload = json.dumps(
        {
            "id": "cmpl-4217",
            "choices": [{"text": "hello async world"}],
            "usage": {"prompt_tokens": 12, "completion_tokens": 6},
        }
    )
    get_calls: List[tuple[str, Dict[str, str], aiohttp.ClientTimeout | None]] = []

    # Return 202 on the first GET poll and 200 with a completion body on the second GET poll.
    def _fake_get(
        url: str,
        headers: Dict[str, str],
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> _MockHttpContext:
        get_calls.append((url, dict(headers), timeout))
        if len(get_calls) == 1:
            return _MockHttpContext(202, json.dumps({"id": "job/4217", "status": "pending"}))
        return _MockHttpContext(200, completion_payload)

    mock_http.get.side_effect = _fake_get

    req = CompletionAPIData(prompt="Summarize async batching.", max_tokens=16)
    await session.process_request(req, stage_id=0, scheduled_time=0.0)

    assert len(get_calls) == 2
    poll_url, poll_headers, poll_timeout = get_calls[0]
    assert poll_url == "http://gateway:8081/v1/requests/job%2F4217"
    assert poll_headers.get("x-llm-d-tenant") == "premium"
    assert poll_headers.get("Authorization") == "Bearer secret-key"
    assert "x-llm-d-async-mode" not in {k.lower() for k in poll_headers}
    assert "content-type" not in {k.lower() for k in poll_headers}
    assert isinstance(poll_timeout, aiohttp.ClientTimeout)

    client.metrics_collector.record_metric.assert_called_once()
    recorded: RequestLifecycleMetric = client.metrics_collector.record_metric.call_args.args[0]
    assert recorded.error is None
    assert recorded.info.request_metrics.text.input_tokens == 12
    assert recorded.info.response_metrics is not None


# Support absolute poll_url_template, extra polling headers, and unbounded timeout when no deadline is configured.
@pytest.mark.asyncio
async def test_enqueue_202_absolute_poll_url_custom_headers_and_no_timeout() -> None:
    api_cfg = APIConfig(
        type=APIType.Completion,
        streaming=False,
        headers={"x-llm-d-tenant": "default"},
        async_polling=AsyncPollingConfig(
            poll_url_template="https://broker.example.com/status/{id}",
            poll_interval_sec=0.005,
            headers={"X-Poll-Source": "bench", "x-llm-d-tenant": "override"},
        ),
    )
    client = _make_mock_client(api_cfg, timeout=None)
    session, mock_http = await _make_session(client)

    mock_http.post.return_value = _MockHttpContext(202, json.dumps({"id": "job-abs"}))
    completion_payload = json.dumps(
        {
            "id": "cmpl-abs",
            "choices": [{"text": "done"}],
            "usage": {"prompt_tokens": 5, "completion_tokens": 2},
        }
    )
    get_calls: List[tuple[str, Dict[str, str], aiohttp.ClientTimeout | None]] = []

    # Record GET arguments and return a 200 completion immediately.
    def _fake_get(
        url: str,
        headers: Dict[str, str],
        timeout: aiohttp.ClientTimeout | None = None,
    ) -> _MockHttpContext:
        get_calls.append((url, dict(headers), timeout))
        return _MockHttpContext(200, completion_payload)

    mock_http.get.side_effect = _fake_get

    req = CompletionAPIData(prompt="Absolute URL", max_tokens=8)
    await session.process_request(req, stage_id=0, scheduled_time=0.0)

    assert len(get_calls) == 1
    poll_url, poll_headers, poll_timeout = get_calls[0]
    assert poll_url == "https://broker.example.com/status/job-abs"
    assert poll_headers["X-Poll-Source"] == "bench"
    assert poll_headers["x-llm-d-tenant"] == "override"
    assert poll_timeout is None
    recorded: RequestLifecycleMetric = client.metrics_collector.record_metric.call_args.args[0]
    assert recorded.error is None


# Record InvalidEnqueueResponse without polling when a 202 response contains neither JSON ID nor fallback header.
@pytest.mark.asyncio
async def test_enqueue_202_missing_request_id_records_invalid_enqueue_response() -> None:
    api_cfg = APIConfig(
        type=APIType.Completion,
        streaming=False,
        async_polling=AsyncPollingConfig(poll_interval_sec=0.005),
    )
    client = _make_mock_client(api_cfg)
    session, mock_http = await _make_session(client)

    mock_http.post.return_value = _MockHttpContext(202, "{}")

    req = CompletionAPIData(prompt="No ID", max_tokens=8)
    await session.process_request(req, stage_id=0, scheduled_time=0.0)

    mock_http.get.assert_not_called()
    recorded: RequestLifecycleMetric = client.metrics_collector.record_metric.call_args.args[0]
    assert recorded.error is not None
    assert recorded.error.error_type == "InvalidEnqueueResponse"


# Record HTTP Error 504 without retrying POST when a polled job terminates with 504 DEADLINE_EXCEEDED.
@pytest.mark.asyncio
async def test_enqueue_202_terminal_504_deadline_exceeded_is_not_retried() -> None:
    api_cfg = APIConfig(
        type=APIConfig.model_fields["type"].default,
        streaming=False,
        headers={
            "x-llm-d-async-mode": "enqueue",
            "x-llm-d-tenant": "batch",
            "x-llm-d-request-timeout-seconds": "10",
        },
        async_polling=AsyncPollingConfig(poll_interval_sec=0.005),
    )
    client = _make_mock_client(api_cfg)
    session, mock_http = await _make_session(client)

    # Body has no id, so response_id_header ('x-request-id') fallback is exercised too.
    mock_http.post.return_value = _MockHttpContext(
        202,
        "{}",
        headers={"X-Request-Id": "job-expired-99"},
    )
    mock_http.get.return_value = _MockHttpContext(
        504,
        json.dumps({"error": {"code": "DEADLINE_EXCEEDED", "message": "deadline exceeded"}}),
    )

    req = CompletionAPIData(prompt="Late job", max_tokens=16)
    await session.process_request(req, stage_id=0, scheduled_time=0.0)

    # POST was accepted (202), so transport retry loop must not re-POST a duplicate job.
    assert mock_http.post.call_count == 1
    recorded: RequestLifecycleMetric = client.metrics_collector.record_metric.call_args.args[0]
    assert recorded.error is not None
    assert recorded.error.error_type == "HTTP Error 504"


# Record InBandError and keep raw body when a 200 poll response carries an in-band error payload.
@pytest.mark.asyncio
async def test_enqueue_202_invalid_200_poll_body_records_in_band_error() -> None:
    api_cfg = APIConfig(
        type=APIType.Completion,
        streaming=False,
        async_polling=AsyncPollingConfig(poll_interval_sec=0.005),
    )
    client = _make_mock_client(api_cfg)
    session, mock_http = await _make_session(client)

    mock_http.post.return_value = _MockHttpContext(202, json.dumps({"id": "job-bad-200"}))
    bad_body = json.dumps({"error": {"message": "upstream failure inside 200"}})
    mock_http.get.return_value = _MockHttpContext(200, bad_body)

    req = CompletionAPIData(prompt="Bad 200 body", max_tokens=16)
    await session.process_request(req, stage_id=0, scheduled_time=0.0)

    assert mock_http.post.call_count == 1
    recorded: RequestLifecycleMetric = client.metrics_collector.record_metric.call_args.args[0]
    assert recorded.error is not None
    assert recorded.error.error_type == "InBandError"
    assert recorded.response_data == bad_body


# Preserve InvalidResponseError.raw_content when process_response raises InvalidResponseError on a 200 poll.
@pytest.mark.asyncio
async def test_enqueue_202_invalid_response_error_preserves_raw_content() -> None:
    api_cfg = APIConfig(
        type=APIType.Completion,
        streaming=False,
        async_polling=AsyncPollingConfig(poll_interval_sec=0.005),
    )
    client = _make_mock_client(api_cfg)
    session, mock_http = await _make_session(client)

    mock_http.post.return_value = _MockHttpContext(202, json.dumps({"id": "job-raw-200"}))
    mock_http.get.return_value = _MockHttpContext(200, "{}")

    req = CompletionAPIData(prompt="Raw content error", max_tokens=16)
    with patch.object(
        CompletionAPIData,
        "process_response",
        new_callable=AsyncMock,
        side_effect=InvalidResponseError("missing choices", raw_content="raw-server-bytes"),
    ):
        await session.process_request(req, stage_id=0, scheduled_time=0.0)

    recorded: RequestLifecycleMetric = client.metrics_collector.record_metric.call_args.args[0]
    assert recorded.error is not None
    assert recorded.error.error_type == "InvalidResponseError"
    assert recorded.response_data == "raw-server-bytes"


# Record TimeoutError without retrying POST when polling exceeds max_poll_duration_sec.
@pytest.mark.asyncio
async def test_enqueue_202_poll_timeout_records_timeout_error() -> None:
    api_cfg = APIConfig(
        type=APIType.Completion,
        streaming=False,
        async_polling=AsyncPollingConfig(
            poll_interval_sec=0.01,
            max_poll_duration_sec=0.025,
        ),
    )
    client = _make_mock_client(api_cfg, timeout=5.0)
    session, mock_http = await _make_session(client)

    mock_http.post.return_value = _MockHttpContext(202, json.dumps({"id": "job-stuck"}))
    mock_http.get.return_value = _MockHttpContext(202, json.dumps({"id": "job-stuck", "status": "pending"}))

    req = CompletionAPIData(prompt="Stuck job", max_tokens=16)
    await session.process_request(req, stage_id=0, scheduled_time=0.0)

    assert mock_http.post.call_count == 1
    recorded: RequestLifecycleMetric = client.metrics_collector.record_metric.call_args.args[0]
    assert recorded.error is not None
    assert recorded.error.error_type == "TimeoutError"


# Record TimeoutError immediately before polling if the 202 enqueue itself exhausted max_poll_duration_sec.
@pytest.mark.asyncio
async def test_enqueue_202_deadline_already_expired_skips_poll() -> None:
    api_cfg = APIConfig(
        type=APIType.Completion,
        streaming=False,
        async_polling=AsyncPollingConfig(
            poll_interval_sec=0.1,
            max_poll_duration_sec=0.01,
        ),
    )
    client = _make_mock_client(api_cfg, timeout=5.0)
    session, mock_http = await _make_session(client)

    mock_http.post.return_value = _MockHttpContext(202, json.dumps({"id": "job-slow-post"}), delay_sec=0.03)

    req = CompletionAPIData(prompt="Slow enqueue", max_tokens=16)
    await session.process_request(req, stage_id=0, scheduled_time=0.0)

    mock_http.get.assert_not_called()
    recorded: RequestLifecycleMetric = client.metrics_collector.record_metric.call_args.args[0]
    assert recorded.error is not None
    assert recorded.error.error_type == "TimeoutError"


# Record StageTeardownCancelled and re-raise CancelledError if stage teardown cancels an active poll.
@pytest.mark.asyncio
async def test_enqueue_202_cancelled_during_poll_records_cancelled_and_reraises() -> None:
    api_cfg = APIConfig(
        type=APIType.Completion,
        streaming=False,
        async_polling=AsyncPollingConfig(poll_interval_sec=5.0, max_poll_duration_sec=30.0),
    )
    client = _make_mock_client(api_cfg)
    session, mock_http = await _make_session(client)

    mock_http.post.return_value = _MockHttpContext(202, json.dumps({"id": "job-cancel"}))

    req = CompletionAPIData(prompt="Cancel while polling", max_tokens=16)
    task = asyncio.create_task(session.process_request(req, stage_id=0, scheduled_time=0.0))
    await asyncio.sleep(0.01)
    task.cancel()

    with pytest.raises(asyncio.CancelledError):
        await task

    recorded: RequestLifecycleMetric = client.metrics_collector.record_metric.call_args.args[0]
    assert recorded.error is not None
    assert recorded.error.error_type == "StageTeardownCancelled"


# Treat HTTP 202 as HTTP Error 202 when api.async_polling is not configured (backwards compatibility).
@pytest.mark.asyncio
async def test_http_202_without_async_polling_records_http_error_202() -> None:
    api_cfg = APIConfig(type=APIType.Completion, streaming=False)
    client = _make_mock_client(api_cfg)
    session, mock_http = await _make_session(client)

    mock_http.post.return_value = _MockHttpContext(202, json.dumps({"id": "job-ignored"}))

    req = CompletionAPIData(prompt="Unexpected 202", max_tokens=8)
    await session.process_request(req, stage_id=0, scheduled_time=0.0)

    mock_http.get.assert_not_called()
    recorded: RequestLifecycleMetric = client.metrics_collector.record_metric.call_args.args[0]
    assert recorded.error is not None
    assert recorded.error.error_type == "HTTP Error 202"
