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
"""Integration tests for issue #713 (#606 Integration tier, per-change lane).

A 200 is only a success if its body is a completion. Some servers and proxies
answer ``200 OK`` and put the failure in the body; before #713 such a request
was recorded as a success with ``output_tokens: 0`` and counted toward
throughput, with nothing in the report pointing at the cause.

These tests drive the real client against a fake that returns a *complete*,
well-formed 200 (the ``Content-Length`` is honest, nothing breaks) whose
payload is either an error object or nothing at all, in both the streaming and
the unary path, and assert the request lands in the failure bucket with the
body preserved. No model server needed.
"""

import json
from typing import Any, Dict

import pytest

from fake_truncating_server import TruncatingSSEServer
from openai_client_harness import generate_reports, run_request_against

from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.config import APIConfig, APIType

# The OpenAI error object as vLLM and SGLang emit it mid-stream and as a proxy
# might return it whole. `code` inside the payload says 503; the transport says 200.
ERROR_PAYLOAD: Dict[str, Any] = {
    "error": {"message": "The model is overloaded, retry later", "type": "server_error", "code": 503}
}
ERROR_FRAME = json.dumps(ERROR_PAYLOAD)


# Streaming completion (default) and streaming chat against a complete 200 whose
# only data frame is the error object, followed by [DONE]. Both must record a
# failure of type InBandError whose error_msg is that object, with the whole
# body (frame and [DONE]) as response_data.
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("api_config", "data"),
    [
        pytest.param(APIConfig(type=APIType.Completion, streaming=True), None, id="completion"),
        pytest.param(
            APIConfig(type=APIType.Chat, streaming=True),
            ChatCompletionAPIData(messages=[ChatMessage(role="user", content="hi")], max_tokens=16),
            id="chat",
        ),
    ],
)
async def test_streaming_in_band_error_frame_is_a_failure(api_config: APIConfig, data: Any) -> None:
    """The #713 streaming shape. The stream is complete and well-formed, so no
    transport error fires; the payload alone has to make this a failure."""
    async with TruncatingSSEServer([ERROR_FRAME, "[DONE]"], missing_bytes=0) as server:
        metric = await run_request_against(server.base_url, api_config, data)
        sent_body = server.sent_body

    assert metric.error is not None, "a 200 carrying an error payload must be recorded as a failure"
    assert metric.error.error_type == "InBandError"
    assert json.loads(metric.error.error_msg) == ERROR_PAYLOAD, "error_msg must be the server's error object"
    assert metric.response_data == sent_body, "the whole body must be preserved, framing included"
    assert "[DONE]" in (metric.response_data or "")


# Streaming completion against a complete 200 whose only frame is [DONE]: no
# content, no usage. Must record an EmptyResponseError failure with the body kept.
@pytest.mark.asyncio
async def test_streaming_empty_stream_is_a_failure() -> None:
    """A stream that says nothing is not a zero-token completion. Before #713 this
    was a success with output_tokens 0 and an empty response."""
    async with TruncatingSSEServer(["[DONE]"], missing_bytes=0) as server:
        metric = await run_request_against(server.base_url)
        sent_body = server.sent_body

    assert metric.error is not None
    assert metric.error.error_type == "EmptyResponseError"
    assert metric.response_data == sent_body == "data: [DONE]\n\n"


# Unary completion against a complete 200 whose JSON body is the error object.
# Must record an InBandError failure whose error_msg is that object and whose
# response_data is the body as sent.
@pytest.mark.asyncio
async def test_unary_in_band_error_body_is_a_failure() -> None:
    """The #713 unary shape: a proxy that answers 200 with an OpenAI error object.
    The client reads the body before process_response, so it is the client's own
    copy that must survive as response_data."""
    async with TruncatingSSEServer([], missing_bytes=0, body=ERROR_FRAME, content_type="application/json") as server:
        metric = await run_request_against(server.base_url, APIConfig(type=APIType.Completion, streaming=False))

    assert metric.error is not None
    assert metric.error.error_type == "InBandError"
    assert json.loads(metric.error.error_msg) == ERROR_PAYLOAD
    assert metric.response_data == ERROR_FRAME


# Unary completion against a complete 200 whose body is `{}`. Must record an
# EmptyResponseError failure with response_data == "{}".
@pytest.mark.asyncio
async def test_unary_empty_body_is_a_failure() -> None:
    async with TruncatingSSEServer([], missing_bytes=0, body="{}", content_type="application/json") as server:
        metric = await run_request_against(server.base_url, APIConfig(type=APIType.Completion, streaming=False))

    assert metric.error is not None
    assert metric.error.error_type == "EmptyResponseError"
    assert metric.response_data == "{}"


# End of the chain: the in-band error request must show up in the summary as
# failures.count == 1 under the "inbanderror" label carrying the server's
# message, successes.count == 0, and the per-request entry must carry both the
# error and the raw body.
@pytest.mark.asyncio
async def test_in_band_error_lands_in_the_failure_bucket_of_the_report() -> None:
    """What #713 is about: the request must not count toward throughput, and the
    report has to say why it failed."""
    async with TruncatingSSEServer([ERROR_FRAME, "[DONE]"], missing_bytes=0) as server:
        metric = await run_request_against(server.base_url)
        sent_body = server.sent_body

    summary, entries = await generate_reports([metric])

    assert summary["successes"]["count"] == 0
    assert summary["failures"]["count"] == 1
    by_label = summary["failures"]["by_label"]
    assert list(by_label) == ["inbanderror"], by_label
    assert by_label["inbanderror"]["count"] == 1
    assert by_label["inbanderror"]["messages"][0]["message"] == ERROR_PAYLOAD["error"]["message"]

    assert len(entries) == 1
    assert entries[0]["error"]["error_type"] == "InBandError"
    assert entries[0]["response"] == sent_body
