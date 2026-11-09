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
"""Integration tests for ``server_request_id`` (#685).

The client reads the server's ID once, after the request, from the body it already
holds, rather than in each API's parser. These tests drive the real client and the
real response parsers against a fake that sends a complete, verbatim body, and assert
the ID on the recorded metric: for each valid SSE framing, for a stream that opens
with an event carrying no ID, for a 200 that the client records as an in-band error,
and for each trace-replay API data, which builds its own ``InferenceInfo`` and so is
where an ID copied by hand used to go missing (#686).
"""

import json

import pytest

from fake_truncating_server import TruncatingSSEServer
from openai_client_harness import ConcreteOpenAIClient, run_request_against

from inference_perf.apis.chat import ChatMessage
from inference_perf.config import APIConfig, APIType
from inference_perf.datagen.replay.replay_graph_session_datagen import (
    EventOutputRegistry,
    SessionAnthropicMessagesAPIData,
    SessionChatCompletionAPIData,
    WorkerSessionTracker,
)

COMPLETION_CHUNK = json.dumps({"id": "cmpl-1", "choices": [{"index": 0, "text": "hello"}]})
USAGE_CHUNK = json.dumps({"id": "cmpl-1", "choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 1}})


# A one-turn replay event of the given class, with its own registry and tracker, as
# the replay data generator would hand to the client.
def make_replay_data(cls: type[SessionChatCompletionAPIData]) -> SessionChatCompletionAPIData:
    return cls(
        messages=[ChatMessage(role="user", content="hi")],
        max_tokens=16,
        event_id="session_1:event_0",
        registry=EventOutputRegistry(),
        worker_tracker=WorkerSessionTracker(),
        completion_queue=None,
        total_events_in_session=1,
        predecessor_event_ids=[],
    )


# Streaming completion bodies whose first event carries id "cmpl-1", framed with LF,
# CRLF and lone-CR line endings, without the space after "data:", with the first
# event's JSON split across two data lines, and with a comment or an event field
# between those lines. Each must record server_request_id "cmpl-1".
@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        pytest.param(f"data: {COMPLETION_CHUNK}\n\ndata: {USAGE_CHUNK}\n\ndata: [DONE]\n\n", id="lf"),
        pytest.param(f"data: {COMPLETION_CHUNK}\r\n\r\ndata: {USAGE_CHUNK}\r\n\r\ndata: [DONE]\r\n\r\n", id="crlf"),
        pytest.param(f"data: {COMPLETION_CHUNK}\r\rdata: {USAGE_CHUNK}\r\rdata: [DONE]\r\r", id="cr"),
        pytest.param(f"data:{COMPLETION_CHUNK}\n\ndata:{USAGE_CHUNK}\n\ndata:[DONE]\n\n", id="no-space"),
        pytest.param(
            'data: {"id": "cmpl-1",\ndata: "choices": [{"index": 0, "text": "hello"}]}\n\n'
            f"data: {USAGE_CHUNK}\n\ndata: [DONE]\n\n",
            id="multi-line-data",
        ),
        pytest.param(
            'data: {"id": "cmpl-1",\n: keepalive\ndata: "choices": [{"index": 0, "text": "hello"}]}\n\n'
            f"data: {USAGE_CHUNK}\n\ndata: [DONE]\n\n",
            id="comment-between-data",
        ),
        pytest.param(
            'data: {"id": "cmpl-1",\nevent: completion\ndata: "choices": [{"index": 0, "text": "hello"}]}\n\n'
            f"data: {USAGE_CHUNK}\n\ndata: [DONE]\n\n",
            id="field-between-data",
        ),
    ],
)
async def test_server_request_id_read_from_each_sse_framing(body: str) -> None:
    """The response parser accepts every one of these framings (#781), so the ID read
    from the same body must not depend on which one the server chose."""
    async with TruncatingSSEServer([], missing_bytes=0, body=body) as server:
        metric = await run_request_against(server.base_url)

    assert metric.error is None
    assert metric.server_request_id == "cmpl-1"


# A streaming completion body whose first event is a ping with no id, followed by
# events with id "cmpl-1". Must record server_request_id "cmpl-1".
@pytest.mark.asyncio
async def test_server_request_id_read_past_an_event_without_one() -> None:
    """A stream may open with an event that carries no ID, so the ID is taken from the
    first event that does."""
    ping = json.dumps({"type": "ping"})
    async with TruncatingSSEServer([ping, COMPLETION_CHUNK, USAGE_CHUNK, "[DONE]"], missing_bytes=0) as server:
        metric = await run_request_against(server.base_url)

    assert metric.error is None
    assert metric.server_request_id == "cmpl-1"


# A complete streaming 200 whose first event carries id "cmpl-1" and whose second is
# an OpenAI error object. Must record an InBandError failure with server_request_id
# "cmpl-1".
@pytest.mark.asyncio
async def test_server_request_id_kept_on_an_in_band_error() -> None:
    """A 200 recorded as a failure keeps its body as the response (#713), and the ID is
    read from that body, so the failed request can still be found in server logs."""
    error_frame = json.dumps({"error": {"message": "The model is overloaded", "type": "server_error", "code": 503}})
    async with TruncatingSSEServer([COMPLETION_CHUNK, error_frame, "[DONE]"], missing_bytes=0) as server:
        metric = await run_request_against(server.base_url)

    assert metric.error is not None
    assert metric.error.error_type == "InBandError"
    assert metric.server_request_id == "cmpl-1"


# Replayed chat turns against a body with id "chatcmpl-r1", and replayed Anthropic turns
# against a body whose message id is "msg_r1", each streamed and unary. Each must
# record that ID as server_request_id.
@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("data_cls", "api_type", "streaming", "body", "expected"),
    [
        pytest.param(
            SessionChatCompletionAPIData,
            APIType.Chat,
            True,
            'data: {"id": "chatcmpl-r1", "choices": [{"index": 0, "delta": {"content": "hello"}}]}\n\n'
            'data: {"id": "chatcmpl-r1", "choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 1}}\n\n'
            "data: [DONE]\n\n",
            "chatcmpl-r1",
            id="chat-streaming",
        ),
        pytest.param(
            SessionChatCompletionAPIData,
            APIType.Chat,
            False,
            json.dumps(
                {
                    "id": "chatcmpl-r1",
                    "choices": [{"index": 0, "message": {"role": "assistant", "content": "hello"}}],
                    "usage": {"prompt_tokens": 4, "completion_tokens": 1},
                }
            ),
            "chatcmpl-r1",
            id="chat-unary",
        ),
        pytest.param(
            SessionAnthropicMessagesAPIData,
            APIType.AnthropicMessages,
            True,
            "event: message_start\n"
            'data: {"type": "message_start", "message": {"id": "msg_r1", "usage": {"input_tokens": 4}}}\n\n'
            "event: content_block_delta\n"
            'data: {"type": "content_block_delta", "index": 0, "delta": {"type": "text_delta", "text": "hello"}}\n\n'
            "event: message_delta\n"
            'data: {"type": "message_delta", "usage": {"output_tokens": 1}}\n\n'
            "event: message_stop\n"
            'data: {"type": "message_stop"}\n\n',
            "msg_r1",
            id="anthropic-streaming",
        ),
        pytest.param(
            SessionAnthropicMessagesAPIData,
            APIType.AnthropicMessages,
            False,
            json.dumps(
                {
                    "id": "msg_r1",
                    "type": "message",
                    "role": "assistant",
                    "content": [{"type": "text", "text": "hello"}],
                    "usage": {"input_tokens": 4, "output_tokens": 1},
                }
            ),
            "msg_r1",
            id="anthropic-unary",
        ),
    ],
)
async def test_server_request_id_read_for_each_replay_api_data(
    data_cls: type[SessionChatCompletionAPIData],
    api_type: APIType,
    streaming: bool,
    body: str,
    expected: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Replay data parses the response with its own code and builds its own
    ``InferenceInfo``. The ID must reach the metric through that path too."""
    # The harness client declares Chat and Completion only; the request path it drives
    # is the same for the Anthropic Messages API.
    monkeypatch.setattr(ConcreteOpenAIClient, "get_supported_apis", lambda self: [api_type])
    content_type = "text/event-stream" if streaming else "application/json"
    data = make_replay_data(data_cls)
    async with TruncatingSSEServer([], missing_bytes=0, body=body, content_type=content_type) as server:
        metric = await run_request_against(server.base_url, APIConfig(type=api_type, streaming=streaming), data)

    assert metric.error is None
    assert metric.server_request_id == expected
