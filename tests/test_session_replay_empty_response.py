#!/usr/bin/env python3

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

"""A replayed chat turn whose 200 body is not a completion fails the turn (#1076).

SessionChatCompletionAPIData.process_response raises EmptyResponseError, as
ChatCompletionAPIData does, when a body or stream has no generated text, no
tool call and no usage. The turn then fails, instead of the session going on
with "" as the assistant's reply.
"""

from __future__ import annotations

import json
from typing import AsyncGenerator, Dict, List
from unittest.mock import AsyncMock, MagicMock

import pytest

from inference_perf.apis.chat import ChatMessage
from inference_perf.apis.response_errors import EmptyResponseError
from inference_perf.datagen.replay.replay_graph_session_datagen import (
    EventOutputRegistry,
    SessionChatCompletionAPIData,
    WorkerSessionTracker,
)

EVENT_ID = "session_1:event_0"
TOOL_CALL = {"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}


def _tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.count_tokens = lambda text, **kw: max(1, len((text or "").split()))
    return tokenizer


def _config(streaming: bool) -> MagicMock:
    config = MagicMock()
    config.streaming = streaming
    return config


def _unary(body: Dict[str, object]) -> MagicMock:
    response = MagicMock()
    response.json = AsyncMock(return_value=body)
    return response


def _stream(frames: List[Dict[str, object]]) -> MagicMock:
    payload = b"".join(f"data: {json.dumps(frame)}\n\n".encode() for frame in frames) + b"data: [DONE]\n\n"

    async def iter_any() -> AsyncGenerator[bytes, None]:
        yield payload

    response = MagicMock()
    response.status = 200
    response.headers = {"content-type": "text/event-stream"}
    response.content.iter_any = iter_any
    return response


def _api_data(registry: EventOutputRegistry) -> SessionChatCompletionAPIData:
    return SessionChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="hi")],
        max_tokens=50,
        event_id=EVENT_ID,
        registry=registry,
        worker_tracker=WorkerSessionTracker(),
        completion_queue=None,
        total_events_in_session=1,
        predecessor_event_ids=[],
    )


@pytest.mark.asyncio
async def test_unary_empty_body_fails_the_turn() -> None:
    registry = EventOutputRegistry()

    with pytest.raises(EmptyResponseError):
        await _api_data(registry).process_response(_unary({}), _config(streaming=False), _tokenizer())

    assert registry.get_output_by_event_id(EVENT_ID) is None


@pytest.mark.asyncio
async def test_empty_stream_fails_the_turn() -> None:
    registry = EventOutputRegistry()

    with pytest.raises(EmptyResponseError):
        await _api_data(registry).process_response(_stream([]), _config(streaming=True), _tokenizer())

    assert registry.get_output_by_event_id(EVENT_ID) is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        pytest.param(
            {
                "choices": [{"message": {"role": "assistant", "content": ""}}],
                "usage": {"prompt_tokens": 3, "completion_tokens": 0},
            },
            id="empty-content-with-usage",
        ),
        pytest.param(
            {"choices": [{"message": {"role": "assistant", "content": None, "tool_calls": [TOOL_CALL]}}]},
            id="tool-only-without-usage",
        ),
    ],
)
async def test_unary_completion_without_text_still_succeeds(body: Dict[str, object]) -> None:
    registry = EventOutputRegistry()

    info = await _api_data(registry).process_response(_unary(body), _config(streaming=False), _tokenizer())

    assert info.output_message is not None


@pytest.mark.asyncio
async def test_usage_only_stream_still_succeeds() -> None:
    registry = EventOutputRegistry()
    frames: List[Dict[str, object]] = [{"choices": [], "usage": {"prompt_tokens": 3, "completion_tokens": 0}}]

    info = await _api_data(registry).process_response(_stream(frames), _config(streaming=True), _tokenizer())

    assert info.request_metrics.text.input_tokens == 3
