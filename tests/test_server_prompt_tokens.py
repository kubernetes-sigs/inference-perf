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

"""Tests for the `api.use_server_prompt_tokens` fast path in
SessionChatCompletionAPIData.process_response (issue #648).

When the flag is enabled and the server reports usage.prompt_tokens, the
client records the server count and skips re-tokenizing the full conversation
history (an O(history) CPU cost on the loadgen event loop per event). When the
flag is disabled (the default), behavior is unchanged: the client tokenizes
the joined conversation text.
"""

import json
import logging
from typing import Any, AsyncGenerator, Dict, List, Optional, cast
from unittest.mock import AsyncMock, MagicMock

import pytest
from aiohttp import ClientResponse

import inference_perf.datagen.replay_graph_session_datagen as replay_mod
from inference_perf.apis.chat import ChatMessage
from inference_perf.config import APIConfig, APIType
from inference_perf.datagen.replay_graph_session_datagen import (
    EventOutputRegistry,
    SessionChatCompletionAPIData,
    WorkerSessionTracker,
)


# The fallback tokenizer below counts whitespace-split words, so this prompt
# counts as 3 client-side tokens; the server-reported count is deliberately
# different so the two paths are distinguishable.
PROMPT_TEXT = "alpha beta gamma"
SERVER_USAGE: Dict[str, Any] = {"prompt_tokens": 4242, "completion_tokens": 7, "total_tokens": 4249}


@pytest.fixture(autouse=True)
def _reset_warning_flag(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(replay_mod, "_warned_missing_server_prompt_tokens", False)


def _make_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.count_tokens = MagicMock(side_effect=lambda text: max(1, len((text or "").split())))
    return tok


def _make_config(streaming: bool, use_server_prompt_tokens: bool) -> APIConfig:
    return APIConfig(type=APIType.Chat, streaming=streaming, use_server_prompt_tokens=use_server_prompt_tokens)


class FakeStreamingResponse:
    """Minimal aiohttp ClientResponse stand-in that yields preset SSE bytes."""

    def __init__(self, chunks: List[bytes]) -> None:
        self.status = 200
        self.headers = {"content-type": "text/event-stream"}
        self._chunks = chunks
        self.content = self._make_content()

    def _make_content(self) -> MagicMock:
        content = MagicMock()

        async def iter_any() -> AsyncGenerator[bytes, None]:
            for chunk in self._chunks:
                yield chunk

        content.iter_any = iter_any
        return content


def _make_streaming_response(deltas: List[Dict[str, Any]], usage: Optional[Dict[str, Any]] = None) -> ClientResponse:
    parts: List[bytes] = []
    for delta in deltas:
        parts.append(f"data: {json.dumps({'choices': [{'delta': delta}]})}\n\n".encode())
    if usage is not None:
        parts.append(f"data: {json.dumps({'choices': [], 'usage': usage})}\n\n".encode())
    parts.append(b"data: [DONE]\n\n")
    return cast(ClientResponse, FakeStreamingResponse([b"".join(parts)]))


def _make_non_streaming_response(usage: Optional[Dict[str, Any]] = None) -> ClientResponse:
    body: Dict[str, Any] = {"choices": [{"message": {"role": "assistant", "content": "The answer"}}]}
    if usage is not None:
        body["usage"] = usage
    response = MagicMock()
    response.json = AsyncMock(return_value=body)
    return cast(ClientResponse, response)


def _make_session_api_data() -> SessionChatCompletionAPIData:
    return SessionChatCompletionAPIData(
        messages=[ChatMessage(role="user", content=PROMPT_TEXT)],
        max_tokens=500,
        event_id="session_1:event_0",
        registry=EventOutputRegistry(),
        worker_tracker=WorkerSessionTracker(),
        completion_queue=None,
        total_events_in_session=1,
        predecessor_event_ids=[],
    )


class TestStreamingServerPromptTokens:
    @pytest.mark.asyncio
    async def test_enabled_uses_server_count_and_skips_tokenizer(self) -> None:
        api_data = _make_session_api_data()
        tokenizer = _make_tokenizer()
        response = _make_streaming_response([{"content": "Hi"}], usage=SERVER_USAGE)

        info = await api_data.process_response(response, _make_config(True, True), tokenizer)

        assert info.request_metrics is not None
        assert info.request_metrics.text.input_tokens == 4242
        assert info.response_metrics is not None
        assert info.response_metrics.output_tokens == 7
        # The whole point: no client-side tokenization at all.
        assert tokenizer.count_tokens.call_count == 0

    @pytest.mark.asyncio
    async def test_enabled_falls_back_when_usage_absent(self) -> None:
        api_data = _make_session_api_data()
        tokenizer = _make_tokenizer()
        response = _make_streaming_response([{"content": "Hi"}], usage=None)

        info = await api_data.process_response(response, _make_config(True, True), tokenizer)

        assert info.request_metrics is not None
        assert info.request_metrics.text.input_tokens == 3  # client-side word count of PROMPT_TEXT
        assert tokenizer.count_tokens.call_count > 0

    @pytest.mark.asyncio
    async def test_enabled_falls_back_when_prompt_tokens_missing(self) -> None:
        api_data = _make_session_api_data()
        tokenizer = _make_tokenizer()
        response = _make_streaming_response([{"content": "Hi"}], usage={"completion_tokens": 7})

        info = await api_data.process_response(response, _make_config(True, True), tokenizer)

        assert info.request_metrics is not None
        assert info.request_metrics.text.input_tokens == 3  # prompt falls back to the client count
        assert info.response_metrics is not None
        assert info.response_metrics.output_tokens == 7  # output still honors server usage
        assert tokenizer.count_tokens.call_count == 1  # prompt only; output came from the server

    @pytest.mark.asyncio
    async def test_disabled_keeps_client_side_count(self) -> None:
        api_data = _make_session_api_data()
        tokenizer = _make_tokenizer()
        response = _make_streaming_response([{"content": "Hi"}], usage=SERVER_USAGE)

        info = await api_data.process_response(response, _make_config(True, False), tokenizer)

        assert info.request_metrics is not None
        assert info.request_metrics.text.input_tokens == 3  # default behavior is unchanged
        assert info.response_metrics is not None
        assert info.response_metrics.output_tokens == 7
        assert tokenizer.count_tokens.call_count == 1

    @pytest.mark.asyncio
    async def test_warns_once_per_process_when_usage_absent(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING, logger=replay_mod.__name__)
        config = _make_config(True, True)
        for _ in range(2):
            api_data = _make_session_api_data()
            response = _make_streaming_response([{"content": "Hi"}], usage=None)
            await api_data.process_response(response, config, _make_tokenizer())

        warnings = [r for r in caplog.records if "usage.prompt_tokens" in r.getMessage()]
        assert len(warnings) == 1

    @pytest.mark.asyncio
    async def test_disabled_never_warns(self, caplog: pytest.LogCaptureFixture) -> None:
        caplog.set_level(logging.WARNING, logger=replay_mod.__name__)
        api_data = _make_session_api_data()
        response = _make_streaming_response([{"content": "Hi"}], usage=None)

        await api_data.process_response(response, _make_config(True, False), _make_tokenizer())

        assert not [r for r in caplog.records if "usage.prompt_tokens" in r.getMessage()]


class TestNonStreamingServerPromptTokens:
    @pytest.mark.asyncio
    async def test_enabled_uses_server_count_and_propagates_usage(self) -> None:
        api_data = _make_session_api_data()
        tokenizer = _make_tokenizer()
        response = _make_non_streaming_response(usage=SERVER_USAGE)

        info = await api_data.process_response(response, _make_config(False, True), tokenizer)

        assert info.request_metrics is not None
        assert info.request_metrics.text.input_tokens == 4242
        assert info.response_metrics is not None
        assert info.response_metrics.output_tokens == 7
        assert info.response_metrics.server_usage == SERVER_USAGE
        assert tokenizer.count_tokens.call_count == 0

    @pytest.mark.asyncio
    async def test_enabled_falls_back_when_usage_absent(self) -> None:
        api_data = _make_session_api_data()
        tokenizer = _make_tokenizer()
        response = _make_non_streaming_response(usage=None)

        info = await api_data.process_response(response, _make_config(False, True), tokenizer)

        assert info.request_metrics is not None
        assert info.request_metrics.text.input_tokens == 3
        assert info.response_metrics is not None
        assert info.response_metrics.output_tokens == 2  # client-side count of "The answer"
        assert info.response_metrics.server_usage is None

    @pytest.mark.asyncio
    async def test_disabled_keeps_client_side_count_but_propagates_usage(self) -> None:
        api_data = _make_session_api_data()
        tokenizer = _make_tokenizer()
        response = _make_non_streaming_response(usage=SERVER_USAGE)

        info = await api_data.process_response(response, _make_config(False, False), tokenizer)

        assert info.request_metrics is not None
        assert info.request_metrics.text.input_tokens == 3  # default behavior is unchanged
        assert info.response_metrics is not None
        assert info.response_metrics.output_tokens == 7  # output-side server preference predates the flag
        # usage propagation into metrics is unconditional, matching chat.py/completion.py
        assert info.response_metrics.server_usage == SERVER_USAGE
