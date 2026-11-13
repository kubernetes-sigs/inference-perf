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
"""Unit tests for the token timeline built from token ids the server returns (#917)."""

import json
from typing import AsyncGenerator, Optional
from unittest.mock import MagicMock, Mock

import pytest

from inference_perf.apis.base import StreamedResponseMetrics
from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.apis.streaming_parser import chunk_token_id_count, parse_sse_stream
from inference_perf.config import APIConfig, APIType
from inference_perf.datagen.replay.replay_graph_session_datagen import (
    EventOutputRegistry,
    SessionChatCompletionAPIData,
    WorkerSessionTracker,
)


# A fake aiohttp response whose body is one SSE frame per payload, then [DONE].
def make_response(payloads: list[dict[str, object]]) -> Mock:
    frames = [f"data: {json.dumps(payload)}\n\n".encode() for payload in payloads] + [b"data: [DONE]\n\n"]
    response = Mock()
    response.content = Mock()

    async def iter_any() -> AsyncGenerator[bytes, None]:
        for frame in frames:
            yield frame

    response.content.iter_any = iter_any
    return response


# A tokenizer that counts whitespace-separated words, so "a b c" is 3 tokens.
def make_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.count_tokens = MagicMock(side_effect=lambda text, **kwargs: len(text.split()))
    return tokenizer


# The chat delta's content field, or None when the chunk has no choices or no content.
def chat_content(data: dict[str, object]) -> Optional[str]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    content = choices[0].get("delta", {}).get("content")
    return str(content) if content else None


# A chat chunk whose delta is `delta` and, when given, carries `token_ids` on the choice.
def chat_chunk(delta: dict[str, object], token_ids: Optional[list[int]] = None) -> dict[str, object]:
    choice: dict[str, object] = {"delta": delta}
    if token_ids is not None:
        choice["token_ids"] = token_ids
    return {"choices": [choice]}


# One choice with ids [1, 2, 3] counts 3. Two choices with 2 and 1 ids count 3.
def test_chunk_token_id_count_sums_every_choice() -> None:
    assert chunk_token_id_count({"choices": [{"token_ids": [1, 2, 3]}]}) == 3
    assert chunk_token_id_count({"choices": [{"token_ids": [1, 2]}, {"token_ids": [3]}]}) == 3


# No choices key, choices that is not a list, a choice that is not a dict, and
# token_ids that is null all count 0. A null next to a real list counts the list.
def test_chunk_token_id_count_ignores_what_is_not_a_list_of_ids() -> None:
    assert chunk_token_id_count({}) == 0
    assert chunk_token_id_count({"choices": "nope"}) == 0
    assert chunk_token_id_count({"choices": ["nope"]}) == 0
    assert chunk_token_id_count({"choices": [{"token_ids": None}]}) == 0
    assert chunk_token_id_count({"choices": ["nope", {"token_ids": None}, {"token_ids": [7]}]}) == 1


# Four chunks: an empty delta with 1 id, "Hello" with 2 ids, " world" with 1 id,
# an empty delta with 1 id. The id timeline has 5 stamps, the text timeline 2,
# and the two ids of "Hello" share the stamp of the chunk that carried them.
@pytest.mark.asyncio
async def test_parser_stamps_one_time_per_id_including_chunks_without_text() -> None:
    parsed = await parse_sse_stream(
        make_response(
            [
                chat_chunk({}, [11]),
                chat_chunk({"content": "Hello"}, [12, 13]),
                chat_chunk({"content": " world"}, [14]),
                chat_chunk({}, [15]),
            ]
        ),
        chat_content,
    )

    assert parsed.output_text == "Hello world"
    assert len(parsed.chunk_times) == 2
    assert len(parsed.token_id_times) == 5
    assert parsed.token_id_times == sorted(parsed.token_id_times)
    assert parsed.token_id_times[1] == parsed.token_id_times[2] == parsed.chunk_times[0]
    assert parsed.token_id_times[3] == parsed.chunk_times[1]
    # the swallowed tokens bracket the text: first id before any text, last id after it
    assert parsed.token_id_times[0] <= parsed.chunk_times[0]
    assert parsed.token_id_times[4] >= parsed.chunk_times[1]


# Two content chunks from a server that sends no token_ids. The id timeline is empty.
@pytest.mark.asyncio
async def test_parser_has_no_id_timeline_when_the_server_sends_no_ids() -> None:
    parsed = await parse_sse_stream(
        make_response([chat_chunk({"content": "Hello"}), chat_chunk({"content": " world"})]), chat_content
    )

    assert parsed.token_id_times == []
    assert len(parsed.chunk_times) == 2


# vLLM's first chunk carries the prompt's ids at the top level as prompt_token_ids.
# 4 prompt ids there and 1 generated id on the next chunk give an id timeline of 1.
@pytest.mark.asyncio
async def test_parser_does_not_count_prompt_token_ids() -> None:
    first: dict[str, object] = {"prompt_token_ids": [1, 2, 3, 4], "choices": [{"delta": {"role": "assistant", "content": ""}}]}
    parsed = await parse_sse_stream(make_response([first, chat_chunk({"content": "Hi"}, [9])]), chat_content)

    assert len(parsed.token_id_times) == 1


# A usage-only trailing chunk has an empty choices list. The 2 ids sent before it
# still count, and the usage is still read.
@pytest.mark.asyncio
async def test_parser_keeps_ids_and_usage_across_an_empty_choices_chunk() -> None:
    usage: dict[str, object] = {"choices": [], "usage": {"completion_tokens": 2}}
    parsed = await parse_sse_stream(make_response([chat_chunk({"content": "Hi there"}, [5, 6]), usage]), chat_content)

    assert len(parsed.token_id_times) == 2
    assert parsed.server_usage == {"completion_tokens": 2}


# Chat stream: a swallowed token (1 id), a tool-call fragment (2 ids), an end token
# (1 id). output_token_times has 4 entries, one per id, and the flag is set.
@pytest.mark.asyncio
async def test_chat_uses_the_id_timeline_when_ids_are_returned() -> None:
    tool_delta: dict[str, object] = {
        "tool_calls": [{"index": 0, "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]
    }
    response = make_response([chat_chunk({}, [1]), chat_chunk(tool_delta, [2, 3]), chat_chunk({}, [4])])
    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="weather?")], max_tokens=16)

    info = await data.process_response(response, APIConfig(type=APIType.Chat, streaming=True), make_tokenizer())

    assert isinstance(info.response_metrics, StreamedResponseMetrics)
    assert info.response_metrics.token_ids_returned is True
    assert len(info.response_metrics.output_token_times) == 4
    # the visible fragment is one chunk, so the chunk timeline alone would have 1 entry
    assert len(info.response_metrics.tool_chunk_times) == 1


# The same chat stream without token_ids. output_token_times falls back to the one
# chunk that carried text, and the flag stays off.
@pytest.mark.asyncio
async def test_chat_keeps_the_chunk_timeline_when_no_ids_are_returned() -> None:
    tool_delta: dict[str, object] = {
        "tool_calls": [{"index": 0, "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]
    }
    response = make_response([chat_chunk({}), chat_chunk(tool_delta), chat_chunk({})])
    data = ChatCompletionAPIData(messages=[ChatMessage(role="user", content="weather?")], max_tokens=16)

    info = await data.process_response(response, APIConfig(type=APIType.Chat, streaming=True), make_tokenizer())

    assert isinstance(info.response_metrics, StreamedResponseMetrics)
    assert info.response_metrics.token_ids_returned is False
    assert info.response_metrics.output_token_times == info.response_metrics.tool_chunk_times
    assert len(info.response_metrics.output_token_times) == 1


# Completion stream: "Hello" with 2 ids, then " world" with 1 id. With ids the
# timeline has 3 entries and the flag is set. Without ids it has 2, one per chunk.
@pytest.mark.asyncio
@pytest.mark.parametrize(("with_ids", "expected_times"), [(True, 3), (False, 2)])
async def test_completion_uses_the_id_timeline_only_when_ids_are_returned(with_ids: bool, expected_times: int) -> None:
    first: dict[str, object] = {"text": "Hello"}
    second: dict[str, object] = {"text": " world"}
    if with_ids:
        first["token_ids"] = [1, 2]
        second["token_ids"] = [3]
    response = make_response([{"choices": [first]}, {"choices": [second]}])
    data = CompletionAPIData(prompt="say hello", max_tokens=16)

    info = await data.process_response(response, APIConfig(type=APIType.Completion, streaming=True), make_tokenizer())

    assert isinstance(info.response_metrics, StreamedResponseMetrics)
    assert info.response_metrics.token_ids_returned is with_ids
    assert len(info.response_metrics.output_token_times) == expected_times
    # output_len stays the client's own count of the text: two words
    assert info.response_metrics.output_tokens == 2


# Trace replay, one event: a swallowed token (1 id), a tool-call fragment (2 ids),
# an end token (1 id). With ids the timeline has 4 entries and the flag is set.
# Without ids it has 1, the chunk that carried the fragment.
@pytest.mark.asyncio
@pytest.mark.parametrize(("with_ids", "expected_times"), [(True, 4), (False, 1)])
async def test_replay_uses_the_id_timeline_only_when_ids_are_returned(with_ids: bool, expected_times: int) -> None:
    tool_delta: dict[str, object] = {
        "tool_calls": [{"index": 0, "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'}}]
    }
    chunks = [
        chat_chunk({}, [1] if with_ids else None),
        chat_chunk(tool_delta, [2, 3] if with_ids else None),
        chat_chunk({}, [4] if with_ids else None),
    ]
    data = SessionChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="weather?")],
        event_id="s:e",
        registry=EventOutputRegistry(),
        worker_tracker=WorkerSessionTracker(),
        completion_queue=None,
        total_events_in_session=1,
        predecessor_event_ids=[],
    )

    info = await data.process_response(make_response(chunks), APIConfig(type=APIType.Chat, streaming=True), make_tokenizer())

    assert isinstance(info.response_metrics, StreamedResponseMetrics)
    assert info.response_metrics.token_ids_returned is with_ids
    assert len(info.response_metrics.output_token_times) == expected_times
