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
"""Unit tests for the Batch API input writer: one JSONL line per record, in the record's own request shape."""

import asyncio
import json
from typing import Generator

import pytest

from inference_perf.apis import ChatCompletionAPIData, ChatMessage, CompletionAPIData
from inference_perf.apis.user_session import LocalUserSession, UserSessionCompletionAPIData
from inference_perf.client.batch.jsonl import build_batch_input
from inference_perf.config import APIConfig, APIType
from inference_perf.config.apis.config import ResponseFormat, ResponseFormatType
from inference_perf.datagen.replay.replay_graph_session_datagen import (
    EventOutputRegistry,
    SessionChatCompletionAPIData,
    WorkerSessionTracker,
)

# The api config of a run that sets neither clean_tool_schemas nor response_format, so a line's body is the
# shape's own body.
NO_API_OPTIONS = APIConfig()


# Empties the LocalUserSession registry before and after a test, so session "u0" starts fresh and does not
# leak into other tests.
@pytest.fixture
def user_session_registry() -> Generator[None, None, None]:
    LocalUserSession.clear_instances()
    yield
    LocalUserSession.clear_instances()


# The completion prompts "hello" and "world" become two POST lines to /v1/completions with custom_ids
# s0-b0-0 and s0-b0-1, each body built with model "m", max_tokens 16, ignore_eos false and stream false.
async def test_each_record_becomes_one_post_line_to_its_route() -> None:
    batch_input = await build_batch_input(
        [CompletionAPIData(prompt="hello"), CompletionAPIData(prompt="world")],
        custom_id_prefix="s0-b0",
        api_config=NO_API_OPTIONS,
        model_name="m",
        max_tokens=16,
        ignore_eos=False,
    )

    assert batch_input.endpoint == "/v1/completions"
    assert batch_input.custom_ids == ["s0-b0-0", "s0-b0-1"]
    assert [json.loads(line) for line in batch_input.content.splitlines()] == [
        {
            "custom_id": "s0-b0-0",
            "method": "POST",
            "url": "/v1/completions",
            "body": {"model": "m", "prompt": "hello", "max_tokens": 16, "ignore_eos": False, "stream": False},
        },
        {
            "custom_id": "s0-b0-1",
            "method": "POST",
            "url": "/v1/completions",
            "body": {"model": "m", "prompt": "world", "max_tokens": 16, "ignore_eos": False, "stream": False},
        },
    ]


# One chat record with the user message "hi" becomes one POST line to /v1/chat/completions with custom_id
# s1-b2-0. ignore_eos true reaches the body as given, and stream stays false.
async def test_chat_record_becomes_a_chat_completions_line() -> None:
    batch_input = await build_batch_input(
        [ChatCompletionAPIData(messages=[ChatMessage(role="user", content="hi")])],
        custom_id_prefix="s1-b2",
        api_config=NO_API_OPTIONS,
        model_name="m",
        max_tokens=16,
        ignore_eos=True,
    )

    assert batch_input.endpoint == "/v1/chat/completions"
    assert batch_input.custom_ids == ["s1-b2-0"]
    assert [json.loads(line) for line in batch_input.content.splitlines()] == [
        {
            "custom_id": "s1-b2-0",
            "method": "POST",
            "url": "/v1/chat/completions",
            "body": {
                "model": "m",
                "messages": [{"role": "user", "content": "hi"}],
                "max_tokens": 16,
                "ignore_eos": True,
                "stream": False,
            },
        },
    ]


# A chat record whose tool schema is {"type": "object", "title": "F"}, written for a run with
# api.clean_tool_schemas on and api.response_format json_object, gets what the HTTP path would send: the
# schema loses "title" and gains "properties": {}, and the body carries "response_format": {"type": "json_object"}.
async def test_lines_get_the_api_options_the_http_path_adds() -> None:
    record = ChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="hi")],
        tool_definitions=[{"type": "function", "name": "f", "parameters": {"type": "object", "title": "F"}}],
    )
    api_config = APIConfig(
        type=APIType.Chat, clean_tool_schemas=True, response_format=ResponseFormat(type=ResponseFormatType.JSON_OBJECT)
    )

    batch_input = await build_batch_input(
        [record], custom_id_prefix="s0-b0", api_config=api_config, model_name="m", max_tokens=16, ignore_eos=False
    )

    (line,) = [json.loads(line) for line in batch_input.content.splitlines()]
    assert line["body"]["tools"][0]["function"]["parameters"] == {"type": "object", "properties": {}}
    assert line["body"]["response_format"] == {"type": "json_object"}


# A completion record followed by a chat record cannot go in one batch, because a batch runs against
# one endpoint: this raises ValueError naming the second route.
async def test_records_with_different_routes_are_rejected() -> None:
    records = [CompletionAPIData(prompt="hello"), ChatCompletionAPIData(messages=[ChatMessage(role="user", content="hi")])]

    with pytest.raises(ValueError, match="/v1/chat/completions"):
        await build_batch_input(
            records, custom_id_prefix="s0-b0", api_config=NO_API_OPTIONS, model_name="m", max_tokens=16, ignore_eos=False
        )


# Rounds 0 and 1 of user session "u0" (shared_prefix and conversation_replay build these): round 1's prompt
# waits for round 0's response, which a batch does not return until it ends, so the writer raises ValueError
# naming the session instead of waiting forever.
@pytest.mark.usefixtures("user_session_registry")
async def test_user_session_turns_are_rejected() -> None:
    records = [
        UserSessionCompletionAPIData(prompt="turn one", user_session_id="u0", target_round=0),
        UserSessionCompletionAPIData(prompt="turn two", user_session_id="u0", target_round=1),
    ]

    with pytest.raises(ValueError, match="u0"):
        batch = build_batch_input(
            records, custom_id_prefix="s0-b0", api_config=NO_API_OPTIONS, model_name="m", max_tokens=16, ignore_eos=False
        )
        await asyncio.wait_for(batch, timeout=5)


# Trace replay event session_1:event_1 lists session_1:event_0 as its predecessor. The load generator fills its
# prompt from event_0's response through wait_for_predecessors_and_substitute, which a batch never calls, so the
# writer raises ValueError for record 0 instead of writing the recorded placeholder text.
async def test_trace_replay_events_are_rejected() -> None:
    record = SessionChatCompletionAPIData(
        messages=[ChatMessage(role="user", content="hi")],
        event_id="session_1:event_1",
        registry=EventOutputRegistry(),
        worker_tracker=WorkerSessionTracker(),
        completion_queue=None,
        total_events_in_session=2,
        predecessor_event_ids=["session_1:event_0"],
    )

    with pytest.raises(ValueError, match="Record 0"):
        await build_batch_input(
            [record], custom_id_prefix="s0-b0", api_config=NO_API_OPTIONS, model_name="m", max_tokens=16, ignore_eos=False
        )


# No records means no endpoint to batch against, so an empty list raises ValueError.
async def test_empty_record_list_is_rejected() -> None:
    with pytest.raises(ValueError):
        await build_batch_input(
            [], custom_id_prefix="s0-b0", api_config=NO_API_OPTIONS, model_name="m", max_tokens=16, ignore_eos=False
        )
