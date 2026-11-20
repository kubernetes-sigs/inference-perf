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


# A completion record followed by a chat record cannot go in one batch, because a batch runs against
# one endpoint: this raises ValueError naming the second route.
async def test_records_with_different_routes_are_rejected() -> None:
    records = [CompletionAPIData(prompt="hello"), ChatCompletionAPIData(messages=[ChatMessage(role="user", content="hi")])]

    with pytest.raises(ValueError, match="/v1/chat/completions"):
        await build_batch_input(records, custom_id_prefix="s0-b0", model_name="m", max_tokens=16, ignore_eos=False)


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
        batch = build_batch_input(records, custom_id_prefix="s0-b0", model_name="m", max_tokens=16, ignore_eos=False)
        await asyncio.wait_for(batch, timeout=5)


# No records means no endpoint to batch against, so an empty list raises ValueError.
async def test_empty_record_list_is_rejected() -> None:
    with pytest.raises(ValueError):
        await build_batch_input([], custom_id_prefix="s0-b0", model_name="m", max_tokens=16, ignore_eos=False)
