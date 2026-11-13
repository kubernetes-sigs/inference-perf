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
"""Tests for the first-token hook: the stream parser reports a response's first token to the task that sent it."""

import asyncio
from typing import AsyncGenerator, List, Optional
from unittest.mock import Mock

from inference_perf.apis.request_phase import notify_first_token, watch_first_token
from inference_perf.apis.streaming_parser import parse_sse_stream


# A response whose body arrives as the given SSE frames, one per read.
def _response(frames: List[bytes]) -> Mock:
    async def iter_any() -> AsyncGenerator[bytes, None]:
        for frame in frames:
            yield frame

    response = Mock()
    response.content.iter_any = iter_any
    return response


# One string field of an OpenAI chat delta, or None when the chunk has no choices or the field is absent.
# ({"choices": [{"delta": {"content": "Hi"}}]}, "content") -> "Hi".
def _delta_field(data: dict[str, object], field: str) -> Optional[str]:
    choices = data.get("choices")
    if not isinstance(choices, list) or not choices:
        return None
    delta = choices[0].get("delta")
    value = delta.get(field) if isinstance(delta, dict) else None
    return value if isinstance(value, str) else None


# The content of a chat delta, e.g. {"choices": [{"delta": {"content": "Hi"}}]} -> "Hi".
def _content(data: dict[str, object]) -> Optional[str]:
    return _delta_field(data, "content")


# The reasoning text of a chat delta, e.g. {"choices": [{"delta": {"reasoning_content": "hm"}}]} -> "hm".
def _reasoning(data: dict[str, object]) -> Optional[str]:
    return _delta_field(data, "reasoning_content")


# Parses the frames in a task of its own that watches for the first token. Returns how many times it was reported.
async def _first_tokens_seen(frames: List[bytes], reasoning: bool = False) -> int:
    seen = 0

    async def send() -> None:
        def first_token() -> None:
            nonlocal seen
            seen += 1

        watch_first_token(first_token)
        await parse_sse_stream(_response(frames), _content, _reasoning if reasoning else None)

    await asyncio.create_task(send())
    return seen


# A role-only delta, two content deltas, a usage chunk and [DONE]. Expects the first token reported once,
# on the first content delta, not once per chunk.
async def test_first_content_chunk_is_reported_once() -> None:
    frames = [
        b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n',
        b'data: {"choices": [], "usage": {"completion_tokens": 2}}\n\n',
        b"data: [DONE]\n\n",
    ]
    assert await _first_tokens_seen(frames) == 1


# A stream with a role-only delta, a usage chunk and [DONE] generates nothing. Expects no first token reported.
async def test_stream_without_generated_chunks_reports_nothing() -> None:
    frames = [
        b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n',
        b'data: {"choices": [], "usage": {"completion_tokens": 0}}\n\n',
        b"data: [DONE]\n\n",
    ]
    assert await _first_tokens_seen(frames) == 0


# A stream that opens with a reasoning delta. Expects the first token reported on it, since TTFT counts
# reasoning tokens, and not again on the content delta that follows.
async def test_reasoning_chunk_counts_as_the_first_token() -> None:
    frames = [
        b'data: {"choices": [{"delta": {"reasoning_content": "hm"}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n',
    ]
    assert await _first_tokens_seen(frames, reasoning=True) == 1


# A stream whose only generated chunk is a tool call argument delta. Expects the first token reported on it.
async def test_tool_chunk_counts_as_the_first_token() -> None:
    frames = [
        b'data: {"choices": [{"delta": {"tool_calls": [{"index": 0, "function": {"arguments": "{\\"a\\""}}]}}]}\n\n',
    ]
    assert await _first_tokens_seen(frames) == 1


# A stream parsed with nobody watching. Expects it to parse as usual.
async def test_no_watcher_is_harmless() -> None:
    frames = [b'data: {"choices": [{"delta": {"content": "Hi"}}]}\n\n']
    parsed = await parse_sse_stream(_response(frames), _content)
    assert parsed.output_text == "Hi"


# Two tasks each watch for their own first token and only task a's response produces one.
# Expects a's callback called and b's not: a callback is seen only by the task that set it.
async def test_a_watcher_sees_only_its_own_task() -> None:
    seen: List[str] = []
    both_watching = asyncio.Event()
    watching = 0

    async def send(name: str, generates: bool) -> None:
        nonlocal watching
        watch_first_token(lambda: seen.append(name))
        watching += 1
        if watching == 2:
            both_watching.set()
        await both_watching.wait()
        if generates:
            notify_first_token()

    await asyncio.gather(asyncio.create_task(send("a", True)), asyncio.create_task(send("b", False)))
    assert seen == ["a"]
