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
from __future__ import annotations

import json
from collections.abc import AsyncIterator
from types import SimpleNamespace
from typing import Any, cast

import pytest
from aiohttp import ClientResponse
from inference_perf.apis import streaming_parser
from inference_perf.apis.anthropic_messages import AnthropicMessagesAPIData
from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.apis.base import InferenceInfo, RequestLifecycleMetric, StreamedResponseMetrics
from inference_perf.config import APIConfig, APIType
from inference_perf.datagen.replay.replay_graph_session_datagen import (
    EventOutputRegistry,
    SessionInferenceInfo,
    WorkerSessionTracker,
    SessionChatCompletionAPIData,
    SessionAnthropicMessagesAPIData,
)
from inference_perf.reportgen.base import retokenize_output_token_times, compute_request_latency_metrics, summarize_requests
from inference_perf.utils.custom_tokenizer import CustomTokenizer


class CharacterTokenizer:
    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        return len(text) + int(add_special_tokens)


class Response:
    def __init__(self, payload: dict[str, Any], events: list[dict[str, Any]]) -> None:
        self.payload, self.events, self.content = payload, events, self

    async def json(self) -> dict[str, Any]:
        return self.payload

    async def iter_any(self) -> AsyncIterator[bytes]:
        for event in self.events:
            yield f"data: {json.dumps(event)}\n\n".encode()
        yield b"data: [DONE]\n\n"


def api(path: str) -> ChatCompletionAPIData | AnthropicMessagesAPIData:
    messages = [ChatMessage(role="user", content="use tools")]
    if path.endswith("replay"):
        cls = SessionAnthropicMessagesAPIData if path.startswith("messages") else SessionChatCompletionAPIData
        return cls(
            messages=messages,
            event_id="s:e",
            registry=EventOutputRegistry(),
            worker_tracker=WorkerSessionTracker(),
            completion_queue=None,
            total_events_in_session=1,
            predecessor_event_ids=[],
        )
    return (AnthropicMessagesAPIData if path == "messages" else ChatCompletionAPIData)(messages=messages)


def response(path: str, mixed: bool, usage_present: bool) -> tuple[Response, int, list[float]]:
    args = ['{"city": "Paris"}', '{"unit": "C"}']
    text, reasoning = ("answer", "think") if mixed else ("", "")
    events: list[dict[str, Any]] = []
    times: list[float] = []

    def add(event: dict[str, Any], generated: str = "") -> None:
        events.append(event)
        times.extend([float(len(events))] * len(generated))

    if path.startswith("messages"):
        usage = {"input_tokens": 17, "output_tokens": 999}
        blocks: list[dict[str, Any]] = []
        for i, arg in enumerate(args):
            block = {"type": "tool_use", "id": f"call-{i}", "name": "tool", "input": json.loads(arg)}
            blocks.append(block)
            add({"type": "content_block_start", "index": i, "content_block": {**block, "input": {}}})
        for part in range(2):
            for i, arg in enumerate(args):
                fragment = arg[:5] if part == 0 else arg[5:]
                add(
                    {
                        "type": "content_block_delta",
                        "index": i,
                        "delta": {"type": "input_json_delta", "partial_json": fragment},
                    },
                    fragment,
                )
        if mixed:
            blocks += [{"type": "thinking", "thinking": reasoning}, {"type": "text", "text": text}]
            add(
                {"type": "content_block_delta", "index": 2, "delta": {"type": "thinking_delta", "thinking": reasoning}},
                reasoning,
            )
            add({"type": "content_block_delta", "index": 3, "delta": {"type": "text_delta", "text": text}}, text)
        payload: dict[str, Any] = {"content": blocks}
        if usage_present:
            add({"type": "message_delta", "usage": usage})
    else:
        usage = {"prompt_tokens": 17, "completion_tokens": 999}
        calls: list[dict[str, Any]] = [
            {"id": f"call-{i}", "type": "function", "function": {"name": "tool", "arguments": arg}}
            for i, arg in enumerate(args)
        ]
        payload = {"choices": [{"message": {"content": text or None, "reasoning_content": reasoning, "tool_calls": calls}}]}
        for i, call in enumerate(calls):
            add(
                {"choices": [{"delta": {"tool_calls": [{**call, "index": i, "function": {"name": "tool", "arguments": ""}}]}}]}
            )
        for part in range(2):
            for i, arg in enumerate(args):
                fragment = arg[:5] if part == 0 else arg[5:]
                add({"choices": [{"delta": {"tool_calls": [{"index": i, "function": {"arguments": fragment}}]}}]}, fragment)
        if mixed:
            add({"choices": [{"delta": {"content": text, "reasoning_content": reasoning}}]}, reasoning + text)
        if usage_present:
            add({"choices": [], "usage": usage})
    if usage_present:
        payload["usage"] = usage
    return Response(payload, events), sum(map(len, args)) + len(text) + len(reasoning), times


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["chat", "messages", "chat_replay", "messages_replay"])
@pytest.mark.parametrize("streaming", [False, True])
@pytest.mark.parametrize("mixed", [False, True])
@pytest.mark.parametrize("usage_present", [False, True])
async def test_tool_accounting(
    path: str, streaming: bool, mixed: bool, usage_present: bool, monkeypatch: pytest.MonkeyPatch
) -> None:
    wire, count, times = response(path, mixed, usage_present)
    clock = iter(float(i) for i in range(1, 100))
    monkeypatch.setattr(streaming_parser, "time", SimpleNamespace(perf_counter=lambda: next(clock)))
    tokenizer = cast(CustomTokenizer, CharacterTokenizer())
    info = await api(path).process_response(
        cast(ClientResponse, wire),
        APIConfig(type=APIType.AnthropicMessages if path.startswith("messages") else APIType.Chat, streaming=streaming),
        tokenizer,
    )
    assert info.response_metrics is not None
    assert info.response_metrics.output_tokens == count
    assert info.response_metrics.server_usage == wire.payload.get("usage")
    if streaming:
        assert isinstance(info.response_metrics, StreamedResponseMetrics)
        assert retokenize_output_token_times(info.response_metrics, tokenizer) == times
        info.response_metrics.output_token_times = times
        metric = RequestLifecycleMetric(scheduled_time=0, start_time=0, end_time=20, request_data="p", info=info, error=None)
        latency = compute_request_latency_metrics(metric)
        assert latency["time_to_first_token"] == 3
        expected_ttfo = (8 if path.startswith("messages") else 7) if mixed else None
        assert latency["time_to_first_output_token"] == expected_ttfo


def metric_for_chunks(
    times: list[float], fragments: list[str], server_count: int, tool: bool = True
) -> RequestLifecycleMetric:
    from inference_perf.payloads import RequestMetrics, Text

    chunks = [
        json.dumps(
            {
                "choices": [
                    {"delta": {"tool_calls": [{"index": 0, "function": {"arguments": part}}]} if tool else {"content": part}}
                ]
            }
        )
        for part in fragments
    ]
    metrics = StreamedResponseMetrics(
        output_tokens=sum(map(len, fragments)),
        server_usage={"completion_tokens": server_count},
        has_tool_calls=tool,
        tool_chunks=chunks if tool else [],
        tool_chunk_times=times if tool else [],
        response_chunks=[] if tool else chunks,
        chunk_times=[] if tool else times,
    )
    return RequestLifecycleMetric(
        scheduled_time=0,
        start_time=0,
        end_time=10,
        request_data="p",
        error=None,
        info=InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=1)), response_metrics=metrics),
    )


@pytest.mark.parametrize("times", [[2.0], [2.0, 2.0]])
@pytest.mark.parametrize("tool", [True, False])
def test_one_arrival_has_ttft_but_no_decode_intervals(times: list[float], tool: bool) -> None:
    from inference_perf.reportgen.base import correct_streamed_response_metrics

    metric = metric_for_chunks(times, ["x"] * len(times), 20, tool=tool)
    correct_streamed_response_metrics(metric, cast(CustomTokenizer, CharacterTokenizer()))
    latency = compute_request_latency_metrics(metric)
    assert latency["time_to_first_token"] == 2
    assert latency["time_to_first_output_token"] == (None if tool else 2)
    assert latency["time_per_output_token"] is None
    assert latency["inter_token_latency"] is None
    assert latency["inter_token_latency_deltas"] == []


@pytest.mark.parametrize("server_count, mismatch", [(3, 0), (4, 0), (5, 1), (0, 1)])
def test_tool_mismatch_tolerance_is_separate(server_count: int, mismatch: int) -> None:
    tool = metric_for_chunks([2, 3], ["ab", "c"], server_count)
    text = metric_for_chunks([2, 3], ["ab", "c"], 4, tool=False)
    result = summarize_requests([tool, text], [50], tokenizer=cast(CustomTokenizer, CharacterTokenizer()))
    assert result.successes["tool_token_count_mismatches"] == mismatch
    assert result.successes["token_count_mismatches"] == 1


def test_tool_mismatch_warning() -> None:
    from inference_perf.reportgen.summary.validation import token_mismatch_findings

    findings = token_mismatch_findings(
        "summary.json",
        {"successes": {"token_count_mismatches": 0, "tool_token_count_mismatches": 2}},
        "summary.token_mismatches",
    )
    assert len(findings) == 1
    assert "tool" in findings[0].message
    assert "25%" in findings[0].message


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["chat", "chat_replay"])
async def test_mixed_delta_with_multiple_calls_is_recorded_once(path: str, monkeypatch: pytest.MonkeyPatch) -> None:
    wire, count, _ = response(path, True, False)
    message = wire.payload["choices"][0]["message"]
    wire.events = [{"choices": [{"delta": message}]}]
    monkeypatch.setattr(streaming_parser, "time", SimpleNamespace(perf_counter=lambda: 2.0))
    tokenizer = cast(CustomTokenizer, CharacterTokenizer())
    info = await api(path).process_response(
        cast(ClientResponse, wire), APIConfig(type=APIType.Chat, streaming=True), tokenizer
    )
    assert isinstance(info.response_metrics, StreamedResponseMetrics)
    metrics = info.response_metrics
    assert metrics.output_tokens == count
    assert len(metrics.response_chunks) == 1
    assert not metrics.reasoning_chunks
    assert not metrics.tool_chunks
    assert retokenize_output_token_times(metrics, tokenizer) == [2.0] * count


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["messages", "messages_replay"])
@pytest.mark.parametrize("streaming", [False, True])
async def test_anthropic_nonempty_start_input_counts_once(path: str, streaming: bool) -> None:
    argument = {"city": "東京"}
    block = {"type": "tool_use", "id": "t", "name": "weather", "input": argument}
    wire = Response({"content": [block]}, [{"type": "content_block_start", "index": 0, "content_block": block}])
    tokenizer = cast(CustomTokenizer, CharacterTokenizer())
    info = await api(path).process_response(
        cast(ClientResponse, wire), APIConfig(type=APIType.AnthropicMessages, streaming=streaming), tokenizer
    )
    assert info.response_metrics is not None
    count = len(json.dumps(argument, ensure_ascii=False))
    assert info.response_metrics.output_tokens == count
    if streaming:
        assert isinstance(info.response_metrics, StreamedResponseMetrics)
        assert len(retokenize_output_token_times(info.response_metrics, tokenizer)) == count


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["chat", "messages", "chat_replay", "messages_replay"])
async def test_tool_only_http_benchmark(path: str) -> None:
    """Actual SSE transport through parsing and the summary report, without a model download."""
    import asyncio
    import time
    from aiohttp import ClientSession, web

    wire, count, _ = response(path, False, True)
    sent: list[float] = []

    async def serve(request: web.Request) -> web.StreamResponse:
        stream = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await stream.prepare(request)
        for event in wire.events:
            await asyncio.sleep(0.01)
            sent.append(time.perf_counter())
            await stream.write(f"data: {json.dumps(event)}\n\n".encode())
        await stream.write(b"data: [DONE]\n\n")
        await stream.write_eof()
        return stream

    app = web.Application()
    app.router.add_post("/", serve)
    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, "127.0.0.1", 0)
    try:
        await site.start()
        port = runner.addresses[0][1]
        tokenizer = cast(CustomTokenizer, CharacterTokenizer())
        async with ClientSession() as session:
            start = time.perf_counter()
            async with session.post(f"http://127.0.0.1:{port}/") as incoming:
                info = await api(path).process_response(
                    incoming,
                    APIConfig(type=APIType.AnthropicMessages if path.startswith("messages") else APIType.Chat, streaming=True),
                    tokenizer,
                )
            end = time.perf_counter()
        metric = RequestLifecycleMetric(
            scheduled_time=start, start_time=start, end_time=end, request_data="p", error=None, info=info
        )
        summary = summarize_requests([metric], [50], tokenizer=tokenizer)
        assert info.response_metrics is not None
        assert info.response_metrics.output_tokens == count
        latency = compute_request_latency_metrics(metric)
        assert sent[2] - start <= latency["time_to_first_token"] < sent[-1] - start
        assert latency["time_to_first_output_token"] is None
        assert latency["time_per_output_token"] > 0
        assert latency["inter_token_latency"] > 0
        assert summary.successes["token_count_mismatches"] == 0
        assert summary.successes["tool_token_count_mismatches"] == 1
    finally:
        await runner.cleanup()


@pytest.mark.parametrize("times, expected_tpot", [([2.0], None), ([2.0, 2.0], None), ([2.0, 3.0], 1 / 3)])
@pytest.mark.parametrize("tool", [True, False])
def test_benchmark_report_matches_native_tool_timing(times: list[float], expected_tpot: float | None, tool: bool) -> None:
    from inference_perf.reportgen.br.v0_2.adapter import _per_request_latencies

    metric = metric_for_chunks(times, ["{}"] * len(times), 20, tool=tool)
    _, ttft, tpot, itl, _ = _per_request_latencies([metric], cast(CustomTokenizer, CharacterTokenizer()), False)
    assert ttft == [2.0]
    assert tpot == [expected_tpot]
    assert bool(itl) == (expected_tpot is not None)


class JoiningTokenizer:
    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        return (1 if text in ("ab", "cd", "abcd") else len(text)) + int(add_special_tokens)


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["chat", "chat_replay"])
async def test_assemble_fragments_but_keep_calls_separate(path: str) -> None:
    events = [
        {"choices": [{"delta": {"tool_calls": [{"index": i, "function": {"arguments": part}}]}}]}
        for i, part in [(0, "a"), (1, "c"), (0, "b"), (1, "d")]
    ]
    info = await api(path).process_response(
        cast(ClientResponse, Response({}, events)),
        APIConfig(type=APIType.Chat, streaming=True),
        cast(CustomTokenizer, JoiningTokenizer()),
    )
    assert isinstance(info.response_metrics, StreamedResponseMetrics)
    assert info.response_metrics.output_tokens == 2
    assert len(retokenize_output_token_times(info.response_metrics, cast(CustomTokenizer, JoiningTokenizer()))) == 4


@pytest.mark.asyncio
@pytest.mark.parametrize("path", ["chat", "chat_replay"])
@pytest.mark.parametrize("streaming", [True, False])
@pytest.mark.parametrize("bad_call", [None, {"function": "x"}, {"index": [], "function": {}}, {"function": {"arguments": 42}}])
async def test_malformed_tool_entries_preserve_valid_sibling_output(path: str, streaming: bool, bad_call: Any) -> None:
    message = {
        "content": "answer",
        "reasoning_content": "think",
        "tool_calls": [bad_call, {"index": 1, "function": {"name": "ok", "arguments": "{}"}}],
    }
    wire = Response({"choices": [{"message": message}]}, [{"choices": [{"delta": message}]}])
    tokenizer = cast(CustomTokenizer, CharacterTokenizer())
    info = await api(path).process_response(
        cast(ClientResponse, wire), APIConfig(type=APIType.Chat, streaming=streaming), tokenizer
    )
    assert info.response_metrics is not None
    assert info.response_metrics.output_tokens == len("answerthink{}")
    if streaming:
        assert isinstance(info.response_metrics, StreamedResponseMetrics)
        assert len(retokenize_output_token_times(info.response_metrics, tokenizer)) == len("answerthink{}")
    if path == "chat_replay":
        assert isinstance(info, SessionInferenceInfo)
        assert info.output_message is not None
        assert info.output_message["tool_calls"][-1]["function"]["arguments"] == "{}"


@pytest.mark.asyncio
async def test_replay_message_and_count_share_missing_index_fallback() -> None:
    fragments = [("a", "c"), ("b", "d")]
    events = [
        {
            "choices": [
                {
                    "delta": {
                        "tool_calls": [{"id": f"call-{i}", "function": {"arguments": part}} for i, part in enumerate(parts)]
                    }
                }
            ]
        }
        for parts in fragments
    ]
    info = await api("chat_replay").process_response(
        cast(ClientResponse, Response({}, events)),
        APIConfig(type=APIType.Chat, streaming=True),
        cast(CustomTokenizer, JoiningTokenizer()),
    )
    assert isinstance(info, SessionInferenceInfo)
    assert info.output_message is not None
    calls = info.output_message["tool_calls"]
    assert [call["function"]["arguments"] for call in calls] == ["ab", "cd"]
    assert [call["id"] for call in calls] == ["call-0", "call-1"]
    assert info.response_metrics is not None
    assert info.response_metrics.output_tokens == 2


@pytest.mark.parametrize(
    "data",
    [
        {"choices": [None]},
        {"choices": [{"delta": {"tool_calls": "bad"}}]},
        {"type": "content_block_start", "content_block": "bad"},
        {"type": "content_block_delta", "delta": "bad"},
        {"type": "content_block_delta", "index": [], "delta": {"type": "input_json_delta"}},
    ],
)
def test_malformed_tool_envelopes_have_no_argument_fragments(data: dict[str, Any]) -> None:
    from inference_perf.apis.tool_output import tool_argument_deltas

    assert tool_argument_deltas(data) == []
