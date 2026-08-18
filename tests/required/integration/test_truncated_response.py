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

"""Integration tests for issue #655 (#606 Integration tier, per-change lane).

A stream the server closes cleanly after a handful of tokens is a well-formed
200 and, before #655, indistinguishable from a legitimately short completion:
no stream break, no token-count mismatch, no error, so it landed in the success
bucket and quietly depressed the output-length distribution.

These tests drive the real client against a fake that returns a complete,
well-formed 200 carrying fewer tokens than the request's ``max_tokens``, with an
honest ``finish_reason`` and ``usage``, and assert:

- with ``ignore_eos`` on (the config default) the request is recorded as a
  ``TruncatedResponseError`` failure with the parsed metrics and body kept;
- with ``ignore_eos`` off it stays a success and the report shows it as a
  ``finish_reasons`` bucket plus one ``output_shortfalls``;
- a response that delivers the full budget is a success either way.

No model server needed.
"""

import json
from typing import Any, Dict, List

import pytest

from fake_truncating_server import TruncatingSSEServer
from openai_client_harness import generate_reports, run_request_against

from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.config import APIConfig, APIType

MAX_TOKENS = 16


# The SSE frames of a completion stream that stops after `words` content words
# with the given finish_reason, followed by a usage frame that reports exactly
# that many completion_tokens (the harness tokenizer also counts words, so the
# client and server counts agree unless a test breaks that on purpose).
def short_completion_stream(words: List[str], finish_reason: str, completion_tokens: int | None = None) -> List[str]:
    frames = [json.dumps({"choices": [{"index": 0, "text": f"{w} ", "finish_reason": None}]}) for w in words[:-1]] + [
        json.dumps({"choices": [{"index": 0, "text": words[-1], "finish_reason": finish_reason}]})
    ]
    usage = {"prompt_tokens": 4, "completion_tokens": completion_tokens if completion_tokens is not None else len(words)}
    frames.append(json.dumps({"choices": [], "usage": usage}))
    frames.append("[DONE]")
    return frames


# A unary chat body whose single choice carries `content`, `finish_reason` and
# a usage block reporting `completion_tokens`.
def unary_chat_body(content: str, finish_reason: str, completion_tokens: int) -> str:
    return json.dumps(
        {
            "choices": [{"index": 0, "message": {"role": "assistant", "content": content}, "finish_reason": finish_reason}],
            "usage": {"prompt_tokens": 4, "completion_tokens": completion_tokens, "total_tokens": 4 + completion_tokens},
        }
    )


# Streaming completion, ignore_eos on, max_tokens 16: the fake sends 3 words then
# finish_reason "stop" and usage completion_tokens 3. Must be recorded as a
# failure of type TruncatedResponseError whose message names 3 of 16 and "stop",
# with the parsed metrics (finish_reason "stop", 3 delivered tokens), max_tokens
# 16 and the whole SSE body still on the record.
@pytest.mark.asyncio
async def test_streaming_short_stop_under_ignore_eos_is_a_truncation_failure() -> None:
    async with TruncatingSSEServer(short_completion_stream(["one", "two", "three"], "stop"), missing_bytes=0) as server:
        metric = await run_request_against(server.base_url, ignore_eos=True)

    assert metric.error is not None
    assert metric.error.error_type == "TruncatedResponseError"
    assert "delivered 3 of 16" in metric.error.error_msg
    assert "finish_reason=stop" in metric.error.error_msg
    assert metric.max_tokens == MAX_TOKENS
    assert metric.info.response_metrics is not None
    assert metric.info.response_metrics.finish_reason == "stop"
    assert metric.info.response_metrics.delivered_output_tokens() == 3
    assert metric.response_data == server.sent_body


# Same short stream, ignore_eos off: a model emitting EOS early is legitimate, so
# the request must stay a success (error None) with finish_reason "stop" recorded,
# and the report must show finish_reasons {"stop": 1} and output_shortfalls 1.
@pytest.mark.asyncio
async def test_streaming_short_stop_without_ignore_eos_is_a_success_with_a_shortfall() -> None:
    async with TruncatingSSEServer(short_completion_stream(["one", "two", "three"], "stop"), missing_bytes=0) as server:
        metric = await run_request_against(server.base_url, ignore_eos=False)

    assert metric.error is None
    assert metric.info.response_metrics is not None
    assert metric.info.response_metrics.finish_reason == "stop"

    summary, entries = await generate_reports([metric])
    assert summary["successes"]["count"] == 1
    assert summary["successes"]["finish_reasons"] == {"stop": 1}
    assert summary["successes"]["output_shortfalls"] == 1
    assert summary["failures"]["count"] == 0
    assert entries[0]["max_tokens"] == MAX_TOKENS


# Streaming completion that delivers exactly max_tokens (16 words, finish_reason
# "length", usage 16), ignore_eos on: the budget was met, so it must be a success
# with no shortfall and finish_reasons {"length": 1}.
@pytest.mark.asyncio
async def test_streaming_full_budget_under_ignore_eos_is_a_success() -> None:
    words = [f"w{i}" for i in range(MAX_TOKENS)]
    async with TruncatingSSEServer(short_completion_stream(words, "length"), missing_bytes=0) as server:
        metric = await run_request_against(server.base_url, ignore_eos=True)

    assert metric.error is None
    assert metric.info.response_metrics is not None
    assert metric.info.response_metrics.finish_reason == "length"

    summary, _ = await generate_reports([metric])
    assert summary["successes"]["finish_reasons"] == {"length": 1}
    assert summary["successes"]["output_shortfalls"] == 0


# The server says finish_reason "length" but its usage reports only 10 of the 16
# requested tokens (the shape of a max_model_len cap): under ignore_eos the
# requested budget was still not delivered, so it is a TruncatedResponseError
# whose message says "finish_reason=length".
@pytest.mark.asyncio
async def test_streaming_capped_length_under_ignore_eos_is_still_a_truncation() -> None:
    words = [f"w{i}" for i in range(10)]
    async with TruncatingSSEServer(short_completion_stream(words, "length"), missing_bytes=0) as server:
        metric = await run_request_against(server.base_url, ignore_eos=True)

    assert metric.error is not None
    assert metric.error.error_type == "TruncatedResponseError"
    assert "delivered 10 of 16" in metric.error.error_msg
    assert "finish_reason=length" in metric.error.error_msg


# The server's usage is what counts: the fake streams 16 words (client count 16)
# but reports completion_tokens 5. Under ignore_eos the server's own count wins,
# so this is a truncation of 5 of 16.
@pytest.mark.asyncio
async def test_server_completion_tokens_take_precedence_over_the_client_count() -> None:
    words = [f"w{i}" for i in range(MAX_TOKENS)]
    async with TruncatingSSEServer(short_completion_stream(words, "stop", completion_tokens=5), missing_bytes=0) as server:
        metric = await run_request_against(server.base_url, ignore_eos=True)

    assert metric.error is not None
    assert metric.error.error_type == "TruncatedResponseError"
    assert "delivered 5 of 16" in metric.error.error_msg


# Unary chat, ignore_eos on: a 200 body with a 2-word answer, finish_reason
# "stop" and usage completion_tokens 2 against max_tokens 16 is a
# TruncatedResponseError; the same body with ignore_eos off is a success whose
# report shows finish_reasons {"stop": 1} and output_shortfalls 1.
@pytest.mark.asyncio
async def test_unary_chat_short_stop_follows_ignore_eos() -> None:
    api_config = APIConfig(type=APIType.Chat, streaming=False)
    body = unary_chat_body("Paris.", "stop", 2)

    def chat_request() -> ChatCompletionAPIData:
        return ChatCompletionAPIData(messages=[ChatMessage(role="user", content="capital of France?")], max_tokens=16)

    async with TruncatingSSEServer([], missing_bytes=0, body=body, content_type="application/json") as server:
        truncated = await run_request_against(server.base_url, api_config, chat_request(), ignore_eos=True)
    async with TruncatingSSEServer([], missing_bytes=0, body=body, content_type="application/json") as server:
        accepted = await run_request_against(server.base_url, api_config, chat_request(), ignore_eos=False)

    assert truncated.error is not None
    assert truncated.error.error_type == "TruncatedResponseError"
    assert "delivered 2 of 16" in truncated.error.error_msg
    assert truncated.response_data == body

    assert accepted.error is None
    summary, _ = await generate_reports([accepted])
    assert summary["successes"]["finish_reasons"] == {"stop": 1}
    assert summary["successes"]["output_shortfalls"] == 1


# One truncated request and one full-budget request through the report, ignore_eos
# on: the failure bucket shows by_label {"truncatedresponseerror": {count 1, the
# delivered-3-of-16 message}} and the success bucket shows finish_reasons
# {"length": 1} with output_shortfalls 0; the per-request entry for the failure
# still carries its response and its max_tokens.
@pytest.mark.asyncio
async def test_truncation_lands_in_the_failure_bucket_of_the_report() -> None:
    async with TruncatingSSEServer(short_completion_stream(["one", "two", "three"], "stop"), missing_bytes=0) as server:
        truncated = await run_request_against(server.base_url, ignore_eos=True)
    full = [f"w{i}" for i in range(MAX_TOKENS)]
    async with TruncatingSSEServer(short_completion_stream(full, "length"), missing_bytes=0) as server:
        complete = await run_request_against(server.base_url, ignore_eos=True)

    summary, entries = await generate_reports([truncated, complete])

    assert summary["successes"]["count"] == 1
    assert summary["successes"]["finish_reasons"] == {"length": 1}
    assert summary["successes"]["output_shortfalls"] == 0
    assert summary["failures"]["count"] == 1
    by_label: Dict[str, Any] = summary["failures"]["by_label"]
    assert list(by_label) == ["truncatedresponseerror"]
    assert by_label["truncatedresponseerror"]["count"] == 1
    assert "delivered 3 of 16" in by_label["truncatedresponseerror"]["messages"][0]["message"]

    failed_entry = next(e for e in entries if e["error"] is not None)
    assert failed_entry["max_tokens"] == MAX_TOKENS
    assert failed_entry["response"]
    assert failed_entry["info"]["response_metrics"]["finish_reason"] == "stop"


# A request the API layer produced no response_metrics for (a 200 with usage but
# an empty choices list keeps its early return from #713) has nothing to compare
# against max_tokens, so under ignore_eos it is neither a truncation nor a
# shortfall: error None and output_shortfalls 0.
@pytest.mark.asyncio
async def test_no_response_metrics_is_not_judged() -> None:
    body = json.dumps({"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 0}})
    async with TruncatingSSEServer([], missing_bytes=0, body=body, content_type="application/json") as server:
        metric = await run_request_against(
            server.base_url,
            APIConfig(type=APIType.Completion, streaming=False),
            CompletionAPIData(prompt="hi", max_tokens=16),
            ignore_eos=True,
        )

    assert metric.error is None
    assert metric.info.response_metrics is None
    summary, _ = await generate_reports([metric])
    assert summary["successes"]["output_shortfalls"] == 0
    assert summary["successes"]["finish_reasons"] == {}
