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
"""Unit tests for SGLang's native /generate route: both streaming modes, the request body and the response parse."""

import json
from typing import AsyncGenerator, Callable, Dict, List, Mapping
from unittest.mock import AsyncMock, MagicMock

import pytest

from inference_perf.apis import RequestLifecycleMetric, SGLangGenerateAPIData, StreamedResponseMetrics, UnaryResponseMetrics
from inference_perf.apis.response_errors import EmptyResponseError, InBandError
from inference_perf.apis.sglang_generate import parse_stream, server_usage, stream_state
from inference_perf.config import APIConfig, APIType
from inference_perf.reportgen.base import compute_request_latency_metrics, correct_streamed_response_metrics

TOKENS = [f" w{i}" for i in range(64)]
FINAL_TEXT = "".join(TOKENS)
STREAMING = APIConfig(type=APIType.SGLangGenerate, streaming=True)
UNARY = APIConfig(type=APIType.SGLangGenerate, streaming=False)


# SGLang's /generate SSE body for TOKENS, `per_chunk` tokens per chunk, ending in [DONE]. Cumulative (the
# default): each chunk has all text and output_ids so far. Incremental (--incremental-streaming-output): only
# the new ones. completion_tokens is the running total in both; prompt_tokens is 10, cached_tokens 4.
def _sglang_stream(incremental: bool, per_chunk: int = 1) -> List[bytes]:
    chunks = []
    for end in range(per_chunk, len(TOKENS) + 1, per_chunk):
        start = end - per_chunk if incremental else 0
        body = {
            "text": "".join(TOKENS[start:end]),
            "output_ids": list(range(start, end)),
            "meta_info": {"prompt_tokens": 10, "completion_tokens": end, "cached_tokens": 4},
        }
        chunks.append(b"data: " + json.dumps(body).encode() + b"\n\n")
    return chunks + [b"data: [DONE]\n\n"]


# A streamed response whose body arrives as `chunks`.
def _streamed(chunks: List[bytes]) -> MagicMock:
    async def iter_any() -> AsyncGenerator[bytes, None]:
        for chunk in chunks:
            yield chunk

    response = MagicMock()
    response.content.iter_any = iter_any
    return response


# A unary response whose JSON body is `body`.
def _unary(body: Mapping[str, object]) -> MagicMock:
    response = MagicMock()
    response.json = AsyncMock(return_value=body)
    return response


# Counts whitespace-separated words, so "a b c" -> 3, with or without special tokens.
def _tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.count_tokens = lambda text, **kwargs: len(text.split())
    return tokenizer


# The type is sglang_generate and the route is SGLang's /generate.
def test_api_type_and_route() -> None:
    data = SGLangGenerateAPIData(prompt="hi")

    assert data.get_api_type() == APIType.SGLangGenerate
    assert data.get_route() == "/generate"


# Prompt "hi", the client's 16 max tokens, ignore_eos, streaming -> text + sampling_params, no model field.
async def test_request_body_takes_the_clients_max_tokens() -> None:
    body = await SGLangGenerateAPIData(prompt="hi").to_request_body("my-model", 16, ignore_eos=True, streaming=True)

    assert body == {"text": "hi", "sampling_params": {"max_new_tokens": 16, "ignore_eos": True}, "stream": True}


# A request with its own 7 max tokens keeps them over the client's 16.
async def test_request_body_keeps_the_requests_own_max_tokens() -> None:
    body = await SGLangGenerateAPIData(prompt="hi", max_tokens=7).to_request_body("my-model", 16, False, False)

    assert body["sampling_params"] == {"max_new_tokens": 7, "ignore_eos": False}
    assert body["stream"] is False


# 64 tokens, cumulative, one per chunk -> the 64 tokens once over 64 content chunks, one token time each.
# Without the per-request state each chunk's full text is kept: 2,080 words, every prefix concatenated.
async def test_cumulative_stream_yields_each_token_once() -> None:
    data = SGLangGenerateAPIData(prompt="p")

    info = await data.process_response(_streamed(_sglang_stream(incremental=False)), STREAMING, _tokenizer())

    assert isinstance(info.response_metrics, StreamedResponseMetrics)
    assert data.model_response == FINAL_TEXT
    assert info.response_metrics.output_tokens == 64
    assert len(info.response_metrics.chunk_times) == 64
    assert info.response_metrics.output_token_times == info.response_metrics.chunk_times


# 64 tokens, incremental (--incremental-streaming-output), one per chunk -> the same text over 64 chunks.
async def test_incremental_stream_yields_each_token_once() -> None:
    data = SGLangGenerateAPIData(prompt="p")

    info = await data.process_response(_streamed(_sglang_stream(incremental=True)), STREAMING, _tokenizer())

    assert isinstance(info.response_metrics, StreamedResponseMetrics)
    assert data.model_response == FINAL_TEXT
    assert len(info.response_metrics.chunk_times) == 64


# 64 tokens, 4 per chunk, in each mode -> the 64 tokens once over 16 content chunks.
@pytest.mark.parametrize("incremental", [False, True])
async def test_multi_token_chunks_in_both_modes(incremental: bool) -> None:
    data = SGLangGenerateAPIData(prompt="p")

    info = await data.process_response(_streamed(_sglang_stream(incremental, per_chunk=4)), STREAMING, _tokenizer())

    assert isinstance(info.response_metrics, StreamedResponseMetrics)
    assert data.model_response == FINAL_TEXT
    assert len(info.response_metrics.chunk_times) == 16


# 64 tokens, one per chunk, then a chunk with no new text, in each mode, through the report's token timeline ->
# each content chunk stored as its new text (one per chunk time), 64 token times, no client/server count mismatch and a TTFT. Stored raw, a cumulative stream's chunks
# re-tokenize to 2,080 tokens.
@pytest.mark.parametrize("incremental", [False, True], ids=["cumulative", "incremental"])
async def test_stream_feeds_the_reports_token_timeline(incremental: bool) -> None:
    data = SGLangGenerateAPIData(prompt="p")
    # A last chunk with no new text, as a cumulative stream repeats its final chunk.
    chunks = _sglang_stream(incremental)
    chunks.insert(-1, b'data: {"text": "", "output_ids": [], "meta_info": {"completion_tokens": 64}}\n\n')
    info = await data.process_response(_streamed(chunks), STREAMING, _tokenizer())
    metric = RequestLifecycleMetric(scheduled_time=0.0, start_time=0.0, end_time=9.0, request_data="", info=info, error=None)

    mismatch = correct_streamed_response_metrics(metric, _tokenizer())

    assert isinstance(info.response_metrics, StreamedResponseMetrics)
    assert info.response_metrics.response_chunks[:2] == ['{"text": " w0"}', '{"text": " w1"}']
    assert len(info.response_metrics.response_chunks) == len(info.response_metrics.chunk_times) == 64
    assert mismatch is False
    assert len(info.response_metrics.output_token_times) == 64
    assert compute_request_latency_metrics(metric)["time_to_first_token"] is not None


# A stream whose last meta_info says 10 prompt, 64 completion and 4 cached tokens -> those counts, as usage.
async def test_stream_token_counts_come_from_meta_info() -> None:
    info = await SGLangGenerateAPIData(prompt="a b").process_response(
        _streamed(_sglang_stream(incremental=False)), STREAMING, _tokenizer()
    )

    assert info.request_metrics.text is not None and info.request_metrics.text.input_tokens == 10
    assert isinstance(info.response_metrics, StreamedResponseMetrics)
    assert info.response_metrics.server_usage == {
        "prompt_tokens": 10,
        "completion_tokens": 64,
        "prompt_tokens_details": {"cached_tokens": 4},
    }


# Chunks "a", "a b" with no meta_info -> text " b" read cumulatively, prompt "x y z" counted client-side as 3.
async def test_stream_without_meta_info_counts_the_prompt_client_side() -> None:
    chunks = [b'data: {"text": "a"}\n\n', b'data: {"text": "a b"}\n\n', b"data: [DONE]\n\n"]
    data = SGLangGenerateAPIData(prompt="x y z")

    info = await data.process_response(_streamed(chunks), STREAMING, _tokenizer())

    assert data.model_response == "a b"
    assert info.request_metrics.text is not None and info.request_metrics.text.input_tokens == 3
    assert isinstance(info.response_metrics, StreamedResponseMetrics)
    assert info.response_metrics.server_usage is None


# A stream of only [DONE] carries no text and no counts -> EmptyResponseError, not a zero-token success.
async def test_empty_stream_is_an_error() -> None:
    with pytest.raises(EmptyResponseError):
        await SGLangGenerateAPIData(prompt="p").process_response(_streamed([b"data: [DONE]\n\n"]), STREAMING, _tokenizer())


# A stream with no text but a meta_info count (max_new_tokens 0) -> a zero-token success.
async def test_stream_with_counts_but_no_text_is_not_an_error() -> None:
    chunks = [b'data: {"text": "", "output_ids": [], "meta_info": {"prompt_tokens": 5}}\n\n', b"data: [DONE]\n\n"]

    info = await SGLangGenerateAPIData(prompt="p").process_response(_streamed(chunks), STREAMING, _tokenizer())

    assert info.request_metrics.text is not None and info.request_metrics.text.input_tokens == 5
    assert isinstance(info.response_metrics, StreamedResponseMetrics)
    assert info.response_metrics.output_tokens == 0


# The error frame SGLang streams when generation fails mid-request -> InBandError.
async def test_stream_error_frame_is_an_error() -> None:
    error = {"error": {"message": "boom", "type": "invalid_request_error", "code": 400, "retryable": False}}
    chunks = [b"data: " + json.dumps(error).encode() + b"\n\n", b"data: [DONE]\n\n"]

    with pytest.raises(InBandError):
        await SGLangGenerateAPIData(prompt="p").process_response(_streamed(chunks), STREAMING, _tokenizer())


# A unary body with text " Paris." and 7 prompt / 2 completion / 3 cached tokens -> those counts and 1 output word.
async def test_unary_response() -> None:
    body = {
        "text": " Paris.",
        "output_ids": [1, 2],
        "meta_info": {"prompt_tokens": 7, "completion_tokens": 2, "cached_tokens": 3},
    }
    data = SGLangGenerateAPIData(prompt="The capital of France is")

    info = await data.process_response(_unary(body), UNARY, _tokenizer())

    assert data.model_response == " Paris."
    assert info.request_metrics.text is not None and info.request_metrics.text.input_tokens == 7
    assert isinstance(info.response_metrics, UnaryResponseMetrics)
    assert info.response_metrics.output_tokens == 1
    assert info.response_metrics.server_usage == {
        "prompt_tokens": 7,
        "completion_tokens": 2,
        "prompt_tokens_details": {"cached_tokens": 3},
    }


# A unary body with text "ok" and no meta_info -> prompt "a b" counted client-side as 2, no usage.
async def test_unary_response_without_meta_info() -> None:
    info = await SGLangGenerateAPIData(prompt="a b").process_response(_unary({"text": "ok"}), UNARY, _tokenizer())

    assert info.request_metrics.text is not None and info.request_metrics.text.input_tokens == 2
    assert isinstance(info.response_metrics, UnaryResponseMetrics)
    assert info.response_metrics.server_usage is None


# A unary body with empty text but 5 prompt tokens (max_new_tokens 0) -> a zero-token success.
async def test_unary_response_with_counts_but_no_text() -> None:
    body = {"text": "", "meta_info": {"prompt_tokens": 5, "completion_tokens": 0}}

    info = await SGLangGenerateAPIData(prompt="p").process_response(_unary(body), UNARY, _tokenizer())

    assert isinstance(info.response_metrics, UnaryResponseMetrics)
    assert info.response_metrics.output_tokens == 0


# Unary bodies with neither text nor counts ({} or a non-string text) -> EmptyResponseError.
@pytest.mark.parametrize("body", [{}, {"text": None}, {"text": 5}], ids=["empty", "null text", "number text"])
async def test_unary_response_without_text_or_counts_is_an_error(body: Mapping[str, object]) -> None:
    with pytest.raises(EmptyResponseError):
        await SGLangGenerateAPIData(prompt="p").process_response(_unary(body), UNARY, _tokenizer())


# A 200 whose unary body is {"error": {...}} -> InBandError.
async def test_unary_error_payload_is_an_error() -> None:
    with pytest.raises(InBandError):
        await SGLangGenerateAPIData(prompt="p").process_response(_unary({"error": {"message": "boom"}}), UNARY, _tokenizer())


# Cumulative "a" then "a b" twice (the finished chunk repeats) -> "a", " b", then None.
def test_a_repeated_cumulative_chunk_adds_nothing() -> None:
    state = stream_state()
    chunk = {"text": "a b", "output_ids": [1, 2], "meta_info": {"completion_tokens": 2}}

    assert parse_stream({"text": "a", "output_ids": [1], "meta_info": {"completion_tokens": 1}}, state) == "a"
    assert parse_stream(chunk, state) == " b"
    assert parse_stream(chunk, state) is None


# After "a": text "b" with 1 output id of 2 completion tokens is a delta -> "b";
# text "ab" with 2 output ids of 2 is cumulative -> only its new tail, "b".
@pytest.mark.parametrize("text, output_ids", [("b", [2]), ("ab", [1, 2])], ids=["fewer ids: delta", "equal ids: cumulative"])
def test_fewer_output_ids_than_completion_tokens_is_a_delta(text: str, output_ids: List[int]) -> None:
    state = stream_state()
    parse_stream({"text": "a", "output_ids": [1], "meta_info": {"completion_tokens": 1}}, state)

    assert parse_stream({"text": text, "output_ids": output_ids, "meta_info": {"completion_tokens": 2}}, state) == "b"


# Cumulative chunks "a", "ab" with output_ids missing, not a list, or completion_tokens missing, not an
# int -> each read as cumulative: "a", "b".
@pytest.mark.parametrize(
    "change",
    [
        lambda c: c.pop("output_ids"),
        lambda c: c.update(output_ids="1"),
        lambda c: c["meta_info"].pop("completion_tokens"),
        lambda c: c["meta_info"].update(completion_tokens="2"),
        lambda c: c.pop("meta_info"),
    ],
    ids=["no output_ids", "output_ids not a list", "no completion_tokens", "completion_tokens not an int", "no meta_info"],
)
def test_without_the_counts_a_chunk_is_read_as_cumulative(change: Callable[[Dict[str, object]], object]) -> None:
    state = stream_state()
    chunks: List[Dict[str, object]] = [
        {"text": "a", "output_ids": [1], "meta_info": {"completion_tokens": 1}},
        {"text": "ab", "output_ids": [2], "meta_info": {"completion_tokens": 2}},
    ]
    for chunk in chunks:
        change(chunk)

    assert [parse_stream(c, state) for c in chunks] == ["a", "b"]


# Two requests streaming at once, each with its own state -> each gets its own text.
def test_state_is_per_request() -> None:
    a, b = stream_state(), stream_state()

    assert parse_stream({"text": "x", "output_ids": [1], "meta_info": {"completion_tokens": 1}}, a) == "x"
    assert parse_stream({"text": "yy", "output_ids": [1], "meta_info": {"completion_tokens": 1}}, b) == "yy"
    assert parse_stream({"text": "xz", "output_ids": [1, 2], "meta_info": {"completion_tokens": 2}}, a) == "z"


# A chunk with no text (token-only output) still updates the counts -> None, state.text_len stays 0.
def test_a_chunk_without_text_keeps_its_counts() -> None:
    state = stream_state()

    assert parse_stream({"output_ids": [1], "meta_info": {"prompt_tokens": 3}}, state) is None
    assert state.text_len == 0
    assert state.meta_info == {"prompt_tokens": 3}


# "a" with 3 prompt tokens, then "a b" with no meta_info -> the counts from the first chunk are kept.
def test_a_chunk_without_meta_info_keeps_the_earlier_counts() -> None:
    state = stream_state()

    parse_stream({"text": "a", "meta_info": {"prompt_tokens": 3}}, state)
    assert parse_stream({"text": "a b"}, state) == " b"
    assert state.meta_info == {"prompt_tokens": 3}


# meta_info with each count alone, or with non-int counts, or not a dict -> only the int counts, or None.
@pytest.mark.parametrize(
    "meta_info, expected",
    [
        ({"prompt_tokens": 3}, {"prompt_tokens": 3}),
        ({"completion_tokens": 2}, {"completion_tokens": 2}),
        ({"cached_tokens": 1}, {"prompt_tokens_details": {"cached_tokens": 1}}),
        ({"prompt_tokens": "3", "completion_tokens": None, "cached_tokens": 1.0}, None),
        ({}, None),
        (None, None),
        ([3], None),
    ],
    ids=["prompt", "completion", "cached", "non-int counts", "empty", "none", "not a dict"],
)
def test_server_usage(meta_info: object, expected: object) -> None:
    assert server_usage(meta_info) == expected


# A request sent for LoRA adapter "a", streamed or unary -> the result names adapter "a".
@pytest.mark.parametrize("config", [STREAMING, UNARY], ids=["streaming", "unary"])
async def test_lora_adapter_is_recorded(config: APIConfig) -> None:
    response = _streamed(_sglang_stream(incremental=False)) if config.streaming else _unary({"text": "ok"})

    info = await SGLangGenerateAPIData(prompt="p").process_response(response, config, _tokenizer(), lora_adapter="a")

    assert info.lora_adapter == "a"
