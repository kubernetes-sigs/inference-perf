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
import json
import logging
from typing import Any, AsyncGenerator
from unittest.mock import AsyncMock, MagicMock

import pytest

from inference_perf.apis import InferenceInfo, StreamedResponseMetrics, TemplateAPIData, UnaryResponseMetrics
from inference_perf.apis import template as template_module
from inference_perf.apis.streaming_parser import StreamInterruptedError
from inference_perf.config import (
    APIConfig,
    APIType,
    TemplateConfig,
    TemplateResponseConfig,
    TemplateStreamChunks,
    TemplateStreamConfig,
    TemplateStreamFraming,
)


def _make_tokenizer() -> MagicMock:
    tok = MagicMock()
    tok.count_tokens = lambda text, **kw: len(text.split())
    return tok


def _make_response(body: Any) -> MagicMock:
    response = MagicMock()
    response.json = AsyncMock(return_value=body)
    return response


# The SGLang native API, as in the example in docs/config.md.
_TEMPLATE = TemplateConfig(
    route="/generate",
    body={"text": "${prompt}", "sampling_params": {"max_new_tokens": "${max_tokens}", "ignore_eos": True}},
    ignore_eos=True,
    response=TemplateResponseConfig(
        text_path="text",
        input_tokens_path="meta_info.prompt_tokens",
        output_tokens_path="meta_info.completion_tokens",
    ),
)
_CONFIG = APIConfig(type=APIType.Template, template=_TEMPLATE)


@pytest.mark.asyncio
async def test_template_route_is_filled_with_the_model() -> None:
    template = TemplateConfig(
        route="/predictions/${model}", body={"text": "${prompt}"}, response=TemplateResponseConfig(text_path="text")
    )
    data = TemplateAPIData(prompt="hello", template=template)

    await data.to_request_body("test-model", 100, False, False)

    assert data.get_api_type() == APIType.Template
    assert data.get_route() == "/predictions/test-model"


def test_template_route_needs_the_request_body_first() -> None:
    # The model name, and with it the route, is only known once the body is built.
    data = TemplateAPIData(prompt="hello", template=_TEMPLATE)

    with pytest.raises(RuntimeError, match="to_request_body"):
        data.get_route()


@pytest.mark.asyncio
async def test_template_request_body_fills_placeholders() -> None:
    data = TemplateAPIData(prompt="Hello, world!", max_tokens=32, template=_TEMPLATE)
    # A string that is only ${max_tokens} is sent as a number. Other values are sent as written.
    assert await data.to_request_body("test-model", 100, True, False) == {
        "text": "Hello, world!",
        "sampling_params": {"max_new_tokens": 32, "ignore_eos": True},
    }


@pytest.mark.asyncio
async def test_template_request_body_fills_model_lists_and_embedded_text() -> None:
    template = TemplateConfig(
        route="/v1/generate",
        body={"model": "${model}", "inputs": ["Question: ${prompt}"], "limit": "${max_tokens}", "note": "costs $$5"},
        response=TemplateResponseConfig(text_path="output"),
    )
    # max_tokens falls back to the client default when the request does not set one.
    data = TemplateAPIData(prompt="hi there", template=template)

    assert await data.to_request_body("test-model", 100, False, False) == {
        "model": "test-model",
        "inputs": ["Question: hi there"],
        "limit": 100,
        "note": "costs $5",
    }


@pytest.mark.asyncio
async def test_template_process_response_reads_text_and_token_counts() -> None:
    data = TemplateAPIData(prompt="one two", template=_TEMPLATE)
    response = _make_response({"text": "a b c", "meta_info": {"prompt_tokens": 9, "completion_tokens": 3}})

    info = await data.process_response(response, _CONFIG, _make_tokenizer())

    assert info.request_metrics.text.input_tokens == 9
    assert isinstance(info.response_metrics, UnaryResponseMetrics)
    assert info.response_metrics.output_tokens == 3
    # The counts are stored under the usage keys the reports read.
    assert info.response_metrics.server_usage == {"prompt_tokens": 9, "completion_tokens": 3}


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "body",
    [
        {"text": "a b"},
        {"text": "a b", "meta_info": {"prompt_tokens": "9", "completion_tokens": True}},
    ],
)
async def test_template_process_response_counts_tokens_without_usable_counts(body: dict[str, Any]) -> None:
    # Counts that are missing or not integers are ignored, and the prompt is tokenized on the client.
    data = TemplateAPIData(prompt="one two three", template=_TEMPLATE)

    info = await data.process_response(_make_response(body), _CONFIG, _make_tokenizer())

    assert info.request_metrics.text.input_tokens == 3
    assert info.response_metrics is not None
    assert info.response_metrics.output_tokens == 2
    assert info.response_metrics.server_usage is None


@pytest.mark.asyncio
async def test_template_warns_once_about_a_count_that_is_not_an_integer(
    caplog: pytest.LogCaptureFixture, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(template_module, "_warned_count_paths", set())
    data = TemplateAPIData(prompt="one two three", template=_TEMPLATE)
    body = {"text": "a b", "meta_info": {"prompt_tokens": 9.0, "completion_tokens": 2}}

    with caplog.at_level(logging.WARNING, logger="inference_perf.apis.template"):
        for _ in range(2):
            info = await data.process_response(_make_response(body), _CONFIG, _make_tokenizer())

    warnings = [record.getMessage() for record in caplog.records if "meta_info.prompt_tokens" in record.getMessage()]
    assert len(warnings) == 1
    assert "selected a float, not an integer" in warnings[0]
    # The float is dropped, so the prompt is counted with the tokenizer.
    assert info.request_metrics.text.input_tokens == 3


@pytest.mark.asyncio
async def test_template_process_response_rejects_a_response_without_text() -> None:
    # A text_path that selects nothing fails the request instead of reporting zero output tokens.
    data = TemplateAPIData(prompt="hello", template=_TEMPLATE)

    with pytest.raises(ValueError, match="text_path 'text' did not select a string"):
        await data.process_response(_make_response({"generated_text": "a b"}), _CONFIG, _make_tokenizer())


def _make_stream(payload: bytes, chunk_size: int = 7) -> MagicMock:
    """A streamed response whose body arrives in pieces of chunk_size bytes."""
    response = MagicMock()

    async def iter_any() -> AsyncGenerator[bytes, None]:
        for offset in range(0, len(payload), chunk_size):
            yield payload[offset : offset + chunk_size]

    response.content.iter_any = iter_any
    return response


def _sse(*chunks: Any) -> bytes:
    return b"".join(b"data: " + json.dumps(chunk).encode() + b"\n\n" for chunk in chunks)


def _stream_template(
    framing: TemplateStreamFraming = TemplateStreamFraming.SSE,
    chunks: TemplateStreamChunks = TemplateStreamChunks.DELTA,
    **paths: str,
) -> TemplateConfig:
    return TemplateConfig(
        route="/generate",
        body={"text": "${prompt}", "stream": True},
        response=TemplateResponseConfig(
            text_path="text", stream=TemplateStreamConfig(framing=framing, chunks=chunks), **paths
        ),
    )


async def _process_stream(template: TemplateConfig, response: MagicMock) -> InferenceInfo:
    data = TemplateAPIData(prompt="one two", template=template)
    config = APIConfig(type=APIType.Template, template=template, streaming=True)
    return await data.process_response(response, config, _make_tokenizer())


@pytest.mark.asyncio
async def test_template_stream_reads_the_new_text_of_cumulative_chunks() -> None:
    # Like the SGLang native API: each chunk repeats the text so far, then [DONE].
    template = _stream_template(
        chunks=TemplateStreamChunks.CUMULATIVE,
        input_tokens_path="meta_info.prompt_tokens",
        output_tokens_path="meta_info.completion_tokens",
    )
    payload = (
        _sse(
            {"text": "a", "meta_info": {"prompt_tokens": 9, "completion_tokens": 1}},
            {"text": "a b", "meta_info": {"prompt_tokens": 9, "completion_tokens": 2}},
            {"text": "a b c", "meta_info": {"prompt_tokens": 9, "completion_tokens": 3}},
        )
        + b"data: [DONE]\n\n"
    )

    info = await _process_stream(template, _make_stream(payload))

    metrics = info.response_metrics
    assert isinstance(metrics, StreamedResponseMetrics)
    assert metrics.chunk_texts == ["a", " b", " c"]
    assert len(metrics.chunk_times) == 3
    # The raw stream is kept once, as raw_response.
    assert metrics.response_chunks == []
    assert metrics.output_tokens == 3
    # The counts are taken from the last chunk that has them.
    assert metrics.server_usage == {"prompt_tokens": 9, "completion_tokens": 3}
    assert info.request_metrics.text.input_tokens == 9
    assert info.extra_info["raw_response"] == payload.decode()


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("framing", "payload"),
    [
        (TemplateStreamFraming.SSE, _sse({"text": "a"}, {"text": " b c"}, {"text": "", "stats": {"generated": 3}})),
        # Blank lines and CRLF are allowed, and the last line has no newline after it.
        (
            TemplateStreamFraming.NDJSON,
            b'{"text": "a"}\n{"text": " b c"}\r\n\n{"text": "", "stats": {"generated": 3}}',
        ),
    ],
)
async def test_template_stream_reads_delta_chunks(framing: TemplateStreamFraming, payload: bytes) -> None:
    template = _stream_template(framing=framing, output_tokens_path="stats.generated")

    metrics = (await _process_stream(template, _make_stream(payload))).response_metrics

    assert isinstance(metrics, StreamedResponseMetrics)
    # The last chunk has no new text, so it is not timed.
    assert metrics.chunk_texts == ["a", " b c"]
    assert len(metrics.chunk_times) == 2
    assert metrics.output_tokens == 3
    assert metrics.server_usage == {"completion_tokens": 3}


@pytest.mark.asyncio
async def test_template_stream_counts_the_last_text_of_a_cumulative_stream() -> None:
    # The server sent a stop string, then removed it from the final text.
    template = _stream_template(chunks=TemplateStreamChunks.CUMULATIVE)

    payload = _sse({"text": "a b"}, {"text": "a b STOP"}, {"text": "a b"})

    metrics = (await _process_stream(template, _make_stream(payload))).response_metrics

    assert isinstance(metrics, StreamedResponseMetrics)
    assert metrics.chunk_texts == ["a b", " STOP"]
    assert metrics.output_tokens == 2


@pytest.mark.asyncio
async def test_template_stream_skips_empty_text_in_a_cumulative_stream() -> None:
    template = _stream_template(chunks=TemplateStreamChunks.CUMULATIVE)
    payload = _sse({"text": "a"}, {"text": ""}, {"text": "a b"}, {"text": "a b c"}, {"text": ""})

    metrics = (await _process_stream(template, _make_stream(payload))).response_metrics

    # An empty text does not clear the text so far.
    assert isinstance(metrics, StreamedResponseMetrics)
    assert metrics.chunk_texts == ["a", " b", " c"]
    assert metrics.output_tokens == 3


@pytest.mark.asyncio
@pytest.mark.parametrize("chunks", list(TemplateStreamChunks))
async def test_template_stream_with_only_empty_text_has_no_output_tokens(chunks: TemplateStreamChunks) -> None:
    # Like a response whose text is "", this is a success with no output tokens.
    template = _stream_template(chunks=chunks)

    metrics = (await _process_stream(template, _make_stream(_sse({"text": ""}, {"text": ""})))).response_metrics

    assert isinstance(metrics, StreamedResponseMetrics)
    assert metrics.output_tokens == 0
    assert metrics.chunk_times == []


@pytest.mark.asyncio
@pytest.mark.parametrize("framing", list(TemplateStreamFraming))
async def test_template_stream_without_text_fails(framing: TemplateStreamFraming) -> None:
    # For example, the body did not ask for a stream and the server sent one JSON object.
    template = _stream_template(framing=framing)
    response = _make_stream(json.dumps({"generated_text": "a b c"}).encode())

    with pytest.raises(ValueError, match="text_path 'text' did not select a string in any chunk"):
        await _process_stream(template, response)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("framing", "first"),
    [(TemplateStreamFraming.SSE, b'data: {"text": "a"}\n\n'), (TemplateStreamFraming.NDJSON, b'{"text": "a"}\n')],
)
async def test_template_stream_keeps_the_bytes_of_a_broken_stream(framing: TemplateStreamFraming, first: bytes) -> None:
    response = MagicMock()

    async def iter_any() -> AsyncGenerator[bytes, None]:
        yield first
        raise ConnectionResetError("connection reset")

    response.content.iter_any = iter_any

    with pytest.raises(StreamInterruptedError) as exc_info:
        await _process_stream(_stream_template(framing=framing), response)

    assert isinstance(exc_info.value.original, ConnectionResetError)
    assert exc_info.value.raw_content == first.decode()
