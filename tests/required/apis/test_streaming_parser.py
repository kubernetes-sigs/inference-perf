# Copyright 2025 The Kubernetes Authors.
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
from typing import Any, AsyncGenerator, Optional
from unittest.mock import Mock

import pytest

from inference_perf.apis.response_errors import InBandError
from inference_perf.apis.streaming_parser import (
    StreamInterruptedError,
    _SSEStreamParser,
    finish_reason_of,
    parse_sse_stream,
)


@pytest.mark.asyncio
async def test_parse_sse_stream() -> None:
    mock_response = Mock()
    mock_content = Mock()
    mock_response.content = mock_content

    chunks = [
        b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]

    async def mock_iter_any() -> AsyncGenerator[bytes, None]:
        for chunk in chunks:
            yield chunk

    mock_content.iter_any = mock_iter_any

    def extract_content(data: dict[str, Any]) -> Optional[str]:
        return data.get("choices", [{}])[0].get("delta", {}).get("content")  # type: ignore[no-any-return]

    output_text, chunk_times, raw_content, response_chunks, server_usage, finish_reason = await parse_sse_stream(
        mock_response, extract_content
    )

    assert output_text == "Hello world"
    assert len(chunk_times) == 2
    assert "Hello" in raw_content
    assert "world" in raw_content
    assert "[DONE]" in raw_content
    assert len(response_chunks) == 2
    assert "Hello" in response_chunks[0]
    assert "world" in response_chunks[1]
    # response_chunks and chunk_times must stay in lockstep — reportgen zips them with strict=True.
    assert len(chunk_times) == len(response_chunks)
    assert server_usage is None
    # No frame carried choices[0].finish_reason, so none is reported.
    assert finish_reason is None


@pytest.mark.asyncio
async def test_parse_sse_stream_timestamps_only_content_events() -> None:
    """Reproduces issue #392: timestamps must only be recorded for content-bearing
    SSE events. Role-only first chunks, trailing usage chunks, and [DONE] signals
    must not appear in chunk_times, since they corrupt TPOT/TTFT/ITL. response_chunks
    is kept 1:1 aligned with chunk_times so reportgen's strict zip stays valid."""
    mock_response = Mock()
    mock_content = Mock()
    mock_response.content = mock_content

    chunks = [
        # Role-only first chunk — no content yet.
        b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n',
        # Two content-bearing chunks.
        b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n',
        # Trailing usage chunk — choices empty, no content.
        b'data: {"choices": [], "usage": {"prompt_tokens": 5, "completion_tokens": 2}}\n\n',
        # End-of-stream signal.
        b"data: [DONE]\n\n",
    ]

    async def mock_iter_any() -> AsyncGenerator[bytes, None]:
        for chunk in chunks:
            yield chunk

    mock_content.iter_any = mock_iter_any

    def extract_content(data: dict[str, Any]) -> Optional[str]:
        return data.get("choices", [{}])[0].get("delta", {}).get("content")  # type: ignore[no-any-return]

    output_text, chunk_times, _, response_chunks, server_usage, _ = await parse_sse_stream(mock_response, extract_content)

    assert output_text == "Hello world"
    assert len(chunk_times) == 2, (
        f"expected 2 timestamps for content-bearing chunks, got {len(chunk_times)} "
        "(role-only, usage, or [DONE] events leaking into chunk_times)"
    )
    assert len(response_chunks) == len(chunk_times), "response_chunks must stay 1:1 aligned with chunk_times"
    assert server_usage == {"prompt_tokens": 5, "completion_tokens": 2}, (
        "usage info from a content-less chunk should still be surfaced separately"
    )


@pytest.mark.asyncio
async def test_parse_sse_stream_interrupted_preserves_partial_body() -> None:
    """A stream that breaks partway (e.g. truncated SSE / dropped connection on a
    200 response) must raise StreamInterruptedError carrying the bytes received so
    far. This is what lets the per-request report show what the server actually sent
    instead of an empty response body, so 200-but-failed requests stay diagnosable."""
    mock_response = Mock()
    mock_content = Mock()
    mock_response.content = mock_content

    chunks = [
        b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n',
    ]
    boom = ConnectionResetError("Response payload is not completed")

    async def mock_iter_any() -> AsyncGenerator[bytes, None]:
        for chunk in chunks:
            yield chunk
        raise boom

    mock_content.iter_any = mock_iter_any

    def extract_content(data: dict[str, Any]) -> Optional[str]:
        return data.get("choices", [{}])[0].get("delta", {}).get("content")  # type: ignore[no-any-return]

    with pytest.raises(StreamInterruptedError) as exc_info:
        await parse_sse_stream(mock_response, extract_content)

    err = exc_info.value
    # The original transport exception is preserved for accurate error_type/error_msg.
    assert err.original is boom
    assert isinstance(err.original, ConnectionResetError)
    # The bytes received before the break are retained, not discarded.
    assert "Hello" in err.raw_content
    assert "world" in err.raw_content


@pytest.mark.asyncio
@pytest.mark.parametrize("ending", [b"\n", b"\r\n", b"\r"])
@pytest.mark.parametrize("space", [b"", b" "])
@pytest.mark.parametrize("chunk_size", [1, 7, 4096])
async def test_sse_line_endings_and_optional_space(ending: bytes, space: bytes, chunk_size: int) -> None:
    """Network boundaries must not change valid SSE event framing or UTF-8."""
    payload = (
        b": heartbeat"
        + ending
        + b"event: message"
        + ending
        + b"data:"
        + space
        + '{"content": "你好"}'.encode()
        + ending * 2
        + b"data:"
        + space
        + b'{"usage": {"completion_tokens": 2}}'
        + ending * 2
        + b"data:"
        + space
        + b"[DONE]"
        + ending * 2
    )
    response = Mock()

    async def chunks() -> AsyncGenerator[bytes, None]:
        for offset in range(0, len(payload), chunk_size):
            yield payload[offset : offset + chunk_size]

    response.content.iter_any = chunks
    output, times, raw, events, usage, _ = await parse_sse_stream(response, lambda data: data.get("content"))
    assert output == "你好"
    assert len(times) == len(events) == 1
    assert raw == payload.decode()
    assert usage == {"completion_tokens": 2}


@pytest.mark.asyncio
async def test_sse_multiline_data_and_incomplete_event() -> None:
    response = Mock()
    payload = b'data: {"content":\ndata: "Hello"}\n\ndata: invalid json\n\ndata: {"content": " discarded"}\n'

    async def chunks() -> AsyncGenerator[bytes, None]:
        yield payload

    response.content.iter_any = chunks
    output, times, raw, events, _, _ = await parse_sse_stream(response, lambda data: data.get("content"))
    assert output == "Hello"
    assert len(times) == len(events) == 1
    assert events == ['{"content":\n"Hello"}']
    assert raw == payload.decode()


@pytest.mark.asyncio
async def test_sse_done_ignores_later_content_but_preserves_raw_body() -> None:
    response = Mock()
    payloads = [
        b'data: {"content": "Hello"}\n\n',
        b"data:  [DONE] \n\n",
        b'data: {"content": " ignored"}\n\n',
    ]

    async def chunks() -> AsyncGenerator[bytes, None]:
        for payload in payloads:
            yield payload

    response.content.iter_any = chunks
    output, times, raw, events, _, _ = await parse_sse_stream(response, lambda data: data.get("content"))
    assert output == "Hello"
    assert len(times) == len(events) == 1
    assert raw == b"".join(payloads).decode()


@pytest.mark.asyncio
async def test_parse_sse_stream_fragmented_chunks() -> None:
    mock_response = Mock()
    mock_content = Mock()
    mock_response.content = mock_content

    # Chunks split across arbitrary byte boundaries
    chunks = [
        b"da",
        b'ta: {"choices": [{"delta": {"content": "Frag"',
        b"}}]}\n",
        b'\ndata: {"choices": [{"delta": {"content": "mented"}}]}\n\ndata: [DONE]\n\n',
    ]

    async def mock_iter_any() -> AsyncGenerator[bytes, None]:
        for chunk in chunks:
            yield chunk

    mock_content.iter_any = mock_iter_any

    def extract_content(data: dict[str, Any]) -> Optional[str]:
        return data.get("choices", [{}])[0].get("delta", {}).get("content")  # type: ignore[no-any-return]

    output_text, chunk_times, raw_content, response_chunks, _, _ = await parse_sse_stream(mock_response, extract_content)

    assert output_text == "Fragmented"
    assert len(chunk_times) == 2
    assert len(response_chunks) == 2
    assert "Frag" in response_chunks[0]
    assert "mented" in response_chunks[1]
    assert "Frag" in raw_content
    assert "mented" in raw_content


@pytest.mark.asyncio
async def test_parse_sse_stream_multiple_events_in_single_chunk() -> None:
    mock_response = Mock()
    mock_content = Mock()
    mock_response.content = mock_content

    chunks = [
        b'data: {"choices": [{"delta": {"content": "One"}}]}\n\n'
        b'data: {"choices": [{"delta": {"content": "Two"}}]}\n\n'
        b'data: {"choices": [{"delta": {"content": "Three"}}]}\n\n'
        b"data: [DONE]\n\n",
    ]

    async def mock_iter_any() -> AsyncGenerator[bytes, None]:
        for chunk in chunks:
            yield chunk

    mock_content.iter_any = mock_iter_any

    def extract_content(data: dict[str, Any]) -> Optional[str]:
        return data.get("choices", [{}])[0].get("delta", {}).get("content")  # type: ignore[no-any-return]

    output_text, chunk_times, raw_content, response_chunks, _, _ = await parse_sse_stream(mock_response, extract_content)

    assert output_text == "OneTwoThree"
    assert len(chunk_times) == 3
    assert len(response_chunks) == 3


@pytest.mark.asyncio
async def test_parse_sse_stream_no_space_prefix() -> None:
    mock_response = Mock()
    mock_content = Mock()
    mock_response.content = mock_content

    chunks = [
        b'data:{"choices": [{"delta": {"content": "A"}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": "B"}}]}\n\n',
        b"data:[DONE]\n\n",
    ]

    async def mock_iter_any() -> AsyncGenerator[bytes, None]:
        for chunk in chunks:
            yield chunk

    mock_content.iter_any = mock_iter_any

    def extract_content(data: dict[str, Any]) -> Optional[str]:
        return data.get("choices", [{}])[0].get("delta", {}).get("content")  # type: ignore[no-any-return]

    output_text, chunk_times, raw_content, response_chunks, _, _ = await parse_sse_stream(mock_response, extract_content)

    assert output_text == "AB"
    assert len(chunk_times) == 2
    assert len(response_chunks) == 2


@pytest.mark.asyncio
async def test_parse_sse_stream_comments_and_multiline_events() -> None:
    mock_response = Mock()
    mock_content = Mock()
    mock_response.content = mock_content

    chunks = [
        b": keepalive ping\n\n",
        b'event: message\ndata: {"choices": [{"delta": {"content": "Data"}}]}\n\n',
        b": trailing comment\n\n",
        b"data: [DONE]\n\n",
    ]

    async def mock_iter_any() -> AsyncGenerator[bytes, None]:
        for chunk in chunks:
            yield chunk

    mock_content.iter_any = mock_iter_any

    def extract_content(data: dict[str, Any]) -> Optional[str]:
        return data.get("choices", [{}])[0].get("delta", {}).get("content")  # type: ignore[no-any-return]

    output_text, chunk_times, raw_content, response_chunks, _, _ = await parse_sse_stream(mock_response, extract_content)

    assert output_text == "Data"
    assert len(chunk_times) == 1
    assert len(response_chunks) == 1
    assert "keepalive ping" in raw_content


@pytest.mark.asyncio
async def test_parse_sse_stream_merges_multiple_usage_updates() -> None:
    mock_response = Mock()
    mock_content = Mock()
    mock_response.content = mock_content

    chunks = [
        b'data: {"choices": [{"delta": {"content": "Hi"}}], "usage": {"prompt_tokens": 10}}\n\n',
        b'data: {"choices": [{"delta": {"content": "!"}}], "usage": {"completion_tokens": 2}}\n\n',
        b"data: [DONE]\n\n",
    ]

    async def mock_iter_any() -> AsyncGenerator[bytes, None]:
        for chunk in chunks:
            yield chunk

    mock_content.iter_any = mock_iter_any

    def extract_content(data: dict[str, Any]) -> Optional[str]:
        return data.get("choices", [{}])[0].get("delta", {}).get("content")  # type: ignore[no-any-return]

    output_text, chunk_times, raw_content, response_chunks, server_usage, _ = await parse_sse_stream(
        mock_response, extract_content
    )

    assert output_text == "Hi!"
    assert len(chunk_times) == 2
    assert server_usage == {"prompt_tokens": 10, "completion_tokens": 2}


@pytest.mark.asyncio
async def test_parse_sse_stream_bare_cr_mid_chunk_does_not_drop_content() -> None:
    """Bare CR mid-chunk with CRLF ending must not be treated as a single data frame."""
    mock_response = Mock()
    mock_content = Mock()
    mock_response.content = mock_content

    # The exact reproduction from differential fuzzing: bare \r before id: 1
    chunks = [
        b'data: {"choices":[{"delta":{"content":"A"}}]}\rid: 1\r\n\r\n',
        b"data: [DONE]\r\n\r\n",
    ]

    async def mock_iter_any() -> AsyncGenerator[bytes, None]:
        for chunk in chunks:
            yield chunk

    mock_content.iter_any = mock_iter_any

    def extract_content(data: dict[str, Any]) -> Optional[str]:
        return data.get("choices", [{}])[0].get("delta", {}).get("content")  # type: ignore[no-any-return]

    output_text, chunk_times, raw_content, response_chunks, _, _ = await parse_sse_stream(mock_response, extract_content)
    assert output_text == "A"
    assert len(chunk_times) == 1
    assert len(response_chunks) == 1


@pytest.mark.asyncio
async def test_parse_sse_stream_chunking_invariance_whole_byte_by_byte_per_event() -> None:
    """Differential test: feeding a stream whole, byte-by-byte, or per-event must yield identical results."""
    event_chunks = [
        b'data: {"choices": [{"delta": {"content": "Hello"}}]}\r\n\r\n',
        b'data: {"choices": [{"delta": {"content": " world"}}]}\n\n',
        b'data: {"message": {"usage": {"output_tokens": 15}}}\n\n',
        b'data: {"choices": [{"delta": {"content": "!"}}]}\rid: 42\r\n\r\n',
        b": ping heartbeat\r\n\r\n",
        b"data: [DONE]\r\n\r\n",
    ]
    full_stream = b"".join(event_chunks)

    def extract_content(data: dict[str, Any]) -> Optional[str]:
        return data.get("choices", [{}])[0].get("delta", {}).get("content")  # type: ignore[no-any-return]

    async def run_with_chunks(
        chunks_to_feed: list[bytes],
    ) -> tuple[str, int, str, list[str], Optional[dict[str, Any]]]:
        mock_response = Mock()
        mock_content = Mock()
        mock_response.content = mock_content

        async def mock_iter_any() -> AsyncGenerator[bytes, None]:
            for c in chunks_to_feed:
                yield c

        mock_content.iter_any = mock_iter_any
        output_text, chunk_times, raw_content, response_chunks, server_usage, _ = await parse_sse_stream(
            mock_response, extract_content
        )
        return output_text, len(chunk_times), raw_content, response_chunks, server_usage

    # 1. Per-event delivery (exercises fast path on standalone frames)
    res_per_event = await run_with_chunks(event_chunks)

    # 2. Whole delivery (exercises multi-frame buffered iteration)
    res_whole = await run_with_chunks([full_stream])

    # 3. Byte-by-byte delivery (exercises fine-grained buffer accumulation & straddling endings)
    res_byte_by_byte = await run_with_chunks([full_stream[i : i + 1] for i in range(len(full_stream))])

    # All three chunking modes must produce identical results
    assert res_per_event == res_whole == res_byte_by_byte

    # Verify expected values
    output_text, num_times, raw_content, response_chunks, server_usage = res_per_event
    assert output_text == "Hello world!"
    assert num_times == 3
    assert len(response_chunks) == 3
    assert server_usage == {"output_tokens": 15}
    assert raw_content == full_stream.decode("utf-8")


def test_sse_stream_parser_process_data_payload_content_and_timing() -> None:
    """Verify process_data_payload records content, timing, and response chunks for content-bearing deltas."""

    def extract_content(data: dict[str, Any]) -> Optional[str]:
        return data.get("choices", [{}])[0].get("delta", {}).get("content")  # type: ignore[no-any-return]

    parser = _SSEStreamParser(extract_content)
    timestamp = 123.456
    payload = b'{"choices": [{"delta": {"content": "foo"}}]}'

    parser.process_data_payload(payload, timestamp)

    assert parser.output_text_parts == ["foo"]
    assert parser.chunk_times == [timestamp]
    assert len(parser.response_chunks) == 1
    assert "foo" in parser.response_chunks[0]
    assert parser.server_usage is None


def test_sse_stream_parser_process_data_payload_skips_empty_or_role_deltas() -> None:
    """Verify process_data_payload skips recording timestamps/response chunks when no content is present."""

    def extract_content(data: dict[str, Any]) -> Optional[str]:
        return data.get("choices", [{}])[0].get("delta", {}).get("content")  # type: ignore[no-any-return]

    parser = _SSEStreamParser(extract_content)
    # Role-only chunk
    parser.process_data_payload(b'{"choices": [{"delta": {"role": "assistant"}}]}', 100.0)
    # Empty choices chunk
    parser.process_data_payload(b'{"choices": []}', 101.0)

    assert parser.output_text_parts == []
    assert parser.chunk_times == []
    assert parser.response_chunks == []
    assert parser.server_usage is None


def test_sse_stream_parser_process_data_payload_usage_extraction_and_merge() -> None:
    """Verify process_data_payload extracts and merges direct and nested usage fields."""
    parser = _SSEStreamParser(lambda d: None)

    # Direct usage (OpenAI style)
    parser.process_data_payload(b'{"usage": {"prompt_tokens": 10, "completion_tokens": 5}}', 1.0)
    assert parser.server_usage == {"prompt_tokens": 10, "completion_tokens": 5}

    # Nested message.usage (Anthropic style)
    parser.process_data_payload(b'{"message": {"usage": {"completion_tokens": 20, "cached_tokens": 4}}}', 2.0)
    assert parser.server_usage == {"prompt_tokens": 10, "completion_tokens": 20, "cached_tokens": 4}

    # Non-dictionary usage ignored gracefully
    parser.process_data_payload(b'{"usage": "invalid"}', 3.0)
    assert parser.server_usage == {"prompt_tokens": 10, "completion_tokens": 20, "cached_tokens": 4}


def test_sse_stream_parser_process_data_payload_malformed_json_handled_silently() -> None:
    """Verify process_data_payload catches json decode and index errors silently."""

    def extract_content(data: dict[str, Any]) -> Optional[str]:
        val = data.get("text")
        return str(val) if val is not None else None

    parser = _SSEStreamParser(extract_content)

    # Malformed JSON should not raise
    parser.process_data_payload(b'{"broken": json', 1.0)
    assert parser.output_text_parts == []
    assert parser.chunk_times == []
    assert parser.response_chunks == []


# Builds a fake aiohttp response whose body is the given SSE chunks, optionally
# raising `broken_by` after the last one, and returns it with the standard chat
# delta extractor.
def _stream(chunks: list[bytes], broken_by: Optional[Exception] = None) -> tuple[Mock, Any]:
    mock_response = Mock()
    mock_content = Mock()
    mock_response.content = mock_content

    async def mock_iter_any() -> AsyncGenerator[bytes, None]:
        for chunk in chunks:
            yield chunk
        if broken_by is not None:
            raise broken_by

    mock_content.iter_any = mock_iter_any

    def extract_content(data: dict[str, Any]) -> Optional[str]:
        return data.get("choices", [{}])[0].get("delta", {}).get("content")  # type: ignore[no-any-return]

    return mock_response, extract_content


# One content frame, then a `{"error": {...}}` frame, then [DONE], all delivered
# cleanly. Must raise InBandError whose message is the error frame's JSON and whose
# raw_content is the whole stream, [DONE] included, rather than return "Hello".
@pytest.mark.asyncio
async def test_parse_sse_stream_raises_on_in_band_error_frame() -> None:
    """A 200 stream that carries its failure as a frame (the #713 shape, what vLLM
    and SGLang emit when generation fails mid-stream) is that failure, not a short
    success. The stream is read to the end first so raw_content is the full body."""
    error_frame = b'{"error": {"message": "The model is overloaded", "type": "server_error", "code": 503}}'
    mock_response, extract_content = _stream(
        [
            b'data: {"choices": [{"delta": {"content": "Hello"}}]}\n\n',
            b"data: " + error_frame + b"\n\n",
            b"data: [DONE]\n\n",
        ]
    )

    with pytest.raises(InBandError) as exc_info:
        await parse_sse_stream(mock_response, extract_content)

    err = exc_info.value
    assert json.loads(str(err)) == json.loads(error_frame)
    assert "Hello" in err.raw_content
    assert "[DONE]" in err.raw_content, "the stream must be drained before raising"


# An error frame followed by a dropped connection. The in-band error must win over
# the transport error: InBandError, not StreamInterruptedError, and the raw body
# still holds the frame.
@pytest.mark.asyncio
async def test_parse_sse_stream_in_band_error_wins_over_a_later_break() -> None:
    """When the server says why it failed and then drops the connection, the reason
    is what belongs in the report; the break is a symptom."""
    error_frame = b'{"error": {"message": "engine died", "type": "server_error"}}'
    mock_response, extract_content = _stream(
        [b"data: " + error_frame + b"\n\n"], broken_by=ConnectionResetError("Response payload is not completed")
    )

    with pytest.raises(InBandError) as exc_info:
        await parse_sse_stream(mock_response, extract_content)

    assert json.loads(str(exc_info.value)) == json.loads(error_frame)
    assert "engine died" in exc_info.value.raw_content


# A normal frame that happens to carry `"error": null` next to its choices. Must
# parse as a plain "Hello" success: only a truthy top-level error counts.
@pytest.mark.asyncio
async def test_parse_sse_stream_ignores_null_error_field() -> None:
    mock_response, extract_content = _stream(
        [
            b'data: {"choices": [{"delta": {"content": "Hello"}}], "error": null}\n\n',
            b"data: [DONE]\n\n",
        ]
    )

    output_text, chunk_times, _, response_chunks, server_usage, _ = await parse_sse_stream(mock_response, extract_content)

    assert output_text == "Hello"
    assert len(chunk_times) == len(response_chunks) == 1
    assert server_usage is None


# A vLLM-shaped stream: two content frames with finish_reason null, a third
# content frame carrying finish_reason "length", then the usage-only frame with
# empty choices, then [DONE]. Must report finish_reason "length" alongside the
# text "Hello world!" and the usage dict, and stay at 3 content chunks.
@pytest.mark.asyncio
async def test_parse_sse_stream_reports_last_finish_reason() -> None:
    mock_response, extract_content = _stream(
        [
            b'data: {"choices": [{"delta": {"content": "Hello"}, "finish_reason": null}]}\n\n',
            b'data: {"choices": [{"delta": {"content": " world"}, "finish_reason": null}]}\n\n',
            b'data: {"choices": [{"delta": {"content": "!"}, "finish_reason": "length"}]}\n\n',
            b'data: {"choices": [], "usage": {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}}\n\n',
            b"data: [DONE]\n\n",
        ]
    )

    output_text, chunk_times, _, response_chunks, server_usage, finish_reason = await parse_sse_stream(
        mock_response, extract_content
    )

    assert output_text == "Hello world!"
    assert len(chunk_times) == len(response_chunks) == 3
    assert server_usage == {"prompt_tokens": 4, "completion_tokens": 3, "total_tokens": 7}
    assert finish_reason == "length"


# The finish_reason arrives on its own frame after the last content (an empty
# delta with finish_reason "stop"), as some servers send it. Must still be
# reported as "stop", and that reason-only frame must not count as content.
@pytest.mark.asyncio
async def test_parse_sse_stream_finish_reason_on_a_content_free_frame() -> None:
    mock_response, extract_content = _stream(
        [
            b'data: {"choices": [{"delta": {"content": "Hi"}, "finish_reason": null}]}\n\n',
            b'data: {"choices": [{"delta": {}, "finish_reason": "stop"}]}\n\n',
            b"data: [DONE]\n\n",
        ]
    )

    output_text, chunk_times, _, response_chunks, _, finish_reason = await parse_sse_stream(mock_response, extract_content)

    assert output_text == "Hi"
    assert len(chunk_times) == len(response_chunks) == 1
    assert finish_reason == "stop"


# finish_reason_of on single payloads: an OpenAI unary body with
# choices[0].finish_reason "stop" gives "stop"; an Anthropic unary body with
# stop_reason "max_tokens" gives "max_tokens"; an Anthropic message_delta frame
# with delta.stop_reason "end_turn" gives "end_turn"; a chunk whose
# finish_reason is null, a usage-only frame with empty choices, and an error
# payload all give None.
def test_finish_reason_of_reads_openai_and_anthropic_shapes() -> None:
    assert finish_reason_of({"choices": [{"text": "x", "finish_reason": "stop"}]}) == "stop"
    assert finish_reason_of({"type": "message", "stop_reason": "max_tokens", "content": []}) == "max_tokens"
    assert finish_reason_of({"type": "message_delta", "delta": {"stop_reason": "end_turn"}, "usage": {}}) == "end_turn"
    assert finish_reason_of({"choices": [{"delta": {"content": "x"}, "finish_reason": None}]}) is None
    assert finish_reason_of({"choices": [], "usage": {"completion_tokens": 3}}) is None
    assert finish_reason_of({"error": {"message": "boom"}}) is None
