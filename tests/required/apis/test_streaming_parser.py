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

from typing import Any, AsyncGenerator, Optional
from unittest.mock import Mock

import pytest

from inference_perf.apis.streaming_parser import (
    StreamInterruptedError,
    _SSEStreamParser,
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

    parsed = await parse_sse_stream(mock_response, extract_content)

    assert parsed.output_text == "Hello world"
    assert len(parsed.chunk_times) == 2
    assert "Hello" in parsed.raw_content
    assert "world" in parsed.raw_content
    assert "[DONE]" in parsed.raw_content
    assert len(parsed.response_chunks) == 2
    assert "Hello" in parsed.response_chunks[0]
    assert "world" in parsed.response_chunks[1]
    # response_chunks and chunk_times must stay in lockstep — reportgen zips them with strict=True.
    assert len(parsed.chunk_times) == len(parsed.response_chunks)
    assert parsed.server_usage is None


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

    parsed = await parse_sse_stream(mock_response, extract_content)

    assert parsed.output_text == "Hello world"
    assert len(parsed.chunk_times) == 2, (
        f"expected 2 timestamps for content-bearing chunks, got {len(parsed.chunk_times)} "
        "(role-only, usage, or [DONE] events leaking into chunk_times)"
    )
    assert len(parsed.response_chunks) == len(parsed.chunk_times), "response_chunks must stay 1:1 aligned with chunk_times"
    assert parsed.server_usage == {"prompt_tokens": 5, "completion_tokens": 2}, (
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
    parsed = await parse_sse_stream(response, lambda data: data.get("content"))
    assert parsed.output_text == "你好"
    assert len(parsed.chunk_times) == len(parsed.response_chunks) == 1
    assert parsed.raw_content == payload.decode()
    assert parsed.server_usage == {"completion_tokens": 2}


@pytest.mark.asyncio
async def test_sse_multiline_data_and_incomplete_event() -> None:
    response = Mock()
    payload = b'data: {"content":\ndata: "Hello"}\n\ndata: invalid json\n\ndata: {"content": " discarded"}\n'

    async def chunks() -> AsyncGenerator[bytes, None]:
        yield payload

    response.content.iter_any = chunks
    parsed = await parse_sse_stream(response, lambda data: data.get("content"))
    assert parsed.output_text == "Hello"
    assert len(parsed.chunk_times) == len(parsed.response_chunks) == 1
    assert parsed.response_chunks == ['{"content":\n"Hello"}']
    assert parsed.raw_content == payload.decode()


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
    parsed = await parse_sse_stream(response, lambda data: data.get("content"))
    assert parsed.output_text == "Hello"
    assert len(parsed.chunk_times) == len(parsed.response_chunks) == 1
    assert parsed.raw_content == b"".join(payloads).decode()


def extract_delta_content(data: dict[str, Any]) -> Optional[str]:
    return data.get("choices", [{}])[0].get("delta", {}).get("content")  # type: ignore[no-any-return]


def extract_delta_reasoning(data: dict[str, Any]) -> Optional[str]:
    delta = data.get("choices", [{}])[0].get("delta", {})
    return delta.get("reasoning_content") or delta.get("reasoning")  # type: ignore[no-any-return]


def make_response(chunks: list[bytes]) -> Mock:
    mock_response = Mock()
    mock_content = Mock()
    mock_response.content = mock_content

    async def mock_iter_any() -> AsyncGenerator[bytes, None]:
        for chunk in chunks:
            yield chunk

    mock_content.iter_any = mock_iter_any
    return mock_response


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

    output_text, chunk_times, raw_content, response_chunks, _, *_ = await parse_sse_stream(mock_response, extract_content)

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

    output_text, chunk_times, raw_content, response_chunks, _, *_ = await parse_sse_stream(mock_response, extract_content)

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

    output_text, chunk_times, raw_content, response_chunks, _, *_ = await parse_sse_stream(mock_response, extract_content)

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

    output_text, chunk_times, raw_content, response_chunks, _, *_ = await parse_sse_stream(mock_response, extract_content)

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

    output_text, chunk_times, raw_content, response_chunks, server_usage, *_ = await parse_sse_stream(
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

    output_text, chunk_times, raw_content, response_chunks, _, *_ = await parse_sse_stream(mock_response, extract_content)
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
        output_text, chunk_times, raw_content, response_chunks, server_usage, *_ = await parse_sse_stream(
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


REASONING_THEN_CONTENT_CHUNKS = [
    b'data: {"choices": [{"delta": {"role": "assistant"}}]}\n\n',
    b'data: {"choices": [{"delta": {"reasoning_content": "Let me"}}]}\n\n',
    b'data: {"choices": [{"delta": {"reasoning_content": " think."}}]}\n\n',
    b'data: {"choices": [{"delta": {"content": "The answer"}}]}\n\n',
    b'data: {"choices": [{"delta": {"content": " is 4."}}]}\n\n',
    b"data: [DONE]\n\n",
]


@pytest.mark.asyncio
async def test_parse_sse_stream_reasoning_tracked_separately_from_content() -> None:
    """Reasoning models (gpt-oss, DeepSeek-R1, QwQ) stream delta.reasoning_content
    before delta.content. The channels must stay separate (#559): reasoning
    timestamps anchor TTFT, while output_text (the basis for output_len) and
    chunk_times (the basis for TPOT/ITL) must remain content-only so reasoning
    doesn't count as user-facing output."""
    parsed = await parse_sse_stream(
        make_response(REASONING_THEN_CONTENT_CHUNKS), extract_delta_content, extract_delta_reasoning
    )

    assert parsed.output_text == "The answer is 4."
    assert parsed.reasoning_text == "Let me think."
    assert len(parsed.chunk_times) == 2
    assert len(parsed.response_chunks) == 2
    assert len(parsed.reasoning_chunk_times) == 2
    # reasoning_chunks and reasoning_chunk_times stay 1:1, mirroring the content lists.
    assert len(parsed.reasoning_chunks) == len(parsed.reasoning_chunk_times)
    assert all("reasoning_content" in chunk for chunk in parsed.reasoning_chunks)
    # Reasoning arrived before content, so its timestamps must precede content's:
    # this ordering is what lets reportgen anchor TTFT to the reasoning channel.
    assert parsed.reasoning_chunk_times[0] <= parsed.chunk_times[0]


@pytest.mark.asyncio
async def test_parse_sse_stream_reasoning_field_variant() -> None:
    """Some OpenAI-compatible servers name the channel delta.reasoning rather
    than delta.reasoning_content; both must be recognized."""
    chunks = [
        b'data: {"choices": [{"delta": {"reasoning": "Step 1."}}]}\n\n',
        b'data: {"choices": [{"delta": {"content": "Result."}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    parsed = await parse_sse_stream(make_response(chunks), extract_delta_content, extract_delta_reasoning)

    assert parsed.output_text == "Result."
    assert parsed.reasoning_text == "Step 1."
    assert len(parsed.reasoning_chunk_times) == 1
    assert len(parsed.chunk_times) == 1


@pytest.mark.asyncio
async def test_parse_sse_stream_reasoning_only_stream() -> None:
    """When the output budget is exhausted mid-reasoning (max_tokens below the
    reasoning length), the stream ends with no content chunk at all. The
    reasoning channel must still be captured: it is the only TTFT anchor such
    a request has (#559's null-TTFT case)."""
    chunks = [
        b'data: {"choices": [{"delta": {"reasoning_content": "Thinking"}}]}\n\n',
        b'data: {"choices": [{"delta": {"reasoning_content": " hard"}}]}\n\n',
        b"data: [DONE]\n\n",
    ]
    parsed = await parse_sse_stream(make_response(chunks), extract_delta_content, extract_delta_reasoning)

    assert parsed.output_text == ""
    assert parsed.reasoning_text == "Thinking hard"
    assert len(parsed.chunk_times) == 0
    assert len(parsed.reasoning_chunk_times) == 2


@pytest.mark.asyncio
async def test_parse_sse_stream_reasoning_ignored_without_extractor() -> None:
    """Callers that pass no extract_reasoning (e.g. the completions API) must
    see exactly the pre-#559 behavior: reasoning chunks contribute nothing."""
    parsed = await parse_sse_stream(make_response(REASONING_THEN_CONTENT_CHUNKS), extract_delta_content)

    assert parsed.output_text == "The answer is 4."
    assert parsed.reasoning_text == ""
    assert len(parsed.chunk_times) == 2
    assert parsed.reasoning_chunks == []
    assert parsed.reasoning_chunk_times == []
