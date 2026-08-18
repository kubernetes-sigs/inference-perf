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
from inference_perf.apis.response_errors import InBandError
from inference_perf.apis.streaming_parser import parse_sse_stream, StreamInterruptedError
import pytest


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

    output_text, chunk_times, raw_content, response_chunks, server_usage = await parse_sse_stream(
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

    output_text, chunk_times, _, response_chunks, server_usage = await parse_sse_stream(mock_response, extract_content)

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

    output_text, chunk_times, _, response_chunks, server_usage = await parse_sse_stream(mock_response, extract_content)

    assert output_text == "Hello"
    assert len(chunk_times) == len(response_chunks) == 1
    assert server_usage is None
