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
