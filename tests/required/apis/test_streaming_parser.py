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
from inference_perf.apis.streaming_parser import parse_sse_stream, resolve_output_token_count, StreamInterruptedError
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


def test_resolve_output_token_count_prefers_server_usage() -> None:
    """When the server reports usage, the exact completion_tokens wins over
    client-side re-tokenization (which the mock tokenizer must therefore not do)."""
    tokenizer = Mock()
    tokenizer.count_tokens = Mock(return_value=999)

    assert resolve_output_token_count({"completion_tokens": 7}, "some text", tokenizer) == 7
    tokenizer.count_tokens.assert_not_called()


@pytest.mark.parametrize(
    "server_usage",
    [
        None,  # server omitted usage entirely
        {},  # usage present but no completion_tokens key
        {"completion_tokens": 0},  # falsy count — treat as unreported
        {"completion_tokens": None},
    ],
)
def test_resolve_output_token_count_falls_back_to_retokenization(server_usage: Optional[dict[str, Any]]) -> None:
    """No usable server count -> fall back to re-tokenizing the output text."""
    tokenizer = Mock()
    tokenizer.count_tokens = Mock(return_value=4)

    assert resolve_output_token_count(server_usage, "four token text", tokenizer) == 4
    tokenizer.count_tokens.assert_called_once_with("four token text")
