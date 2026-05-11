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
from inference_perf.apis.streaming_parser import parse_sse_stream
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
