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

"""
Shared utilities for parsing Server-Sent Events (SSE) streaming responses.

This module provides common functionality for parsing streaming responses from
LLM APIs, reducing code duplication across different API types.
"""

import json
import time
from typing import Any, Callable, List, Optional, Tuple

from aiohttp import ClientResponse


class StreamInterruptedError(Exception):
    """Raised when an SSE stream fails partway through being read.

    Carries the raw bytes received before the failure (``raw_content``) so
    callers can still surface what the server actually sent, which is the whole
    point of per-request error capture. The triggering exception is preserved as
    ``original`` so callers can report its real type and message rather than this
    wrapper's.
    """

    def __init__(self, original: Exception, raw_content: str) -> None:
        super().__init__(str(original))
        self.original = original
        self.raw_content = raw_content


async def parse_sse_stream(
    response: ClientResponse, extract_content: Callable[[dict[str, Any]], Optional[str]]
) -> Tuple[str, List[float], str, List[str], Optional[dict[str, Any]]]:
    """
    Parse Server-Sent Events (SSE) stream and extract content.

    This function handles the common SSE parsing logic used across different
    API types (chat completions, text completions, etc.). It processes the
    streaming response chunk by chunk, extracting content using the provided
    extraction function.

    Args:
        response: The HTTP response with streaming content
        extract_content: Function to extract text content from parsed JSON data.
                        Should return the text content or None if not found.
                        Example: lambda data: data.get("choices", [{}])[0].get("delta", {}).get("content")

    Returns:
        Tuple of (output_text, chunk_times, raw_content, response_chunks, server_usage):
        - output_text: The concatenated text content from all chunks
        - chunk_times: Timestamps for content-bearing chunks only. Role-only
          deltas, usage-only chunks, [DONE] signals, and unparseable messages
          are excluded so they don't corrupt downstream TPOT/TTFT/ITL.
        - raw_content: The raw string content of the stream
        - response_chunks: Raw JSON strings of content-bearing chunks, 1:1 with
          chunk_times.
        - server_usage: Last-seen `usage` dict from any chunk that carried one
          (e.g. trailing `{"choices":[],"usage":{...}}` when stream_options
          include_usage=true). None if the server didn't emit usage.
    """
    output_text = ""
    chunk_times: List[float] = []
    buffer = b""
    raw_content = b""
    response_chunks: List[str] = []
    server_usage: Optional[dict[str, Any]] = None

    try:
        async for chunk in response.content.iter_any():
            raw_content += chunk
            buffer += chunk
            while b"\n\n" in buffer:
                message, buffer = buffer.split(b"\n\n", 1)
                message_time = time.perf_counter()
                done = False
                for line in message.split(sep=b"\n"):
                    if line.startswith(b"data:"):
                        data_str = line.removeprefix(b"data: ").strip()
                        if data_str == b"[DONE]":
                            done = True
                            break
                        try:
                            data = json.loads(data_str)
                            if usage := data.get("usage"):
                                server_usage = usage
                            if content := extract_content(data):
                                output_text += content
                                chunk_times.append(message_time)
                                response_chunks.append(data_str.decode("utf-8", errors="ignore"))
                        except (json.JSONDecodeError, IndexError):
                            continue
                if done:
                    break
    except Exception as e:
        # The stream broke partway (e.g. a truncated SSE stream, a dropped
        # connection, or a proxy that 200s then sends an error page). Re-raise
        # with the bytes received so far attached so the caller can still record
        # what the server actually sent instead of an empty response body.
        raise StreamInterruptedError(e, raw_content.decode("utf-8", errors="ignore")) from e

    return output_text, chunk_times, raw_content.decode("utf-8", errors="ignore"), response_chunks, server_usage
