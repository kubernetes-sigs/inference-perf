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
import re
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


class _SSEStreamParser:
    """Internal stateful parser for Server-Sent Events (SSE) streaming responses."""

    _LINE_ENDING = re.compile(rb"\r\n|\r|\n")

    def __init__(self, extract_content: Callable[[dict[str, Any]], Optional[str]]) -> None:
        self.extract_content = extract_content
        self.output_text_parts: List[str] = []
        self.chunk_times: List[float] = []
        self.raw_content_chunks: List[bytes] = []
        self.response_chunks: List[str] = []
        self.server_usage: Optional[dict[str, Any]] = None
        self.buffer = bytearray()
        self.data_lines: List[bytes] = []
        self.skip_lf = False
        self.done = False

    def process_data_payload(self, data_bytes: bytes, message_time: float) -> None:
        """Parse JSON payload, extract usage, extract text content, and record metrics."""
        try:
            data_str = data_bytes.decode("utf-8", errors="ignore")
            data = json.loads(data_str)
            if b"usage" in data_bytes or b"message" in data_bytes:
                usage = data.get("usage")
                if not isinstance(usage, dict):
                    message_data = data.get("message")
                    if isinstance(message_data, dict):
                        usage = message_data.get("usage")
                if isinstance(usage, dict):
                    if self.server_usage is None:
                        self.server_usage = dict(usage)
                    else:
                        self.server_usage.update(usage)
            if content := self.extract_content(data):
                self.output_text_parts.append(content)
                self.chunk_times.append(message_time)
                self.response_chunks.append(data_str)
        except (json.JSONDecodeError, IndexError):
            pass

    async def parse(self, response: ClientResponse) -> Tuple[str, List[float], str, List[str], Optional[dict[str, Any]]]:
        buffer = self.buffer
        data_lines = self.data_lines
        raw_chunks_append = self.raw_content_chunks.append
        process_data_payload = self.process_data_payload
        perf_counter = time.perf_counter
        line_ending = self._LINE_ENDING

        try:
            async for chunk in response.content.iter_any():
                raw_chunks_append(chunk)
                if self.done:
                    continue

                message_time = perf_counter()

                # Fast-path: standalone single SSE frame when buffer is empty
                if not buffer and not data_lines and not self.skip_lf and chunk.startswith(b"data:"):
                    data_bytes: Optional[bytes] = None
                    if chunk.endswith(b"\n\n") and chunk.count(b"\n") == 2 and b"\r" not in chunk:
                        data_bytes = chunk[6:-2] if chunk.startswith(b"data: ") else chunk[5:-2]
                    elif (
                        chunk.endswith(b"\r\n\r\n")
                        and chunk.count(b"\r") == 2
                        and chunk.count(b"\n") == 2
                        and chunk.count(b"\r\n") == 2
                    ):
                        data_bytes = chunk[6:-4] if chunk.startswith(b"data: ") else chunk[5:-4]

                    if data_bytes is not None:
                        if data_bytes.strip() == b"[DONE]":
                            self.done = True
                            continue
                        process_data_payload(data_bytes, message_time)
                        continue

                # Fallback path: fragmented or multi-message stream buffering
                buffer.extend(chunk)
                # A CR terminates a line immediately; swallow its optional LF even
                # when the pair straddles network chunks.
                if self.skip_lf and buffer:
                    if buffer[0:1] == b"\n":
                        del buffer[0:1]
                    self.skip_lf = False
                scan_pos = 0
                while match := line_ending.search(buffer, scan_pos):
                    line = bytes(buffer[scan_pos : match.start()])
                    self.skip_lf = match.group() == b"\r" and match.end() == len(buffer)
                    scan_pos = match.end()
                    if line:
                        field, separator, value = line.partition(b":")
                        if field == b"data":
                            data_lines.append(bytes(value.removeprefix(b" ") if separator else b""))
                        continue
                    if not data_lines:
                        continue
                    data_bytes = b"\n".join(data_lines)
                    data_lines.clear()
                    message_time = perf_counter()
                    if data_bytes.strip() == b"[DONE]":
                        self.done = True
                        break
                    process_data_payload(data_bytes, message_time)
                if scan_pos > 0:
                    if scan_pos == len(buffer):
                        buffer.clear()
                    else:
                        del buffer[:scan_pos]
        except Exception as e:
            # The stream broke partway (e.g. a truncated SSE stream, a dropped
            # connection, or a proxy that 200s then sends an error page). Re-raise
            # with the bytes received so far attached so the caller can still record
            # what the server actually sent instead of an empty response body.
            raw_str = b"".join(self.raw_content_chunks).decode("utf-8", errors="ignore")
            raise StreamInterruptedError(e, raw_str) from e

        output_text = "".join(self.output_text_parts)
        raw_content = b"".join(self.raw_content_chunks).decode("utf-8", errors="ignore")
        return output_text, self.chunk_times, raw_content, self.response_chunks, self.server_usage


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
        - server_usage: Merged `usage` fields from chunks that carried one
          (e.g. OpenAI trailing `{"choices":[],"usage":{...}}` or Anthropic
          `message.usage`/`message_delta.usage`). None if the server didn't
          emit usage.
    """
    return await _SSEStreamParser(extract_content).parse(response)
