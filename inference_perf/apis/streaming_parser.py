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


async def parse_sse_stream(
    response: ClientResponse, extract_content: Callable[[dict[str, Any]], Optional[str]]
) -> Tuple[str, List[float]]:
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
        Tuple of (output_text, output_token_times) where:
        - output_text: The concatenated text content from all chunks
        - output_token_times: List of timestamps when each token was received

    Example:
        # For chat completions
        output_text, times = await parse_sse_stream(
            response,
            lambda d: d.get("choices", [{}])[0].get("delta", {}).get("content")
        )

        # For text completions
        output_text, times = await parse_sse_stream(
            response,
            lambda d: d.get("choices", [{}])[0].get("text")
        )
    """
    output_text = ""
    output_token_times: List[float] = []
    buffer = b""

    async for chunk in response.content.iter_any():
        buffer += chunk
        while b"\n\n" in buffer:
            message, buffer = buffer.split(b"\n\n", 1)
            output_token_times.append(time.perf_counter())
            for line in message.split(sep=b"\n"):
                if line.startswith(b"data:"):
                    data_str = line.removeprefix(b"data: ").strip()
                    if data_str == b"[DONE]":
                        break
                    try:
                        data = json.loads(data_str)
                        if content := extract_content(data):
                            output_text += content
                    except (json.JSONDecodeError, IndexError):
                        continue
            else:
                continue
            break

    return output_text, output_token_times
