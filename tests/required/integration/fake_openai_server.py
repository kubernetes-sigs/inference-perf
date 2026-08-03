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
"""In-process OpenAI-compatible streaming fake for the integration tier.

The sim-backed e2e tier (llm-d-inference-sim) exercises realistic server
behavior, but the sim cannot emit reasoning-channel deltas or pace chunks on
cue. This fake serves a scripted SSE event sequence over real HTTP with
controlled per-event delays, and records a timestamp for every chunk it
actually sent, so tests can assert client-side metrics against the server's
own timeline rather than against configured intent (the #606 Integration
tier's "fake the conditions, never the oracle" principle).
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import List, Optional, Type

from aiohttp import web


@dataclass(frozen=True)
class StreamEvent:
    """One scripted SSE delta: which channel carries the text and how long the
    server waits before sending it."""

    channel: str  # "reasoning" | "content"
    text: str
    delay_before: float = 0.0


@dataclass
class ServedStream:
    """What the fake actually did for one request.

    Timestamps are ``perf_counter()`` stamped immediately after each event was
    written and drained. The client can only observe a chunk at or after its
    send stamp (same clock, same process), so these bound TTFT assertions
    tightly without trusting ``asyncio.sleep`` accuracy.
    """

    reasoning_send_times: List[float] = field(default_factory=list)
    content_send_times: List[float] = field(default_factory=list)


class FakeOpenAIServer:
    """Serves /v1/chat/completions, streaming the configured script for every
    request. Use as an async context manager; ``url`` is the endpoint."""

    def __init__(self, script: List[StreamEvent], completion_tokens: Optional[int] = None) -> None:
        self.script = script
        # Emitted as a trailing usage chunk (stream_options.include_usage
        # style) when set.
        self.completion_tokens = completion_tokens
        self.served: List[ServedStream] = []
        self._runner: Optional[web.AppRunner] = None
        self.port = 0

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1/chat/completions"

    async def __aenter__(self) -> "FakeOpenAIServer":
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._handle)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = self._runner.addresses[0][1]
        return self

    async def __aexit__(
        self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]
    ) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        record = ServedStream()
        self.served.append(record)
        for event in self.script:
            if event.delay_before > 0:
                await asyncio.sleep(event.delay_before)
            key = "reasoning_content" if event.channel == "reasoning" else "content"
            payload = {"choices": [{"delta": {key: event.text}}]}
            await response.write(f"data: {json.dumps(payload)}\n\n".encode())
            stamp = time.perf_counter()
            if event.channel == "reasoning":
                record.reasoning_send_times.append(stamp)
            else:
                record.content_send_times.append(stamp)
        if self.completion_tokens is not None:
            usage = {"choices": [], "usage": {"completion_tokens": self.completion_tokens}}
            await response.write(f"data: {json.dumps(usage)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response
