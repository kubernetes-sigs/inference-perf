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
from typing import Dict, List, Optional, Tuple, Type

from aiohttp import web


@dataclass(frozen=True)
class StreamEvent:
    """One scripted SSE delta: which channel carries the text and how long the
    server waits before sending it."""

    # "tool" streams text as tool-call arguments. "hidden" sends a delta with
    # no text, the way vLLM does for a token its tool parser swallowed.
    channel: str  # "reasoning" | "content" | "tool" | "hidden"
    text: str
    delay_before: float = 0.0
    # Ids of the tokens this event generated. Sent as the choice's token_ids
    # only to a request that set return_token_ids, as vLLM does.
    token_ids: Tuple[int, ...] = ()
    # Function name, on the tool event that opens a call.
    tool_name: str = ""


@dataclass
class ServedStream:
    """What the fake actually did for one request.

    Timestamps are ``perf_counter()`` stamped immediately after each event was
    written and drained. The client can only observe a chunk at or after its
    send stamp (same clock, same process), so these bound TTFT assertions
    tightly without trusting ``asyncio.sleep`` accuracy.
    """

    # Stamped when the handler starts, before the stream is opened.
    arrival_time: float = 0.0
    reasoning_send_times: List[float] = field(default_factory=list)
    content_send_times: List[float] = field(default_factory=list)
    # One stamp per token id sent, so an event with three ids appears three times.
    token_id_send_times: List[float] = field(default_factory=list)
    # The JSON body the client posted.
    request_body: Dict[str, object] = field(default_factory=dict)


class FakeOpenAIServer:
    """Serves /v1/chat/completions, streaming the configured script for every
    request. Use as an async context manager; ``url`` is the endpoint.

    ``reasoning_key`` is the delta field that carries reasoning text: vLLM has
    shipped both ``reasoning_content`` and ``reasoning``. /health and
    /v1/models answer too, so benchmark tools that probe the server before
    sending load (the e2e cross-tool parity test) can run against it."""

    def __init__(
        self,
        script: List[StreamEvent],
        completion_tokens: Optional[int] = None,
        *,
        reasoning_key: str = "reasoning_content",
        model: str = "fake-model",
    ) -> None:
        self.script = script
        self.reasoning_key = reasoning_key
        self.model = model
        # Emitted as a trailing usage chunk (stream_options.include_usage
        # style) when set.
        self.completion_tokens = completion_tokens
        self.served: List[ServedStream] = []
        self._runner: Optional[web.AppRunner] = None
        self.port = 0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    @property
    def url(self) -> str:
        return f"{self.base_url}/v1/chat/completions"

    async def __aenter__(self) -> "FakeOpenAIServer":
        app = web.Application()
        app.router.add_post("/v1/chat/completions", self._handle)
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/v1/models", self._handle_models)
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

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.Response(status=200)

    async def _handle_models(self, request: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [{"id": self.model, "object": "model"}]})

    async def _handle(self, request: web.Request) -> web.StreamResponse:
        record = ServedStream(arrival_time=time.perf_counter(), request_body=await request.json())
        send_token_ids = record.request_body.get("return_token_ids") is True
        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        self.served.append(record)
        for event in self.script:
            if event.delay_before > 0:
                await asyncio.sleep(event.delay_before)
            delta: Dict[str, object]
            if event.channel == "tool":
                function = {"arguments": event.text, **({"name": event.tool_name} if event.tool_name else {})}
                delta = {"tool_calls": [{"index": 0, "function": function}]}
            elif event.channel == "hidden":
                delta = {}
            else:
                delta = {self.reasoning_key if event.channel == "reasoning" else "content": event.text}
            choice: Dict[str, object] = {"delta": delta}
            if send_token_ids:
                choice["token_ids"] = list(event.token_ids)
            payload = {"choices": [choice]}
            await response.write(f"data: {json.dumps(payload)}\n\n".encode())
            stamp = time.perf_counter()
            if event.channel == "reasoning":
                record.reasoning_send_times.append(stamp)
            elif event.channel == "content":
                record.content_send_times.append(stamp)
            if send_token_ids:
                record.token_id_send_times.extend([stamp] * len(event.token_ids))
        if self.completion_tokens is not None:
            usage = {"choices": [], "usage": {"completion_tokens": self.completion_tokens}}
            await response.write(f"data: {json.dumps(usage)}\n\n".encode())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response
