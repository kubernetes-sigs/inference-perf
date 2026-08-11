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
"""OpenAI-compatible absorber: records the load a client offers it.

The absorber is the oracle for the tool-parity tests: any benchmark tool is
pointed at it, and afterwards the recorded requests ARE the workload that tool
actually offered: request count, arrival times, prompt payloads, sampling
flags, and how many requests were in flight at once. Divergence between two
tools configured for "the same" workload shows up here as a diff over
recorded requests, naming the knob that leaked, instead of as a noisy delta
between the tools' reported metrics.

It is deliberately NOT dumb: closed-loop clients issue their next request when
the previous one finishes, so response pacing shapes the arrival process being
measured. Responses therefore stream ``max_tokens`` chunks paced by a
configured TTFT and inter-chunk interval. It is also NOT a token oracle: chunk
text is arbitrary filler and the usage numbers it returns are naive counts.
Token-accounting fidelity belongs to the golden sim (golden_sim.py) and the
live vLLM tier, never to parity cases.
"""

import asyncio
import itertools
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aiohttp import web

_FILLER_WORDS = ("alpha", "bravo", "carol", "delta", "echo", "fox", "golf", "hotel")


@dataclass(frozen=True)
class AbsorbedRequest:
    """One request exactly as the absorber received it."""

    route: str  # "/v1/completions" or "/v1/chat/completions"
    arrival_s: float  # perf_counter() at handler entry
    in_flight_at_arrival: int  # concurrent requests the moment this one arrived, including it
    body: Dict[str, Any]  # raw parsed JSON body

    @property
    def prompt_text(self) -> Optional[str]:
        """The prompt as text, or None when it cannot be interpreted as text."""
        if self.route == "/v1/chat/completions":
            parts = []
            for message in self.body.get("messages", []):
                content = message.get("content")
                if isinstance(content, str):
                    parts.append(content)
            return "\n".join(parts)
        prompt = self.body.get("prompt")
        if isinstance(prompt, str):
            return prompt
        if isinstance(prompt, list) and all(isinstance(p, str) for p in prompt):
            return "\n".join(prompt)
        return None

    @property
    def prompt_token_ids(self) -> Optional[List[int]]:
        """Pre-tokenized prompt ids, when the client sent ids instead of text."""
        prompt = self.body.get("prompt")
        if isinstance(prompt, list) and prompt and all(isinstance(p, int) for p in prompt):
            return prompt
        return None

    @property
    def max_tokens(self) -> Optional[int]:
        value = self.body.get("max_tokens", self.body.get("max_completion_tokens"))
        return int(value) if isinstance(value, int) else None

    @property
    def stream(self) -> bool:
        return bool(self.body.get("stream", False))

    @property
    def ignore_eos(self) -> bool:
        return bool(self.body.get("ignore_eos", False))


@dataclass
class AbsorberServer:
    """Serves /v1/completions and /v1/chat/completions, recording every request.

    ``ttft_s`` delays the first streamed chunk (or the whole unary response)
    and ``itl_s`` paces subsequent chunks, so closed-loop clients see realistic
    request durations. ``default_output_tokens`` bounds responses when a client
    omits ``max_tokens``.
    """

    port: int
    model: str
    ttft_s: float = 0.04
    itl_s: float = 0.005
    default_output_tokens: int = 16
    host: str = "127.0.0.1"
    requests: List[AbsorbedRequest] = field(default_factory=list)

    def __post_init__(self) -> None:
        self._in_flight = 0
        self._runner: Optional[web.AppRunner] = None

    @property
    def base_url(self) -> str:
        return f"http://{self.host}:{self.port}"

    async def __aenter__(self) -> "AbsorberServer":
        app = web.Application()
        app.router.add_post("/v1/completions", self._handle_completions)
        app.router.add_post("/v1/chat/completions", self._handle_chat)
        app.router.add_get("/v1/models", self._handle_models)
        app.router.add_get("/health", self._handle_health)
        app.router.add_get("/metrics", self._handle_metrics)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    async def _handle_models(self, request: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [{"id": self.model, "object": "model"}]})

    async def _handle_health(self, request: web.Request) -> web.Response:
        return web.Response(text="")

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        # An empty but valid exposition, so clients that scrape do not error.
        return web.Response(text="", content_type="text/plain")

    def _absorb(self, route: str, body: Dict[str, Any]) -> AbsorbedRequest:
        absorbed = AbsorbedRequest(
            route=route,
            arrival_s=time.perf_counter(),
            in_flight_at_arrival=self._in_flight,
            body=body,
        )
        self.requests.append(absorbed)
        return absorbed

    def _naive_prompt_tokens(self, absorbed: AbsorbedRequest) -> int:
        ids = absorbed.prompt_token_ids
        if ids is not None:
            return len(ids)
        text = absorbed.prompt_text or ""
        return len(text.split())

    def _chunk_texts(self, n: int) -> List[str]:
        words = itertools.cycle(_FILLER_WORDS)
        return [f" {next(words)}" for _ in range(n)]

    async def _handle_completions(self, request: web.Request) -> web.StreamResponse:
        return await self._respond(request, "/v1/completions")

    async def _handle_chat(self, request: web.Request) -> web.StreamResponse:
        return await self._respond(request, "/v1/chat/completions")

    async def _respond(self, request: web.Request, route: str) -> web.StreamResponse:
        self._in_flight += 1
        try:
            body = await request.json()
            absorbed = self._absorb(route, body)
            n_out = absorbed.max_tokens or self.default_output_tokens
            chunks = self._chunk_texts(n_out)
            usage = {
                "prompt_tokens": self._naive_prompt_tokens(absorbed),
                "completion_tokens": n_out,
                "total_tokens": self._naive_prompt_tokens(absorbed) + n_out,
            }
            if absorbed.stream:
                return await self._stream(request, route, chunks, body, usage)
            await asyncio.sleep(self.ttft_s + self.itl_s * max(0, len(chunks) - 1))
            return web.json_response(self._unary_body(route, chunks, usage))
        finally:
            self._in_flight -= 1

    def _envelope(self, route: str) -> Dict[str, Any]:
        kind = "text_completion" if route == "/v1/completions" else "chat.completion"
        return {"id": "absorber-0", "object": kind, "created": int(time.time()), "model": self.model}

    def _unary_body(self, route: str, chunks: List[str], usage: Dict[str, int]) -> Dict[str, Any]:
        text = "".join(chunks)
        if route == "/v1/completions":
            choice: Dict[str, Any] = {"index": 0, "text": text, "finish_reason": "length"}
        else:
            choice = {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "length"}
        return {**self._envelope(route), "choices": [choice], "usage": usage}

    async def _stream(
        self,
        request: web.Request,
        route: str,
        chunks: List[str],
        body: Dict[str, Any],
        usage: Dict[str, int],
    ) -> web.StreamResponse:
        response = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache", "Connection": "keep-alive"},
        )
        await response.prepare(request)

        envelope = self._envelope(route)
        if route == "/v1/chat/completions":
            envelope["object"] = "chat.completion.chunk"

        async def send(event: Dict[str, Any]) -> None:
            await response.write(f"data: {json.dumps(event)}\n\n".encode("utf-8"))

        await asyncio.sleep(self.ttft_s)
        for i, text in enumerate(chunks):
            if i > 0:
                await asyncio.sleep(self.itl_s)
            last = i == len(chunks) - 1
            finish = "length" if last else None
            if route == "/v1/completions":
                choice: Dict[str, Any] = {"index": 0, "text": text, "finish_reason": finish}
            else:
                choice = {"index": 0, "delta": {"content": text}, "finish_reason": finish}
            await send({**envelope, "choices": [choice]})

        include_usage = bool((body.get("stream_options") or {}).get("include_usage"))
        if include_usage:
            await send({**envelope, "choices": [], "usage": usage})
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response
