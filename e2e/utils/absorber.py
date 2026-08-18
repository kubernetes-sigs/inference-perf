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
"""A fake OpenAI-compatible server that records every request sent to it.

Point any benchmark tool at the absorber and, when the run finishes, the list
of recorded requests is exactly what that tool sent: how many, when each
arrived and when its reply finished, the prompt, ``max_tokens``, and the
``stream`` / ``ignore_eos`` flags. The arrival/finish pairs feed the shared
load-shape helpers in ``e2e/utils/load_shape.py``, which turn them into a
request rate and an in-flight concurrency; the absorber itself only records.
The tool-parity tests compare those recordings between tools. When two tools configured for "the
same" workload differ, the difference shows up as a concrete request-level
mismatch that names the setting responsible, rather than as a fuzzy gap in the
latency numbers the tools print.

Two things to know about the responses it sends back:

- They are paced on purpose. Closed-loop tools (a fixed number of requests in
  flight) send their next request only when the previous one finishes, so how
  quickly the server answers changes the traffic pattern being measured. The
  absorber therefore streams ``max_tokens`` chunks with a configurable wait
  before the first chunk (TTFT) and between chunks (ITL).
- They are not real tokens. Chunk text is filler words and the ``usage``
  counts are rough. Anything about token accounting is tested elsewhere
  (golden_sim.py and the live vLLM tier), never here.
"""

import asyncio
import itertools
import json
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aiohttp import web

_FILLER_WORDS = ("alpha", "bravo", "carol", "delta", "echo", "fox", "golf", "hotel")


@dataclass
class AbsorbedRequest:
    """One request exactly as the absorber received it, plus when it arrived and when its reply finished.

    The two timestamps are ``time.perf_counter()`` readings, so only differences
    between them mean anything. Together they are the raw material for the
    load-shape helpers in ``e2e/utils/load_shape.py`` (request rate, in-flight
    concurrency); the absorber itself computes nothing from them.
    """

    route: str  # "/v1/completions" or "/v1/chat/completions"
    arrival_s: float  # when the request arrived (before its body was read)
    body: Dict[str, Any]  # the request's JSON body, as sent
    done_s: Optional[float] = None  # when the last byte of the reply was sent; None while the reply is still going

    # The (start_time, end_time) pair in the shape e2e/utils/load_shape.py reads,
    # so a list of these can go straight into its rate and concurrency helpers.
    # Fails if the reply has not finished yet.
    def lifecycle_entry(self) -> Dict[str, float]:
        assert self.done_s is not None, "reply not finished; read requests only after the tool has exited"
        return {"start_time": self.arrival_s, "end_time": self.done_s}

    @property
    def prompt_text(self) -> Optional[str]:
        """The prompt as one string (chat messages joined by newlines), or None if it is not text."""
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
        """The prompt as a list of token ids, when the tool sent ids instead of text; else None."""
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
    """The fake server. Use as ``async with AbsorberServer(...) as s:``; afterwards ``s.requests`` is the recording.

    Answers on /v1/completions and /v1/chat/completions (plus /v1/models,
    /health and /metrics so tools that probe those do not error). ``ttft_s`` is
    the wait before the first chunk (or before a non-streaming reply) and
    ``itl_s`` the wait between chunks. ``default_output_tokens`` is how many
    chunks to send when the request has no ``max_tokens``.
    """

    port: int
    model: str
    ttft_s: float = 0.04
    itl_s: float = 0.005
    default_output_tokens: int = 16
    host: str = "127.0.0.1"
    requests: List[AbsorbedRequest] = field(default_factory=list)

    def __post_init__(self) -> None:
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
        # Empty but valid Prometheus output, so tools that scrape metrics do not error.
        return web.Response(text="", content_type="text/plain")

    # Records one request with the arrival time taken by the caller, and appends
    # it to self.requests. done_s is filled in later, when the reply is finished.
    def _absorb(self, route: str, body: Dict[str, Any], arrival_s: float) -> AbsorbedRequest:
        absorbed = AbsorbedRequest(route=route, arrival_s=arrival_s, body=body)
        self.requests.append(absorbed)
        return absorbed

    # Rough prompt token count for the usage field: list length for token ids,
    # word count for text. Not accurate, and nothing here relies on it being.
    def _naive_prompt_tokens(self, absorbed: AbsorbedRequest) -> int:
        ids = absorbed.prompt_token_ids
        if ids is not None:
            return len(ids)
        text = absorbed.prompt_text or ""
        return len(text.split())

    # n filler chunks: " alpha", " bravo", ... cycling through _FILLER_WORDS.
    def _chunk_texts(self, n: int) -> List[str]:
        words = itertools.cycle(_FILLER_WORDS)
        return [f" {next(words)}" for _ in range(n)]

    async def _handle_completions(self, request: web.Request) -> web.StreamResponse:
        return await self._respond(request, "/v1/completions")

    async def _handle_chat(self, request: web.Request) -> web.StreamResponse:
        return await self._respond(request, "/v1/chat/completions")

    # Shared handler for both endpoints. Takes the arrival time first, records
    # the request, then replies with max_tokens (or default_output_tokens)
    # chunks: streamed with pacing if the request said stream=true, otherwise
    # as one JSON body after an equivalent total wait. Stamps done_s once the
    # reply is finished (for a streamed reply, after the final "[DONE]" is
    # written; for a one-piece reply, just before aiohttp writes it).
    async def _respond(self, request: web.Request, route: str) -> web.StreamResponse:
        arrival_s = time.perf_counter()
        body = await request.json()
        absorbed = self._absorb(route, body, arrival_s)
        try:
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
            absorbed.done_s = time.perf_counter()

    # The fixed id/object/created/model fields every reply carries.
    def _envelope(self, route: str) -> Dict[str, Any]:
        kind = "text_completion" if route == "/v1/completions" else "chat.completion"
        return {"id": "absorber-0", "object": kind, "created": int(time.time()), "model": self.model}

    # One complete non-streaming reply with all chunks joined into one text.
    def _unary_body(self, route: str, chunks: List[str], usage: Dict[str, int]) -> Dict[str, Any]:
        text = "".join(chunks)
        if route == "/v1/completions":
            choice: Dict[str, Any] = {"index": 0, "text": text, "finish_reason": "length"}
        else:
            choice = {"index": 0, "message": {"role": "assistant", "content": text}, "finish_reason": "length"}
        return {**self._envelope(route), "choices": [choice], "usage": usage}

    # Server-sent-events reply: wait ttft_s, send the first chunk, then wait
    # itl_s before each later chunk. Sends a final usage chunk only if the
    # request asked for it (stream_options.include_usage), then "data: [DONE]".
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
