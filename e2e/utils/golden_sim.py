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
"""OpenAI-compatible sim server with fully controlled ground truth.

Unlike llm-d-inference-sim, whose chunking follows its own whitespace notion
of tokens, this sim constructs responses FROM a real tokenizer's token ids, so
the exact token count of every response is known by construction and chunk
boundaries can be placed deliberately (token-aligned or mid-token). Chunk
pacing is controlled and every actual send timestamp is recorded, which lets
tests assert token accounting at zero tolerance and inter-chunk timing against
what the sim really did rather than what it was configured to do (#631).

Each response is self-verified at construction: the concatenated chunk text
must re-encode (add_special_tokens=False) to exactly ``n_tokens`` ids. A case
that cannot be built exactly fails loudly at setup instead of silently
testing something weaker.
"""

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from aiohttp import web

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class GoldenCase:
    """One controlled response: exactly ``n_tokens`` tokens split into
    ``n_chunks`` SSE deltas.

    ``n_tokens`` must be unique across the cases given to a server: it is
    reported as ``usage.completion_tokens`` and is how tests map an observed
    request back to its ground truth.

    ``split`` places chunk boundaries:
      - "token": boundaries on token boundaries (via offset mapping)
      - "codepoint": evenly by character, deliberately mid-token/mid-word
    """

    n_tokens: int
    n_chunks: int
    split: str = "token"


@dataclass
class ServedRequest:
    """What the sim actually did for one request."""

    route: str
    case: GoldenCase
    prompt_text: str
    # perf_counter() stamped immediately after each content chunk was written
    # and drained. Tests compare client-observed inter-chunk gaps against
    # diffs of these, so asyncio.sleep overshoot cannot fail the test.
    send_times: List[float] = field(default_factory=list)


class GoldenSimServer:
    """Serves /v1/completions and /v1/chat/completions, streaming and unary,
    cycling deterministically through ``cases`` in request-arrival order."""

    def __init__(
        self,
        tokenizer: Any,
        cases: List[GoldenCase],
        *,
        model: str,
        ttft_delay: float = 0.15,
        inter_chunk_interval: float = 0.08,
        corpus: Optional[str] = None,
        host: str = "127.0.0.1",
        port: int = 8000,
    ) -> None:
        ns = [c.n_tokens for c in cases]
        if len(set(ns)) != len(ns):
            raise ValueError(f"n_tokens must be unique per case (they key ground truth lookups), got {ns}")
        self.tokenizer = tokenizer
        self.cases = cases
        self.model = model
        self.ttft_delay = ttft_delay
        self.inter_chunk_interval = inter_chunk_interval
        self.host = host
        self.port = port
        self.requests: List[ServedRequest] = []
        self._request_count = 0
        self._runner: Optional[web.AppRunner] = None

        corpus = corpus or _DEFAULT_CORPUS
        self._chunks_by_case: Dict[int, List[str]] = {c.n_tokens: self._build_chunks(corpus, c) for c in cases}

    def get_chunks(self, n_tokens: int) -> List[str]:
        """The exact chunk texts served for the case keyed by ``n_tokens``.
        Tests use these to derive ground truth with an independent tokenizer."""
        return list(self._chunks_by_case[n_tokens])

    # -- response construction ------------------------------------------------

    def _build_chunks(self, corpus: str, case: GoldenCase) -> List[str]:
        if not 1 <= case.n_chunks <= case.n_tokens:
            raise ValueError(f"need 1 <= n_chunks <= n_tokens, got {case}")

        enc = self.tokenizer(corpus, add_special_tokens=False, return_offsets_mapping=True)
        ids = enc.input_ids
        if len(ids) < case.n_tokens:
            raise ValueError(f"corpus has {len(ids)} tokens, case needs {case.n_tokens}")
        # Cut the text at the character where token n_tokens starts, so the
        # response text corresponds to exactly the first n_tokens ids.
        offsets = enc.offset_mapping
        end_char = offsets[case.n_tokens - 1][1]
        full_text = corpus[:end_char]

        if case.split == "token":
            # Chunk boundaries at token-start character offsets: exact
            # concatenation is guaranteed because we slice one string.
            per_chunk, extra = divmod(case.n_tokens, case.n_chunks)
            boundaries = []
            t = 0
            for i in range(case.n_chunks):
                t += per_chunk + (1 if i < extra else 0)
                boundaries.append(end_char if t >= case.n_tokens else offsets[t][0])
            chunks = []
            start = 0
            for b in boundaries:
                chunks.append(full_text[start:b])
                start = b
        elif case.split == "codepoint":
            per_chunk, extra = divmod(len(full_text), case.n_chunks)
            chunks = []
            start = 0
            for i in range(case.n_chunks):
                end = start + per_chunk + (1 if i < extra else 0)
                chunks.append(full_text[start:end])
                start = end
        else:
            raise ValueError(f"unknown split mode {case.split!r}")

        # Self-verification: the fixture must actually be golden.
        if "".join(chunks) != full_text:
            raise AssertionError(f"chunk concatenation != full text for {case}")
        if any(not c for c in chunks):
            raise AssertionError(f"empty chunk built for {case}")
        recount = len(self.tokenizer(full_text, add_special_tokens=False).input_ids)
        if recount != case.n_tokens:
            raise AssertionError(f"constructed text re-encodes to {recount} tokens, case needs {case.n_tokens}")
        return chunks

    # -- lifecycle ------------------------------------------------------------

    async def __aenter__(self) -> "GoldenSimServer":
        app = web.Application()
        app.router.add_post("/v1/completions", self._handle_completions)
        app.router.add_post("/v1/chat/completions", self._handle_chat)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, self.host, self.port)
        await site.start()
        logger.debug("golden sim listening on %s:%d", self.host, self.port)
        return self

    async def __aexit__(self, *exc: Any) -> None:
        if self._runner:
            await self._runner.cleanup()

    # -- request handling -----------------------------------------------------

    def _next_case(self) -> GoldenCase:
        case = self.cases[self._request_count % len(self.cases)]
        self._request_count += 1
        return case

    def _usage(self, prompt_text: str, case: GoldenCase) -> Dict[str, int]:
        # Like vLLM: prompt_tokens includes the sequence-start special tokens,
        # completion_tokens counts generated tokens only.
        prompt_tokens = len(self.tokenizer(prompt_text, add_special_tokens=True).input_ids)
        return {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": case.n_tokens,
            "total_tokens": prompt_tokens + case.n_tokens,
        }

    async def _handle_completions(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        prompt_text = body.get("prompt", "")
        case = self._next_case()
        served = ServedRequest(route="/v1/completions", case=case, prompt_text=prompt_text)
        self.requests.append(served)

        def content_event(text: str, finish: Optional[str]) -> Dict[str, Any]:
            return {
                "id": "cmpl-golden",
                "object": "text_completion",
                "created": 0,
                "model": self.model,
                "choices": [{"index": 0, "text": text, "logprobs": None, "finish_reason": finish}],
            }

        if body.get("stream"):
            return await self._stream(
                request,
                served,
                content_events=[
                    content_event(text, "length" if i == case.n_chunks - 1 else None)
                    for i, text in enumerate(self._chunks_by_case[case.n_tokens])
                ],
                preamble_events=[],
                usage=self._usage(prompt_text, case),
            )

        await asyncio.sleep(self.ttft_delay + self.inter_chunk_interval * (case.n_chunks - 1))
        served.send_times.append(time.perf_counter())
        return web.json_response(
            {
                "id": "cmpl-golden",
                "object": "text_completion",
                "created": 0,
                "model": self.model,
                "choices": [
                    {
                        "index": 0,
                        "text": "".join(self._chunks_by_case[case.n_tokens]),
                        "logprobs": None,
                        "finish_reason": "length",
                    }
                ],
                "usage": self._usage(prompt_text, case),
            }
        )

    async def _handle_chat(self, request: web.Request) -> web.StreamResponse:
        body = await request.json()
        messages = body.get("messages", [])
        prompt_text = "".join(m.get("content", "") for m in messages if isinstance(m.get("content"), str))
        case = self._next_case()
        served = ServedRequest(route="/v1/chat/completions", case=case, prompt_text=prompt_text)
        self.requests.append(served)

        def delta_event(delta: Dict[str, Any], finish: Optional[str]) -> Dict[str, Any]:
            return {
                "id": "chatcmpl-golden",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": self.model,
                "choices": [{"index": 0, "delta": delta, "finish_reason": finish}],
            }

        if body.get("stream"):
            chunks = self._chunks_by_case[case.n_tokens]
            content_events = [{"delta": {"content": text}} for text in chunks]
            return await self._stream(
                request,
                served,
                content_events=[delta_event(e["delta"], None) for e in content_events],
                # Realistic vLLM skeleton: a role-only first delta (no content)
                # and an empty-delta finish event. The client must exclude both
                # from token and timing accounting.
                preamble_events=[delta_event({"role": "assistant", "content": ""}, None)],
                usage=self._usage(prompt_text, case),
                finish_event=delta_event({}, "length"),
            )

        await asyncio.sleep(self.ttft_delay + self.inter_chunk_interval * (case.n_chunks - 1))
        served.send_times.append(time.perf_counter())
        return web.json_response(
            {
                "id": "chatcmpl-golden",
                "object": "chat.completion",
                "created": 0,
                "model": self.model,
                "choices": [
                    {
                        "index": 0,
                        "message": {"role": "assistant", "content": "".join(self._chunks_by_case[case.n_tokens])},
                        "finish_reason": "length",
                    }
                ],
                "usage": self._usage(prompt_text, case),
            }
        )

    async def _stream(
        self,
        request: web.Request,
        served: ServedRequest,
        *,
        content_events: List[Dict[str, Any]],
        preamble_events: List[Dict[str, Any]],
        usage: Dict[str, int],
        finish_event: Optional[Dict[str, Any]] = None,
    ) -> web.StreamResponse:
        resp = web.StreamResponse(
            status=200,
            headers={"Content-Type": "text/event-stream", "Cache-Control": "no-cache"},
        )
        await resp.prepare(request)

        async def send(event: Dict[str, Any]) -> None:
            await resp.write(b"data: " + json.dumps(event).encode() + b"\n\n")

        for event in preamble_events:
            await send(event)

        await asyncio.sleep(self.ttft_delay)
        for i, event in enumerate(content_events):
            if i > 0:
                await asyncio.sleep(self.inter_chunk_interval)
            await send(event)
            # Stamp after the write so the recorded time is as close as
            # possible to when bytes actually left for the client.
            served.send_times.append(time.perf_counter())

        if finish_event is not None:
            await send(finish_event)
        # Trailing usage-only event, as vLLM emits with
        # stream_options.include_usage=true.
        await send(
            {
                "id": "golden-usage",
                "object": "chat.completion.chunk",
                "created": 0,
                "model": self.model,
                "choices": [],
                "usage": usage,
            }
        )
        await resp.write(b"data: [DONE]\n\n")
        await resp.write_eof()
        return resp


_DEFAULT_CORPUS = (
    "The quick brown fox jumps over the lazy dog while seventeen curious "
    "penguins waddle across the frozen harbor, pausing to inspect a bright "
    "red bucket that someone left beside the pier. Meanwhile the lighthouse "
    "keeper brews a second pot of coffee and watches the morning ferry carve "
    "a white line through the gray water, wondering whether the storm the "
    "radio promised will arrive before the afternoon tide turns."
)
