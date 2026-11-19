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
"""In-process stand-in for SGLang's native /generate route (#986).

llm-d-inference-sim does not serve /generate, so the e2e tier points
inference-perf at this instead. It streams the way SGLang's tokenizer_manager
does: by default every chunk repeats all the text and output_ids so far
(cumulative); a server started with --incremental-streaming-output sends only
the new ones. In both modes meta_info.completion_tokens is the running total.
Every request gets the same scripted tokens, one per chunk, and the server
records when each request arrived and when each chunk was sent.
"""

import asyncio
import json
import time
from dataclasses import dataclass, field
from types import TracebackType
from typing import Dict, List, Optional, Type

from aiohttp import web


@dataclass
class ServedGenerate:
    """What the fake did for one /generate request. Times are perf_counter()
    stamps, taken after each chunk was written and drained."""

    body: Dict[str, object]
    arrival_time: float = 0.0
    send_times: List[float] = field(default_factory=list)


class FakeSGLangServer:
    """Serves POST /generate, streamed or unary, and GET /v1/models (the
    readiness probe of the sglang client). Use as an async context manager;
    ``base_url`` is the server root."""

    def __init__(
        self,
        tokens: List[str],
        *,
        incremental: bool,
        delay: float = 0.0,
        prompt_tokens: int = 4,
        cached_tokens: int = 0,
    ) -> None:
        self.tokens = tokens
        self.incremental = incremental
        self.delay = delay
        self.prompt_tokens = prompt_tokens
        self.cached_tokens = cached_tokens
        self.served: List[ServedGenerate] = []
        self._runner: Optional[web.AppRunner] = None
        self.port = 0

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def _meta_info(self, completion_tokens: int, finished: bool) -> Dict[str, object]:
        n = len(self.tokens)
        return {
            "id": "fake",
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": completion_tokens,
            "cached_tokens": self.cached_tokens,
            "finish_reason": {"type": "length", "length": n} if finished else None,
        }

    async def _models(self, request: web.Request) -> web.Response:
        return web.json_response({"object": "list", "data": [{"id": "fake-sglang", "object": "model"}]})

    async def _generate(self, request: web.Request) -> web.StreamResponse:
        served = ServedGenerate(body=await request.json(), arrival_time=time.perf_counter())
        self.served.append(served)
        n = len(self.tokens)
        if not served.body.get("stream"):
            await asyncio.sleep(self.delay * n)
            return web.json_response(
                {"text": "".join(self.tokens), "output_ids": list(range(n)), "meta_info": self._meta_info(n, finished=True)}
            )

        response = web.StreamResponse(headers={"Content-Type": "text/event-stream"})
        await response.prepare(request)
        for end in range(1, n + 1):
            await asyncio.sleep(self.delay)
            start = end - 1 if self.incremental else 0
            chunk = {
                "text": "".join(self.tokens[start:end]),
                "output_ids": list(range(start, end)),
                "meta_info": self._meta_info(end, finished=end == n),
            }
            await response.write(b"data: " + json.dumps(chunk).encode() + b"\n\n")
            served.send_times.append(time.perf_counter())
        await response.write(b"data: [DONE]\n\n")
        await response.write_eof()
        return response

    async def __aenter__(self) -> "FakeSGLangServer":
        app = web.Application()
        app.router.add_get("/v1/models", self._models)
        app.router.add_post("/generate", self._generate)
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
