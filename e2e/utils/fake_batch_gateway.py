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
"""An in-process stand-in for an OpenAI Batch API server, for running the CLI with `load.type: batch`.

It serves /health, /v1/files and /v1/batches over real HTTP, for any number of batches. A batch is in_progress
at its first poll and completed at its second, and its output file then answers every input line with status
200, except lines whose custom_id ends in "-1", which get 400. Every upload is kept, so a test can check the
lines the CLI wrote.
"""

import json
import time
from types import TracebackType
from typing import Optional, Type

from aiohttp import web


class FakeBatchGateway:
    """Use as an async context manager; base_url is valid inside it."""

    # A gateway with no files and no batches yet.
    def __init__(self) -> None:
        self.files: dict[str, bytes] = {}
        self.batches: dict[str, dict[str, object]] = {}
        self.polls: dict[str, int] = {}
        self.uploads: list[bytes] = []
        self._runner: Optional[web.AppRunner] = None
        self.port = 0

    # "http://127.0.0.1:<port>".
    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # Starts serving on a free port.
    async def __aenter__(self) -> "FakeBatchGateway":
        app = web.Application()
        app.router.add_get("/health", self._health)
        app.router.add_post("/v1/files", self._upload)
        app.router.add_get("/v1/files/{file_id}/content", self._content)
        app.router.add_post("/v1/batches", self._create)
        app.router.add_get("/v1/batches/{batch_id}", self._retrieve)
        self._runner = web.AppRunner(app)
        await self._runner.setup()
        site = web.TCPSite(self._runner, "127.0.0.1", 0)
        await site.start()
        self.port = self._runner.addresses[0][1]
        return self

    # Stops serving.
    async def __aexit__(
        self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]
    ) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    # GET /health: 200, so the vLLM client's readiness wait passes.
    async def _health(self, request: web.Request) -> web.Response:
        return web.Response(text="ok")

    # POST /v1/files: stores the multipart file part as file-<n> and answers with its id.
    async def _upload(self, request: web.Request) -> web.Response:
        form = await request.post()
        part = form["file"]
        assert isinstance(part, web.FileField)
        content = part.file.read()
        file_id = f"file-{len(self.files)}"
        self.files[file_id] = content
        self.uploads.append(content)
        return web.json_response({"id": file_id, "object": "file", "purpose": form["purpose"]})

    # POST /v1/batches: creates batch-<n> for the named input file, in status validating.
    async def _create(self, request: web.Request) -> web.Response:
        body = await request.json()
        batch_id = f"batch-{len(self.batches)}"
        self.batches[batch_id] = {
            "id": batch_id,
            "object": "batch",
            "status": "validating",
            "endpoint": body["endpoint"],
            "input_file_id": body["input_file_id"],
            "completion_window": body["completion_window"],
            "created_at": int(time.time()),
        }
        self.polls[batch_id] = 0
        return web.json_response(self.batches[batch_id])

    # GET /v1/batches/{id}: in_progress at the first poll; at the second, completed with an output file that
    # answers each input line 200, or 400 when its custom_id ends in "-1".
    async def _retrieve(self, request: web.Request) -> web.Response:
        batch = self.batches[request.match_info["batch_id"]]
        batch_id = str(batch["id"])
        self.polls[batch_id] += 1
        if self.polls[batch_id] == 1:
            batch.update(status="in_progress", in_progress_at=int(time.time()))
        elif batch["status"] != "completed":
            answers = []
            for line in self.files[str(batch["input_file_id"])].splitlines():
                custom_id = json.loads(line)["custom_id"]
                status = 400 if custom_id.endswith("-1") else 200
                answers.append(json.dumps({"custom_id": custom_id, "response": {"status_code": status, "body": {}}}))
            self.files[f"output-{batch_id}"] = ("\n".join(answers) + "\n").encode()
            batch.update(status="completed", completed_at=int(time.time()), output_file_id=f"output-{batch_id}")
        return web.json_response(batch)

    # GET /v1/files/{id}/content: the stored bytes.
    async def _content(self, request: web.Request) -> web.Response:
        return web.Response(body=self.files[request.match_info["file_id"]])
