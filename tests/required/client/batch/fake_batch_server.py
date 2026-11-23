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
"""In-process fake of the OpenAI Batch API (/v1/files and /v1/batches) for the batch client tests.

It answers over real HTTP the way llm-d-batch-gateway does: a multipart upload with `purpose` and `file`
parts, a batch create, one batch that moves through a scripted list of states as it is polled, and file
content downloads. A call without the expected Authorization header gets a 401. Every upload and create it
accepts is recorded, so tests assert what the client sent rather than what it meant to send.
"""

from dataclasses import dataclass
from types import TracebackType
from typing import Optional, Type

from aiohttp import web
from aiohttp.typedefs import Handler

AUTHORIZATION = "Bearer test-key"
# The created_at of every batch, in Unix seconds: 2026-01-01T00:00:00Z.
CREATED_AT = 1767225600


@dataclass(frozen=True)
class Upload:
    """One accepted upload: its purpose, and the name and bytes of its file part."""

    purpose: str
    filename: str
    content: bytes


# A 4xx reply with an OpenAI error body, {"error": {"message": message, ...}}.
def _error(status: int, message: str) -> web.Response:
    return web.json_response({"error": {"message": message, "type": "invalid_request_error", "code": None}}, status=status)


# Passes a request with "Authorization: Bearer test-key" on to its handler; anything else gets a 401
# whose message is "Incorrect API key provided".
@web.middleware
async def _require_authorization(request: web.Request, handler: Handler) -> web.StreamResponse:
    if request.headers.get("Authorization") != AUTHORIZATION:
        return _error(401, "Incorrect API key provided")
    return await handler(request)


class FakeBatchServer:
    """Serves one batch, created through POST /v1/batches. Use as an async context manager.

    Each GET of the batch first updates it with the next entry of `poll_states`; once the list runs out,
    the batch stays in its last state. `files` are served at /v1/files/{id}/content, next to uploads.
    """

    # A server whose batch goes through `poll_states`, e.g. [{"status": "in_progress"}, {"status": "completed"}],
    # and that already holds `files`, e.g. {"file-out": b"..."}, before any upload.
    def __init__(self, poll_states: list[dict[str, object]], files: Optional[dict[str, bytes]] = None) -> None:
        self.poll_states = poll_states
        self.files: dict[str, bytes] = dict(files or {})
        self.uploads: list[Upload] = []
        self.creates: list[dict[str, object]] = []
        self.polls = 0
        self.batch: dict[str, object] = {}
        self._runner: Optional[web.AppRunner] = None
        self.port = 0

    # "http://127.0.0.1:<port>", the root the routes hang off.
    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    # Starts the server on a free port; base_url is valid from here on.
    async def __aenter__(self) -> "FakeBatchServer":
        app = web.Application(middlewares=[_require_authorization])
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

    # Stops the server.
    async def __aexit__(
        self, exc_type: Optional[Type[BaseException]], exc: Optional[BaseException], tb: Optional[TracebackType]
    ) -> None:
        if self._runner is not None:
            await self._runner.cleanup()

    # POST /v1/files: a multipart body with purpose "batch" and a file part that has a filename is stored
    # as file-<n> and answered with its file object; any other body gets a 400.
    async def _upload(self, request: web.Request) -> web.Response:
        form = await request.post()
        purpose = form.get("purpose")
        part = form.get("file")
        if purpose != "batch" or not isinstance(part, web.FileField):
            return _error(400, "expected a purpose of 'batch' and a file part with a filename")
        file_id = f"file-{len(self.files)}"
        content = part.file.read()
        self.files[file_id] = content
        self.uploads.append(Upload(purpose=purpose, filename=part.filename, content=content))
        return web.json_response(
            {
                "id": file_id,
                "object": "file",
                "bytes": len(content),
                "created_at": CREATED_AT,
                "expires_at": None,
                "filename": part.filename,
                "purpose": "batch",
                "status": "processed",
            }
        )

    # GET /v1/files/{file_id}/content: the stored bytes, or a 404 for an unknown id.
    async def _content(self, request: web.Request) -> web.Response:
        content = self.files.get(request.match_info["file_id"])
        if content is None:
            return _error(404, "No such file")
        return web.Response(body=content, content_type="application/octet-stream")

    # POST /v1/batches: a JSON body naming a stored input file, an endpoint and a completion window creates
    # batch-0 in status "validating", with every other timestamp and file id null; anything else gets a 400.
    async def _create(self, request: web.Request) -> web.Response:
        body: dict[str, object] = await request.json()
        if body.get("input_file_id") not in self.files or not body.get("endpoint") or not body.get("completion_window"):
            return _error(400, "expected input_file_id of an uploaded file, endpoint and completion_window")
        self.creates.append(body)
        self.batch = {
            "id": "batch-0",
            "object": "batch",
            "endpoint": body["endpoint"],
            "input_file_id": body["input_file_id"],
            "completion_window": body["completion_window"],
            "metadata": body.get("metadata"),
            "status": "validating",
            "created_at": CREATED_AT,
            "in_progress_at": None,
            "finalizing_at": None,
            "completed_at": None,
            "failed_at": None,
            "expired_at": None,
            "expires_at": None,
            "cancelling_at": None,
            "cancelled_at": None,
            "output_file_id": None,
            "error_file_id": None,
            "request_counts": {"total": 0, "completed": 0, "failed": 0},
            "errors": None,
            "usage": None,
        }
        return web.json_response(self.batch)

    # GET /v1/batches/{batch_id}: batch-0 after applying the next poll state, or a 404 for any other id.
    async def _retrieve(self, request: web.Request) -> web.Response:
        if request.match_info["batch_id"] != self.batch.get("id"):
            return _error(404, "No such batch")
        self.batch.update(self.poll_states[min(self.polls, len(self.poll_states) - 1)])
        self.polls += 1
        return web.json_response(self.batch)
