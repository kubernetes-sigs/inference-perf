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
"""A client for the OpenAI Batch API: upload an input file, create a batch, poll it until it reaches an end
state, and download its output and error files.

The caller passes the aiohttp session and the headers to send, so TLS, timeouts and auth stay the caller's.
"""

import asyncio
import time
from typing import Callable, Mapping

import aiohttp
from pydantic import BaseModel

from inference_perf.client.batch.models import Batch, BatchFile, BatchOutputLine

FILES_ROUTE = "/v1/files"
BATCHES_ROUTE = "/v1/batches"
# A batch in one of these statuses does not change again.
END_STATES = frozenset({"completed", "failed", "expired", "cancelled"})


class BatchAPIError(Exception):
    """The server answered a Batch API call with an error status."""


class BatchPollResult(BaseModel):
    """A batch as the poll that saw it reach an end state returned it."""

    batch: Batch
    # For each status the polls returned, the client clock at the first poll that returned it.
    status_first_seen: dict[str, float]


async def _read_ok(response: aiohttp.ClientResponse) -> bytes:
    body = await response.read()
    if not response.ok:
        raise BatchAPIError(f"{response.method} {response.url} returned {response.status}: {body.decode(errors='replace')}")
    return body


def parse_batch_output(content: bytes) -> list[BatchOutputLine]:
    """The lines of an output or error file, in file order. Blank lines are skipped."""
    return [BatchOutputLine.model_validate_json(line) for line in content.splitlines() if line.strip()]


class BatchClient:
    def __init__(
        self,
        session: aiohttp.ClientSession,
        base_url: str,
        headers: Mapping[str, str],
        clock: Callable[[], float] = time.perf_counter,
    ) -> None:
        self._session = session
        # Each route starts with "/", so a trailing one here would make "//v1/files".
        self._base_url = base_url.rstrip("/")
        # aiohttp sets the Content-Type each call needs: JSON to create a batch, multipart to upload a file.
        # The JSON Content-Type the model-server headers carry would break the upload, so it is dropped.
        self._headers = {key: value for key, value in headers.items() if key.lower() != "content-type"}
        # The client clock each poll's status_first_seen reads, and that callers time a whole job with.
        self.clock = clock

    async def upload_file(self, content: bytes, filename: str) -> BatchFile:
        form = aiohttp.FormData()
        form.add_field("purpose", "batch")
        form.add_field("file", content, filename=filename)
        async with self._session.post(self._base_url + FILES_ROUTE, data=form, headers=self._headers) as response:
            return BatchFile.model_validate_json(await _read_ok(response))

    async def create_batch(
        self, input_file_id: str, endpoint: str, completion_window: str, metadata: Mapping[str, str]
    ) -> Batch:
        payload = {
            "input_file_id": input_file_id,
            "endpoint": endpoint,
            "completion_window": completion_window,
            "metadata": dict(metadata),
        }
        async with self._session.post(self._base_url + BATCHES_ROUTE, json=payload, headers=self._headers) as response:
            return Batch.model_validate_json(await _read_ok(response))

    async def retrieve_batch(self, batch_id: str) -> Batch:
        async with self._session.get(f"{self._base_url}{BATCHES_ROUTE}/{batch_id}", headers=self._headers) as response:
            return Batch.model_validate_json(await _read_ok(response))

    async def wait_for_batch(
        self, batch_id: str, poll_interval_sec: float, should_stop: Callable[[], bool] = lambda: False
    ) -> BatchPollResult:
        """Poll the batch every `poll_interval_sec` until it reaches an end state.

        `should_stop` is asked after each poll. Once it answers true, the wait returns the batch as that poll
        saw it, which may not have ended; the batch goes on running on the server.
        """
        status_first_seen: dict[str, float] = {}
        while True:
            batch = await self.retrieve_batch(batch_id)
            status_first_seen.setdefault(batch.status, self.clock())
            if batch.status in END_STATES or should_stop():
                return BatchPollResult(batch=batch, status_first_seen=status_first_seen)
            await asyncio.sleep(poll_interval_sec)

    async def download_file(self, file_id: str) -> bytes:
        async with self._session.get(f"{self._base_url}{FILES_ROUTE}/{file_id}/content", headers=self._headers) as response:
            return await _read_ok(response)
