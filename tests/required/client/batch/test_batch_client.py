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
"""Unit tests for the Batch API client, run against an in-process fake of /v1/files and /v1/batches."""

import asyncio
import time
from contextlib import asynccontextmanager
from typing import AsyncIterator, Callable

from unittest.mock import MagicMock

import aiohttp
import pytest
from fake_batch_server import CREATED_AT, FakeBatchServer, Upload

from inference_perf.client.batch.client import BatchAPIError, BatchClient, parse_batch_output
from inference_perf.client.modelserver.openai_client import openAIModelServerClient, openAIModelServerClientSession
from inference_perf.config import APIConfig

# What the model-server client sends: the key, and a JSON Content-Type that the multipart upload must not use.
HEADERS = {"Authorization": "Bearer test-key", "Content-Type": "application/json"}


# A BatchClient for `server` on a fresh aiohttp session. It sends `headers` (HEADERS unless given) on every
# call and reads `clock` (perf_counter unless given) at each poll.
@asynccontextmanager
async def _client(
    server: FakeBatchServer, headers: dict[str, str] = HEADERS, clock: Callable[[], float] = time.perf_counter
) -> AsyncIterator[BatchClient]:
    async with aiohttp.ClientSession() as session:
        yield BatchClient(session, server.base_url, headers, clock=clock)


# Uploading b'{"a": 1}\n' as "s0-b0.jsonl" sends one multipart file with purpose "batch", even though the
# headers carry a JSON Content-Type, and returns the id the server gave it, file-0.
async def test_upload_sends_a_batch_purpose_file() -> None:
    async with FakeBatchServer(poll_states=[]) as server, _client(server) as client:
        uploaded = await client.upload_file(b'{"a": 1}\n', "s0-b0.jsonl")

    assert uploaded.id == "file-0"
    assert server.uploads == [Upload(purpose="batch", filename="s0-b0.jsonl", content=b'{"a": 1}\n')]


# A base_url with a trailing slash, "http://127.0.0.1:<port>/", still uploads to /v1/files rather than
# //v1/files, which the server does not route, and gets file-0 back.
async def test_a_trailing_slash_on_the_base_url_is_dropped() -> None:
    async with FakeBatchServer(poll_states=[]) as server, aiohttp.ClientSession() as session:
        client = BatchClient(session, server.base_url + "/", HEADERS)
        uploaded = await client.upload_file(b'{"a": 1}\n', "s0-b0.jsonl")

    assert uploaded.id == "file-0"


# Creating a batch for file-0 against /v1/chat/completions with a 24h window and metadata {"stage": "0"}
# posts exactly those four fields, and returns batch-0 in status "validating".
async def test_create_sends_the_input_file_endpoint_window_and_metadata() -> None:
    async with FakeBatchServer(poll_states=[], files={"file-0": b"{}\n"}) as server, _client(server) as client:
        batch = await client.create_batch("file-0", "/v1/chat/completions", "24h", {"stage": "0"})

    assert (batch.id, batch.status, batch.created_at) == ("batch-0", "validating", CREATED_AT)
    assert server.creates == [
        {"input_file_id": "file-0", "endpoint": "/v1/chat/completions", "completion_window": "24h", "metadata": {"stage": "0"}}
    ]


# Polls that return validating, validating, in_progress, finalizing and completed, read at clock 1 to 5, stop
# at the fifth poll with the completed batch. Each status keeps the clock of the first poll that returned it:
# validating 1, in_progress 3, finalizing 4, completed 5.
async def test_wait_polls_to_an_end_state_and_records_when_each_status_was_first_seen() -> None:
    states: list[dict[str, object]] = [
        {"status": "validating"},
        {"status": "validating"},
        {
            "status": "in_progress",
            "in_progress_at": CREATED_AT + 2,
            "request_counts": {"total": 2, "completed": 0, "failed": 0},
        },
        {"status": "finalizing", "finalizing_at": CREATED_AT + 9, "request_counts": {"total": 2, "completed": 1, "failed": 1}},
        {"status": "completed", "completed_at": CREATED_AT + 10, "output_file_id": "file-out", "error_file_id": "file-err"},
    ]
    clock = iter([1.0, 2.0, 3.0, 4.0, 5.0]).__next__
    async with FakeBatchServer(states, files={"file-0": b"{}\n"}) as server, _client(server, clock=clock) as client:
        created = await client.create_batch("file-0", "/v1/completions", "30m", {})
        result = await asyncio.wait_for(client.wait_for_batch(created.id, poll_interval_sec=0), timeout=5)

    assert server.polls == 5
    assert result.status_first_seen == {"validating": 1.0, "in_progress": 3.0, "finalizing": 4.0, "completed": 5.0}
    batch = result.batch
    assert (batch.status, batch.in_progress_at, batch.finalizing_at, batch.completed_at) == (
        "completed",
        CREATED_AT + 2,
        CREATED_AT + 9,
        CREATED_AT + 10,
    )
    assert (batch.output_file_id, batch.error_file_id) == ("file-out", "file-err")
    assert (batch.request_counts.total, batch.request_counts.completed, batch.request_counts.failed) == (2, 1, 1)


# A batch that is in_progress and then `status` is done at the second poll, for each of the four end
# states: completed, failed, expired and cancelled.
@pytest.mark.parametrize("status", ["completed", "failed", "expired", "cancelled"])
async def test_each_end_state_stops_the_wait(status: str) -> None:
    states: list[dict[str, object]] = [{"status": "in_progress"}, {"status": status}]
    async with FakeBatchServer(states, files={"file-0": b"{}\n"}) as server, _client(server) as client:
        created = await client.create_batch("file-0", "/v1/completions", "24h", {})
        result = await asyncio.wait_for(client.wait_for_batch(created.id, poll_interval_sec=0), timeout=5)

    assert (result.batch.status, server.polls) == (status, 2)


# "cancelling" is on the way to "cancelled", not an end state: a batch that polls cancelling and then
# cancelled is done at the second poll, not the first.
async def test_cancelling_does_not_stop_the_wait() -> None:
    states: list[dict[str, object]] = [{"status": "cancelling"}, {"status": "cancelled"}]
    async with FakeBatchServer(states, files={"file-0": b"{}\n"}) as server, _client(server) as client:
        created = await client.create_batch("file-0", "/v1/completions", "24h", {})
        result = await asyncio.wait_for(client.wait_for_batch(created.id, poll_interval_sec=0), timeout=5)

    assert (result.batch.status, server.polls) == ("cancelled", 2)


# A wait whose should_stop answers true returns after the first poll with the batch still in_progress, rather
# than polling on until it is completed.
async def test_should_stop_ends_the_wait_before_the_batch_ends() -> None:
    states: list[dict[str, object]] = [{"status": "in_progress"}, {"status": "completed"}]
    async with FakeBatchServer(states, files={"file-0": b"{}\n"}) as server, _client(server) as client:
        created = await client.create_batch("file-0", "/v1/completions", "24h", {})
        result = await asyncio.wait_for(
            client.wait_for_batch(created.id, poll_interval_sec=0, should_stop=lambda: True), timeout=5
        )

    assert (result.batch.status, server.polls) == ("in_progress", 1)


# Downloading file-out returns its bytes unchanged, here two output lines.
async def test_download_returns_the_file_bytes() -> None:
    content = b'{"custom_id": "s0-b0-0"}\n{"custom_id": "s0-b0-1"}\n'
    async with FakeBatchServer(poll_states=[], files={"file-out": content}) as server, _client(server) as client:
        downloaded = await client.download_file("file-out")

    assert downloaded == content


# A call the server refuses raises BatchAPIError with the status and the server's message: a wrong key
# gets 401 "Incorrect API key provided".
async def test_a_refused_call_raises_with_the_status_and_message() -> None:
    async with FakeBatchServer(poll_states=[]) as server, _client(server, headers={"Authorization": "Bearer wrong"}) as client:
        with pytest.raises(BatchAPIError, match="401.*Incorrect API key provided"):
            await client.download_file("file-0")


# An output line and an error line parse in file order: s0-b0-0 answered 200 with its body, and s0-b0-1 has
# no response and the error code batch_expired.
def test_parse_batch_output_reads_each_line_in_order() -> None:
    content = (
        b'{"id": "r0", "custom_id": "s0-b0-0", "response": {"status_code": 200, "request_id": "q0",'
        b' "body": {"usage": {"completion_tokens": 3}}}, "error": null}\n'
        b'{"id": "r1", "custom_id": "s0-b0-1", "response": null,'
        b' "error": {"code": "batch_expired", "message": "The completion window expired."}}\n'
    )

    first, second = parse_batch_output(content)

    assert first.custom_id == "s0-b0-0" and first.error is None
    assert first.response is not None
    assert (first.response.status_code, first.response.body) == (200, {"usage": {"completion_tokens": 3}})
    assert second.custom_id == "s0-b0-1" and second.response is None
    assert second.error is not None and second.error.code == "batch_expired"


# A file with an empty line between its two lines and a whitespace-only line at the end parses to those two
# lines, s0-b0-0 and s0-b0-1, instead of failing on the blank ones.
def test_parse_batch_output_skips_blank_lines() -> None:
    content = b'{"custom_id": "s0-b0-0"}\n\n{"custom_id": "s0-b0-1"}\n \n'

    lines = parse_batch_output(content)

    assert [line.custom_id for line in lines] == ["s0-b0-0", "s0-b0-1"]


# A model-server session for a client at the fake server's address with api_key "test-key" hands out a batch
# client that sends "Authorization: Bearer test-key", so the fake, which refuses any other key, takes its upload.
async def test_the_model_server_session_hands_out_a_batch_client_with_its_key() -> None:
    async with FakeBatchServer(poll_states=[]) as server:
        model_client = MagicMock(spec=openAIModelServerClient)
        model_client.uri = server.base_url
        model_client.api_key = "test-key"
        model_client.api_config = APIConfig()
        model_client.timeout = None
        model_client.cert_path = None
        model_client.key_path = None
        model_client.max_tcp_connections = 10
        session = openAIModelServerClientSession(model_client)
        try:
            uploaded = await session.batch_client().upload_file(b"{}\n", "s0-b0.jsonl")
        finally:
            await session.close()

    assert uploaded.id == "file-0"
