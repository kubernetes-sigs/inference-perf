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
"""Unit tests for running one batch job end to end against an in-process fake of /v1/files and /v1/batches."""

import asyncio
import json
import logging
from typing import Optional

import aiohttp
import pytest
from fake_batch_server import FakeBatchServer, Upload

from inference_perf.client.batch.client import BatchAPIError, BatchClient
from inference_perf.client.batch.job import run_batch_job
from inference_perf.client.batch.jsonl import BatchInput

HEADERS = {"Authorization": "Bearer test-key"}
# Four completion lines, s0-b0-0 to s0-b0-3, for the job to upload.
BATCH_INPUT = BatchInput(
    endpoint="/v1/completions",
    custom_ids=[f"s0-b0-{i}" for i in range(4)],
    content=b"".join(f'{{"custom_id": "s0-b0-{i}"}}\n'.encode() for i in range(4)),
)


# One line of an output or error file: the answer to custom_id, with an HTTP status or an error code.
def _line(custom_id: str, status_code: Optional[int] = None, error_code: Optional[str] = None) -> bytes:
    response = {"status_code": status_code, "body": {}} if status_code is not None else None
    error = {"code": error_code, "message": error_code} if error_code is not None else None
    return json.dumps({"custom_id": custom_id, "response": response, "error": error}).encode() + b"\n"


# The four lines are answered as: s0-b0-0 200 and s0-b0-1 400 in the output file, s0-b0-2 batch_expired in
# the error file, and s0-b0-3 nowhere. The output file also answers "other-0", which the input does not hold.
# The job uploads the input as s0-b0.jsonl, creates the batch with window 30m and metadata {"stage": "0"},
# sees in_progress at clock 11 and completed at clock 12, and counts 1 succeeded, 2 failed and 1 unanswered,
# from clock 10 before the upload to clock 13 after the downloads.
async def test_a_completed_job_counts_each_line_by_its_answer() -> None:
    files = {
        "file-out": _line("s0-b0-0", status_code=200) + _line("s0-b0-1", status_code=400) + _line("other-0", status_code=200),
        "file-err": _line("s0-b0-2", error_code="batch_expired"),
    }
    states: list[dict[str, object]] = [
        {"status": "in_progress"},
        {"status": "completed", "output_file_id": "file-out", "error_file_id": "file-err"},
    ]
    clock = iter([10.0, 11.0, 12.0, 13.0]).__next__
    async with FakeBatchServer(states, files=files) as server, aiohttp.ClientSession() as session:
        client = BatchClient(session, server.base_url, HEADERS, clock=clock)
        result = await asyncio.wait_for(
            run_batch_job(
                client,
                BATCH_INPUT,
                filename="s0-b0.jsonl",
                completion_window="30m",
                metadata={"stage": "0"},
                poll_interval_sec=0,
                should_stop=lambda: False,
            ),
            timeout=5,
        )

    assert server.uploads == [Upload(purpose="batch", filename="s0-b0.jsonl", content=BATCH_INPUT.content)]
    assert server.creates == [
        {"input_file_id": "file-2", "endpoint": "/v1/completions", "completion_window": "30m", "metadata": {"stage": "0"}}
    ]
    assert result.batch.status == "completed"
    assert result.status_first_seen == {"in_progress": 11.0, "completed": 12.0}
    assert (result.submitted_at, result.finished_at) == (10.0, 13.0)
    assert (result.lines, result.succeeded, result.failed, result.unanswered) == (4, 1, 2, 1)


# A job whose should_stop answers true stops waiting at the first poll, with the batch still in_progress and
# no output to count: all 4 lines are unanswered, and the job finishes at clock 12, after that poll at 11.
async def test_a_stopped_job_leaves_every_line_unanswered() -> None:
    states: list[dict[str, object]] = [{"status": "in_progress"}, {"status": "completed"}]
    clock = iter([10.0, 11.0, 12.0]).__next__
    async with FakeBatchServer(states) as server, aiohttp.ClientSession() as session:
        client = BatchClient(session, server.base_url, HEADERS, clock=clock)
        result = await asyncio.wait_for(
            run_batch_job(
                client,
                BATCH_INPUT,
                filename="s0-b0.jsonl",
                completion_window="24h",
                metadata={},
                poll_interval_sec=0,
                should_stop=lambda: True,
            ),
            timeout=5,
        )

    assert (result.batch.status, server.polls) == ("in_progress", 1)
    assert result.finished_at == 12.0
    assert (result.lines, result.succeeded, result.failed, result.unanswered) == (4, 0, 0, 4)


# The job's batch is created as batch-0 and its first poll then fails with a 500. The id was already logged with
# the job's file name, s0-b0.jsonl, so a batch whose job fails can still be found on the server.
async def test_the_batch_id_is_logged_when_the_batch_is_created(caplog: pytest.LogCaptureFixture) -> None:
    async with FakeBatchServer(poll_states=[]) as server, aiohttp.ClientSession() as session:
        client = BatchClient(session, server.base_url, HEADERS)
        with (
            caplog.at_level(logging.INFO, logger="inference_perf.client.batch.job"),
            pytest.raises(BatchAPIError, match="500"),
        ):
            await asyncio.wait_for(
                run_batch_job(
                    client,
                    BATCH_INPUT,
                    filename="s0-b0.jsonl",
                    completion_window="24h",
                    metadata={},
                    poll_interval_sec=0,
                    should_stop=lambda: False,
                ),
                timeout=5,
            )

    assert any("batch-0" in record.message and "s0-b0.jsonl" in record.message for record in caplog.records)
