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
"""Runs one batch job: upload its input file, create the batch, wait for it to end, then download its output
and error files and count how each input line was answered."""

import logging
from typing import Callable, Mapping

from pydantic import BaseModel

from inference_perf.client.batch.client import BatchClient, parse_batch_output
from inference_perf.client.batch.jsonl import BatchInput
from inference_perf.client.batch.models import Batch, BatchOutputLine

logger = logging.getLogger(__name__)


class BatchJobResult(BaseModel):
    """How one batch job went. Times are on the batch client's clock."""

    # The batch as the last poll returned it. Its status is an end state unless the wait was stopped first.
    batch: Batch
    # For each status the polls returned, the time of the first poll that returned it.
    status_first_seen: dict[str, float]
    # Before the upload began, and after the last download, or after the last poll when nothing was downloaded.
    submitted_at: float
    finished_at: float
    # Input lines, and how many were answered with status 200, with a failure (another status or an error
    # line), or not at all.
    lines: int
    succeeded: int
    failed: int
    unanswered: int


async def run_batch_job(
    client: BatchClient,
    batch_input: BatchInput,
    *,
    filename: str,
    completion_window: str,
    metadata: Mapping[str, str],
    poll_interval_sec: float,
    should_stop: Callable[[], bool],
) -> BatchJobResult:
    """Submit `batch_input` as one batch and wait for it to end, or until `should_stop` answers true."""
    submitted_at = client.clock()
    uploaded = await client.upload_file(batch_input.content, filename)
    created = await client.create_batch(uploaded.id, batch_input.endpoint, completion_window, metadata)
    # Logged before the wait, so a batch whose job fails later can still be found on the server.
    logger.info("Created batch %s from %s", created.id, filename)
    poll = await client.wait_for_batch(created.id, poll_interval_sec, should_stop)

    # A batch that has not ended has neither file yet.
    answers: dict[str, BatchOutputLine] = {}
    for file_id in (poll.batch.output_file_id, poll.batch.error_file_id):
        if file_id is not None:
            answers.update((line.custom_id, line) for line in parse_batch_output(await client.download_file(file_id)))

    # Counted by the input's custom_ids, so a line the input does not hold counts for nothing.
    succeeded = failed = unanswered = 0
    for custom_id in batch_input.custom_ids:
        answer = answers.get(custom_id)
        if answer is None:
            unanswered += 1
        elif answer.response is not None and answer.response.status_code == 200:
            succeeded += 1
        else:
            failed += 1

    return BatchJobResult(
        batch=poll.batch,
        status_first_seen=poll.status_first_seen,
        submitted_at=submitted_at,
        finished_at=client.clock(),
        lines=len(batch_input.custom_ids),
        succeeded=succeeded,
        failed=failed,
        unanswered=unanswered,
    )
