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
"""Unit tests for load.type batch: how the load generator turns batch stages into batch jobs.

run_batch_job is replaced by a recorder, so these tests pin what each job is given and how its outcome
ends the stage. The job itself is tested against a fake Batch API in tests/required/client/batch.
"""

import asyncio
import json
import logging
from dataclasses import dataclass
from typing import Awaitable, Callable, Generator, Mapping, Optional, Sequence
from unittest.mock import AsyncMock, MagicMock, patch

import aiohttp
import pytest
from pydantic import ValidationError

from inference_perf.apis import CompletionAPIData, InferenceAPIData
from inference_perf.client.batch.client import BatchAPIError
from inference_perf.client.batch.job import BatchJobResult
from inference_perf.client.batch.jsonl import BatchInput, build_batch_input
from inference_perf.client.batch.models import Batch, BatchError, BatchErrors
from inference_perf.client.modelserver import MockModelServerClient
from inference_perf.client.modelserver.openai_client import openAIModelServerClient
from inference_perf.client.server_metrics.base import StageStatus
from inference_perf.config import (
    APIConfig,
    APIType,
    BatchJobConfig,
    BatchLoadStage,
    DataConfig,
    DataGenType,
    LoadConfig,
    LoadType,
    StandardLoadStage,
)
from inference_perf.config.apis.config import ResponseFormat, ResponseFormatType
from inference_perf.datagen import MockDataGenerator, SessionGenerator
from inference_perf.loadgen.load_generator import LoadGenerator


@dataclass
class _Job:
    """One call the load generator made to run_batch_job."""

    batch_input: BatchInput
    filename: str
    completion_window: str
    metadata: Mapping[str, str]
    poll_interval_sec: float
    should_stop: Callable[[], bool]

    # The JSON body of each line, in order.
    @property
    def bodies(self) -> list[dict[str, object]]:
        return [json.loads(line)["body"] for line in self.batch_input.content.splitlines()]


class _JobRecorder:
    """Stands in for run_batch_job: records each call and answers with its batch completed and every line succeeded.

    `answer` replaces that answer for one job, keyed by the job's filename such as "s0-b1.jsonl".
    """

    def __init__(self) -> None:
        self.jobs: list[_Job] = []
        self.answer: dict[str, Callable[[_Job], Awaitable[BatchJobResult]]] = {}

    async def __call__(
        self,
        client: object,
        batch_input: BatchInput,
        *,
        filename: str,
        completion_window: str,
        metadata: Mapping[str, str],
        poll_interval_sec: float,
        should_stop: Callable[[], bool],
    ) -> BatchJobResult:
        job = _Job(batch_input, filename, completion_window, metadata, poll_interval_sec, should_stop)
        self.jobs.append(job)
        answer = self.answer.get(filename)
        if answer is None:
            return _result(batch_input, "completed")
        return await answer(job)


# The result of a job whose batch, named "batch-" and its first custom_id, ended in `status`. Every line
# succeeded if it completed, and none did otherwise.
def _result(batch_input: BatchInput, status: str) -> BatchJobResult:
    lines = len(batch_input.custom_ids)
    batch = Batch(
        id=f"batch-{batch_input.custom_ids[0]}",
        status=status,
        endpoint=batch_input.endpoint,
        input_file_id="file-0",
        completion_window="24h",
        created_at=0,
    )
    succeeded = lines if status == "completed" else 0
    return BatchJobResult(
        batch=batch,
        status_first_seen={status: 1.0},
        submitted_at=0.0,
        finished_at=1.0,
        lines=lines,
        succeeded=succeeded,
        failed=0,
        unanswered=lines - succeeded,
    )


class _Session:
    """Stands in for the model-server session a batch run opens: its batch client is a placeholder, since
    run_batch_job, the only thing that would use it, is replaced. It records whether it was closed."""

    def __init__(self) -> None:
        self.closed = False

    def batch_client(self) -> str:
        return "batch client"

    async def close(self) -> None:
        self.closed = True


# A client of an OpenAI-compatible server type for model "m" with 16 max tokens and ignore_eos false.
def _client(api_config: Optional[APIConfig] = None) -> MagicMock:
    client = MagicMock(spec=openAIModelServerClient)
    client.api_config = api_config if api_config is not None else APIConfig()
    client.model_name = "m"
    client.max_completion_tokens = 16
    client.ignore_eos = False
    return client


# A load generator for load.type batch with `stages`, 2 worker processes (which a batch run must not start)
# and no wait between stages. Its records are the mock generator's: completion prompts "1 2 3 1", "1 2 3 2", ...
# afresh for each stage, or chat prompts "mock prompt 1", ... for api_type chat.
def _load_generator(stages: list[BatchLoadStage], api_type: APIType = APIType.Completion) -> LoadGenerator:
    datagen = MockDataGenerator(APIConfig(type=api_type), DataConfig(type=DataGenType.Mock), None)
    return LoadGenerator(datagen, LoadConfig(type=LoadType.BATCH, stages=stages, num_workers=2, interval=0))


# Runs `loadgen` with run_batch_job replaced by `recorder`, and each model-server session it opens replaced by
# a _Session added to `sessions`. Starting worker processes fails the test.
async def _run(
    loadgen: LoadGenerator,
    recorder: _JobRecorder,
    client: Optional[MagicMock] = None,
    sessions: Optional[list[_Session]] = None,
) -> None:
    opened = sessions if sessions is not None else []

    def open_session(model_client: object) -> _Session:
        opened.append(_Session())
        return opened[-1]

    with (
        patch("inference_perf.loadgen.load_generator.run_batch_job", recorder),
        patch("inference_perf.loadgen.load_generator.openAIModelServerClientSession", side_effect=open_session),
        patch.object(LoadGenerator, "mp_run", side_effect=AssertionError("a batch run started worker processes")),
    ):
        await asyncio.wait_for(loadgen.run(client if client is not None else _client()), timeout=5)


# Stage 0 holds jobs of 2 and 1 records, and stage 1 one job of 1 record. Each job gets the next records of its
# stage, which starts the data over: s0-b0 gets prompts 1 and 2, s0-b1 prompt 3 and s1-b0 prompt 1 again. Both
# stages complete, and no worker process is started.
async def test_each_job_gets_the_next_records_of_its_stage() -> None:
    recorder = _JobRecorder()
    loadgen = _load_generator(
        [
            BatchLoadStage(batches=[BatchJobConfig(batch_size=2), BatchJobConfig(batch_size=1)]),
            BatchLoadStage(batches=[BatchJobConfig(batch_size=1)]),
        ]
    )

    await _run(loadgen, recorder)

    assert [(job.batch_input.custom_ids, [body["prompt"] for body in job.bodies]) for job in recorder.jobs] == [
        (["s0-b0-0", "s0-b0-1"], ["1 2 3 1", "1 2 3 2"]),
        (["s0-b1-0"], ["1 2 3 3"]),
        (["s1-b0-0"], ["1 2 3 1"]),
    ]
    assert {stage_id: info.status for stage_id, info in loadgen.stage_runtime_info.items()} == {
        0: StageStatus.COMPLETED,
        1: StageStatus.COMPLETED,
    }


# A job of a 30m window in a stage that polls every 0.5 s is uploaded as s0-b0.jsonl, with that window and
# poll interval, and with metadata naming its stage and position: {"stage": "0", "batch": "0"}.
async def test_a_job_is_submitted_with_its_window_and_the_stage_poll_interval() -> None:
    recorder = _JobRecorder()
    loadgen = _load_generator(
        [BatchLoadStage(batches=[BatchJobConfig(batch_size=1, completion_window="30m")], poll_interval_sec=0.5)]
    )

    await _run(loadgen, recorder)

    (job,) = recorder.jobs
    assert (job.filename, job.completion_window, job.poll_interval_sec) == ("s0-b0.jsonl", "30m", 0.5)
    assert job.metadata == {"stage": "0", "batch": "0"}


# A chat run whose api config sets response_format json_object writes each line for the client's model "m",
# 16 max tokens and ignore_eos false, with streaming off and the response_format the HTTP path would add.
async def test_lines_are_built_with_the_client_settings_and_api_options() -> None:
    recorder = _JobRecorder()
    api_config = APIConfig(type=APIType.Chat, response_format=ResponseFormat(type=ResponseFormatType.JSON_OBJECT))
    loadgen = _load_generator([BatchLoadStage(batches=[BatchJobConfig(batch_size=1)])], api_type=APIType.Chat)

    await _run(loadgen, recorder, _client(api_config))

    (job,) = recorder.jobs
    assert job.bodies == [
        {
            "model": "m",
            "messages": [{"role": "user", "content": "mock prompt 1"}],
            "max_tokens": 16,
            "ignore_eos": False,
            "stream": False,
            "response_format": {"type": "json_object"},
        }
    ]


# Job s0-b0 waits until job s0-b1 has been submitted. The jobs of a stage go out together, so the stage
# completes; one at a time, it would wait forever and fail on the 5 s timeout.
async def test_the_jobs_of_a_stage_run_together() -> None:
    recorder = _JobRecorder()
    second_job_started = asyncio.Event()

    async def first_job(job: _Job) -> BatchJobResult:
        await second_job_started.wait()
        return _result(job.batch_input, "completed")

    async def second_job(job: _Job) -> BatchJobResult:
        second_job_started.set()
        return _result(job.batch_input, "completed")

    recorder.answer = {"s0-b0.jsonl": first_job, "s0-b1.jsonl": second_job}
    loadgen = _load_generator([BatchLoadStage(batches=[BatchJobConfig(batch_size=1), BatchJobConfig(batch_size=1)])])

    await _run(loadgen, recorder)

    assert loadgen.stage_runtime_info[0].status == StageStatus.COMPLETED


# In stage 0, job s0-b0 completes and job s0-b1 ends expired: stage 0 is FAILED, and stage 1 still runs and
# completes.
async def test_a_batch_that_does_not_complete_fails_its_stage() -> None:
    recorder = _JobRecorder()

    async def expire(job: _Job) -> BatchJobResult:
        return _result(job.batch_input, "expired")

    recorder.answer = {"s0-b1.jsonl": expire}
    loadgen = _load_generator(
        [
            BatchLoadStage(batches=[BatchJobConfig(batch_size=1), BatchJobConfig(batch_size=1)]),
            BatchLoadStage(batches=[BatchJobConfig(batch_size=1)]),
        ]
    )

    await _run(loadgen, recorder)

    assert [job.filename for job in recorder.jobs] == ["s0-b0.jsonl", "s0-b1.jsonl", "s1-b0.jsonl"]
    assert loadgen.stage_runtime_info[0].status == StageStatus.FAILED
    assert loadgen.stage_runtime_info[1].status == StageStatus.COMPLETED


# The error pydantic raises when a 2xx reply is not a batch object, such as a proxy's HTML page.
def _not_a_batch_error() -> ValidationError:
    try:
        Batch.model_validate_json(b"<html>proxy page</html>")
    except ValidationError as e:
        return e
    raise AssertionError("an HTML page parsed as a batch")


# Stage 0's only job raises one of the errors a Batch API call can end in: the server's refusal
# ("POST /v1/batches returned 500"), a dropped connection, a request timeout, or a 2xx reply that is not a
# batch object. The error is logged, stage 0 is FAILED, and stage 1 still runs and completes.
@pytest.mark.parametrize(
    "error, logged",
    [
        (BatchAPIError("POST /v1/batches returned 500"), "returned 500"),
        (aiohttp.ServerDisconnectedError("server went away"), "ServerDisconnectedError"),
        (TimeoutError(), "TimeoutError"),
        (_not_a_batch_error(), "validation error for Batch"),
    ],
)
async def test_a_job_that_raises_fails_its_stage_and_the_run_goes_on(
    error: Exception, logged: str, caplog: pytest.LogCaptureFixture
) -> None:
    recorder = _JobRecorder()

    async def refuse(job: _Job) -> BatchJobResult:
        raise error

    recorder.answer = {"s0-b0.jsonl": refuse}
    loadgen = _load_generator(
        [BatchLoadStage(batches=[BatchJobConfig(batch_size=1)]), BatchLoadStage(batches=[BatchJobConfig(batch_size=1)])]
    )

    with caplog.at_level(logging.ERROR, logger="inference_perf.loadgen.load_generator"):
        await _run(loadgen, recorder)

    assert any(logged in record.message for record in caplog.records)
    assert loadgen.stage_runtime_info[0].status == StageStatus.FAILED
    assert loadgen.stage_runtime_info[1].status == StageStatus.COMPLETED


# An interrupt (Ctrl-C) arrives while stage 0's job waits: its should_stop then answers true, the job returns
# with its batch still in_progress, stage 0 is INTERRUPTED, stage 1 never starts, and a warning says the batch
# is still running on the server.
async def test_an_interrupt_stops_the_wait_and_the_run(caplog: pytest.LogCaptureFixture) -> None:
    recorder = _JobRecorder()
    loadgen = _load_generator(
        [BatchLoadStage(batches=[BatchJobConfig(batch_size=1)]), BatchLoadStage(batches=[BatchJobConfig(batch_size=1)])]
    )

    async def interrupted(job: _Job) -> BatchJobResult:
        loadgen.interrupt_sig = True
        assert job.should_stop()
        return _result(job.batch_input, "in_progress")

    recorder.answer = {"s0-b0.jsonl": interrupted}

    with caplog.at_level(logging.WARNING, logger="inference_perf.loadgen.load_generator"):
        await _run(loadgen, recorder)

    assert [job.filename for job in recorder.jobs] == ["s0-b0.jsonl"]
    assert loadgen.stage_runtime_info[0].status == StageStatus.INTERRUPTED
    assert 1 not in loadgen.stage_runtime_info
    assert any("batch-s0-b0-0" in record.message and "still running" in record.message for record in caplog.records)


# A data generator that holds a single record cannot fill a job of 2: the run raises ValueError naming the
# job, rather than sending a smaller batch than the config asks for, and still closes its session.
async def test_a_job_the_data_cannot_fill_is_refused() -> None:
    class OneRecord(MockDataGenerator):
        def get_data(self) -> Generator[InferenceAPIData, None, None]:
            yield CompletionAPIData(prompt="only")

    datagen = OneRecord(APIConfig(), DataConfig(type=DataGenType.Mock), None)
    loadgen = LoadGenerator(
        datagen, LoadConfig(type=LoadType.BATCH, stages=[BatchLoadStage(batches=[BatchJobConfig(batch_size=2)])], interval=0)
    )
    sessions: list[_Session] = []

    with pytest.raises(ValueError, match="Stage 0 batch 0 needs 2 records"):
        await _run(loadgen, _JobRecorder(), sessions=sessions)

    assert [session.closed for session in sessions] == [True]


# The mock server type has no Batch API: a batch run refuses its client with a TypeError.
async def test_a_client_without_the_batch_api_is_refused() -> None:
    loadgen = _load_generator([BatchLoadStage(batches=[BatchJobConfig(batch_size=1)])])
    client = MockModelServerClient(metrics_collector=MagicMock(), api_config=APIConfig())

    with pytest.raises(TypeError, match="OpenAI-compatible"):
        await loadgen.run(client)


# A batch run whose stage list holds a constant-rate stage, which config validation would have refused, raises
# TypeError instead of running it.
async def test_a_stage_that_is_not_a_batch_stage_is_refused() -> None:
    loadgen = _load_generator([BatchLoadStage(batches=[BatchJobConfig(batch_size=1)])])
    loadgen.stages = [StandardLoadStage(rate=1, duration=1)]

    with pytest.raises(TypeError, match="BatchLoadStage"):
        await _run(loadgen, _JobRecorder())


# Two stages with load.interval 7: the run waits 7 s once, between the stages, and not after the last one.
async def test_stages_are_spaced_by_the_load_interval() -> None:
    datagen = MockDataGenerator(APIConfig(), DataConfig(type=DataGenType.Mock), None)
    stages = [BatchLoadStage(batches=[BatchJobConfig(batch_size=1)]), BatchLoadStage(batches=[BatchJobConfig(batch_size=1)])]
    loadgen = LoadGenerator(datagen, LoadConfig(type=LoadType.BATCH, stages=stages, interval=7))

    with patch("inference_perf.loadgen.load_generator.sleep", new_callable=AsyncMock) as sleep:
        await _run(loadgen, _JobRecorder())

    sleep.assert_awaited_once_with(7)


# A batch run whose data generator is a session generator, which the load generator would have refused when
# built, raises TypeError instead of reading records from it.
async def test_a_session_generator_is_refused() -> None:
    loadgen = _load_generator([BatchLoadStage(batches=[BatchJobConfig(batch_size=1)])])
    loadgen.datagen = MagicMock(spec=SessionGenerator)

    with pytest.raises(TypeError, match="DataGenerator"):
        await _run(loadgen, _JobRecorder())


# Two batch stages reach the stage observer as start 0, end 0, start 1, end 1. Each stage is reported started
# before its job runs, so the run's metrics show the stage running for the whole wait on its batches.
async def test_each_stage_is_reported_started_while_its_jobs_run() -> None:
    events: list[tuple[str, int]] = []
    observer = MagicMock()
    observer.on_stage_start.side_effect = lambda context: events.append(("start", context.stage_id))
    observer.on_stage_end.side_effect = lambda context: events.append(("end", context.stage_id))
    datagen = MockDataGenerator(APIConfig(), DataConfig(type=DataGenType.Mock), None)
    stages = [BatchLoadStage(batches=[BatchJobConfig(batch_size=1)]), BatchLoadStage(batches=[BatchJobConfig(batch_size=1)])]
    loadgen = LoadGenerator(datagen, LoadConfig(type=LoadType.BATCH, stages=stages, interval=0), stage_observer=observer)
    seen_by_jobs: list[list[tuple[str, int]]] = []

    async def note_events(job: _Job) -> BatchJobResult:
        seen_by_jobs.append(list(events))
        return _result(job.batch_input, "completed")

    recorder = _JobRecorder()
    recorder.answer = {"s0-b0.jsonl": note_events, "s1-b0.jsonl": note_events}

    await _run(loadgen, recorder)

    assert events == [("start", 0), ("end", 0), ("start", 1), ("end", 1)]
    assert seen_by_jobs == [[("start", 0)], [("start", 0), ("end", 0), ("start", 1)]]


# A run of two batch stages opens one model-server session for all of its Batch API calls and closes it when
# the run ends.
async def test_one_session_serves_the_run_and_is_closed_at_its_end() -> None:
    loadgen = _load_generator(
        [BatchLoadStage(batches=[BatchJobConfig(batch_size=1)]), BatchLoadStage(batches=[BatchJobConfig(batch_size=1)])]
    )
    sessions: list[_Session] = []

    await _run(loadgen, _JobRecorder(), sessions=sessions)

    assert [session.closed for session in sessions] == [True]


# An interrupt (Ctrl-C) arrives during the 7 s wait after stage 0: stage 1 is never started, so nothing of it
# is written or submitted.
async def test_an_interrupt_during_the_wait_between_stages_ends_the_run() -> None:
    datagen = MockDataGenerator(APIConfig(), DataConfig(type=DataGenType.Mock), None)
    stages = [BatchLoadStage(batches=[BatchJobConfig(batch_size=1)]), BatchLoadStage(batches=[BatchJobConfig(batch_size=1)])]
    loadgen = LoadGenerator(datagen, LoadConfig(type=LoadType.BATCH, stages=stages, interval=7))
    recorder = _JobRecorder()

    async def interrupted_while_waiting(seconds: float) -> None:
        loadgen.interrupt_sig = True

    with patch("inference_perf.loadgen.load_generator.sleep", new=interrupted_while_waiting):
        await _run(loadgen, recorder)

    assert [job.filename for job in recorder.jobs] == ["s0-b0.jsonl"]
    assert 1 not in loadgen.stage_runtime_info


# An interrupt arrives while stage 0 writes the input file of the first of its two jobs: the second file is not
# written, neither job is submitted, and stage 0 is INTERRUPTED.
async def test_an_interrupt_while_the_inputs_are_written_submits_nothing() -> None:
    loadgen = _load_generator([BatchLoadStage(batches=[BatchJobConfig(batch_size=1), BatchJobConfig(batch_size=1)])])
    recorder = _JobRecorder()
    written: list[str] = []

    async def write_then_interrupt(
        records: Sequence[InferenceAPIData],
        *,
        custom_id_prefix: str,
        api_config: APIConfig,
        model_name: str,
        max_tokens: int,
        ignore_eos: bool,
    ) -> BatchInput:
        written.append(custom_id_prefix)
        loadgen.interrupt_sig = True
        return await build_batch_input(
            records,
            custom_id_prefix=custom_id_prefix,
            api_config=api_config,
            model_name=model_name,
            max_tokens=max_tokens,
            ignore_eos=ignore_eos,
        )

    with patch("inference_perf.loadgen.load_generator.build_batch_input", new=write_then_interrupt):
        await _run(loadgen, recorder)

    assert written == ["s0-b0"]
    assert recorder.jobs == []
    assert loadgen.stage_runtime_info[0].status == StageStatus.INTERRUPTED


# An interrupt arrives during stage 0's job in a run with load.interval 7: the run ends without the 7 s wait.
async def test_an_interrupt_during_a_stage_skips_the_wait_after_it() -> None:
    datagen = MockDataGenerator(APIConfig(), DataConfig(type=DataGenType.Mock), None)
    stages = [BatchLoadStage(batches=[BatchJobConfig(batch_size=1)]), BatchLoadStage(batches=[BatchJobConfig(batch_size=1)])]
    loadgen = LoadGenerator(datagen, LoadConfig(type=LoadType.BATCH, stages=stages, interval=7))

    async def interrupted(job: _Job) -> BatchJobResult:
        loadgen.interrupt_sig = True
        return _result(job.batch_input, "in_progress")

    recorder = _JobRecorder()
    recorder.answer = {"s0-b0.jsonl": interrupted}

    with patch("inference_perf.loadgen.load_generator.sleep", new_callable=AsyncMock) as sleep:
        await _run(loadgen, recorder)

    sleep.assert_not_awaited()


# Stage 0's batch ends failed with the error invalid_json "line 2 is not valid JSON": a warning names the batch
# and that error, since a failed batch has no output file to explain it.
async def test_the_errors_of_a_failed_batch_are_logged(caplog: pytest.LogCaptureFixture) -> None:
    async def failed_validation(job: _Job) -> BatchJobResult:
        result = _result(job.batch_input, "failed")
        assert result.batch is not None
        result.batch.errors = BatchErrors(data=[BatchError(code="invalid_json", message="line 2 is not valid JSON")])
        return result

    recorder = _JobRecorder()
    recorder.answer = {"s0-b0.jsonl": failed_validation}
    loadgen = _load_generator([BatchLoadStage(batches=[BatchJobConfig(batch_size=1)])])

    with caplog.at_level(logging.WARNING, logger="inference_perf.loadgen.load_generator"):
        await _run(loadgen, recorder)

    assert any(
        "batch-s0-b0-0" in record.message and "invalid_json" in record.message and "line 2 is not valid JSON" in record.message
        for record in caplog.records
    )


# An interrupt arrives while stage 0's job uploads its file, so the job returns with no batch created: an INFO line
# says the job stopped before creating its batch, and stage 0 is INTERRUPTED.
async def test_a_job_stopped_before_its_batch_is_created_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    loadgen = _load_generator([BatchLoadStage(batches=[BatchJobConfig(batch_size=1)])])

    async def stopped_while_uploading(job: _Job) -> BatchJobResult:
        loadgen.interrupt_sig = True
        lines = len(job.batch_input.custom_ids)
        return BatchJobResult(
            batch=None,
            status_first_seen={},
            submitted_at=0.0,
            finished_at=1.0,
            lines=lines,
            succeeded=0,
            failed=0,
            unanswered=lines,
        )

    recorder = _JobRecorder()
    recorder.answer = {"s0-b0.jsonl": stopped_while_uploading}

    with caplog.at_level(logging.INFO, logger="inference_perf.loadgen.load_generator"):
        await _run(loadgen, recorder)

    assert any("Stage 0 batch 0 stopped before it created its batch" in record.message for record in caplog.records)
    assert loadgen.stage_runtime_info[0].status == StageStatus.INTERRUPTED


# The level of the line that reports stage 0's only batch, a job of 2 records.
async def _outcome_level(failed: int, unanswered: int, caplog: pytest.LogCaptureFixture) -> int:
    async def answered(job: _Job) -> BatchJobResult:
        result = _result(job.batch_input, "completed")
        return result.model_copy(
            update={"succeeded": result.lines - failed - unanswered, "failed": failed, "unanswered": unanswered}
        )

    recorder = _JobRecorder()
    recorder.answer = {"s0-b0.jsonl": answered}
    loadgen = _load_generator([BatchLoadStage(batches=[BatchJobConfig(batch_size=2)])])
    with caplog.at_level(logging.INFO, logger="inference_perf.loadgen.load_generator"):
        await _run(loadgen, recorder)
    (record,) = [record for record in caplog.records if "lines succeeded" in record.message]
    return record.levelno


# A completed batch of 2 lines with 1 failed line, or with 1 line the output never answered, is reported at
# WARNING: until a batch report exists that line is the run's only record of the lines that did not succeed.
@pytest.mark.parametrize("failed, unanswered", [(1, 0), (0, 1)])
async def test_a_batch_with_lines_that_did_not_succeed_is_a_warning(
    failed: int, unanswered: int, caplog: pytest.LogCaptureFixture
) -> None:
    assert await _outcome_level(failed, unanswered, caplog) == logging.WARNING


# A completed batch of 2 lines that both succeeded is reported at INFO.
async def test_a_batch_whose_lines_all_succeeded_is_info(caplog: pytest.LogCaptureFixture) -> None:
    assert await _outcome_level(0, 0, caplog) == logging.INFO
