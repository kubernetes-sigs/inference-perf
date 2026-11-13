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
"""Tests for the load generator's request phases: queued, waiting and streaming."""

import asyncio
import multiprocessing as mp
import time
import unittest
from typing import Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from inference_perf.apis.request_phase import notify_first_token
from inference_perf.config import LoadConfig, LoadType, StandardLoadStage
from inference_perf.datagen import DataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.utils.request_queue import RequestQueue


# A CONSTANT-load LoadGenerator with 2 workers over a MagicMock datagen, a fresh one unless given.
# No worker process is started.
def _make_load_generator(datagen: Optional[MagicMock] = None) -> LoadGenerator:
    datagen = datagen if datagen is not None else MagicMock(spec=DataGenerator)
    datagen.trace = None
    datagen.is_preferred_worker_requested.return_value = False
    load_config = LoadConfig(
        type=LoadType.CONSTANT, num_workers=2, stages=[StandardLoadStage(rate=1, duration=1)], base_seed=42
    )
    with patch("inference_perf.loadgen.load_generator.get_circuit_breaker"):
        return LoadGenerator(datagen, load_config)


# A multiprocess-style LoadGenerator whose shared counters read `in_flight` and `streaming`, with a stage
# schedule of `due` send times in the past and `later` an hour ahead, of which `finished` have finished.
# With schedule=False there is no running rate stage.
def _load_generator_with(
    in_flight: int = 0, streaming: int = 0, due: int = 0, later: int = 0, finished: int = 0, schedule: bool = True
) -> LoadGenerator:
    loadgen = _make_load_generator()
    loadgen._active_requests_counter = mp.Value("i", in_flight)
    loadgen._streaming_requests_counter = mp.Value("i", streaming)
    if schedule:
        now = time.perf_counter()
        loadgen._stage_schedule = ([now - 10.0] * due + [now + 3600.0] * later, mp.Value("i", finished))
    return loadgen


# 5 in flight of which 2 have a first token. Expects waiting 3, streaming 2.
def test_in_flight_splits_into_waiting_and_streaming() -> None:
    assert _load_generator_with(in_flight=5, streaming=2, schedule=False).requests_by_phase() == {
        "queued": 0,
        "waiting": 3,
        "streaming": 2,
    }


# 10 requests due, 3 finished and 2 in flight leaves 5 unsent. Expects queued 5.
def test_queued_is_due_less_finished_and_in_flight() -> None:
    assert _load_generator_with(in_flight=2, due=10, finished=3).requests_by_phase()["queued"] == 5


# 4 requests due and all 4 in flight, with 6 more scheduled an hour ahead. Expects queued 0:
# a request is not queued before its send time.
def test_requests_not_yet_due_are_not_queued() -> None:
    assert _load_generator_with(in_flight=4, due=4, later=6).requests_by_phase()["queued"] == 0


# 2 due, 2 finished and 1 in flight from a counter read a moment apart. Expects queued 0, not -1.
def test_queued_is_never_negative() -> None:
    assert _load_generator_with(in_flight=1, due=2, finished=2).requests_by_phase()["queued"] == 0


# No running rate stage, with 3 in flight. Expects queued 0: there is no schedule to be behind.
def test_queued_is_zero_without_a_schedule() -> None:
    assert _load_generator_with(in_flight=3, schedule=False).requests_by_phase()["queued"] == 0


# The streaming counter reads 3 while in flight already dropped to 2, the two being read one after
# the other. Expects streaming 2 and waiting 0, not waiting -1.
def test_streaming_is_clamped_to_in_flight() -> None:
    phases = _load_generator_with(in_flight=2, streaming=3, schedule=False).requests_by_phase()
    assert (phases["waiting"], phases["streaming"]) == (0, 2)


# The in-process path: one request that waits, receives its first token, waits again and finishes.
# Expects waiting 1, then streaming 1, then neither. A second first token does not count twice.
async def test_in_process_request_moves_from_waiting_to_streaming() -> None:
    loadgen = _make_load_generator()
    first_token, finish = asyncio.Event(), asyncio.Event()
    seen: List[Dict[str, int]] = []

    async def request() -> None:
        seen.append(loadgen.requests_by_phase())
        await first_token.wait()
        notify_first_token()
        notify_first_token()
        seen.append(loadgen.requests_by_phase())
        await finish.wait()

    task = asyncio.create_task(loadgen._track_in_flight(request()))
    await asyncio.sleep(0)
    first_token.set()
    await asyncio.sleep(0)
    finish.set()
    await task

    assert seen == [{"queued": 0, "waiting": 1, "streaming": 0}, {"queued": 0, "waiting": 0, "streaming": 1}]
    assert loadgen.requests_by_phase() == {"queued": 0, "waiting": 0, "streaming": 0}
    # the raw count too: requests_by_phase clamps streaming to the requests in flight
    assert loadgen._local_streaming == 0


# The in-process path: a request cancelled while streaming. Expects it to leave both phases.
async def test_in_process_request_cancelled_while_streaming_leaves_its_phase() -> None:
    loadgen = _make_load_generator()

    async def request() -> None:
        notify_first_token()
        await asyncio.Event().wait()

    task = asyncio.create_task(loadgen._track_in_flight(request()))
    await asyncio.sleep(0)
    assert loadgen.requests_by_phase()["streaming"] == 1
    task.cancel()
    await asyncio.gather(task, return_exceptions=True)
    assert loadgen.requests_by_phase() == {"queued": 0, "waiting": 0, "streaming": 0}
    assert loadgen._local_streaming == 0


class TestRunStageSchedule(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.datagen = MagicMock(spec=DataGenerator)
        self.load_generator = _make_load_generator(self.datagen)
        self.request_queue = MagicMock(spec=RequestQueue)
        self.request_queue.num_channels = 1
        self.finished_counter = mp.Value("i", 0)
        self.active_counter = mp.Value("i", 0)
        self.load_generator._active_requests_counter = self.active_counter
        self.queued_mid_stage: Optional[int] = None

    # Runs one run_stage of 4 requests. The patched sleep stands in for one second passing with nothing
    # sent: it records queued, then marks all 4 finished so the stage returns.
    async def _run_stage(self, concurrency_level: Optional[int] = None) -> None:
        self.datagen.get_data.return_value = iter([MagicMock(preferred_worker_id=-1) for _ in range(4)])

        async def a_second_passes(*_args: object, **_kwargs: object) -> None:
            with patch("inference_perf.loadgen.load_generator.time.perf_counter", return_value=time.perf_counter() + 3600):
                self.queued_mid_stage = self.load_generator.requests_by_phase()["queued"]
            self.finished_counter.value = 4

        with patch("inference_perf.loadgen.load_generator.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = a_second_passes
            await self.load_generator.run_stage(
                stage_id=0,
                rate=4,
                duration=1,
                request_queue=self.request_queue,
                active_requests_counter=self.active_counter,
                finished_requests_counter=self.finished_counter,
                skipped_requests_counter=mp.Value("i", 0),
                request_phase=MagicMock(),
                cancel_signal=MagicMock(),
                concurrency_level=concurrency_level,
            )

    # A rate stage of 4 requests whose workers send nothing. Expects all 4 queued once their send times
    # have passed, and queued 0 an hour after the stage has ended with 1 of the 4 finished: an ended
    # stage has no schedule left to be behind.
    async def test_rate_stage_reports_its_unsent_requests_as_queued(self) -> None:
        await self._run_stage()
        self.assertEqual(self.queued_mid_stage, 4)
        self.finished_counter.value = 1
        with patch("inference_perf.loadgen.load_generator.time.perf_counter", return_value=time.perf_counter() + 3600):
            self.assertEqual(self.load_generator.requests_by_phase()["queued"], 0)

    # The same 4 requests as a concurrent stage at level 2. Expects queued 0 throughout: a concurrent
    # stage sends as slots free up and has no schedule to fall behind.
    async def test_concurrent_stage_has_nothing_queued(self) -> None:
        await self._run_stage(concurrency_level=2)
        self.assertEqual(self.queued_mid_stage, 0)
