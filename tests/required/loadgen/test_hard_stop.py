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
"""load.hard_stop: a stage ends at start + duration, with no drain and no grace."""

import asyncio
import multiprocessing as mp
import time
import unittest
from typing import Any, List, Optional
from unittest.mock import MagicMock, patch

import pytest

from inference_perf.config import LoadConfig, LoadType, StandardLoadStage, TraceSessionReplayLoadStage
from inference_perf.datagen import DataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.utils.request_queue import RequestQueue


# A CONSTANT-load LoadGenerator over a MagicMock datagen yielding MagicMock requests,
# with `num_workers` workers (none started) and the given hard_stop setting.
def _make_load_generator(hard_stop: bool, num_workers: int = 2, rate: float = 10, duration: int = 1) -> LoadGenerator:
    datagen = MagicMock(spec=DataGenerator)
    datagen.trace = None
    datagen.is_preferred_worker_requested.return_value = False
    datagen.get_data.side_effect = lambda: iter(MagicMock(preferred_worker_id=-1) for _ in range(100000))
    load_config = LoadConfig(
        type=LoadType.CONSTANT,
        num_workers=num_workers,
        stages=[StandardLoadStage(rate=rate, duration=duration)],
        base_seed=42,
        hard_stop=hard_stop,
        stage_teardown_grace_seconds=120,
    )
    with patch("inference_perf.loadgen.load_generator.get_circuit_breaker"):
        return LoadGenerator(datagen, load_config)


# hard_stop with the trace_session_replay load type is rejected at config time:
# session stages already take their own duration.
def test_hard_stop_rejected_for_trace_session_replay() -> None:
    with pytest.raises(ValueError, match="hard_stop is not supported"):
        LoadConfig(
            type=LoadType.TRACE_SESSION_REPLAY,
            stages=[TraceSessionReplayLoadStage(concurrent_sessions=1)],
            hard_stop=True,
        )


# With hard_stop the configured stage_teardown_grace_seconds (120) is not applied:
# the effective grace is 0, so in-flight requests are cancelled at once. Without
# hard_stop the configured 120 s stays.
def test_hard_stop_zeroes_the_teardown_grace() -> None:
    assert _make_load_generator(hard_stop=True).teardown_grace_seconds == 0.0
    assert _make_load_generator(hard_stop=False).teardown_grace_seconds == 120.0


class TestRunStageStopAfter(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.load_generator = _make_load_generator(hard_stop=True)
        self.load_generator.workers = []  # no worker processes: teardown has nobody to wait for
        self.request_queue = MagicMock(spec=RequestQueue)
        self.request_queue.num_channels = 1
        self.finished = mp.Value("i", 0)

    # Runs one stage of rate=10 for duration=0.5 s on which no request ever finishes.
    async def _run(self, stop_after: Optional[float], timeout: Optional[float] = None) -> float:
        started = time.perf_counter()
        await self.load_generator.run_stage(
            stage_id=0,
            rate=10,
            duration=1,
            request_queue=self.request_queue,
            active_requests_counter=mp.Value("i", 0),
            finished_requests_counter=self.finished,
            skipped_requests_counter=mp.Value("i", 0),
            request_phase=MagicMock(),
            cancel_signal=MagicMock(),
            timeout=timeout,
            stop_after=stop_after,
        )
        return time.perf_counter() - started

    # stop_after=0.5 on a stage whose requests never finish: the stage ends as
    # COMPLETED 1 s (schedule lead-in) + 0.5 s after it began, not when the last
    # request finishes (never) and not at the 1 s poll granularity (it would be 2 s).
    async def test_stage_ends_at_planned_end_as_completed(self) -> None:
        elapsed = await self._run(stop_after=0.5)

        self.assertEqual(self.load_generator.stage_runtime_info[0].status.name, "COMPLETED")
        self.assertGreaterEqual(elapsed, 1.45)
        self.assertLess(elapsed, 1.9)

    # The same stuck stage without stop_after but with timeout=0.5 ends TIMED_OUT,
    # which is how a cap reads, unlike a planned end.
    async def test_timeout_without_stop_after_still_reads_timed_out(self) -> None:
        await self._run(stop_after=None, timeout=0.5)

        self.assertEqual(self.load_generator.stage_runtime_info[0].status.name, "TIMED_OUT")


# A client whose requests never finish; records which ones were cancelled.
class _HangingClient:
    def __init__(self) -> None:
        self.started: List[float] = []
        self.cancelled: List[float] = []

    async def process_request(self, data: Any, stage_id: int, scheduled_time: float, lora: Optional[str] = None) -> None:
        self.started.append(scheduled_time)
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            self.cancelled.append(time.perf_counter())
            raise


# In-process run (num_workers=0) of one stage at 20 qps for 1 s against a server
# that never answers: with hard_stop the stage returns about 1 s after it began,
# with all ~20 sent requests cancelled at the planned end (none survives it).
@pytest.mark.asyncio
async def test_in_process_hard_stop_cancels_in_flight_requests_at_planned_end() -> None:
    load_generator = _make_load_generator(hard_stop=True, num_workers=0, rate=20, duration=1)
    client = _HangingClient()

    started = time.perf_counter()
    await load_generator.run(client)  # type: ignore[arg-type]
    elapsed = time.perf_counter() - started

    assert 0.95 <= elapsed < 1.6
    assert 18 <= len(client.started) <= 21
    assert len(client.cancelled) == len(client.started)
    assert all(t - started < 1.3 for t in client.cancelled)
    assert load_generator.stage_runtime_info[0].status.name == "COMPLETED"
