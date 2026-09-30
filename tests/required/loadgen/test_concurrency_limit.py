# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""A worker's concurrency limit can change while requests are in flight.

The unit tests drive _ConcurrencyLimit directly. The worker tests run a real
forked Worker, change its shared limit in the middle of a stage, and sample
the shared in-flight counter from the test process, which is what the main
process sees too.
"""

import asyncio
import multiprocessing as mp
import time
from typing import Generator, List, Optional

import pytest

from inference_perf.apis.base import InferenceAPIData
from inference_perf.client.modelserver.base import ModelServerClient
from inference_perf.client.modelserver.metrics import BaseMetrics
from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType, LoadConfig, LoadType, StandardLoadStage
from inference_perf.datagen import MockDataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator, RequestQueueData, Worker, _ConcurrencyLimit
from inference_perf.utils.numeric.concurrency_schedule import ConcurrencySchedule
from inference_perf.utils.request_queue import RequestQueue


# Acquire a limit of 2 twice, then try a third time.
# Expected: the third acquire is still waiting 50ms later; 2 permits are in use.
async def test_acquire_waits_at_the_limit() -> None:
    limit = _ConcurrencyLimit(2)
    await limit.acquire()
    await limit.acquire()
    third = asyncio.ensure_future(limit.acquire())
    await asyncio.sleep(0.05)
    assert not third.done()
    assert limit.in_use == 2
    third.cancel()


# A third acquire is waiting at limit 2, then the limit is raised to 3.
# Expected: the waiting acquire gets in straight away, with nothing released.
async def test_raising_the_limit_admits_a_waiter_without_a_release() -> None:
    limit = _ConcurrencyLimit(2)
    await limit.acquire()
    await limit.acquire()
    third = asyncio.ensure_future(limit.acquire())
    await asyncio.sleep(0.01)
    limit.set_limit(3)
    await asyncio.wait_for(third, timeout=1.0)
    assert limit.in_use == 3


# 4 permits are held at limit 4, then the limit is lowered to 2.
# Expected: nothing held is taken back (4 still in use). A new acquire waits
# through 2 releases (4 -> 2 in use) and gets in after the 3rd (2 -> 1).
async def test_lowering_the_limit_waits_for_in_flight_to_drain_below_it() -> None:
    limit = _ConcurrencyLimit(4)
    for _ in range(4):
        await limit.acquire()
    limit.set_limit(2)
    assert limit.in_use == 4
    waiter = asyncio.ensure_future(limit.acquire())
    for _ in range(2):
        limit.release()
        await asyncio.sleep(0.01)
        assert not waiter.done()
    limit.release()
    await asyncio.wait_for(waiter, timeout=1.0)
    assert limit.in_use == 2


# Limit 0, one acquire waiting; then the limit goes to 1.
# Expected: the acquire waits while the limit is 0 and gets in once it is 1.
async def test_zero_parks_until_the_limit_rises() -> None:
    limit = _ConcurrencyLimit(0)
    waiter = asyncio.ensure_future(limit.acquire())
    await asyncio.sleep(0.05)
    assert not waiter.done()
    limit.set_limit(1)
    await asyncio.wait_for(waiter, timeout=1.0)
    assert limit.in_use == 1


# An acquire times out (the worker loop bounds each acquire with wait_for),
# then the limit is raised.
# Expected: the timed-out acquire took no permit (0 in use), and a new one
# gets in at once.
async def test_a_timed_out_acquire_takes_no_permit() -> None:
    limit = _ConcurrencyLimit(0)
    with pytest.raises(asyncio.TimeoutError):
        await asyncio.wait_for(limit.acquire(), timeout=0.05)
    assert limit.in_use == 0
    limit.set_limit(1)
    await asyncio.wait_for(limit.acquire(), timeout=1.0)
    assert limit.in_use == 1


# Worker tests below: every request takes REQUEST_SECONDS, so the in-flight
# count follows the limit closely and a drain takes one request's time.
REQUEST_SECONDS = 0.4


# A client whose requests take REQUEST_SECONDS and do nothing else.
# Input: any request. Expected: returns after REQUEST_SECONDS.
class _SleepClient(ModelServerClient):
    def __init__(self) -> None:
        super().__init__(APIConfig(type=APIType.Chat))

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Chat]

    def get_prometheus_metric_metadata(self) -> BaseMetrics:
        raise NotImplementedError("not used in concurrency tests")

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        await asyncio.sleep(REQUEST_SECONDS)


# Workers must fork, as in production, so test state is inherited rather than
# pickled. Skips on platforms without fork.
@pytest.fixture(autouse=True)
def _fork_start_method() -> Generator[None, None, None]:
    old = mp.get_start_method(allow_none=True)
    if old != "fork":
        if "fork" not in mp.get_all_start_methods():
            pytest.skip("fork start method unavailable on this platform")
        mp.set_start_method("fork", force=True)
    yield
    if old is not None and old != "fork":
        mp.set_start_method(old, force=True)


# One forked Worker with a shared concurrency limit, driven like a concurrent
# stage: every request is queued at once and the limit alone paces them.
# Input: the worker's starting limit. Expected: a running worker whose limit
# the test can change with set_limit() and whose in-flight count it can read.
class _WorkerHarness:
    def __init__(self, initial_limit: int) -> None:
        api_config = APIConfig(type=APIType.Chat)
        self.datagen = MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)
        load_config = LoadConfig(type=LoadType.CONSTANT, stages=[StandardLoadStage(rate=1, duration=1)], num_workers=1)
        self.loadgen = LoadGenerator(self.datagen, load_config)
        self.request_queue: RequestQueue[RequestQueueData] = RequestQueue(1)
        self.finished_counter = mp.Value("i", 0)
        self.active_counter = mp.Value("i", 0)
        self.shared_limit = mp.Value("i", initial_limit)
        self.request_phase = mp.Event()
        self.stop_signal = mp.Event()
        self.cancel_signal = mp.Event()
        self.request_phase.set()
        self.loadgen._force_stop_signal = mp.Event()
        self.loadgen._stage_boundary_seq = mp.Value("i", 0)
        worker = Worker(
            0,
            _SleepClient(),
            self.request_queue.get_channel(0),
            self.datagen,
            initial_limit,
            self.stop_signal,
            self.cancel_signal,
            self.request_phase,
            self.finished_counter,
            self.active_counter,
            self.shared_limit,
            base_seed=42,
            force_stop_signal=self.loadgen._force_stop_signal,
            stage_done_counter=mp.Value("i", 0),
            stage_boundary_seq=self.loadgen._stage_boundary_seq,
            teardown_grace_seconds=5.0,
        )
        worker.start()
        self.loadgen.workers = [worker]

    def set_limit(self, limit: int) -> None:
        with self.shared_limit.get_lock():
            self.shared_limit.value = limit

    def in_flight(self) -> int:
        return int(self.active_counter.value)

    async def run_stage(self, num_requests: int, concurrency: Optional[ConcurrencySchedule] = None) -> None:
        # Same shape main.py gives a concurrent stage: rate = num_requests over 1s.
        await self.loadgen.run_stage(
            0,
            rate=num_requests,
            duration=1,
            request_queue=self.request_queue,
            active_requests_counter=self.active_counter,
            finished_requests_counter=self.finished_counter,
            request_phase=self.request_phase,
            cancel_signal=self.cancel_signal,
            timeout=60.0,
            concurrency=concurrency,
        )

    def shutdown(self) -> None:
        self.stop_signal.set()
        self.request_phase.set()
        for worker in self.loadgen.workers:
            worker.join(timeout=3.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=3.0)


# Sample the in-flight count every 10ms for `seconds`.
# Input: a harness and a duration. Expected: the list of samples, in order.
async def _sample(harness: _WorkerHarness, seconds: float) -> List[int]:
    samples = []
    deadline = time.perf_counter() + seconds
    while time.perf_counter() < deadline:
        samples.append(harness.in_flight())
        await asyncio.sleep(0.01)
    return samples


# The longest stretch of consecutive samples below `floor`, in samples (10ms each).
# Input: samples [2, 1, 1, 2, 0, 2] and floor 2. Expected: 2 (the "1, 1").
# A handoff between one request finishing and the next starting dips for a
# sample or two; a drain lasts a whole request (REQUEST_SECONDS = 40 samples).
def _longest_run_below(samples: List[int], floor: int) -> int:
    longest = run = 0
    for sample in samples:
        run = run + 1 if sample < floor else 0
        longest = max(longest, run)
    return longest


def test_longest_run_below_counts_consecutive_dips() -> None:
    assert _longest_run_below([2, 1, 1, 2, 0, 2], 2) == 2
    assert _longest_run_below([3, 3], 2) == 0


# A worker at limit 3 serves 30 requests that all arrive together.
# Expected: in flight never exceeds 3, reaches 3, and all 30 finish.
# This is the fixed-concurrency behaviour a plain integer level has always had.
async def test_fixed_limit_caps_in_flight() -> None:
    harness = _WorkerHarness(initial_limit=3)
    try:
        stage = asyncio.ensure_future(harness.run_stage(30))
        samples = await _sample(harness, 3.5)
        await stage
        assert max(samples) == 3
        assert harness.finished_counter.value == 30
    finally:
        harness.shutdown()


# A worker starts at limit 2 with 80 requests queued. Once 2 are in flight the
# limit rises to 6, and later falls to 1.
# Expected: in flight reaches 6 within one request time of the rise, and never
# dips below 2 for longer than a handoff (no drain before resizing up). After
# the fall it settles at 1 within one request time (at most 1 in flight, and
# 1 nearly all the time), and never sits at 0 longer than a handoff.
async def test_limit_changes_mid_stage_without_draining() -> None:
    harness = _WorkerHarness(initial_limit=2)
    try:
        stage = asyncio.ensure_future(harness.run_stage(80))
        # Requests start 1s after run_stage is called; let the first ones run.
        before = await _sample(harness, 1.8)
        assert max(before) == 2

        harness.set_limit(6)
        after_rise = await _sample(harness, REQUEST_SECONDS + 0.2)
        assert max(after_rise) == 6
        assert _longest_run_below(after_rise, 2) <= 5, "in flight drained while resizing up"

        harness.set_limit(1)
        settling = await _sample(harness, REQUEST_SECONDS + 0.2)
        settled = await _sample(harness, 1.0)
        assert _longest_run_below(settling, 1) <= 5, "resizing down must not drain the worker"
        assert max(settled) == 1
        assert settled.count(1) >= 0.8 * len(settled), f"1 in flight only {settled.count(1)}/{len(settled)} samples"

        harness.set_limit(20)
        await stage
        assert harness.finished_counter.value == 80
    finally:
        harness.shutdown()


# A worker starts at limit 0 (a concurrency level below the worker count gives
# some workers 0), then its limit rises to 2 mid-stage.
# Expected: nothing is in flight while the limit is 0, then the worker picks
# the requests up and all 10 finish.
async def test_worker_at_zero_resumes_when_the_limit_rises() -> None:
    harness = _WorkerHarness(initial_limit=0)
    try:
        stage = asyncio.ensure_future(harness.run_stage(10))
        parked = await _sample(harness, 1.5)
        assert max(parked) == 0
        harness.set_limit(2)
        resumed = await _sample(harness, REQUEST_SECONDS + 0.3)
        assert max(resumed) == 2
        await stage
        assert harness.finished_counter.value == 10
    finally:
        harness.shutdown()


# The full path: a stage whose level is an expression, 1 before t=1.5s and 4
# after, run by the load generator against a real worker.
# Expected: 1 in flight before the step, 4 after it (the load generator moves
# the worker's limit on its own), all 40 requests finish, and the stage
# records its peak (4) and the expression.
async def test_stage_follows_a_concurrency_expression() -> None:
    raw = "Piecewise((1, t < 1.5), (4, True))"
    schedule = ConcurrencySchedule(raw)
    harness = _WorkerHarness(initial_limit=schedule.initial)
    try:
        stage = asyncio.ensure_future(harness.run_stage(40, concurrency=schedule))
        # t=0 is 1s after run_stage starts, so the step lands at about 2.5s.
        before_step = await _sample(harness, 2.3)
        await asyncio.sleep(0.3)
        after_step = await _sample(harness, 1.0)
        await stage
        assert max(before_step) == 1
        assert max(after_step) == 4
        assert harness.finished_counter.value == 40
        info = harness.loadgen.stage_runtime_info[0]
        assert (info.concurrency_level, info.concurrency_expression) == (4, raw)
    finally:
        harness.shutdown()
