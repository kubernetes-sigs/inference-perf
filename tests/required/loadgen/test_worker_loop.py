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
"""In-process tests for the real Worker.loop dispatch path.

The Worker is constructed directly with multiprocessing primitives and its
async loop() is awaited on the test's event loop. It is never .start()ed, so the
dispatch loop that produces every latency metric runs under coverage and
plain CI. Stage-boundary signals (request_phase plus the stage_done_counter /
stage_boundary_seq rendezvous) are driven from a controller thread in the
order _teardown_stage uses, and stop_signal is set last, as mp_run does after
its final stage.
"""

import asyncio
import multiprocessing as mp
import threading
import time
import unittest
from typing import Any, List, Optional, Tuple, cast
from unittest.mock import AsyncMock, MagicMock

from inference_perf.apis import CompletionAPIData, InferenceAPIData, LazyLoadInferenceAPIData
from inference_perf.client.modelserver import ModelServerClient
from inference_perf.datagen import DataGenerator
from inference_perf.loadgen.load_generator import RequestQueueData, Worker
from inference_perf.utils.request_queue import RequestQueue


# A real CompletionAPIData (prompt "hello", max_tokens 5). Queue items are pickled by the mp.Queue feeder thread,
# so a MagicMock would not survive the trip.
def _make_data() -> CompletionAPIData:
    return CompletionAPIData(prompt="hello", max_tokens=5)


class _ProbeClient:
    """Client double that records calls and observed in-flight concurrency."""

    def __init__(self, latency: float = 0.0) -> None:
        self.latency = latency
        self.active = 0
        self.max_active = 0
        self.calls: List[Tuple[InferenceAPIData, int, float, Optional[str]]] = []

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        self.active += 1
        self.max_active = max(self.max_active, self.active)
        self.calls.append((data, stage_id, scheduled_time, lora_adapter))
        try:
            if self.latency:
                await asyncio.sleep(self.latency)
        finally:
            self.active -= 1


class TestWorkerLoop(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.request_queue: RequestQueue[RequestQueueData] = RequestQueue(1)
        self.channel = self.request_queue.get_channel(0)
        self.stop_signal = mp.Event()
        self.cancel_signal = mp.Event()
        self.request_phase = mp.Event()
        # Lets a stage end cancel in-flight work now instead of after the worker's 120 s teardown grace.
        self.force_stop_signal = mp.Event()
        self.finished_counter = mp.Value("i", 0)
        self.active_counter = mp.Value("i", 0)
        self.skipped_counter = mp.Value("i", 0)
        self.datagen = MagicMock(spec=DataGenerator)
        # The stage rendezvous is a published sequence number, not a Barrier:
        # the main side publishes stage_boundary_seq and each worker copies it
        # into its own stage_done_counter once it reaches the boundary.
        self.stage_done_counter = mp.Value("i", 0)
        self.stage_boundary_seq = mp.Value("i", 0)
        self.controller: Optional[threading.Thread] = None

    def tearDown(self) -> None:
        if self.controller is not None:
            self.controller.join(timeout=15)

    # Worker 0 on this test's queue channel and signals, base_seed 42, max_concurrency 4 unless given.
    # Passing shared_max_concurrency models the CONCURRENT load type, where main can resize it mid-stage.
    def _make_worker(
        self,
        client: Any,
        max_concurrency: int = 4,
        shared_max_concurrency: Optional[Any] = None,
    ) -> Worker:
        return Worker(
            0,
            cast(ModelServerClient, client),
            self.channel,
            self.datagen,
            max_concurrency,
            self.stop_signal,
            self.cancel_signal,
            self.request_phase,
            self.finished_counter,
            self.active_counter,
            self.skipped_counter,
            shared_max_concurrency,
            base_seed=42,
            force_stop_signal=self.force_stop_signal,
            stage_done_counter=self.stage_done_counter,
            stage_boundary_seq=self.stage_boundary_seq,
        )

    # Queues n requests on channel 0, scheduled at time 0.0, so they dispatch at once.
    # _put(3, stage_id=7) adds three stage-7 items. stage_id defaults to 3.
    def _put(self, n: int, stage_id: int = 3, lora_adapter: Optional[str] = None) -> None:
        for _ in range(n):
            self.request_queue.put(RequestQueueData(stage_id, _make_data(), 0.0, lora_adapter), 0)

    # Polls predicate every 10 ms while yielding to the worker. Fails the test if it is still false after
    # timeout seconds (10 by default).
    async def _wait_until(self, predicate: Any, timeout: float = 10.0) -> None:
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if predicate():
                return
            await asyncio.sleep(0.01)
        self.fail("condition not met within timeout")

    # Ends one stage in _teardown_stage's order and fails unless the worker acks it. The first call publishes
    # stage_boundary_seq=1 and expects stage_done_counter=1. With a task it then sets stop_signal last, as mp_run
    # does, and waits for loop() to return. Without one the worker must carry on into the next stage within 15 s.
    async def _end_stage(self, task: "Optional[asyncio.Task[None]]" = None, cancel: bool = False) -> None:
        acked = threading.Event()

        def _run() -> None:
            try:
                # Publish before signalling so a worker that reaches the
                # boundary can only read the current sequence.
                with self.stage_boundary_seq.get_lock():
                    self.stage_boundary_seq.value += 1
                    expected = self.stage_boundary_seq.value
                if cancel:
                    self.cancel_signal.set()
                    # Stands in for an expired teardown grace, so in-flight work
                    # is cancelled now instead of after teardown_grace_seconds.
                    self.force_stop_signal.set()
                self.request_phase.clear()
                deadline = time.perf_counter() + 10
                while time.perf_counter() < deadline:
                    if self.stage_done_counter.value >= expected:
                        acked.set()
                        break
                    time.sleep(0.01)
            finally:
                # Release the worker even without an ack, so the test fails on
                # its assertions instead of hanging.
                self.force_stop_signal.clear()
                self.cancel_signal.clear()
                self.request_phase.set()
                # Stopping any earlier lets the worker leave before it acks.
                if task is not None:
                    self.stop_signal.set()

        # A thread, not a coroutine: once the worker parks in its boundary
        # wait it blocks the shared event loop in mp.Event.wait().
        self.controller = threading.Thread(target=_run, daemon=True)
        # With no stop to follow, a worker that never leaves the boundary wait
        # would hold the event loop for good. Stopping it lets the test fail.
        watchdog = threading.Timer(15, self.stop_signal.set) if task is None else None
        self.controller.start()
        if watchdog is not None:
            watchdog.daemon = True
            watchdog.start()
        if task is not None:
            await asyncio.wait_for(task, timeout=15)
        await asyncio.to_thread(self.controller.join, 15)
        if watchdog is not None:
            watchdog.cancel()
            self.assertFalse(self.stop_signal.is_set(), "worker never left the boundary wait for the next stage")
        self.assertTrue(acked.is_set(), "worker never acknowledged the stage boundary")
        self.assertEqual(self.stage_done_counter.value, self.stage_boundary_seq.value)

    # 3 stage-7 requests on adapter-a. Expects 3 client calls carrying stage 7, scheduled time 0.0 and adapter-a,
    # with finished=3 and active=0 after the stage end.
    async def test_processes_requests_and_updates_counters(self) -> None:
        client = _ProbeClient()
        worker = self._make_worker(client)
        self.request_phase.set()
        self._put(3, stage_id=7, lora_adapter="adapter-a")

        task = asyncio.get_event_loop().create_task(worker.loop())
        await self._wait_until(lambda: self.finished_counter.value == 3)
        await self._end_stage(task)

        self.assertEqual(len(client.calls), 3)
        for data, stage_id, scheduled_time, lora_adapter in client.calls:
            self.assertIsInstance(data, CompletionAPIData)
            self.assertEqual(stage_id, 7)
            self.assertEqual(scheduled_time, 0.0)
            self.assertEqual(lora_adapter, "adapter-a")
        self.assertEqual(self.finished_counter.value, 3)
        self.assertEqual(self.active_counter.value, 0)

    # A lazy item the datagen cannot materialize (it is not a LazyLoadDataMixin), then 1 normal request.
    # Expects 1 client call and finished=2: the failed item still counts and the loop keeps serving.
    async def test_lazy_load_failure_is_counted_and_loop_continues(self) -> None:
        client = _ProbeClient()
        worker = self._make_worker(client)
        self.request_phase.set()
        self.request_queue.put(RequestQueueData(0, LazyLoadInferenceAPIData(data_index=0), 0.0, None), 0)
        self._put(1)

        task = asyncio.get_event_loop().create_task(worker.loop())
        await self._wait_until(lambda: self.finished_counter.value == 2)
        await self._end_stage(task)

        self.assertEqual(len(client.calls), 1, "only the materializable request reaches the client")
        self.assertEqual(self.finished_counter.value, 2, "the failed request still counts as finished")
        self.assertEqual(self.active_counter.value, 0)

    # 6 requests at 50 ms each with max_concurrency=2. Expects all 6 served and never more than 2 in flight.
    async def test_semaphore_bounds_in_flight_concurrency(self) -> None:
        client = _ProbeClient(latency=0.05)
        worker = self._make_worker(client, max_concurrency=2)
        self.request_phase.set()
        self._put(6)

        task = asyncio.get_event_loop().create_task(worker.loop())
        await self._wait_until(lambda: self.finished_counter.value == 6)
        await self._end_stage(task)

        self.assertEqual(len(client.calls), 6)
        self.assertLessEqual(client.max_active, 2)

    # 2 requests that would each run 30 s, then a stage end with cancel and force set.
    # Expects both cancelled right away (active=0, finished=2) and the boundary acked.
    async def test_cancel_signal_cancels_in_flight_requests(self) -> None:
        client = _ProbeClient(latency=30.0)
        worker = self._make_worker(client)
        self.request_phase.set()
        self._put(2)

        task = asyncio.get_event_loop().create_task(worker.loop())
        await self._wait_until(lambda: self.active_counter.value == 2)

        await self._end_stage(task, cancel=True)

        self.assertEqual(self.active_counter.value, 0, "cancelled requests are no longer in flight")
        self.assertEqual(self.finished_counter.value, 2, "cancelled requests still count as finished")

    # CONCURRENT load, shared concurrency 0, 2 queued requests. Expects no calls after 0.3 s,
    # then both served once the shared value is raised to 2.
    async def test_zero_shared_concurrency_skips_until_raised(self) -> None:
        shared = mp.Value("i", 0)
        client = _ProbeClient()
        worker = self._make_worker(client, max_concurrency=4, shared_max_concurrency=shared)
        self.request_phase.set()
        self._put(2)

        task = asyncio.get_event_loop().create_task(worker.loop())
        await asyncio.sleep(0.3)
        self.assertEqual(len(client.calls), 0, "worker with 0 concurrency must not consume requests")
        self.assertEqual(self.finished_counter.value, 0)

        with shared.get_lock():
            shared.value = 2
        await self._wait_until(lambda: self.finished_counter.value == 2)
        await self._end_stage(task)
        self.assertEqual(len(client.calls), 2)

    # max_concurrency=4 but shared concurrency 1, 4 requests at 50 ms each.
    # Expects all 4 served one at a time (max_active=1), so the old semaphore was drained and replaced.
    async def test_shared_concurrency_update_rebinds_semaphore(self) -> None:
        shared = mp.Value("i", 1)
        client = _ProbeClient(latency=0.05)
        worker = self._make_worker(client, max_concurrency=4, shared_max_concurrency=shared)
        self.request_phase.set()
        self._put(4)

        task = asyncio.get_event_loop().create_task(worker.loop())
        await self._wait_until(lambda: self.finished_counter.value == 4)
        await self._end_stage(task)

        self.assertEqual(len(client.calls), 4)
        self.assertEqual(client.max_active, 1)

    # One stage of 2 requests, then one stage end. Expects stage_boundary_seq=1 and stage_done_counter=1:
    # the worker acks the published boundary once and never runs ahead of it.
    async def test_stage_rendezvous_acknowledges_published_boundary(self) -> None:
        client = _ProbeClient()
        worker = self._make_worker(client)
        self.request_phase.set()
        self._put(2)

        task = asyncio.get_event_loop().create_task(worker.loop())
        await self._wait_until(lambda: self.finished_counter.value == 2)
        await self._end_stage(task)

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(self.stage_boundary_seq.value, 1, "main side must publish exactly one boundary")
        self.assertEqual(
            self.stage_done_counter.value,
            self.stage_boundary_seq.value,
            "worker must acknowledge the published boundary, never run ahead of it",
        )

    # Stage 0 serves 2 requests and ends with seq 1 acked. The same loop then serves 2 stage-1 requests.
    # Expects calls tagged [0, 0, 1, 1] and stage_done_counter=2 after the second stage end.
    async def test_worker_resumes_after_stage_boundary(self) -> None:
        client = _ProbeClient()
        worker = self._make_worker(client)
        self.request_phase.set()
        self._put(2, stage_id=0)

        task = asyncio.get_event_loop().create_task(worker.loop())
        await self._wait_until(lambda: self.finished_counter.value == 2)
        await self._end_stage()
        self.assertEqual(self.stage_done_counter.value, 1)

        self._put(2, stage_id=1)
        await self._wait_until(lambda: self.finished_counter.value == 4)
        await self._end_stage(task)

        self.assertEqual([stage_id for _, stage_id, _, _ in client.calls], [0, 0, 1, 1])
        self.assertEqual(self.stage_done_counter.value, 2)

    # stop_signal and request_phase both set before loop() starts, nothing queued.
    # Expects loop() to return at once without calling the client.
    async def test_stop_signal_exits_loop_without_work(self) -> None:
        client = AsyncMock(spec=ModelServerClient)
        worker = self._make_worker(client)
        self.stop_signal.set()
        self.request_phase.set()
        await asyncio.wait_for(worker.loop(), timeout=15)
        client.process_request.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
