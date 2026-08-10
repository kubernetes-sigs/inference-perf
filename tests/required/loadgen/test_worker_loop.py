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
async loop() is awaited on the test's event loop — never .start()ed — so the
dispatch loop that produces every latency metric runs under coverage and
plain CI. Stage-boundary signals (request_phase / stage_barrier) are driven
from a controller thread, mirroring the main process's side of the protocol
in mp_run.
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


def _make_data() -> CompletionAPIData:
    # A real, picklable InferenceAPIData: items cross a JoinableQueue feeder thread.
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
        self.finished_counter = mp.Value("i", 0)
        self.active_counter = mp.Value("i", 0)
        self.datagen = MagicMock(spec=DataGenerator)
        self.barrier = mp.Barrier(2)
        self.controller: Optional[threading.Thread] = None

    def tearDown(self) -> None:
        if self.controller is not None:
            self.controller.join(timeout=15)

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
            shared_max_concurrency,
            base_seed=42,
            stage_barrier=self.barrier,
        )

    def _put(self, n: int, stage_id: int = 3, lora_adapter: Optional[str] = None) -> None:
        for _ in range(n):
            self.request_queue.put(RequestQueueData(stage_id, _make_data(), 0.0, lora_adapter), 0)

    async def _wait_until(self, predicate: Any, timeout: float = 10.0) -> None:
        deadline = time.perf_counter() + timeout
        while time.perf_counter() < deadline:
            if predicate():
                return
            await asyncio.sleep(0.01)
        self.fail("condition not met within timeout")

    def _end_stage_and_stop(self) -> None:
        """Drive the main-process side of the stage-end protocol from a thread.

        Mirrors mp_run: clear request_phase to end the stage, pair the
        worker's stage_barrier arrival, and only after the barrier trips
        re-set request_phase (with stop_signal already set) so the worker's
        final wait() releases. The barrier pairing is what makes re-setting
        the phase race-free: without it the worker can miss the cleared
        phase entirely and spin in the dispatch loop forever.
        """

        def _run() -> None:
            try:
                self.stop_signal.set()
                self.request_phase.clear()
                self.barrier.wait(timeout=10)
            finally:
                # Always release the worker from request_phase.wait(), even if
                # the barrier timed out — the test then fails on assertions
                # instead of hanging.
                self.request_phase.set()

        self.controller = threading.Thread(target=_run, daemon=True)
        self.controller.start()

    async def test_processes_requests_and_updates_counters(self) -> None:
        client = _ProbeClient()
        worker = self._make_worker(client)
        self.request_phase.set()
        self._put(3, stage_id=7, lora_adapter="adapter-a")

        task = asyncio.get_event_loop().create_task(worker.loop())
        await self._wait_until(lambda: self.finished_counter.value == 3)
        self._end_stage_and_stop()
        await asyncio.wait_for(task, timeout=15)

        self.assertEqual(len(client.calls), 3)
        for data, stage_id, scheduled_time, lora_adapter in client.calls:
            self.assertIsInstance(data, CompletionAPIData)
            self.assertEqual(stage_id, 7)
            self.assertEqual(scheduled_time, 0.0)
            self.assertEqual(lora_adapter, "adapter-a")
        self.assertEqual(self.finished_counter.value, 3)
        self.assertEqual(self.active_counter.value, 0)

    async def test_lazy_load_failure_is_counted_and_loop_continues(self) -> None:
        # self.datagen is not a LazyLoadDataMixin, so a lazy item fails to
        # materialize; the worker must count it finished, ack the queue item,
        # and keep serving later requests.
        client = _ProbeClient()
        worker = self._make_worker(client)
        self.request_phase.set()
        self.request_queue.put(RequestQueueData(0, LazyLoadInferenceAPIData(data_index=0), 0.0, None), 0)
        self._put(1)

        task = asyncio.get_event_loop().create_task(worker.loop())
        await self._wait_until(lambda: self.finished_counter.value == 2)
        self._end_stage_and_stop()
        await asyncio.wait_for(task, timeout=15)

        self.assertEqual(len(client.calls), 1, "only the materializable request reaches the client")
        self.assertEqual(self.finished_counter.value, 2, "the failed request still counts as finished")
        self.assertEqual(self.active_counter.value, 0)

    async def test_semaphore_bounds_in_flight_concurrency(self) -> None:
        client = _ProbeClient(latency=0.05)
        worker = self._make_worker(client, max_concurrency=2)
        self.request_phase.set()
        self._put(6)

        task = asyncio.get_event_loop().create_task(worker.loop())
        await self._wait_until(lambda: self.finished_counter.value == 6)
        self._end_stage_and_stop()
        await asyncio.wait_for(task, timeout=15)

        self.assertEqual(len(client.calls), 6)
        self.assertLessEqual(client.max_active, 2)

    async def test_cancel_signal_cancels_in_flight_requests(self) -> None:
        client = _ProbeClient(latency=30.0)
        worker = self._make_worker(client)
        self.request_phase.set()
        self._put(2)

        task = asyncio.get_event_loop().create_task(worker.loop())
        await self._wait_until(lambda: self.active_counter.value == 2)

        # Mirror run_stage's cancellation protocol: set cancel_signal, wait for
        # workers to unwind their in-flight tasks, clear cancel_signal, and
        # only then end the stage.
        self.cancel_signal.set()
        await self._wait_until(lambda: self.active_counter.value == 0 and self.finished_counter.value == 2)
        self.cancel_signal.clear()
        self._end_stage_and_stop()
        await asyncio.wait_for(task, timeout=15)

        self.assertEqual(self.active_counter.value, 0, "cancelled requests are no longer in flight")
        self.assertEqual(self.finished_counter.value, 2, "cancelled requests still count as finished")

    async def test_zero_shared_concurrency_skips_until_raised(self) -> None:
        # CONCURRENT load type: shared value 0 means this worker sits out the
        # stage; raising it mid-run must resume consumption.
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
        self._end_stage_and_stop()
        await asyncio.wait_for(task, timeout=15)
        self.assertEqual(len(client.calls), 2)

    async def test_shared_concurrency_update_rebinds_semaphore(self) -> None:
        # Worker starts with max_concurrency=4 but the shared value says 1:
        # the loop must drain the old semaphore and enforce the new bound.
        shared = mp.Value("i", 1)
        client = _ProbeClient(latency=0.05)
        worker = self._make_worker(client, max_concurrency=4, shared_max_concurrency=shared)
        self.request_phase.set()
        self._put(4)

        task = asyncio.get_event_loop().create_task(worker.loop())
        await self._wait_until(lambda: self.finished_counter.value == 4)
        self._end_stage_and_stop()
        await asyncio.wait_for(task, timeout=15)

        self.assertEqual(len(client.calls), 4)
        self.assertEqual(client.max_active, 1)

    async def test_stage_barrier_pairing_contract(self) -> None:
        # The worker arrives at stage_barrier exactly once per stage end, and
        # proceeds only when the main side pairs the arrival (mp_run's wait()
        # after each stage). The sweep pre-pass historically violated this
        # pairing; this locks the seam any stage-driving caller must respect.
        client = _ProbeClient()
        worker = self._make_worker(client)
        self.request_phase.set()
        self._put(2)

        task = asyncio.get_event_loop().create_task(worker.loop())
        await self._wait_until(lambda: self.finished_counter.value == 2)
        self._end_stage_and_stop()
        await asyncio.wait_for(task, timeout=15)

        self.assertEqual(len(client.calls), 2)
        self.assertFalse(self.barrier.broken, "worker-side arrival must pair with the main-side wait")
        self.assertEqual(self.barrier.n_waiting, 0)

    async def test_stop_signal_exits_loop_without_work(self) -> None:
        client = AsyncMock(spec=ModelServerClient)
        worker = self._make_worker(client)
        self.stop_signal.set()
        self.request_phase.set()
        await asyncio.wait_for(worker.loop(), timeout=15)
        client.process_request.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
