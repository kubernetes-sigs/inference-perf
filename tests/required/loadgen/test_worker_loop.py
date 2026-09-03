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
plain CI. Stage-boundary signals (request_phase plus the stage_done_counter /
stage_boundary_seq rendezvous) are driven from a controller thread, mirroring
the main process's side of the protocol in mp_run.
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
        # Ends the teardown grace early; _teardown_stage sets this when the
        # grace expires, and it is the only thing that cancels in-flight work.
        self.force_stop_signal = mp.Event()
        self.finished_counter = mp.Value("i", 0)
        self.active_counter = mp.Value("i", 0)
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
            force_stop_signal=self.force_stop_signal,
            stage_done_counter=self.stage_done_counter,
            stage_boundary_seq=self.stage_boundary_seq,
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

    def _end_stage_and_stop(self, before: Optional[Any] = None) -> None:
        """Drive the main-process side of the stage-end protocol from a thread.

        `before` runs first on the same thread, for protocol steps that must
        happen while the worker is parked at the boundary. It has to be a
        thread and not the test coroutine: the worker's boundary wait is a
        blocking mp.Event.wait(), so once the worker parks there it owns the
        shared event loop and no coroutine can run to release it.

        Mirrors _teardown_stage: publish the next stage_boundary_seq, clear
        request_phase to end the stage, then wait for the worker to copy that
        sequence into stage_done_counter. Publishing before signalling is what
        makes the rendezvous race-free: a worker that reaches the boundary
        reads a sequence that is already current, so it can never acknowledge
        a stale one and run ahead of the main side.
        """

        def _run() -> None:
            try:
                if before is not None:
                    before()
                self.stop_signal.set()
                # Publish the boundary before signalling, as _teardown_stage does.
                with self.stage_boundary_seq.get_lock():
                    self.stage_boundary_seq.value += 1
                    expected = self.stage_boundary_seq.value
                self.request_phase.clear()
                deadline = time.perf_counter() + 10
                while time.perf_counter() < deadline:
                    if self.stage_done_counter.value >= expected:
                        break
                    time.sleep(0.01)
            finally:
                # Always release the worker from request_phase.wait(), even if
                # the rendezvous timed out, so the test fails on assertions
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

        # Mirror _teardown_stage's cancellation protocol: cancel_signal ends the
        # stage, then force_stop_signal cuts the teardown grace short so the
        # in-flight requests are cancelled instead of running their full 30s.
        # Without the force signal the worker would wait out
        # teardown_grace_seconds first, the behaviour #662 introduced.
        self.cancel_signal.set()
        self.force_stop_signal.set()

        def _release_cancel() -> None:
            # Wait for both requests to unwind, then drop the signals, exactly
            # as _teardown_stage does once every worker reaches the boundary.
            deadline = time.perf_counter() + 10
            while time.perf_counter() < deadline:
                if self.active_counter.value == 0 and self.finished_counter.value == 2:
                    break
                time.sleep(0.01)
            self.force_stop_signal.clear()
            self.cancel_signal.clear()

        self._end_stage_and_stop(before=_release_cancel)
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

    async def test_stage_rendezvous_acknowledges_published_boundary(self) -> None:
        # Input: one stage serving 2 requests, then a single stage end.
        # Expected: the worker acknowledges the boundary exactly once, so
        # stage_done_counter ends equal to the published stage_boundary_seq
        # (1 after one stage end) and never runs ahead of it. The sweep
        # pre-pass historically violated this pairing; this locks the seam
        # any stage-driving caller must respect.
        client = _ProbeClient()
        worker = self._make_worker(client)
        self.request_phase.set()
        self._put(2)

        task = asyncio.get_event_loop().create_task(worker.loop())
        await self._wait_until(lambda: self.finished_counter.value == 2)
        self._end_stage_and_stop()
        await asyncio.wait_for(task, timeout=15)

        self.assertEqual(len(client.calls), 2)
        self.assertEqual(self.stage_boundary_seq.value, 1, "main side must publish exactly one boundary")
        self.assertEqual(
            self.stage_done_counter.value,
            self.stage_boundary_seq.value,
            "worker must acknowledge the published boundary, never run ahead of it",
        )

    async def test_stop_signal_exits_loop_without_work(self) -> None:
        client = AsyncMock(spec=ModelServerClient)
        worker = self._make_worker(client)
        self.stop_signal.set()
        self.request_phase.set()
        await asyncio.wait_for(worker.loop(), timeout=15)
        client.process_request.assert_not_awaited()


if __name__ == "__main__":
    unittest.main()
