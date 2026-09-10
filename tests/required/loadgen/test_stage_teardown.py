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

"""Regression tests for bounded stage teardown.

A stage that times out with work still in flight must never hang the run:
in-flight requests get the configured grace to finish, whatever remains is
cancelled, and a worker whose event loop is wedged (never observes the
cancellation) is terminated and respawned so later stages still run.

Timing assertions are two-sided on purpose. The lower bound pins that the
grace was actually waited for; the upper bound pins that teardown returned as
soon as the work was done instead of exhausting the grace, force, or reap
windows. Deterministic components per stage (with the default constants and
the per-test grace G): ~2s of stage run (1s scheduling offset + 1s stage
timeout), then G of wind-down when tasks are still pending, then sub-second
cancellation/rendezvous overhead.
"""

import asyncio
import os
import sys
import time
from typing import Generator, List, Optional, Tuple

import pytest

import inference_perf.loadgen.load_generator as lg_module
from inference_perf.apis.base import InferenceAPIData
from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.client.modelserver.base import ModelServerClient
from inference_perf.client.modelserver.metrics import BaseMetrics
from inference_perf.config import (
    APIConfig,
    APIType,
    DataConfig,
    DataGenType,
    LoadConfig,
    LoadType,
    StageGenType,
    StandardLoadStage,
    SweepConfig,
)
from inference_perf.datagen import MockDataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator, RequestQueueData, Worker
from inference_perf.utils.mp_context import MP_CONTEXT
from inference_perf.utils.request_queue import RequestQueue


class _TestClientBase(ModelServerClient):
    def __init__(self) -> None:
        super().__init__(APIConfig(type=APIType.Chat))

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Chat]

    def get_prometheus_metric_metadata(self) -> BaseMetrics:
        raise NotImplementedError("not used in teardown tests")


class CompletingSlowClient(_TestClientBase):
    """Requests take a fixed time, longer than the stage timeout but shorter
    than the teardown grace: they should complete during teardown. A completion
    line is appended only when the request actually ran to the end, so a
    teardown that cancels instead of waiting is detected."""

    def __init__(self, completion_log: str) -> None:
        super().__init__()
        self.completion_log = completion_log

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        await asyncio.sleep(2.0)
        with open(self.completion_log, "a") as f:
            f.write("completed\n")


class HangingAsyncClient(_TestClientBase):
    """Requests hang forever but respond to cancellation."""

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        await asyncio.sleep(3600)


class QuickClient(_TestClientBase):
    """Requests complete almost immediately."""

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        await asyncio.sleep(0.05)


class CancelHostileClient(_TestClientBase):
    """Requests hang forever and raise a non-cancellation error when
    cancelled. The wind-down gather must absorb the error (return_exceptions)
    instead of letting it kill the worker process."""

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        try:
            await asyncio.sleep(3600)
        except asyncio.CancelledError:
            raise RuntimeError("client failure during cancellation") from None


class WedgedSyncClient(_TestClientBase):
    """Blocks the worker's event loop synchronously: cancellation is never
    delivered, so only terminate-and-respawn can end the stage."""

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        time.sleep(3600)


class DieOnceClient(_TestClientBase):
    """The first request ever dispatched kills the worker process abruptly
    (simulating an OOM kill). Requests served by the respawned worker behave
    like CompletingSlowClient."""

    def __init__(self, death_marker: str, completion_log: str) -> None:
        super().__init__()
        self.death_marker = death_marker
        self.completion_log = completion_log

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        if not os.path.exists(self.death_marker):
            with open(self.death_marker, "w") as f:
                f.write("died\n")
            os._exit(1)
        await asyncio.sleep(2.0)
        with open(self.completion_log, "a") as f:
            f.write("completed\n")


class ChainedChatData(ChatCompletionAPIData):
    """Chat data that optionally waits for a flag file before dispatch,
    modeling a session-replay successor parked on its predecessor."""

    wait_flag_path: Optional[str] = None

    async def wait_for_predecessors_and_substitute(self) -> None:
        if self.wait_flag_path is None:
            return
        while not os.path.exists(self.wait_flag_path):
            await asyncio.sleep(0.05)


class ChainDataGenerator(MockDataGenerator):
    """Yields alternating plain / predecessor-gated requests."""

    def __init__(self, api_config: APIConfig, config: DataConfig, wait_flag_path: str) -> None:
        super().__init__(api_config, config, None)
        self.wait_flag_path = wait_flag_path

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        i = 0
        while True:
            i += 1
            yield ChainedChatData(
                messages=[ChatMessage(role="user", content=f"chained prompt {i}")],
                wait_flag_path=self.wait_flag_path if i % 2 == 0 else None,
            )


class ChainClient(_TestClientBase):
    """Logs each dispatch, then completes after a delay and raises the flag
    the gated request is waiting on."""

    def __init__(self, dispatch_log: str, done_flag: str) -> None:
        super().__init__()
        self.dispatch_log = dispatch_log
        self.done_flag = done_flag

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        with open(self.dispatch_log, "a") as f:
            f.write("dispatched\n")
        await asyncio.sleep(3.0)
        with open(self.done_flag, "w") as f:
            f.write("done\n")


def _line_count(path: str) -> int:
    if not os.path.exists(path):
        return 0
    with open(path) as f:
        return len(f.readlines())


# How long the harness waits for a worker process to finish booting. Only a
# guard against a wedged worker hanging the suite; a healthy forkserver
# worker signals ready in well under a second.
_WORKER_BOOT_TIMEOUT_SEC = 60.0


class _Harness:
    def __init__(
        self,
        client: ModelServerClient,
        teardown_grace_seconds: float,
        datagen: Optional[MockDataGenerator] = None,
        worker_max_concurrency: int = 4,
    ) -> None:
        api_config = APIConfig(type=APIType.Chat)
        self.datagen = (
            datagen if datagen is not None else MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)
        )
        load_config = LoadConfig(
            type=LoadType.CONSTANT,
            stages=[StandardLoadStage(rate=2, duration=1)],
            num_workers=1,
            worker_max_concurrency=worker_max_concurrency,
            stage_teardown_grace_seconds=teardown_grace_seconds,
        )
        self.loadgen = LoadGenerator(self.datagen, load_config)
        self.request_queue: RequestQueue[RequestQueueData] = RequestQueue(1)
        self.finished_counter = MP_CONTEXT.Value("i", 0)
        self.active_counter = MP_CONTEXT.Value("i", 0)
        self.request_phase = MP_CONTEXT.Event()
        self.stop_signal = MP_CONTEXT.Event()
        self.cancel_signal = MP_CONTEXT.Event()
        self.force_stop_signal = MP_CONTEXT.Event()
        self.stage_boundary_seq = MP_CONTEXT.Value("i", 0)
        self.workers_ready = MP_CONTEXT.Value("i", 0)
        self.loadgen._force_stop_signal = self.force_stop_signal
        self.loadgen._stage_boundary_seq = self.stage_boundary_seq
        self.request_phase.set()

        worker = Worker(
            0,
            client,
            self.request_queue.get_channel(0),
            self.datagen,
            worker_max_concurrency,
            self.stop_signal,
            self.cancel_signal,
            self.request_phase,
            self.finished_counter,
            self.active_counter,
            None,
            base_seed=42,
            force_stop_signal=self.force_stop_signal,
            stage_done_counter=MP_CONTEXT.Value("i", 0),
            stage_boundary_seq=self.stage_boundary_seq,
            teardown_grace_seconds=teardown_grace_seconds,
            ready_counter=self.workers_ready,
        )
        worker.start()
        self._await_worker_ready()
        self.loadgen.workers = [worker]

    # Blocks until the worker process has finished booting and is about to start
    # consuming the request queue. Input: none; reads self.workers_ready, which
    # the worker increments once its payload is unpickled. Output: returns once
    # the count reaches 1, or raises RuntimeError if the worker never gets there.
    #
    # Forkserver workers take real time to boot, unlike the forked workers these
    # tests were first written against. Production keeps that startup off the
    # stage clock (LoadGenerator._await_workers_ready); these tests must too,
    # because their timing assertions begin when the stage does.
    def _await_worker_ready(self) -> None:
        deadline = time.monotonic() + _WORKER_BOOT_TIMEOUT_SEC
        while time.monotonic() < deadline:
            with self.workers_ready.get_lock():
                if self.workers_ready.value >= 1:
                    return
            time.sleep(0.02)
        raise RuntimeError(f"worker did not signal ready within {_WORKER_BOOT_TIMEOUT_SEC}s")

    async def run_stage(self, stage_id: int, timeout: float, rate: float = 2) -> float:
        start = time.perf_counter()
        await self.loadgen.run_stage(
            stage_id,
            rate=rate,
            duration=1,
            request_queue=self.request_queue,
            active_requests_counter=self.active_counter,
            finished_requests_counter=self.finished_counter,
            request_phase=self.request_phase,
            cancel_signal=self.cancel_signal,
            timeout=timeout,
        )
        return time.perf_counter() - start

    def worker_pids(self) -> Tuple[Optional[int], ...]:
        return tuple(w.pid for w in self.loadgen.workers)

    def shutdown(self) -> None:
        self.stop_signal.set()
        self.request_phase.set()
        for worker in self.loadgen.workers:
            worker.join(timeout=3.0)
            if worker.is_alive():
                worker.terminate()
                worker.join(timeout=3.0)


async def test_grace_lets_inflight_requests_complete(tmp_path: object) -> None:
    """Requests in flight at stage timeout finish during the teardown grace
    (proven by completion markers written at request end, which cancellation
    would skip); teardown returns at completion, far below the grace bound."""
    completion_log = os.path.join(str(tmp_path), "completions.log")
    harness = _Harness(CompletingSlowClient(completion_log), teardown_grace_seconds=15.0)
    try:
        pids_before = harness.worker_pids()
        elapsed = await harness.run_stage(0, timeout=1.0)

        # Lower bound: requests dispatch at >= 1s and take 2s, so a teardown
        # that really waits cannot return before ~3s. Upper bound: it must
        # return at completion (~4s), not at grace + margin (30s).
        assert 3.0 <= elapsed < 12.0, f"teardown window violated: {elapsed:.1f}s"
        assert harness.loadgen.stage_runtime_info[0].status.name == "FAILED"  # timed out
        # Both requests (rate*duration = 2) ran to completion during the grace.
        assert _line_count(completion_log) == 2
        assert harness.finished_counter.value == 2
        assert harness.loadgen.stage_runtime_info[0].dropped_requests == 0
        assert harness.worker_pids() == pids_before, "worker should not be respawned"
        assert harness.loadgen.workers[0].is_alive()
    finally:
        harness.shutdown()


async def test_stuck_requests_cancelled_at_grace_expiry() -> None:
    """Requests that never finish are cancelled once the grace expires; the
    worker survives and the next stage still runs."""
    harness = _Harness(HangingAsyncClient(), teardown_grace_seconds=1.0)
    try:
        pids_before = harness.worker_pids()

        elapsed = await harness.run_stage(0, timeout=1.0)
        # Lower bound: ~2s of stage run plus the full 1s grace (the tasks
        # never finish, so the grace cannot be cut short). Upper bound:
        # cancellation must end the wind-down promptly; a reap that waits out
        # _WIND_DOWN_REAP_SECONDS (10s) because tasks were not cancelled would
        # exceed it.
        assert 3.0 <= elapsed < 8.0, f"teardown window violated: {elapsed:.1f}s"
        assert harness.loadgen.stage_runtime_info[0].status.name == "FAILED"
        assert harness.worker_pids() == pids_before, "cancellable tasks must not force a respawn"

        # Multi-stage: the next stage must run through the same worker.
        elapsed = await harness.run_stage(1, timeout=1.0)
        assert 3.0 <= elapsed < 8.0, f"second stage teardown window violated: {elapsed:.1f}s"
        assert 1 in harness.loadgen.stage_runtime_info
    finally:
        harness.shutdown()


async def test_saturated_worker_reaches_boundary_and_drops_backlog() -> None:
    """A worker whose concurrency permits are all held by hung requests must
    still observe the stage boundary (bounded semaphore acquire), cancel at
    grace expiry without being terminated, and the undispatched backlog must
    be counted as dropped."""
    harness = _Harness(HangingAsyncClient(), teardown_grace_seconds=1.0, worker_max_concurrency=4)
    try:
        pids_before = harness.worker_pids()

        # 20 requests, 4 permits: 4 hang in flight, 16 never leave the queue.
        elapsed = await harness.run_stage(0, timeout=1.0, rate=20)

        # Same window as the unsaturated hang case: the boundary must be
        # reached within the 0.5s acquire timeout, not via the force path
        # (grace + 15s margin + terminate).
        assert 3.0 <= elapsed < 9.0, f"teardown window violated: {elapsed:.1f}s"
        assert harness.loadgen.stage_runtime_info[0].status.name == "FAILED"
        assert harness.loadgen.stage_runtime_info[0].dropped_requests == 16
        assert harness.finished_counter.value == 4  # the cancelled in-flight tasks
        assert harness.worker_pids() == pids_before, "saturated worker must drain gracefully, not be terminated"
        assert harness.loadgen.workers[0].is_alive()
    finally:
        harness.shutdown()


async def test_cancel_hostile_task_failure_does_not_kill_worker() -> None:
    """A task that raises a non-cancellation error while being cancelled must
    be absorbed by the wind-down gather; the worker survives and serves the
    next stage."""
    harness = _Harness(CancelHostileClient(), teardown_grace_seconds=1.0)
    try:
        pids_before = harness.worker_pids()

        elapsed = await harness.run_stage(0, timeout=1.0)
        assert 3.0 <= elapsed < 8.0, f"teardown window violated: {elapsed:.1f}s"
        assert harness.worker_pids() == pids_before, "task failure at cancellation must not kill the worker"
        assert harness.loadgen.workers[0].is_alive()

        elapsed = await harness.run_stage(1, timeout=1.0)
        assert 1 in harness.loadgen.stage_runtime_info
    finally:
        harness.shutdown()


async def test_wedged_worker_terminated_and_respawned(monkeypatch: pytest.MonkeyPatch) -> None:
    """A worker whose event loop is blocked never observes cancellation: it
    must be terminated at the force deadline and respawned so subsequent
    stages run at full capacity."""
    monkeypatch.setattr(lg_module, "_TEARDOWN_MARGIN_SECONDS", 2.0)
    monkeypatch.setattr(lg_module, "_FORCE_REAP_SECONDS", 2.0)

    harness = _Harness(WedgedSyncClient(), teardown_grace_seconds=0.5)
    try:
        pids_before = harness.worker_pids()

        elapsed = await harness.run_stage(0, timeout=1.0)
        # Lower bound: ~2s run + 0.5s grace + 2s margin + 2s force reap = 6.5s
        # of deterministic waiting before terminate. Upper bound: terminate
        # must kill the worker promptly; skipping terminate and relying on the
        # join(5s)-then-kill fallback would add ~5s and exceed it.
        assert 6.0 <= elapsed < 11.0, f"teardown window violated: {elapsed:.1f}s"
        assert harness.loadgen.stage_runtime_info[0].status.name == "FAILED"
        assert harness.worker_pids() != pids_before, "wedged worker should be respawned"
        assert harness.loadgen.workers[0].is_alive(), "replacement worker should be running"

        # Multi-stage: the respawned worker serves the next stage, which wedges
        # and is bounded again.
        elapsed = await harness.run_stage(1, timeout=1.0)
        assert 6.0 <= elapsed < 15.0, f"second stage teardown window violated: {elapsed:.1f}s"
        assert 1 in harness.loadgen.stage_runtime_info
    finally:
        harness.shutdown()


async def test_respawned_worker_resyncs_stage_rendezvous(tmp_path: object) -> None:
    """A worker respawned after dying mid-stage may pass a stage boundary it
    never served (it starts while request_phase is cleared). Its stage-done
    counter must converge on the published boundary sequence instead of
    running ahead, so the next stage's teardown still waits for real work."""
    death_marker = os.path.join(str(tmp_path), "death.marker")
    completion_log = os.path.join(str(tmp_path), "completions.log")
    harness = _Harness(DieOnceClient(death_marker, completion_log), teardown_grace_seconds=15.0)
    try:
        pids_before = harness.worker_pids()

        # Stage 0: the first dispatched request kills the worker; the stage
        # fails and the worker is respawned during teardown.
        await harness.run_stage(0, timeout=5.0)
        assert harness.loadgen.stage_runtime_info[0].status.name == "FAILED"
        assert harness.worker_pids() != pids_before, "dead worker should be respawned"
        assert harness.loadgen.workers[0].is_alive()
        pids_after_respawn = harness.worker_pids()

        # Gap between stages: the respawned worker passes the idle stage
        # boundary here. With a relative counter this desynchronized the
        # rendezvous permanently; the published sequence keeps it aligned.
        await asyncio.sleep(3.0)

        # Stage 1: two 2s requests against a 1s timeout. The teardown must
        # wait for both to complete during the grace; a desynchronized
        # rendezvous returns immediately with nothing completed.
        elapsed = await harness.run_stage(1, timeout=1.0)
        assert 3.0 <= elapsed < 12.0, f"teardown window violated: {elapsed:.1f}s"
        assert _line_count(completion_log) == 2, "in-flight requests must complete during the grace"
        assert harness.finished_counter.value == 2
        assert harness.worker_pids() == pids_after_respawn, "respawned worker must survive the next stage"
    finally:
        harness.shutdown()


async def test_draining_gate_blocks_dependent_dispatch(tmp_path: object) -> None:
    """Session-replay dependency chains: a request parked on its predecessor
    that wakes up during teardown must exit without dispatching (draining
    gate), so the chain unwinds well before the grace expires."""
    dispatch_log = os.path.join(str(tmp_path), "dispatch.log")
    done_flag = os.path.join(str(tmp_path), "done.flag")
    api_config = APIConfig(type=APIType.Chat)
    datagen = ChainDataGenerator(api_config, DataConfig(type=DataGenType.Mock), done_flag)
    harness = _Harness(ChainClient(dispatch_log, done_flag), teardown_grace_seconds=15.0, datagen=datagen)
    try:
        pids_before = harness.worker_pids()

        # Request A dispatches (~1.5s) and completes at ~4.5s, raising the
        # flag. Request B waits on the flag; teardown starts at ~2s, so B
        # wakes while draining and must not dispatch.
        elapsed = await harness.run_stage(0, timeout=1.0)

        # Lower bound: teardown cannot end before A completes (~4.5s). Upper
        # bound: the chain must unwind at A's completion, not at grace +
        # margin (30s), and without B's extra 3s dispatch.
        assert 3.5 <= elapsed < 10.0, f"teardown window violated: {elapsed:.1f}s"
        assert _line_count(dispatch_log) == 1, "dependent request must not dispatch during teardown"
        assert harness.finished_counter.value == 2  # A completed, B exited via the gate
        assert harness.worker_pids() == pids_before
    finally:
        harness.shutdown()


async def test_zero_request_stage_boundary_is_not_missed() -> None:
    """A stage with zero requests sets and tears down its signals within
    milliseconds, faster than a parked worker samples them. The worker must
    still acknowledge that boundary (via the published sequence) instead of
    waiting out the grace and being killed as unresponsive."""
    harness = _Harness(QuickClient(), teardown_grace_seconds=1.0)
    try:
        pids_before = harness.worker_pids()

        # Stage 0: normal, completes.
        elapsed = await harness.run_stage(0, timeout=10.0)
        assert harness.loadgen.stage_runtime_info[0].status.name == "COMPLETED"

        # Stage 1: rate*duration < 1 -> zero requests -> immediate teardown.
        elapsed = await harness.run_stage(1, timeout=10.0, rate=0.5)
        # A missed boundary parks the worker until the grace + margin (16s)
        # expires and then respawns it; an acknowledged boundary returns fast.
        assert elapsed < 5.0, f"zero-request stage teardown not acknowledged: {elapsed:.1f}s"
        assert harness.worker_pids() == pids_before, "healthy worker must not be killed at an empty boundary"

        # Stage 2: normal again, must run through the same worker.
        elapsed = await harness.run_stage(2, timeout=10.0)
        assert elapsed < 10.0, f"stage after empty boundary not bounded: {elapsed:.1f}s"
        assert harness.loadgen.stage_runtime_info[2].status.name == "COMPLETED"
        assert harness.finished_counter.value == 2
        assert harness.worker_pids() == pids_before
    finally:
        harness.shutdown()


async def test_multi_stage_happy_path_through_mp_run() -> None:
    """Normal multi-stage runs must be unaffected by the teardown rework: both
    stages complete cleanly through the real mp_run worker lifecycle."""
    from inference_perf.client.modelserver import MockModelServerClient
    from inference_perf.metrics.request_collector.local import LocalRequestMetricCollector

    api_config = APIConfig(type=APIType.Chat)
    datagen = MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)
    load_config = LoadConfig(
        type=LoadType.CONSTANT,
        interval=0.1,
        stages=[StandardLoadStage(rate=4, duration=1), StandardLoadStage(rate=4, duration=1)],
        num_workers=2,
        worker_max_concurrency=4,
        stage_teardown_grace_seconds=10.0,
    )
    loadgen = LoadGenerator(datagen, load_config)
    client = MockModelServerClient(LocalRequestMetricCollector(), api_config, mock_latency=0.05)

    start = time.perf_counter()
    await loadgen.run(client)
    elapsed = time.perf_counter() - start

    try:
        assert elapsed < 60, f"multi-stage run not bounded: {elapsed:.1f}s"
        assert loadgen.stage_runtime_info[0].status.name == "COMPLETED"
        assert loadgen.stage_runtime_info[1].status.name == "COMPLETED"
        assert loadgen.stage_runtime_info[0].dropped_requests == 0
        assert loadgen.stage_runtime_info[1].dropped_requests == 0
    finally:
        await loadgen.stop()


async def test_sweep_preprocess_timeout_is_bounded() -> None:
    """The sweep preprocess stage is designed to time out through the
    teardown machinery (#608 regression surface). With a hanging server it
    must still return promptly and record runtime info for the probe stage
    instead of deadlocking."""
    api_config = APIConfig(type=APIType.Chat)
    datagen = MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)
    load_config = LoadConfig(
        type=LoadType.CONSTANT,
        stages=[],
        sweep=SweepConfig(type=StageGenType.GEOM, num_requests=8, timeout=2.0, num_stages=2, stage_duration=1),
        num_workers=1,
        worker_max_concurrency=4,
        stage_teardown_grace_seconds=1.0,
    )
    loadgen = LoadGenerator(datagen, load_config)

    start = time.perf_counter()
    await loadgen.run(HangingAsyncClient())
    elapsed = time.perf_counter() - start

    try:
        assert elapsed < 25, f"sweep preprocess not bounded: {elapsed:.1f}s"
        # The probe stage (stage_id=-1) timed out but still produced runtime info.
        assert -1 in loadgen.stage_runtime_info
        assert loadgen.stage_runtime_info[-1].status.name == "FAILED"
    finally:
        await loadgen.stop()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
