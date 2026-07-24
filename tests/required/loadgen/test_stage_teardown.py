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
"""

import asyncio
import multiprocessing as mp
import sys
import time
from typing import Generator, List, Optional, Tuple

import pytest

import inference_perf.loadgen.load_generator as lg_module
from inference_perf.apis.base import InferenceAPIData
from inference_perf.client.modelserver.base import ModelServerClient, PrometheusMetricMetadata
from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType, LoadConfig, LoadType, StandardLoadStage
from inference_perf.datagen import MockDataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator, RequestQueueData, Worker
from inference_perf.utils.request_queue import RequestQueue


class _TestClientBase(ModelServerClient):
    def __init__(self) -> None:
        super().__init__(APIConfig(type=APIType.Chat))

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Chat]

    def get_prometheus_metric_metadata(self) -> PrometheusMetricMetadata:
        raise NotImplementedError("not used in teardown tests")


class SlowClient(_TestClientBase):
    """Requests take a fixed time, longer than the stage timeout but shorter
    than the teardown grace: they should complete during teardown."""

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        await asyncio.sleep(2.0)


class HangingAsyncClient(_TestClientBase):
    """Requests hang forever but respond to cancellation."""

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        await asyncio.sleep(3600)


class WedgedSyncClient(_TestClientBase):
    """Blocks the worker's event loop synchronously: cancellation is never
    delivered, so only terminate-and-respawn can end the stage."""

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        time.sleep(3600)


@pytest.fixture(autouse=True)
def _fork_start_method() -> Generator[None, None, None]:
    # Workers must fork (matching production: Linux default, forced on macOS in
    # main_cli) so unpicklable test state is inherited rather than pickled.
    old = mp.get_start_method(allow_none=True)
    if old != "fork":
        if "fork" not in mp.get_all_start_methods():
            pytest.skip("fork start method unavailable on this platform")
        mp.set_start_method("fork", force=True)
    yield
    if old is not None and old != "fork":
        mp.set_start_method(old, force=True)


class _Harness:
    def __init__(self, client: ModelServerClient, teardown_grace_seconds: float) -> None:
        api_config = APIConfig(type=APIType.Chat)
        self.datagen = MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)
        load_config = LoadConfig(
            type=LoadType.CONSTANT,
            stages=[StandardLoadStage(rate=2, duration=1)],
            num_workers=1,
            worker_max_concurrency=4,
            stage_teardown_grace_seconds=teardown_grace_seconds,
        )
        self.loadgen = LoadGenerator(self.datagen, load_config)
        self.request_queue: RequestQueue[RequestQueueData] = RequestQueue(1)
        self.finished_counter = mp.Value("i", 0)
        self.active_counter = mp.Value("i", 0)
        self.request_phase = mp.Event()
        self.stop_signal = mp.Event()
        self.cancel_signal = mp.Event()
        self.force_stop_signal = mp.Event()
        self.loadgen._force_stop_signal = self.force_stop_signal
        self.request_phase.set()

        worker = Worker(
            0,
            client,
            self.request_queue.get_channel(0),
            self.datagen,
            load_config.worker_max_concurrency,
            self.stop_signal,
            self.cancel_signal,
            self.request_phase,
            self.finished_counter,
            self.active_counter,
            None,
            base_seed=42,
            force_stop_signal=self.force_stop_signal,
            stage_done_counter=mp.Value("i", 0),
            teardown_grace_seconds=teardown_grace_seconds,
        )
        worker.start()
        self.loadgen.workers = [worker]

    async def run_stage(self, stage_id: int, timeout: float) -> float:
        start = time.perf_counter()
        await self.loadgen.run_stage(
            stage_id,
            rate=2,
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


async def test_grace_lets_inflight_requests_complete() -> None:
    """Requests in flight at stage timeout finish during the teardown grace
    and their completions are counted; the worker survives untouched."""
    harness = _Harness(SlowClient(), teardown_grace_seconds=15.0)
    try:
        pids_before = harness.worker_pids()
        elapsed = await harness.run_stage(0, timeout=1.0)

        assert elapsed < 30, f"teardown not bounded: {elapsed:.1f}s"
        assert harness.loadgen.stage_runtime_info[0].status.name == "FAILED"  # timed out
        # Both requests (rate*duration = 2) completed during the grace window.
        assert harness.finished_counter.value == 2
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
        assert elapsed < 30, f"teardown not bounded: {elapsed:.1f}s"
        assert harness.loadgen.stage_runtime_info[0].status.name == "FAILED"
        assert harness.worker_pids() == pids_before, "cancellable tasks must not force a respawn"

        # Multi-stage: the next stage must run through the same worker.
        elapsed = await harness.run_stage(1, timeout=1.0)
        assert elapsed < 30, f"second stage teardown not bounded: {elapsed:.1f}s"
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
        assert elapsed < 45, f"teardown not bounded: {elapsed:.1f}s"
        assert harness.loadgen.stage_runtime_info[0].status.name == "FAILED"
        assert harness.worker_pids() != pids_before, "wedged worker should be respawned"
        assert harness.loadgen.workers[0].is_alive(), "replacement worker should be running"

        # Multi-stage: the respawned worker serves the next stage, which wedges
        # and is bounded again.
        elapsed = await harness.run_stage(1, timeout=1.0)
        assert elapsed < 45, f"second stage teardown not bounded: {elapsed:.1f}s"
        assert 1 in harness.loadgen.stage_runtime_info
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
    finally:
        await loadgen.stop()


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
