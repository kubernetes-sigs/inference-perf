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

import asyncio
import logging
import multiprocessing as mp
from multiprocessing.process import BaseProcess
from multiprocessing.synchronize import Event as ProcessEvent
import threading
import time
from typing import Callable, Generator, List, Optional, Sequence, Union
from unittest.mock import MagicMock, patch
import pytest

from inference_perf.apis import (
    InferenceInfo,
    RequestLifecycleMetric,
    UnaryResponseMetrics,
)
from inference_perf.circuit_breaker import _initialized_circuit_breakers, feed_breakers
from inference_perf.metrics.request_collector import RequestMetricCollector
from inference_perf.metrics.request_collector.multiprocess import (
    _STAGE_COUNTERS,
    MultiprocessRequestMetricCollector,
    _stage_counter,
)
from inference_perf.payloads import RequestMetrics, Text


def _create_dummy_metric(req_id: int = 1) -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=0.0,
        start_time=1.0,
        end_time=2.0,
        request_data=f"request-{req_id}",
        response_data=f"response-{req_id}",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=10)),
            response_metrics=UnaryResponseMetrics(output_tokens=5),
        ),
        error=None,
    )


@pytest.mark.asyncio
async def test_collector_lifecycle_start_context() -> None:
    collector = MultiprocessRequestMetricCollector()

    total_metrics = 10
    async with collector.start():
        for i in range(total_metrics):
            collector.record_metric(_create_dummy_metric(i))

    collected = collector.get_metrics()
    assert len(collected) == total_metrics
    assert [m.request_data for m in collected] == [f"request-{i}" for i in range(total_metrics)]


@pytest.mark.asyncio
async def test_collector_drains_mixed_batches_and_singletons() -> None:
    collector = MultiprocessRequestMetricCollector()

    m1 = _create_dummy_metric(1)
    m2 = _create_dummy_metric(2)
    m3 = _create_dummy_metric(3)

    # Put a list batch and a singleton into the queue directly
    collector.queue.put([m1, m2])
    collector.queue.put(m3)
    collector.queue.put(None)

    collected = await collector.collect_metrics()
    assert len(collected) == 3
    assert [m.request_data for m in collected] == ["request-1", "request-2", "request-3"]


@pytest.mark.asyncio
async def test_collector_bulk_drain_capacity() -> None:
    collector = MultiprocessRequestMetricCollector()

    num_items = 100
    for i in range(num_items):
        collector.record_metric(_create_dummy_metric(i))
    collector.queue.put(None)

    collected = await collector.collect_metrics()
    assert len(collected) == num_items


@pytest.mark.asyncio
async def test_collector_feeds_circuit_breakers_when_initialized() -> None:
    collector = MultiprocessRequestMetricCollector()
    # main.py wires breakers in as a collector observer; mirror that here.
    collector.add_observer(feed_breakers)
    m1 = _create_dummy_metric(1)

    mock_breaker = MagicMock()
    with patch.dict(_initialized_circuit_breakers, {"test_breaker": mock_breaker}):
        async with collector.start():
            collector.record_metric(m1)

    mock_breaker.feed.assert_called_once_with(m1)


# Stage-end reports (#817) read a finished stage's metrics while the collector
# is still running, so the tests below exercise the collector mid-run: what
# snapshot() exposes, and how wait_for_recorded() closes the gap between a
# worker handing a metric over and this process having drained it.

_COLLECTOR_LOGGER = "inference_perf.metrics.request_collector.multiprocess"


def _reap_children() -> None:
    for child in mp.active_children():
        child.kill()
        child.join(timeout=5)


@pytest.fixture
def fork_start_method() -> Generator[None, None, None]:
    # Producers must fork (matching production: Linux default, forced on macOS
    # in main_cli). The collector's queue and counter take the start method
    # current when it is built, so this runs before the test constructs one.
    old = mp.get_start_method(allow_none=True)
    if old != "fork":
        if "fork" not in mp.get_all_start_methods():
            pytest.skip("fork start method unavailable on this platform")
        mp.set_start_method("fork", force=True)
    yield
    _reap_children()
    if old != "fork":
        mp.set_start_method(old, force=True)


@pytest.fixture
def spawn_start_method() -> Generator[None, None, None]:
    # spawn pickles everything a new process gets, as forkserver (Python
    # 3.14's Linux default) does, without starting a fork server.
    old = mp.get_start_method(allow_none=True)
    mp.set_start_method("spawn", force=True)
    yield
    _reap_children()
    mp.set_start_method(old, force=True)


async def _until(condition: Callable[[], bool], timeout: float = 5.0) -> None:
    async def poll() -> None:
        while not condition():
            await asyncio.sleep(0.005)

    await asyncio.wait_for(poll(), timeout)


async def _stop(processes: Sequence[BaseProcess], timeout: float) -> None:
    """Give producers time to flush their metrics and exit, then kill any left.

    Polls rather than calling join(): a producer only exits once the collector
    has drained what it put, and the drain needs this event loop to run.
    """
    try:
        await _until(lambda: not any(p.is_alive() for p in processes), timeout)
    except asyncio.TimeoutError:
        pass
    for process in processes:
        if process.is_alive():
            process.kill()
            process.join(timeout=5)


def _record_then_signal(
    collector: MultiprocessRequestMetricCollector, count: int, payload_size: int, finished: ProcessEvent
) -> None:
    payload = "x" * payload_size
    for i in range(count):
        collector.record_metric(_create_dummy_metric(i).model_copy(update={"response_data": payload}))
    finished.set()


def _record_when_released(collector: MultiprocessRequestMetricCollector, count: int, go: ProcessEvent) -> None:
    metric = _create_dummy_metric(0)
    go.wait()
    for _ in range(count):
        collector.record_metric(metric)


def _record_what_was_inherited(collector: MultiprocessRequestMetricCollector) -> None:
    inherited = len(collector.snapshot())
    collector.record_metric(_create_dummy_metric(0).model_copy(update={"request_data": f"inherited-{inherited}"}))


async def test_snapshot_returns_metrics_collected_so_far_while_running() -> None:
    """snapshot() serves the metrics drained so far without stopping the collector.

    Stage-end reports are built from it while the run goes on. Before #817 the
    collector only exposed its metrics once start() exited at the end of the
    run, so a run killed in a later stage lost every stage it had finished.
    Each call returns a copy: a stage's report must not change under it as the
    drain keeps appending, and a caller editing its copy must not change what
    the end-of-run reports see. Reading mid-run must not duplicate or drop
    anything in get_metrics() once start() exits either.
    """
    collector = MultiprocessRequestMetricCollector()

    async with collector.start():
        for i in range(3):
            collector.record_metric(_create_dummy_metric(i))
        await _until(lambda: len(collector.snapshot()) == 3)

        mid_run = collector.snapshot()
        assert [m.request_data for m in mid_run] == ["request-0", "request-1", "request-2"]
        mid_run.clear()
        assert len(collector.snapshot()) == 3

        held = collector.snapshot()
        for i in range(3, 5):
            collector.record_metric(_create_dummy_metric(i))
        await _until(lambda: len(collector.snapshot()) == 5)
        assert len(held) == 3

    assert [m.request_data for m in collector.get_metrics()] == [f"request-{i}" for i in range(5)]


@pytest.mark.timeout(30)
async def test_wait_for_recorded_catches_up_with_metrics_still_in_flight(
    fork_start_method: None, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    """wait_for_recorded() returns only once every metric a worker recorded has arrived.

    A worker's queue.put() only hands the metric to its feeder thread, which
    pickles it and writes it to a pipe this process drains asynchronously; the
    end-of-stage rendezvous only guarantees put() was called. A stage-end
    report built straight after the stage would miss the tail of the stage,
    and object storage (GCS/S3 skip names that already exist) would keep that
    short copy over the complete end-of-run one.

    The drain is held back until the producer has finished so its metrics are
    certainly in flight when it signals; on a real run the drain falls behind
    on its own (about 1% of 2000 10 KB metrics had arrived when a forked
    producer finished). Large payloads make the catch-up take many polls.
    """
    total = 2000
    caplog.set_level(logging.WARNING, logger=_COLLECTOR_LOGGER)
    collector = MultiprocessRequestMetricCollector()

    drain_released = threading.Event()
    real_get = collector.queue.get

    def held_get(
        block: bool = True, timeout: Optional[float] = None
    ) -> Optional[Union[RequestLifecycleMetric, List[RequestLifecycleMetric]]]:
        drain_released.wait()
        return real_get(block, timeout)

    monkeypatch.setattr(collector.queue, "get", held_get)

    finished = mp.Event()
    producer = mp.Process(target=_record_then_signal, args=(collector, total, 10_000, finished))
    async with collector.start():
        producer.start()
        try:
            await _until(finished.is_set, timeout=10)
            # The producer is done, yet none of its metrics have reached this
            # process: a wait that only looked at what had arrived would build
            # an empty stage.
            assert collector.snapshot() == []

            drain_released.set()
            assert await asyncio.wait_for(collector.wait_for_recorded(0), timeout=10)
            arrived = collector.snapshot()
        finally:
            drain_released.set()
            await _stop([producer], timeout=10)

    assert producer.exitcode == 0
    assert [m.request_data for m in arrived] == [f"request-{i}" for i in range(total)]
    assert not [r for r in caplog.records if r.name == _COLLECTOR_LOGGER and r.levelno >= logging.WARNING]


@pytest.mark.timeout(15)
async def test_wait_for_recorded_gives_up_when_recorded_metrics_never_arrive(caplog: pytest.LogCaptureFixture) -> None:
    """wait_for_recorded() stops waiting once no metric has arrived for stall_timeout.

    A worker killed (e.g. OOM) after counting a metric but before its feeder
    flushed it takes that metric with it, so the count never catches up.
    Waiting for it would hold the run at the stage boundary forever; the
    collector warns with the missing count and reports that it gave up, so
    the stage's reports wait for the end of the run. Counts are kept per
    stage, so the next stage is not held up by what the previous one lost.
    """
    stall_timeout = 1.0
    caplog.set_level(logging.WARNING, logger=_COLLECTOR_LOGGER)
    collector = MultiprocessRequestMetricCollector()

    async with collector.start():
        for i in range(2):
            collector.record_metric(_create_dummy_metric(i))
        await _until(lambda: len(collector.snapshot()) == 2)
        # Counted as recorded, never put: what a worker that died with
        # metrics still in its feeder buffer leaves behind.
        with collector._recorded.get_lock():
            collector._recorded[0] += 3

        started = time.monotonic()
        complete = await asyncio.wait_for(collector.wait_for_recorded(0, stall_timeout=stall_timeout), timeout=5)
        stalled_wait = time.monotonic() - started
        assert complete is False
        assert [m.request_data for m in collector.snapshot()] == ["request-0", "request-1"]

        for i in range(2, 4):
            collector.record_metric(_create_dummy_metric(i).model_copy(update={"stage_id": 1}))
        started = time.monotonic()
        complete = await asyncio.wait_for(collector.wait_for_recorded(1, stall_timeout=stall_timeout), timeout=5)
        next_stage_wait = time.monotonic() - started
        assert complete is True
        assert len(collector.snapshot()) == 4

    assert stall_timeout <= stalled_wait < stall_timeout + 2
    assert next_stage_wait < stall_timeout / 2
    warnings = [r for r in caplog.records if r.name == _COLLECTOR_LOGGER and r.levelno == logging.WARNING]
    assert len(warnings) == 1
    assert "Stage 0: 3 of 5" in warnings[0].getMessage()


async def test_wait_for_recorded_returns_at_once_when_nothing_is_draining() -> None:
    """wait_for_recorded() does not wait when the collector is not running.

    Before start() and after it exits there is no drain, so nothing more can
    arrive and waiting could only run into the stall timeout. Both calls must
    return straight away, and say whether the stage's metrics are all in: a
    stage reported complete while its metrics are missing would be written
    short, and stay that way on object storage.
    """
    collector = MultiprocessRequestMetricCollector()
    assert await asyncio.wait_for(collector.wait_for_recorded(0, stall_timeout=30), timeout=1) is True

    with collector._recorded.get_lock():
        collector._recorded[0] += 1
    assert await asyncio.wait_for(collector.wait_for_recorded(0, stall_timeout=30), timeout=1) is False

    async with collector.start():
        pass

    assert await asyncio.wait_for(collector.wait_for_recorded(0, stall_timeout=30), timeout=1) is False


@pytest.mark.timeout(60)
async def test_recorded_count_is_exact_with_concurrent_producers(fork_start_method: None) -> None:
    """Workers recording at the same time are each counted exactly once.

    wait_for_recorded() trusts the shared count. An undercount lets it return
    before a stage's metrics are all in, so its reports miss requests; an
    overcount makes the stage's wait sit out the stall timeout and leaves its
    reports to the end of the run. Without the lock around the increment,
    four producers lose a sizeable share of 20000 increments here.
    """
    producers, per_producer = 4, 5000
    collector = MultiprocessRequestMetricCollector()
    go = mp.Event()
    processes = [mp.Process(target=_record_when_released, args=(collector, per_producer, go)) for _ in range(producers)]

    async with collector.start():
        for process in processes:
            process.start()
        try:
            go.set()
            await _until(lambda: not any(p.is_alive() for p in processes), timeout=30)
        finally:
            go.set()
            await _stop(processes, timeout=10)
        recorded = collector._recorded[0]

    assert [p.exitcode for p in processes] == [0] * producers
    assert recorded == producers * per_producer
    assert len(collector.get_metrics()) == producers * per_producer


async def test_pickled_state_leaves_out_collected_metrics_and_the_drain_task() -> None:
    """The state workers receive carries the queue and counter, not this process's metrics.

    Under spawn and forkserver (Python 3.14's Linux default) every worker,
    including the ones respawned between stages, gets the collector pickled
    through its client. Mid-run the collector now holds everything collected
    so far and its drain task: shipping the metrics would copy the whole run
    into each new worker, and the task cannot be pickled at all. The worker
    still needs the queue and counter to report through, and this process
    must keep its metrics.
    """
    collector = MultiprocessRequestMetricCollector()

    async with collector.start():
        collector.record_metric(_create_dummy_metric(1))
        await _until(lambda: len(collector.snapshot()) == 1)

        state = collector.__getstate__()

        assert state["metrics"] == []
        assert state["_collector_task"] is None
        assert state["queue"] is collector.queue
        assert state["_recorded"] is collector._recorded
        assert [m.request_data for m in collector.snapshot()] == ["request-1"]
        assert collector._collector_task is not None


@pytest.mark.timeout(60)
async def test_worker_spawned_mid_run_reports_through_the_collector(spawn_start_method: None) -> None:
    """A worker started under spawn while the collector runs inherits no metrics and still reports.

    This is the respawn path end to end: starting the worker pickles the live
    collector (a TypeError if its drain task went along), the worker must
    start from an empty list rather than a copy of the parent's, and what it
    records must reach the parent's queue and counter so wait_for_recorded()
    covers it.
    """
    collector = MultiprocessRequestMetricCollector()

    async with collector.start():
        for i in range(3):
            collector.record_metric(_create_dummy_metric(i))
        await asyncio.wait_for(collector.wait_for_recorded(0), timeout=10)
        assert len(collector.snapshot()) == 3

        worker = mp.Process(target=_record_what_was_inherited, args=(collector,))
        worker.start()
        try:
            await _until(lambda: not worker.is_alive(), timeout=30)
            await asyncio.wait_for(collector.wait_for_recorded(0), timeout=10)
            arrived = [m.request_data for m in collector.snapshot()]
        finally:
            await _stop([worker], timeout=10)

    assert worker.exitcode == 0
    assert arrived == ["request-0", "request-1", "request-2", "inherited-0"]


def test_pickled_state_keeps_what_the_base_class_leaves_out(monkeypatch: pytest.MonkeyPatch) -> None:
    """The state workers receive starts from RequestMetricCollector's.

    The base class may leave out parent-only state of its own (#740 drops its
    observers, which hold locks); building on super() keeps that out of the
    workers too, where copying __dict__ would put it back and fail every
    worker start under spawn and forkserver.
    """

    def base_state(self: RequestMetricCollector) -> dict[str, object]:
        return {k: v for k, v in self.__dict__.items() if k != "_parent_only"}

    monkeypatch.setattr(RequestMetricCollector, "__getstate__", base_state, raising=False)
    collector = MultiprocessRequestMetricCollector()
    collector.__dict__["_parent_only"] = object()

    state = collector.__getstate__()

    assert "_parent_only" not in state
    assert state["queue"] is collector.queue


def test_stage_counters_wrap_around_and_keep_stages_apart() -> None:
    """Each stage counts into its own shared counter until stage ids wrap around.

    The counters are shared memory, sized before workers fork, so stage ids
    map onto them modulo their number: consecutive stages never share one,
    and the sweep's preprocessing stage (-1) and metrics without a stage take
    the last one.
    """
    assert len({_stage_counter(stage_id) for stage_id in range(_STAGE_COUNTERS)}) == _STAGE_COUNTERS
    assert _stage_counter(_STAGE_COUNTERS) == _stage_counter(0)
    assert _stage_counter(-1) == _stage_counter(None) == _STAGE_COUNTERS - 1
