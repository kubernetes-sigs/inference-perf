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

"""Regression tests for LoadGenerator.after_stage (#817).

Reports used to be written once, after the last stage, so a run OOM-killed in
stage N lost everything stages 0..N-1 had measured. InferencePerfRunner now
sets after_stage to write each stage's reports as soon as that stage ends.
These tests pin the load generator's side of that contract: which stages the
hook sees and when, that a failing hook cannot abort the run, and that the
hook does not stretch the pause between stages.

The end-to-end mp_run ordering, the sweep probe stage and failed (worker
crash) stages are covered by additions to test_stage_teardown.py and
test_worker_crash_diagnostics.py, which already pay for those runs.
"""

import asyncio
import logging
import multiprocessing as mp
import time
from typing import Awaitable, Callable, Concatenate, Dict, Generator, List, ParamSpec

import pytest

from inference_perf.client.modelserver import MockModelServerClient
from inference_perf.client.server_metrics.base import StageRuntimeInfo, StageStatus
from inference_perf.config import (
    APIConfig,
    APIType,
    DataConfig,
    DataGenType,
    LoadConfig,
    LoadType,
    StandardLoadStage,
)
from inference_perf.datagen import MockDataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.metrics.request_collector.local import LocalRequestMetricCollector

_LOADGEN_LOGGER = "inference_perf.loadgen.load_generator"


P = ParamSpec("P")


# Wraps LoadGenerator.run_stage so a test sees each stage start and end:
# before(0) runs as stage 0 starts, after(0) once it has returned.
def _around_run_stage(
    run_stage: Callable[Concatenate[int, P], Awaitable[None]],
    before: Callable[[int], None],
    after: Callable[[int], None],
) -> Callable[Concatenate[int, P], Awaitable[None]]:
    async def wrapped(stage_id: int, /, *args: P.args, **kwargs: P.kwargs) -> None:
        before(stage_id)
        await run_stage(stage_id, *args, **kwargs)
        after(stage_id)

    return wrapped


@pytest.fixture
def fork_workers() -> Generator[None, None, None]:
    # Workers must fork (matching production: Linux default, forced on macOS in
    # main_cli) so unpicklable test state is inherited rather than pickled.
    old = mp.get_start_method(allow_none=True)
    if old != "fork":
        if "fork" not in mp.get_all_start_methods():
            pytest.skip("fork start method unavailable on this platform")
        mp.set_start_method("fork", force=True)
    children_before = set(mp.active_children())
    yield
    # A test that fails before loadgen.stop() must not leave its workers
    # running into the rest of the session.
    for child in mp.active_children():
        if child not in children_before:
            child.terminate()
            child.join(timeout=5.0)
    if old is not None and old != "fork":
        mp.set_start_method(old, force=True)


def _make_loadgen(num_workers: int, num_stages: int, interval: float, rate: float) -> LoadGenerator:
    api_config = APIConfig(type=APIType.Chat)
    datagen = MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)
    load_config = LoadConfig(
        type=LoadType.CONSTANT,
        interval=interval,
        stages=[StandardLoadStage(rate=rate, duration=1) for _ in range(num_stages)],
        num_workers=num_workers,
        worker_max_concurrency=4,
        stage_teardown_grace_seconds=5.0,
    )
    return LoadGenerator(datagen, load_config)


def _make_client() -> MockModelServerClient:
    return MockModelServerClient(LocalRequestMetricCollector(), APIConfig(type=APIType.Chat), mock_latency=0)


@pytest.mark.timeout(30)
async def test_hook_runs_after_each_stage_in_order_without_workers() -> None:
    """With num_workers=0, run() drives the stages itself rather than mp_run,
    so it has its own call site. Each stage must reach the hook exactly once,
    in order, with its own runtime info, and the hook must finish before the
    next stage starts: a stage's reports are only safe once they are written,
    and the next stage is where the process may die."""
    loadgen = _make_loadgen(num_workers=0, num_stages=3, interval=0, rate=4)
    seen: List[StageRuntimeInfo] = []
    recorded_at_call: List[List[int]] = []
    finished_at: Dict[int, float] = {}

    async def after_stage(info: StageRuntimeInfo) -> None:
        seen.append(info)
        recorded_at_call.append(sorted(loadgen.stage_runtime_info))
        # Long enough that a hook nobody waited for would still be running
        # when the next stage starts (the interval is 0).
        await asyncio.sleep(0.1)
        finished_at[info.stage_id] = time.time()

    loadgen.after_stage = after_stage
    await asyncio.wait_for(loadgen.run(_make_client()), timeout=20)

    assert [info.stage_id for info in seen] == [0, 1, 2]
    assert seen == [loadgen.stage_runtime_info[stage_id] for stage_id in range(3)]
    # Stage N's hook runs before stage N+1 has finished...
    assert recorded_at_call == [[0], [0, 1], [0, 1, 2]]
    # ...and returns before stage N+1 has even started.
    for stage_id in (0, 1):
        next_start = loadgen.stage_runtime_info[stage_id + 1].start_time
        assert finished_at[stage_id] <= next_start, f"stage {stage_id + 1} started before stage {stage_id}'s hook returned"


@pytest.mark.timeout(30)
async def test_failing_hook_is_logged_and_the_run_continues(caplog: pytest.LogCaptureFixture) -> None:
    """A stage-end write can fail (full disk, storage outage). The end-of-run
    reports are still written, so the failure may cost at most that stage's
    early copy: the run must not abort, later stages must still run and still
    get their hook, and the error must be logged with its traceback so the
    missing files can be explained."""
    loadgen = _make_loadgen(num_workers=0, num_stages=2, interval=0, rate=4)
    calls: List[int] = []

    async def after_stage(info: StageRuntimeInfo) -> None:
        calls.append(info.stage_id)
        if info.stage_id == 0:
            raise RuntimeError("disk full while writing stage 0 reports")

    loadgen.after_stage = after_stage
    with caplog.at_level(logging.ERROR, logger=_LOADGEN_LOGGER):
        await asyncio.wait_for(loadgen.run(_make_client()), timeout=20)

    assert calls == [0, 1], "the stage after a failed hook must still run and get its own hook"
    assert loadgen.stage_runtime_info[1].status == StageStatus.COMPLETED

    failures = [r for r in caplog.records if r.name == _LOADGEN_LOGGER and r.exc_info]
    assert len(failures) == 1, f"expected one logged hook failure, got {[r.getMessage() for r in failures]}"
    record = failures[0]
    assert record.levelno == logging.ERROR
    assert record.exc_info is not None
    assert record.exc_info[0] is RuntimeError
    assert str(record.exc_info[1]) == "disk full while writing stage 0 reports"
    assert "Stage 0" in record.getMessage(), "the log line must say which stage's hook failed"


@pytest.mark.timeout(60)
async def test_interrupted_stage_still_runs_its_hook(fork_workers: None, monkeypatch: pytest.MonkeyPatch) -> None:
    """Ctrl-C ends the run after the stage in progress. That stage's data is
    still worth keeping, so its hook must run, and be awaited, before mp_run
    breaks out of the stage loop; no later stage may start."""
    loadgen = _make_loadgen(num_workers=1, num_stages=2, interval=0, rate=1)
    started: List[int] = []

    def interrupt(stage_id: int) -> None:
        # What LoadGenerator._sigint_handler does. A real SIGINT is not sent:
        # if it landed before the handler was installed it would stop pytest.
        loadgen.interrupt_sig = True

    calls: List[int] = []
    finished: List[int] = []

    async def after_stage(info: StageRuntimeInfo) -> None:
        calls.append(info.stage_id)
        await asyncio.sleep(0.2)
        finished.append(info.stage_id)

    monkeypatch.setattr(loadgen, "run_stage", _around_run_stage(loadgen.run_stage, started.append, interrupt))
    loadgen.after_stage = after_stage
    try:
        await asyncio.wait_for(loadgen.run(_make_client()), timeout=45)

        assert started == [0], "the interrupt must stop the run after stage 0"
        assert 1 not in loadgen.stage_runtime_info
        assert calls == [0], "the interrupted run must still hand stage 0 to the hook"
        assert finished == [0], "the hook must be awaited before the loop breaks, not left running"
    finally:
        await loadgen.stop()


@pytest.mark.timeout(60)
@pytest.mark.parametrize("num_workers", [0, 1], ids=["run", "mp_run"])
async def test_hook_overlaps_the_stage_interval(num_workers: int, fork_workers: None) -> None:
    """Writing a stage's reports must not lengthen the configured pause
    between stages. No load is in flight during the interval, so the hook runs
    alongside it, and only a hook slower than the interval delays the next
    stage. A hook half as long as the interval must vanish into it: the gap
    stays one interval. Hook then interval would take 1.5 intervals, and a
    hook that replaced the interval would leave only 0.5.

    Both loops are covered because each has its own call site."""
    interval = 1.0
    loadgen = _make_loadgen(num_workers=num_workers, num_stages=2, interval=interval, rate=2)
    calls: List[int] = []

    async def after_stage(info: StageRuntimeInfo) -> None:
        calls.append(info.stage_id)
        await asyncio.sleep(0.5 * interval)

    loadgen.after_stage = after_stage
    try:
        await asyncio.wait_for(loadgen.run(_make_client()), timeout=45)

        assert calls == [0, 1]
        stage_0 = loadgen.stage_runtime_info[0]
        stage_1 = loadgen.stage_runtime_info[1]
        # end_time closes stage 0's load window. In mp_run the stage teardown
        # comes after it and before the interval, so it is taken out.
        gap = stage_1.start_time - stage_0.end_time - (stage_0.teardown_duration or 0.0)
        # Lower bound: the interval is still waited out in full. Upper bound:
        # hook and interval overlap. Sleeps never end early, so running them
        # one after the other cannot come in under 1.5 intervals.
        assert 0.95 * interval <= gap < 1.45 * interval, f"gap between stages {gap:.2f}s, interval {interval:.2f}s"
    finally:
        await loadgen.stop()


async def test_run_after_stage_calls_nothing_without_a_hook_or_runtime_info() -> None:
    """Two cases must be no-ops. A LoadGenerator used without
    InferencePerfRunner has no hook. A session stage that finds no sessions
    left returns before recording runtime info; that stage never ran, so there
    is nothing to report and the hook must not be called for it."""
    loadgen = _make_loadgen(num_workers=0, num_stages=2, interval=0, rate=1)
    loadgen.stage_runtime_info[0] = StageRuntimeInfo(
        stage_id=0, rate=1.0, start_time=0.0, end_time=1.0, status=StageStatus.COMPLETED
    )

    assert loadgen.after_stage is None
    await loadgen._run_after_stage(0)  # No hook: returns without raising.

    # Records the argument itself: a hook that read info.stage_id would raise
    # on None, and _run_after_stage would swallow that, hiding a bad call.
    calls: List[StageRuntimeInfo] = []

    async def after_stage(info: StageRuntimeInfo) -> None:
        calls.append(info)

    loadgen.after_stage = after_stage
    await loadgen._run_after_stage(1)  # Stage 1 recorded no runtime info.
    assert calls == []

    # Control: the same hook is called for a stage that did record its info,
    # so the empty list above is not vacuous.
    await loadgen._run_after_stage(0)
    assert calls == [loadgen.stage_runtime_info[0]]
