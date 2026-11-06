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
"""``stop_condition`` must be indistinguishable from ``duration`` in the load generator.

With ``t`` as the only symbol, ``stop_condition: "t >= 60"`` and ``duration: 60``
bound the same dispatch window. These tests pin that equivalence at every layer
the stage flows through: the timer schedule, the in-process run loop, and the
multiprocess run loop.
"""

import time
import unittest.mock
from typing import Optional

import numpy as np

from inference_perf.apis import InferenceInfo, RequestLifecycleMetric
from inference_perf.client.modelserver import MockModelServerClient
from inference_perf.client.server_metrics.base import StageRuntimeInfo
from inference_perf.config import (
    APIConfig,
    APIType,
    DataConfig,
    DataGenType,
    LoadConfig,
    LoadType,
    StandardLoadStage,
    StopConditionConfig,
)
from inference_perf.datagen import DataGenerator
from inference_perf.datagen.synthetic.mock_datagen import MockDataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.metrics.request_collector import MultiprocessRequestMetricCollector
from inference_perf.metrics.request_collector.local import LocalRequestMetricCollector
from inference_perf.payloads import RequestMetrics, Text


# Builds a CONSTANT-load LoadGenerator around a mocked datagen, with the circuit-breaker lookup patched out.
def _loadgen(load_config: LoadConfig) -> LoadGenerator:
    with unittest.mock.patch("inference_perf.loadgen.load_generator.get_circuit_breaker"):
        return LoadGenerator(unittest.mock.MagicMock(spec=DataGenerator), load_config)


# Two stages at rate 4, one with duration=60 and one with stop_condition 't >= 60', are handed the same
# (rate, duration) pair; with the timer rng seeded identically both yield the identical 240-entry schedule.
def test_timer_schedule_identical_for_duration_and_stop_condition() -> None:
    by_duration = StandardLoadStage(rate=4, duration=60)
    by_condition = StandardLoadStage(rate=4, stop_condition="t >= 60")
    loadgen = _loadgen(LoadConfig(type=LoadType.CONSTANT, stages=[by_duration, by_condition], num_workers=1))

    schedules = []
    for stage in (by_duration, by_condition):
        timer = loadgen.get_timer(stage.rate, stage.effective_duration)
        timer._rand = np.random.default_rng(7)  # type: ignore[attr-defined]
        schedules.append(list(timer.start_timer(initial=0.0)))

    assert len(schedules[0]) == int(4 * 60) == 240
    assert schedules[0] == schedules[1]


# For the POISSON timer the schedule cannot be pinned by seeding (it nests a fresh unseeded ConstantLoadTimer per
# second), so this pins the layer below: both stages hand the timer the same rate 4 and the same duration 10.0.
def test_poisson_timer_receives_same_rate_and_duration() -> None:
    by_duration = StandardLoadStage(rate=4, duration=10)
    by_condition = StandardLoadStage(rate=4, stop_condition="t >= 10")
    loadgen = _loadgen(LoadConfig(type=LoadType.POISSON, stages=[by_duration, by_condition], num_workers=1))

    timers = [loadgen.get_timer(stage.rate, stage.effective_duration) for stage in (by_duration, by_condition)]
    assert [(t._rate, t._duration) for t in timers] == [(4.0, 10.0), (4.0, 10.0)]  # type: ignore[attr-defined]


# Runs one CONSTANT stage in-process (num_workers=0) against the mock server and returns how many requests it recorded.
async def _run_in_process(stage: StandardLoadStage) -> tuple[str, int]:
    api_config = APIConfig(type=APIType.Chat)
    datagen = MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)
    load_config = LoadConfig(type=LoadType.CONSTANT, stages=[stage], num_workers=0, worker_max_concurrency=4)
    loadgen = LoadGenerator(datagen, load_config)
    collector = LocalRequestMetricCollector()
    client = MockModelServerClient(collector, api_config, mock_latency=0.01)
    await loadgen.run(client)
    return loadgen.stage_runtime_info[0].status.name, len(collector.get_metrics())


# In-process run at rate 4: duration=1 and stop_condition 't >= 1' both COMPLETE and each records 3 or 4 requests.
# The count is not pinned to 4 because the in-process loop already drops the last scheduled request whenever its
# dispatch time rounds up to the stage end (the timer normalises intervals to sum to exactly the duration, and the
# loop requires dispatch time < end time); that is pre-existing and independent of which field bounds the window.
async def test_in_process_run_dispatches_same_count() -> None:
    for stage in (StandardLoadStage(rate=4, duration=1), StandardLoadStage(rate=4, stop_condition="t >= 1")):
        status, count = await _run_in_process(stage)
        assert status == "COMPLETED"
        assert count in (3, 4), f"{stage}: dispatched {count}"


# Multiprocess run with two workers: a duration=1 stage followed by a stop_condition 't >= 1' stage, both at rate 4;
# both stages COMPLETE with zero dropped requests, so the stop_condition stage survives the real worker lifecycle.
async def test_mp_run_completes_stop_condition_stage() -> None:
    api_config = APIConfig(type=APIType.Chat)
    datagen = MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)
    load_config = LoadConfig(
        type=LoadType.CONSTANT,
        interval=0.1,
        stages=[StandardLoadStage(rate=4, duration=1), StandardLoadStage(rate=4, stop_condition="t >= 1")],
        num_workers=2,
        worker_max_concurrency=4,
        stage_teardown_grace_seconds=10.0,
    )
    loadgen = LoadGenerator(datagen, load_config)
    client = MockModelServerClient(LocalRequestMetricCollector(), api_config, mock_latency=0.05)
    await loadgen.run(client)
    try:
        for stage_id in (0, 1):
            assert loadgen.stage_runtime_info[stage_id].status.name == "COMPLETED"
            assert loadgen.stage_runtime_info[stage_id].dropped_requests == 0
    finally:
        await loadgen.stop()


# Runs one CONSTANT stage in-process against a mock server, by default one whose every request times out (so every
# result is an error), with the load generator observing the collector as main wires it; returns the stage's
# runtime info and how many results were recorded.
async def _run_in_process_with_errors(
    stage: StandardLoadStage, *, timeout: Optional[float] = 0.001, mock_latency: float = 0.01
) -> tuple[StageRuntimeInfo, int]:
    api_config = APIConfig(type=APIType.Chat)
    datagen = MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)
    load_config = LoadConfig(type=LoadType.CONSTANT, stages=[stage], num_workers=0, worker_max_concurrency=4)
    loadgen = LoadGenerator(datagen, load_config)
    collector = LocalRequestMetricCollector()
    collector.add_observer(loadgen.observe_result)
    client = MockModelServerClient(collector, api_config, timeout=timeout, mock_latency=mock_latency)
    await loadgen.run(client)
    return loadgen.stage_runtime_info[0], len(collector.get_metrics())


# In-process at rate 50 with '(t >= 10) | (errors >= 5)': the stage ends COMPLETED on its fifth failed result, long
# before the 500 requests its 10s deadline would send, and stop_condition_met_at records the stage second.
async def test_in_process_run_ends_on_measured_condition() -> None:
    stage = StandardLoadStage(rate=50, stop_condition="(t >= 10) | (errors >= 5)")
    info, count = await _run_in_process_with_errors(stage)
    assert info.status.name == "COMPLETED"
    assert info.stop_condition_met_at is not None and info.stop_condition_met_at < 5
    assert 5 <= count < 100, count


# The object form flows through the run: hold=3 on the same condition needs errors >= 5 on three consecutive
# results, so at least seven results are recorded before the stage ends COMPLETED.
async def test_in_process_run_honours_hold() -> None:
    stage = StandardLoadStage(rate=50, stop_condition=StopConditionConfig(condition="(t >= 10) | (errors >= 5)", hold=3))
    info, count = await _run_in_process_with_errors(stage)
    assert info.status.name == "COMPLETED"
    assert info.stop_condition_met_at is not None
    assert 7 <= count < 100, count


# A window term ends the run too: at rate 50 with '(t >= 10) | (avg_over_last(error, 5) >= 1)' and every request
# failing, the stage ends COMPLETED once five results fill the window, long before its 500 scheduled requests.
async def test_in_process_run_ends_on_a_window_term() -> None:
    stage = StandardLoadStage(rate=50, stop_condition="(t >= 10) | (avg_over_last(error, 5) >= 1)")
    info, count = await _run_in_process_with_errors(stage)
    assert info.status.name == "COMPLETED"
    assert info.stop_condition_met_at is not None and info.stop_condition_met_at < 5
    assert 5 <= count < 100, count


# The loop records dispatch order, so an _over_sent term completes: at rate 50 with every request failing,
# '(t >= 10) | (count_over_sent(error, 0, 5) >= 5)' ends the stage once the first five requests sent have finished.
async def test_in_process_run_places_results_by_dispatch_order() -> None:
    stage = StandardLoadStage(rate=50, stop_condition="(t >= 10) | (count_over_sent(error, 0, 5) >= 5)")
    info, count = await _run_in_process_with_errors(stage)
    assert info.status.name == "COMPLETED"
    assert info.stop_condition_met_at is not None and info.stop_condition_met_at < 5
    assert 5 <= count < 100, count


# The load generator passes the report's output-token rule to each stage's monitor.
def test_monitor_follows_use_server_output_tokens() -> None:
    stage = StandardLoadStage(rate=1, stop_condition="(t >= 10) | (errors >= 1)")
    load_config = LoadConfig(type=LoadType.CONSTANT, stages=[stage], num_workers=0)
    for flag in (False, True):
        loadgen = LoadGenerator(unittest.mock.MagicMock(spec=DataGenerator), load_config, use_server_output_tokens=flag)
        monitor = loadgen._stop_monitor_for(0, stage)
        assert monitor is not None and monitor.use_server_output_tokens is flag
    assert loadgen._stop_monitor_for(0, StandardLoadStage(rate=1, duration=10)) is None


# observe_result hands the monitor the live in-flight count: with 3 requests in flight, one result ends a stage on
# '(t >= 300) | (in_flight >= 3)' right away, without waiting for a tick.
def test_observe_result_reads_in_flight() -> None:
    stage = StandardLoadStage(rate=1, stop_condition="(t >= 300) | (in_flight >= 3)")
    loadgen = _loadgen(LoadConfig(type=LoadType.CONSTANT, stages=[stage], num_workers=0))
    monitor = loadgen._stop_monitor_for(0, stage)
    assert monitor is not None
    monitor.start(time.perf_counter())
    loadgen._stop_monitor = monitor
    loadgen._local_in_flight = 3
    metric = RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=0.0,
        start_time=0.0,
        end_time=0.0,
        request_data="x",
        info=InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=1))),
        error=None,
    )
    loadgen.observe_result(metric)
    assert monitor.met_at is not None
    assert monitor.values_when_met == {"in_flight": 3.0}


# A stalled server ends the stage on the loop's tick, with no result arriving: at rate 20 every request takes 3s,
# and '(t >= 10) | ((in_flight > 0) & (since_last_completion > 1))' holds after t=1, before the first one finishes.
async def test_in_process_run_ends_a_stall_on_tick() -> None:
    stage = StandardLoadStage(rate=20, stop_condition="(t >= 10) | ((in_flight > 0) & (since_last_completion > 1))")
    info, _count = await _run_in_process_with_errors(stage, timeout=None, mock_latency=3.0)
    assert info.status.name == "COMPLETED"
    assert info.stop_condition_met_at is not None and 1 < info.stop_condition_met_at < 3


# A time-only stage (duration=1 at rate 4, every request failing) runs to its deadline: COMPLETED with
# stop_condition_met_at None.
async def test_time_only_stage_records_no_measured_stop() -> None:
    info, count = await _run_in_process_with_errors(StandardLoadStage(rate=4, duration=1))
    assert info.status.name == "COMPLETED"
    assert info.stop_condition_met_at is None
    assert count in (3, 4), count


# Multiprocess run, two workers at rate 20 with '(t >= 10) | (count_over_sent(error, 0, 5) >= 5)', every request
# timing out, results drained by the multiprocess collector in the main process: each result's scheduled time comes
# back from its worker unchanged, so dispatch order places it, and the stage COMPLETES early with
# stop_condition_met_at set, far fewer than its 200 scheduled requests recorded, the remainder reported as dropped.
async def test_mp_run_ends_on_measured_condition() -> None:
    api_config = APIConfig(type=APIType.Chat)
    datagen = MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)
    load_config = LoadConfig(
        type=LoadType.CONSTANT,
        interval=0.1,
        stages=[StandardLoadStage(rate=20, stop_condition="(t >= 10) | (count_over_sent(error, 0, 5) >= 5)")],
        num_workers=2,
        worker_max_concurrency=4,
        stage_teardown_grace_seconds=10.0,
    )
    loadgen = LoadGenerator(datagen, load_config)
    collector = MultiprocessRequestMetricCollector()
    collector.add_observer(loadgen.observe_result)
    client = MockModelServerClient(collector, api_config, timeout=0.001, mock_latency=0.01)
    try:
        async with collector.start():
            await loadgen.run(client)
        info = loadgen.stage_runtime_info[0]
        assert info.status.name == "COMPLETED"
        assert info.stop_condition_met_at is not None
        assert info.dropped_requests is not None and info.dropped_requests > 0
        assert 5 <= len(collector.get_metrics()) < 200, len(collector.get_metrics())
    finally:
        await loadgen.stop()
