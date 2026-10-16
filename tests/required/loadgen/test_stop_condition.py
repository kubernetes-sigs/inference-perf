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

import unittest.mock

import numpy as np

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


# Runs one CONSTANT stage in-process against a mock server whose every request times out (so every result is an
# error), with the load generator observing the collector as main wires it; returns the stage's runtime info and
# how many results were recorded.
async def _run_in_process_with_errors(stage: StandardLoadStage) -> tuple[StageRuntimeInfo, int]:
    api_config = APIConfig(type=APIType.Chat)
    datagen = MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)
    load_config = LoadConfig(type=LoadType.CONSTANT, stages=[stage], num_workers=0, worker_max_concurrency=4)
    loadgen = LoadGenerator(datagen, load_config)
    collector = LocalRequestMetricCollector()
    collector.add_observer(loadgen.observe_result)
    client = MockModelServerClient(collector, api_config, timeout=0.001, mock_latency=0.01)
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


# A time-only stage (duration=1 at rate 4, every request failing) runs to its deadline: COMPLETED with
# stop_condition_met_at None.
async def test_time_only_stage_records_no_measured_stop() -> None:
    info, count = await _run_in_process_with_errors(StandardLoadStage(rate=4, duration=1))
    assert info.status.name == "COMPLETED"
    assert info.stop_condition_met_at is None
    assert count in (3, 4), count


# Multiprocess run, two workers at rate 20 with '(t >= 10) | (errors >= 5)', every request timing out, results
# drained by the multiprocess collector in the main process: the stage COMPLETES early with stop_condition_met_at
# set, far fewer than its 200 scheduled requests recorded, and the undispatched remainder reported as dropped.
async def test_mp_run_ends_on_measured_condition() -> None:
    api_config = APIConfig(type=APIType.Chat)
    datagen = MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)
    load_config = LoadConfig(
        type=LoadType.CONSTANT,
        interval=0.1,
        stages=[StandardLoadStage(rate=20, stop_condition="(t >= 10) | (errors >= 5)")],
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
