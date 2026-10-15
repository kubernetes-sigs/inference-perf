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
"""A stage may set ``request_interval`` instead of ``rate``.

``request_interval`` is the gap in seconds between consecutive requests, as a number
or an expression that may draw from a distribution. It is the arrival process
itself, so ``load.type`` is not set alongside it, and the stage sends however
many requests fit in its window.
"""

import logging
import unittest.mock
from typing import Any

import numpy as np
import pytest
from pydantic import ValidationError

from inference_perf.client.modelserver import MockModelServerClient
from inference_perf.config import (
    APIConfig,
    APIType,
    DataConfig,
    DataGenType,
    LoadConfig,
    LoadType,
    StandardLoadStage,
)
from inference_perf.config.loadgen.config import StageGenType, SweepConfig
from inference_perf.datagen import DataGenerator
from inference_perf.datagen.synthetic.mock_datagen import MockDataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.loadgen.load_timer import ConstantLoadTimer, RequestIntervalLoadTimer
from inference_perf.metrics.request_collector.local import LocalRequestMetricCollector


# Config: request_interval 0.1, "0.1" and "1/10" over 60s are all 600 evenly spaced requests.
# "Exponential(10)" over 60s is accepted, sends roughly 600, and works with stop_condition
# as well as duration. A stage that sets rate instead is unaffected.
def test_request_interval_forms_accepted() -> None:
    for request_interval in (0.1, "0.1", "1/10"):
        stage = StandardLoadStage(request_interval=request_interval, duration=60)
        assert stage.expected_requests == 600
        assert stage.mean_rate == 10.0
    for window in ({"duration": 60}, {"stop_condition": "t >= 60"}):
        stage = StandardLoadStage(request_interval="Exponential(10)", **window)
        assert 400 < stage.expected_requests < 800
    assert StandardLoadStage(rate=10, duration=60).expected_requests == 600


# Stage rejections, each at config load: both rate and request_interval; neither; a gap that
# can go negative (Normal); a gap that depends on t; a zero gap; a fixed gap longer than
# the stage (100s gap, 60s stage).
@pytest.mark.parametrize(
    "fields, message",
    [
        ({"request_interval": 0.1, "rate": 10}, "Exactly one of rate or request_interval"),
        ({}, "Exactly one of rate or request_interval"),
        ({"request_interval": "Normal(0.1, 0.05)"}, "outside the permitted range"),
        ({"request_interval": "0.1 * t"}, "disallowed symbol"),
        ({"request_interval": 0}, "greater than 0"),
        ({"request_interval": 100}, "longer than the stage"),
    ],
)
def test_bad_request_interval_rejected(fields: dict[str, Any], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        StandardLoadStage(duration=60, **fields)


# load.type is rejected when a stage sets request_interval, whether it says poisson or
# constant: the gap distribution already is the arrival process. Leaving load.type out
# is accepted, and so is mixing request_interval stages with rate stages. A sweep, which
# generates rate stages, is rejected too.
def test_load_type_cannot_be_set_with_request_interval() -> None:
    stage = StandardLoadStage(request_interval="Exponential(10)", duration=60)
    for load_type in (LoadType.POISSON, LoadType.CONSTANT):
        with pytest.raises(ValidationError, match="load.type .* cannot be set alongside it"):
            LoadConfig(type=load_type, stages=[stage])
    LoadConfig(stages=[stage])
    LoadConfig(stages=[stage, StandardLoadStage(rate=10, duration=60)])
    with pytest.raises(ValidationError, match="sweep generates rate stages"):
        LoadConfig(stages=[stage], sweep=SweepConfig(type=StageGenType.LINEAR))


# load.type: poisson logs a deprecation warning naming the replacement (request_interval
# Exponential) when the config loads; the config still works. load.type: constant does not warn.
def test_poisson_load_type_warns_deprecated(caplog: pytest.LogCaptureFixture) -> None:
    stage = StandardLoadStage(rate=10, duration=60)
    with caplog.at_level(logging.WARNING, logger="inference_perf.config.loadgen.config"):
        LoadConfig(type=LoadType.POISSON, stages=[stage])
    assert [r.message for r in caplog.records if "deprecated" in r.message] == [
        'load.type: poisson is deprecated. Set request_interval: "Exponential(<rate>)" on each stage instead of rate, '
        "and remove load.type."
    ]
    caplog.clear()
    with caplog.at_level(logging.WARNING, logger="inference_perf.config.loadgen.config"):
        LoadConfig(type=LoadType.CONSTANT, stages=[stage])
    assert not [r for r in caplog.records if "deprecated" in r.message]


# The arrivals are drawn once: reading the stage twice gives the same schedule object,
# so the count read before the run is the count the run dispatches. Changing the
# request_interval after construction (0.1 -> 0.2 over 60s) redraws it: 600 -> 300 requests.
def test_schedule_is_kept_until_the_stage_changes() -> None:
    stage = StandardLoadStage(request_interval="Exponential(10)", duration=60)
    assert stage.request_schedule is stage.request_schedule
    assert stage.expected_requests == len(stage.request_schedule.offsets)
    fixed = StandardLoadStage(request_interval=0.1, duration=60)
    assert fixed.expected_requests == 600
    fixed.request_interval = 0.2
    assert fixed.expected_requests == 300


# The load generator gives a request_interval stage the request_interval timer, and a rate
# stage in the same config the constant timer as before. The request_interval timer started
# at time 100 yields exactly the stage's arrival offsets shifted by 100, then stops.
def test_timer_follows_the_drawn_arrivals() -> None:
    by_gap = StandardLoadStage(request_interval="Uniform(0.05, 0.15)", duration=10)
    by_rate = StandardLoadStage(rate=10, duration=10)
    with unittest.mock.patch("inference_perf.loadgen.load_generator.get_circuit_breaker"):
        loadgen = LoadGenerator(unittest.mock.MagicMock(spec=DataGenerator), LoadConfig(stages=[by_gap, by_rate]))
    timer = loadgen.get_timer(by_gap.schedule, 10.0)
    assert isinstance(timer, RequestIntervalLoadTimer)
    assert isinstance(loadgen.get_timer(by_rate.schedule, 10.0), ConstantLoadTimer)
    times = list(timer.start_timer(initial=100.0))
    assert times == pytest.approx(100.0 + np.asarray(by_gap.request_schedule.offsets))


# In-process run of a 0.25s fixed gap over 1s (4 requests, at 0.25, 0.5, 0.75 and 1.0)
# against the mock server: the stage completes and records 3 or 4 requests. The last one
# is due exactly at the window's end, where the in-process loop can drop it (a known
# pre-existing edge that rate stages share). The stage's runtime info carries the
# request_interval "0.25" and no rate, which is what the per-stage report then shows.
async def test_in_process_run_sends_the_scheduled_requests() -> None:
    api_config = APIConfig(type=APIType.Chat)
    datagen = MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)
    stage = StandardLoadStage(request_interval=0.25, duration=1)
    loadgen = LoadGenerator(datagen, LoadConfig(stages=[stage], num_workers=0, worker_max_concurrency=4))
    collector = LocalRequestMetricCollector()
    await loadgen.run(MockModelServerClient(collector, api_config, mock_latency=0.01))
    info = loadgen.stage_runtime_info[0]
    assert info.status.name == "COMPLETED"
    assert (info.rate, info.request_interval) == (None, "0.25")
    assert len(collector.get_metrics()) in (3, 4)


# Multiprocess run with two workers: a random-gap stage (Exponential(8) over 1s) followed
# by a rate stage. Both complete with nothing dropped, so a stage whose request count is
# only known from the draw survives the real worker lifecycle. The first stage's runtime
# info carries the request_interval string and no rate; the second carries rate 4 as before.
async def test_mp_run_completes_an_request_interval_stage() -> None:
    api_config = APIConfig(type=APIType.Chat)
    datagen = MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)
    load_config = LoadConfig(
        interval=0.1,
        stages=[StandardLoadStage(request_interval="Exponential(8)", duration=1), StandardLoadStage(rate=4, duration=1)],
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
        assert (loadgen.stage_runtime_info[0].rate, loadgen.stage_runtime_info[0].request_interval) == (None, "Exponential(8)")
        assert (loadgen.stage_runtime_info[1].rate, loadgen.stage_runtime_info[1].request_interval) == (4.0, None)
    finally:
        await loadgen.stop()
