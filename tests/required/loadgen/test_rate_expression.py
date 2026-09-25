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
"""A stage ``rate`` may be an expression over stage time ``t``.

A number compiles to a constant expression and must schedule exactly as before;
an expression is integrated over the stage window, the stage dispatches the
integral's worth of requests, and the timers spread them by the rate.
"""

import unittest.mock

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
from inference_perf.datagen import DataGenerator
from inference_perf.datagen.synthetic.mock_datagen import MockDataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.loadgen.load_timer import ConstantLoadTimer, LoadTimer, PoissonLoadTimer
from inference_perf.metrics.request_collector.local import LocalRequestMetricCollector


# Builds a LoadGenerator of the given type around a mocked datagen, with the circuit-breaker lookup patched out.
def _loadgen(load_type: LoadType, *stages: StandardLoadStage) -> LoadGenerator:
    with unittest.mock.patch("inference_perf.loadgen.load_generator.get_circuit_breaker"):
        return LoadGenerator(unittest.mock.MagicMock(spec=DataGenerator), LoadConfig(type=load_type, stages=list(stages)))


# Collects a timer's schedule as offsets from 0, with its rng seeded to `seed`.
def _schedule(timer: LoadTimer, seed: int, limit: int) -> list[float]:
    timer._rand = np.random.default_rng(seed)  # type: ignore[attr-defined]
    generator = timer.start_timer(initial=0.0)
    return [next(generator) for _ in range(limit)]


# Config: rate 4, rate "4" and rate "2*2" over 60s all compile to a constant schedule with
# 240 expected requests and mean rate 4. A ramp "t/10" over 60s expects 60^2/20 = 180
# requests at a mean of 3, and works with stop_condition as well as duration.
def test_rate_forms_accepted() -> None:
    for rate in (4, "4", "2*2"):
        stage = StandardLoadStage(rate=rate, duration=60)
        assert stage.rate_schedule.is_constant
        assert (stage.expected_requests, stage.mean_rate) == (240, 4.0)
    for bound in ({"duration": 60}, {"stop_condition": "t >= 60"}):
        ramp = StandardLoadStage(rate="t/10", **bound)
        assert ramp.expected_requests == 180
        assert ramp.mean_rate == pytest.approx(3.0)


# Config rejections, each at load time: a rate that goes negative inside the window
# ("10 - t" over 60s), a random rate, a symbol other than t, a rate that is zero over the
# whole stage, and a nonpositive number (the existing gt=0 rule).
@pytest.mark.parametrize(
    ("rate", "message"),
    [
        ("10 - t", "outside the permitted range"),
        ("Uniform(1, 5)", "random variable"),
        ("requests/10", "disallowed symbol"),
        ("0*t", "zero over the whole stage"),
        (0, "greater than 0"),
        (-1, "greater than 0"),
    ],
)
def test_bad_rates_rejected(rate: object, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        StandardLoadStage(rate=rate, duration=60)


# The schedule is rebuilt when the stage is mutated after construction (pydantic does not
# re-validate on assignment): changing rate 4 -> "t" over 10s moves the expected count 40 -> 50.
def test_schedule_follows_mutation() -> None:
    stage = StandardLoadStage(rate=4, duration=10)
    assert stage.expected_requests == 40
    stage.rate = "t"
    assert stage.expected_requests == 50


# A numeric rate handed to the CONSTANT timer as a schedule produces the byte-identical
# schedule it did as a bare float (same seed, 240 entries): numbers keep today's exact
# arithmetic. The POISSON timer can't be pinned by seeding (it nests a fresh unseeded timer
# per second), so for it this checks the layer below: the schedule unwraps to the float 4.0
# and takes the original code path.
def test_numeric_rate_schedule_is_unchanged() -> None:
    stage = StandardLoadStage(rate=4, duration=60)
    as_float = _schedule(ConstantLoadTimer(4.0, 60.0), seed=7, limit=240)
    as_schedule = _schedule(ConstantLoadTimer(stage.rate_schedule, 60.0), seed=7, limit=240)
    assert as_float == as_schedule
    poisson = PoissonLoadTimer(stage.rate_schedule, 60.0)
    assert (poisson._rate, poisson._schedule) == (4.0, None)


# CONSTANT timer on the ramp rate = t over 20s: it dispatches the integral, 200 requests;
# the last lands at the window end (20s); and because cumulative requests grow as t^2, a
# quarter of them (50) fall in the first half of the window, give or take jitter.
def test_constant_timer_follows_a_ramp() -> None:
    stage = StandardLoadStage(rate="t", duration=20)
    loadgen = _loadgen(LoadType.CONSTANT, stage)
    times = np.array(_schedule(loadgen.get_timer(stage.rate_schedule, 20.0), seed=3, limit=200))
    with pytest.raises(StopIteration):
        generator = loadgen.get_timer(stage.rate_schedule, 20.0).start_timer(initial=0.0)
        for _ in range(201):
            next(generator)
    assert times[-1] == pytest.approx(20.0)
    assert np.all(np.diff(times) >= 0)
    assert 35 <= int(np.sum(times < 10.0)) <= 65


# POISSON timer on a step rate, 2 req/s for 50s then 20 req/s for 50s (1100 expected): the
# first 1100 dispatches put about 100 in the first half and about 1000 in the second.
def test_poisson_timer_follows_a_step() -> None:
    stage = StandardLoadStage(rate="2 + 18*Heaviside(t - 50)", duration=100)
    loadgen = _loadgen(LoadType.POISSON, stage)
    assert stage.expected_requests == pytest.approx(1100, abs=1)
    times = np.array(_schedule(loadgen.get_timer(stage.rate_schedule, 100.0), seed=11, limit=1000))
    assert 70 <= int(np.sum(times < 50.0)) <= 130


# POISSON timer on a ramp down to zero (10 - t/10 over 100s, 500 expected): pulling well past
# the expected count keeps yielding (the overrun runs at the window's mean) instead of hanging.
def test_poisson_ramp_to_zero_does_not_stall() -> None:
    stage = StandardLoadStage(rate="10 - t/10", duration=100)
    loadgen = _loadgen(LoadType.POISSON, stage)
    times = _schedule(loadgen.get_timer(stage.rate_schedule, 100.0), seed=5, limit=700)
    assert len(times) == 700


# In-process run of a ramp rate = 4*t over 1s (2 expected requests) against the mock server:
# the stage completes, records 1 or 2 requests (the in-process loop can drop the last one at
# the window edge, a known pre-existing issue), and reports the mean rate 2.
async def test_in_process_run_reports_mean_rate() -> None:
    api_config = APIConfig(type=APIType.Chat)
    datagen = MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)
    stage = StandardLoadStage(rate="4*t", duration=1)
    loadgen = LoadGenerator(
        datagen, LoadConfig(type=LoadType.CONSTANT, stages=[stage], num_workers=0, worker_max_concurrency=4)
    )
    collector = LocalRequestMetricCollector()
    await loadgen.run(MockModelServerClient(collector, api_config, mock_latency=0.01))
    info = loadgen.stage_runtime_info[0]
    assert info.status.name == "COMPLETED"
    assert info.rate == pytest.approx(2.0)
    assert len(collector.get_metrics()) in (1, 2)
