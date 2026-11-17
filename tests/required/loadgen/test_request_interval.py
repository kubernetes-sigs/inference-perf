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
# can go negative (Normal(0.1, 0.05)); a gap that is zero at t=0 ("0.1 * t"); and a gap of 0.
@pytest.mark.parametrize(
    "fields, message",
    [
        ({"request_interval": 0.1, "rate": 10}, "Exactly one of rate or request_interval"),
        ({}, "Exactly one of rate or request_interval"),
        ({"request_interval": "Normal(0.1, 0.05)"}, "outside the permitted range"),
        ({"request_interval": "0.1 * t"}, "is zero at t=0"),
        ({"request_interval": 0}, "greater than 0"),
    ],
)
def test_bad_request_interval_rejected(fields: dict[str, object], message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        StandardLoadStage.model_validate({"duration": 60, **fields})


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


# A random rate points at request_interval: rate "Exponential(10)" over 60s is rejected with
# "A rate is never random" and told to set request_interval instead.
def test_random_rate_error_points_at_request_interval() -> None:
    with pytest.raises(ValidationError, match="A rate is never random. For random arrivals set request_interval instead"):
        StandardLoadStage(rate="Exponential(10)", duration=60)


# load.type poisson next to the rate expression "t/10" is rejected, told to remove load.type and
# that Poisson arrivals are written as request_interval: "Exponential(<rate>)".
def test_load_type_with_rate_expression_points_at_request_interval() -> None:
    message = 'Remove load.type. Poisson arrivals are written as request_interval: "Exponential\\(<rate>\\)".'
    with pytest.raises(ValidationError, match=message):
        LoadConfig(type=LoadType.POISSON, stages=[StandardLoadStage(rate="t/10", duration=60)])


# A zero-rate stage and a request_interval stage load side by side with no load.type: rate 0
# for 60s sends nothing, request_interval 0.1 for 60s sends 600, and only the rate stage is a zero rate.
def test_zero_rate_stage_next_to_a_request_interval_stage() -> None:
    idle = StandardLoadStage(rate=0, duration=60)
    paced = StandardLoadStage(request_interval=0.1, duration=60)
    LoadConfig(stages=[idle, paced])
    assert (idle.expected_requests, paced.expected_requests) == (0, 600)
    assert (idle.rate_is_zero, paced.rate_is_zero) == (True, False)


# A request_interval may vary with t. "1/t" over 10s is the rate t and sends 50 requests, bounded by a
# duration or by stop_condition "t >= 10", and the random "Exponential(5 + t/10)" over 60s loads.
def test_request_interval_over_time_loads() -> None:
    for window in ({"duration": 10}, {"stop_condition": "t >= 10"}):
        stage = StandardLoadStage(request_interval="1/t", **window)
        assert stage.expected_requests == 50
        assert stage.request_schedule.offsets[0] == pytest.approx(1.4142, abs=1e-3)
    assert StandardLoadStage(request_interval="Exponential(5 + t/10)", duration=60).expected_requests > 0


# A random request_interval over t that cannot be drawn from at every t is rejected at load: a rate of 0 at
# t=0, and a Pareto whose mean is infinite. The error names the condition and the t, and says a load that
# pauses or starts from zero is written with rate.
@pytest.mark.parametrize(
    ("request_interval", "message"),
    [
        (
            "Exponential(t)",
            r"requires 'rate > 0', which fails at t = 0 over \[0, 60\]. A load that pauses or starts from zero",
        ),
        (
            "Pareto(1 + t, 1)",
            r"requires 'alpha > 1 for a finite mean', which fails at t = 0 over \[0, 60\]. A load that pauses or starts",
        ),
    ],
)
def test_random_request_interval_over_time_rejected(request_interval: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message) as raised:
        StandardLoadStage(request_interval=request_interval, duration=60)
    assert "is written with rate" in str(raised.value)


# A request_interval stage that sends nothing loads with a warning that says why: stage 0 has an
# infinite gap, stage 1 a 100s gap in a 60s stage. Stage 2, at 0.1s, gets none.
def test_request_interval_that_sends_nothing_warns_at_config_load(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="inference_perf.config.loadgen.config"):
        LoadConfig(
            stages=[
                StandardLoadStage(request_interval="oo", duration=60),
                StandardLoadStage(request_interval=100, duration=60),
                StandardLoadStage(request_interval=0.1, duration=60),
            ]
        )
    assert [record.getMessage() for record in caplog.records] == [
        "Stage 0 will send zero requests: its request_interval 'oo' is infinite for the whole stage.",
        "Stage 1 will send zero requests: no gap of its request_interval 100.0 fits in the stage.",
    ]


# In-process run of a stage whose gap is infinite for its whole 1s: it completes, sends nothing,
# lasts its 1s instead of ending at once, reports its request_interval, and logs why it sent nothing.
@pytest.mark.timeout(60)
async def test_in_process_infinite_request_interval_stage_lasts_its_duration(caplog: pytest.LogCaptureFixture) -> None:
    api_config = APIConfig(type=APIType.Chat)
    datagen = MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)
    stage = StandardLoadStage(request_interval="oo", duration=1)
    loadgen = LoadGenerator(datagen, LoadConfig(stages=[stage], num_workers=0, worker_max_concurrency=4))
    collector = LocalRequestMetricCollector()
    with caplog.at_level(logging.WARNING, logger="inference_perf.loadgen.load_generator"):
        await loadgen.run(MockModelServerClient(collector, api_config, mock_latency=0.01))
    info = loadgen.stage_runtime_info[0]
    assert info.status.name == "COMPLETED"
    assert (info.rate, info.request_interval) == (None, "oo")
    assert collector.get_metrics() == []
    assert 1.0 <= info.end_time - info.start_time < 5.0
    assert "Stage 0 sends zero requests: its request_interval 'oo' is infinite for the whole stage." in caplog.text


# Worker-process run of rate 4 for 1s, then an infinite gap for 1s, then rate 4 for 1s: all three
# complete with nothing dropped, and the middle stage holds its window (1s lead-in plus 1s).
@pytest.mark.timeout(120)
async def test_worker_run_infinite_request_interval_stage_between_two_normal_stages() -> None:
    api_config = APIConfig(type=APIType.Chat)
    datagen = MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)
    stages = [
        StandardLoadStage(rate=4, duration=1),
        StandardLoadStage(request_interval="oo", duration=1),
        StandardLoadStage(rate=4, duration=1),
    ]
    load_config = LoadConfig(
        interval=0, stages=stages, num_workers=1, worker_max_concurrency=4, stage_teardown_grace_seconds=10.0
    )
    loadgen = LoadGenerator(datagen, load_config)
    try:
        await loadgen.run(MockModelServerClient(LocalRequestMetricCollector(), api_config, mock_latency=0.01))
    finally:
        await loadgen.stop()
    infos = [loadgen.stage_runtime_info[stage_id] for stage_id in range(3)]
    assert [info.status.name for info in infos] == ["COMPLETED"] * 3
    assert [info.dropped_requests for info in infos] == [0, 0, 0]
    assert 2.0 <= infos[1].end_time - infos[1].start_time < 6.0
