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
"""A stage whose rate is zero for the whole stage is legal and sends nothing.

The config loads it with a warning, the timers yield no send time, and the
stage still lasts its duration or stop condition, in process and across worker
processes. A rate that is negative anywhere is still rejected.
"""

import logging
import time

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
from inference_perf.datagen.synthetic.mock_datagen import MockDataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.loadgen.load_timer import ConstantLoadTimer, PoissonLoadTimer
from inference_perf.metrics.request_collector.local import LocalRequestMetricCollector

ZERO_RATES = [0, "0", "0*t", "Max(0, -t)"]


# Runs the given load config against the mock server (10ms latency) and returns the
# load generator and the collector that recorded the requests it sent.
async def _run(load_config: LoadConfig) -> tuple[LoadGenerator, LocalRequestMetricCollector]:
    api_config = APIConfig(type=APIType.Chat)
    datagen = MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)
    loadgen = LoadGenerator(datagen, load_config)
    collector = LocalRequestMetricCollector()
    try:
        await loadgen.run(MockModelServerClient(collector, api_config, mock_latency=0.01))
    finally:
        await loadgen.stop()
    return loadgen, collector


# The number 0 and the expressions "0", "0*t" and "Max(0, -t)" over 60s each load,
# expect 0 requests at a mean rate of 0, and say rate_is_zero.
@pytest.mark.parametrize("rate", ZERO_RATES)
def test_zero_rate_loads_and_sends_nothing(rate: float | str) -> None:
    stage = StandardLoadStage(rate=rate, duration=60)
    assert (stage.expected_requests, stage.mean_rate) == (0, 0.0)
    assert stage.rate_is_zero


# A zero rate also loads with stop_condition "t >= 60" in place of duration: 0 requests over a 60s window.
def test_zero_rate_loads_with_a_stop_condition() -> None:
    stage = StandardLoadStage(rate=0, stop_condition="t >= 60")
    assert (stage.expected_requests, stage.effective_duration) == (0, 60.0)


# A rate that is zero for only part of the stage is not a zero rate: "Max(0, 10 - t)" over 60s
# is a ramp from 10 down to 0 in the first 10s, so it sends 50 requests and rate_is_zero is False.
def test_partly_zero_rate_still_sends() -> None:
    stage = StandardLoadStage(rate="Max(0, 10 - t)", duration=60)
    assert stage.expected_requests == 50
    assert not stage.rate_is_zero


# A rate above zero whose area is under one request also sends nothing, and says why: 0.1 req/s for
# 5s is 0.5 requests. It is not a zero rate. A stage that sends has no reason to give.
def test_rate_that_rounds_down_to_zero_requests() -> None:
    stage = StandardLoadStage(rate=0.1, duration=5)
    assert stage.expected_requests == 0
    assert not stage.rate_is_zero
    assert stage.zero_requests_reason == "its rate 0.1 adds up to 0.5 requests over 5s, which is less than one request"
    assert StandardLoadStage(rate=0, duration=5).zero_requests_reason == "its rate 0.0 is zero for the whole stage"
    assert StandardLoadStage(rate="t/100", duration=5).zero_requests_reason == (
        "its rate 't/100' adds up to 0.125 requests over 5s, which is less than one request"
    )
    assert StandardLoadStage(rate=0.2, duration=5).zero_requests_reason is None


# Negative is still rejected at load: the number -1 and -0.5, the expression "-1", and "5 - t"
# over 60s, which is negative after t=5.
@pytest.mark.parametrize(
    ("rate", "message"),
    [
        (-1, "greater than or equal to 0"),
        (-0.5, "greater than or equal to 0"),
        ("-1", "outside the permitted range"),
        ("5 - t", "outside the permitted range"),
    ],
)
def test_negative_rate_rejected(rate: float | str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        StandardLoadStage(rate=rate, duration=60)


# Stages of rate 5, 0, "Max(0, -t)" and 0.001 over 60s log one warning each for stages 1, 2 and 3
# at config load, saying the stage will send zero requests and why: a rate that is zero for the
# whole stage, or one that adds up to less than one request. Stage 0 gets no warning.
def test_zero_rate_warns_at_config_load(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="inference_perf.config.loadgen.config"):
        LoadConfig(
            stages=[
                StandardLoadStage(rate=5, duration=60),
                StandardLoadStage(rate=0, duration=60),
                StandardLoadStage(rate="Max(0, -t)", duration=60),
                StandardLoadStage(rate=0.001, duration=60),
            ]
        )
    messages = [record.getMessage() for record in caplog.records]
    assert messages == [
        "Stage 1 will send zero requests: its rate 0.0 is zero for the whole stage.",
        "Stage 2 will send zero requests: its rate 'Max(0, -t)' is zero for the whole stage.",
        "Stage 3 will send zero requests: its rate 0.001 adds up to 0.06 requests over 60s, which is less than one request.",
    ]


# A config with only nonzero rates, 5 and "Max(0, 10 - t)", logs no zero-rate warning.
def test_nonzero_rate_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="inference_perf.config.loadgen.config"):
        LoadConfig(stages=[StandardLoadStage(rate=5, duration=60), StandardLoadStage(rate="Max(0, 10 - t)", duration=60)])
    assert caplog.records == []


# Both timers on a zero rate over 60s yield no send time and stop, for the number 0 and for
# "Max(0, -t)". The Poisson timer used to loop forever here, hence the timeout.
@pytest.mark.timeout(30)
@pytest.mark.parametrize("rate", [0, "Max(0, -t)"])
def test_zero_rate_timers_yield_nothing(rate: float | str) -> None:
    schedule = StandardLoadStage(rate=rate, duration=60).rate_schedule
    assert list(ConstantLoadTimer(schedule, 60.0).start_timer(initial=0.0)) == []
    if schedule.is_constant:
        assert list(PoissonLoadTimer(schedule, 60.0).start_timer(initial=0.0)) == []
        assert list(PoissonLoadTimer(0.0, 60.0).start_timer(initial=0.0)) == []


# In-process run of one zero-rate stage for 1s, under the default load type and under poisson:
# the stage completes, sends 0 requests, reports a rate of 0, lasts its 1s duration instead of
# ending at once, and logs why it sent nothing.
@pytest.mark.timeout(60)
@pytest.mark.parametrize("load_type", [None, LoadType.POISSON])
async def test_in_process_zero_rate_stage_lasts_its_duration(
    load_type: LoadType | None, caplog: pytest.LogCaptureFixture
) -> None:
    stage = StandardLoadStage(rate=0, duration=1)
    load_config = LoadConfig(stages=[stage], num_workers=0, worker_max_concurrency=4)
    if load_type is not None:
        load_config = LoadConfig(type=load_type, stages=[stage], num_workers=0, worker_max_concurrency=4)
    with caplog.at_level(logging.WARNING, logger="inference_perf.loadgen.load_generator"):
        loadgen, collector = await _run(load_config)
    info = loadgen.stage_runtime_info[0]
    assert info.status.name == "COMPLETED"
    assert info.rate == 0
    assert collector.get_metrics() == []
    assert 1.0 <= info.end_time - info.start_time < 5.0
    assert "Stage 0 sends zero requests: its rate 0.0 is zero for the whole stage." in caplog.text


# In-process run of a zero-rate expression bounded by stop_condition "t >= 1" instead of a
# duration: it completes after its 1s window with 0 requests.
@pytest.mark.timeout(60)
async def test_in_process_zero_rate_stage_ends_on_its_stop_condition() -> None:
    stage = StandardLoadStage(rate="Max(0, -t)", stop_condition="t >= 1")
    loadgen, collector = await _run(LoadConfig(stages=[stage], num_workers=0, worker_max_concurrency=4))
    info = loadgen.stage_runtime_info[0]
    assert info.status.name == "COMPLETED"
    assert collector.get_metrics() == []
    assert 1.0 <= info.end_time - info.start_time < 5.0


# In-process run of three stages, rate 4 for 1s, rate 0 for 1s, rate 4 for 1s: all three complete,
# the middle one sends nothing, and the stages on either side still send (3 or 4 each, since the
# in-process loop can drop the last request at the window edge).
@pytest.mark.timeout(60)
async def test_in_process_zero_rate_stage_between_two_normal_stages() -> None:
    stages = [
        StandardLoadStage(rate=4, duration=1),
        StandardLoadStage(rate=0, duration=1),
        StandardLoadStage(rate=4, duration=1),
    ]
    loadgen, collector = await _run(LoadConfig(interval=0, stages=stages, num_workers=0, worker_max_concurrency=4))
    assert [loadgen.stage_runtime_info[stage].status.name for stage in range(3)] == ["COMPLETED"] * 3
    per_stage = [sum(1 for metric in collector.get_metrics() if metric.stage_id == stage) for stage in range(3)]
    assert per_stage[1] == 0
    assert per_stage[0] in (3, 4) and per_stage[2] in (3, 4)


# Worker-process run of the same three stages on one worker: all three complete with nothing
# dropped, the zero-rate stage holds its window (1s lead-in plus its 1s duration) instead of
# tearing down at once, and the run takes well under the teardown grace.
@pytest.mark.timeout(120)
async def test_worker_run_zero_rate_stage_between_two_normal_stages(caplog: pytest.LogCaptureFixture) -> None:
    stages = [
        StandardLoadStage(rate=4, duration=1),
        StandardLoadStage(rate=0, duration=1),
        StandardLoadStage(rate=4, duration=1),
    ]
    load_config = LoadConfig(
        interval=0, stages=stages, num_workers=1, worker_max_concurrency=4, stage_teardown_grace_seconds=10.0
    )
    start = time.perf_counter()
    with caplog.at_level(logging.WARNING, logger="inference_perf.loadgen.load_generator"):
        loadgen, _ = await _run(load_config)
    elapsed = time.perf_counter() - start
    infos = [loadgen.stage_runtime_info[stage] for stage in range(3)]
    assert [info.status.name for info in infos] == ["COMPLETED"] * 3
    assert [info.dropped_requests for info in infos] == [0, 0, 0]
    assert [info.rate for info in infos] == [4, 0, 4]
    assert 2.0 <= infos[1].end_time - infos[1].start_time < 6.0
    assert elapsed < 60
    assert "Stage 1 sends zero requests: its rate 0.0 is zero for the whole stage." in caplog.text


# In-process run of one stage at 0.1 req/s for 1s, which is a tenth of a request: it completes, sends
# nothing, reports its rate of 0.1, lasts its 1s instead of ending at once, and logs why.
@pytest.mark.timeout(60)
async def test_in_process_stage_that_rounds_down_to_zero_lasts_its_duration(caplog: pytest.LogCaptureFixture) -> None:
    stage = StandardLoadStage(rate=0.1, duration=1)
    with caplog.at_level(logging.WARNING, logger="inference_perf.loadgen.load_generator"):
        loadgen, collector = await _run(LoadConfig(stages=[stage], num_workers=0, worker_max_concurrency=4))
    info = loadgen.stage_runtime_info[0]
    assert info.status.name == "COMPLETED"
    assert info.rate == 0.1
    assert collector.get_metrics() == []
    assert 1.0 <= info.end_time - info.start_time < 5.0
    assert (
        "Stage 0 sends zero requests: its rate 0.1 adds up to 0.1 requests over 1s, which is less than one request."
        in caplog.text
    )


# Worker-process run of rate 4 for 1s, rate 0.1 for 1s, rate 4 for 1s on one worker: all three
# complete with nothing dropped, and the middle stage holds its window (1s lead-in plus its 1s)
# and logs why it sent nothing.
@pytest.mark.timeout(120)
async def test_worker_run_stage_that_rounds_down_to_zero_between_two_normal_stages(
    caplog: pytest.LogCaptureFixture,
) -> None:
    stages = [
        StandardLoadStage(rate=4, duration=1),
        StandardLoadStage(rate=0.1, duration=1),
        StandardLoadStage(rate=4, duration=1),
    ]
    load_config = LoadConfig(
        interval=0, stages=stages, num_workers=1, worker_max_concurrency=4, stage_teardown_grace_seconds=10.0
    )
    with caplog.at_level(logging.WARNING, logger="inference_perf.loadgen.load_generator"):
        loadgen, _ = await _run(load_config)
    infos = [loadgen.stage_runtime_info[stage] for stage in range(3)]
    assert [info.status.name for info in infos] == ["COMPLETED"] * 3
    assert [info.dropped_requests for info in infos] == [0, 0, 0]
    assert 2.0 <= infos[1].end_time - infos[1].start_time < 6.0
    assert "Stage 1 sends zero requests: its rate 0.1 adds up to 0.1 requests over 1s" in caplog.text
