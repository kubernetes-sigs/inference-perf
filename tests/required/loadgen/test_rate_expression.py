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

import os
import pathlib
import tempfile
import unittest.mock

import numpy as np
import pytest
import yaml
from pydantic import ValidationError

from inference_perf.client.modelserver import MockModelServerClient
from inference_perf.config import (
    APIConfig,
    APIType,
    Config,
    DataConfig,
    DataGenType,
    LoadConfig,
    LoadType,
    StandardLoadStage,
    read_config,
)
from inference_perf.datagen import DataGenerator
from inference_perf.datagen.synthetic.mock_datagen import MockDataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.loadgen.load_timer import ConstantLoadTimer, LoadTimer, PoissonLoadTimer
from inference_perf.metrics.request_collector.local import LocalRequestMetricCollector
from inference_perf.reportgen.base import ReportGenerator


# Builds a LoadGenerator with no load.type around a mocked datagen, with the circuit-breaker lookup patched out.
def _loadgen(*stages: StandardLoadStage) -> LoadGenerator:
    with unittest.mock.patch("inference_perf.loadgen.load_generator.get_circuit_breaker"):
        return LoadGenerator(unittest.mock.MagicMock(spec=DataGenerator), LoadConfig(stages=list(stages)))


# Collects a timer's schedule as offsets from 0, with its rng seeded to `seed`.
def _schedule(timer: LoadTimer, seed: int, limit: int) -> list[float]:
    assert isinstance(timer, (ConstantLoadTimer, PoissonLoadTimer))
    timer._rand = np.random.default_rng(seed)
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
# ("10 - t" over 60s), a random rate, a symbol other than t, and a negative number.
@pytest.mark.parametrize(
    ("rate", "message"),
    [
        ("10 - t", "outside the permitted range"),
        ("Uniform(1, 5)", "never random"),
        ("requests/10", "disallowed symbol"),
        (-1, "greater than or equal to 0"),
    ],
)
def test_bad_rates_rejected(rate: object, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        StandardLoadStage(rate=rate, duration=60)


# A rate need not mention t: "2+3" over 60s is a constant 5 req/s and sends 300 requests.
def test_rate_without_t_is_valid() -> None:
    stage = StandardLoadStage(rate="2+3", duration=60)
    assert stage.rate_schedule.is_constant
    assert (stage.expected_requests, stage.mean_rate) == (300, 5.0)


# A random rate is rejected with the rate's own message even when it could also go negative:
# "Normal(10, 1)" says "never random", and the text no longer says "function of t".
def test_random_rate_error_does_not_mention_t() -> None:
    with pytest.raises(ValidationError, match="Rate 'Normal\\(10, 1\\)' is random. A rate is never random") as raised:
        StandardLoadStage(rate="Normal(10, 1)", duration=60)
    assert "function of t" not in str(raised.value)


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
# per second), so for it this checks the layer below: the schedule unwraps to the float 4.0.
def test_numeric_rate_schedule_is_unchanged() -> None:
    stage = StandardLoadStage(rate=4, duration=60)
    as_float = _schedule(ConstantLoadTimer(4.0, 60.0), seed=7, limit=240)
    as_schedule = _schedule(ConstantLoadTimer(stage.rate_schedule, 60.0), seed=7, limit=240)
    assert as_float == as_schedule
    poisson = PoissonLoadTimer(stage.rate_schedule, 60.0)
    assert poisson._rate == 4.0


# With no load.type, the ramp rate = t over 10s gets the CONSTANT timer and is sent with no jitter:
# 50 requests, request k where the area t^2/2 reaches k, so at sqrt(2k): 1.414s, 2.0s, 2.449s and
# on to exactly 10s. The timer stops after the 50th.
def test_rate_expression_is_sent_at_exact_times() -> None:
    stage = StandardLoadStage(rate="t", duration=10)
    loadgen = _loadgen(stage)
    timer = loadgen.get_timer(stage.rate_schedule, 10.0)
    assert isinstance(timer, ConstantLoadTimer)
    times = list(timer.start_timer(initial=0.0))
    assert len(times) == stage.expected_requests == 50
    assert times == pytest.approx([np.sqrt(2 * k) for k in range(1, 51)], abs=1e-3)
    assert times[:3] == pytest.approx([1.414, 2.0, 2.449], abs=1e-3)
    assert times[-1] == 10.0


# A rate expression gives the same send times on every run: timers seeded 3 and 4 produce identical
# schedules for the ramp "t" over 10s and for the constant expression "10" over 60s.
@pytest.mark.parametrize(("rate", "duration"), [("t", 10), ("10", 60)])
def test_rate_expression_send_times_do_not_depend_on_the_seed(rate: str, duration: int) -> None:
    schedule = StandardLoadStage(rate=rate, duration=duration).rate_schedule
    count = schedule.expected_requests
    first = _schedule(ConstantLoadTimer(schedule, float(duration)), seed=3, limit=count)
    assert first == _schedule(ConstantLoadTimer(schedule, float(duration)), seed=4, limit=count)


# A constant expression is exact too, and a number is not. rate "10" over 60s sends 600 requests at
# 0.1s, 0.2s, ... 60.0s, every gap 0.1s, the last one exactly at the end of the window. rate 10
# sends the same 600 with gaps that vary (spread over mean near 1) and that differ between seeds.
def test_constant_expression_is_evenly_spaced_and_a_number_is_not() -> None:
    written = StandardLoadStage(rate="10", duration=60)
    times = np.array(list(ConstantLoadTimer(written.rate_schedule, 60.0).start_timer(initial=0.0)))
    assert len(times) == written.expected_requests == 600
    assert times == pytest.approx(0.1 * np.arange(1, 601))
    assert times[-1] == 60.0

    numeric = StandardLoadStage(rate=10, duration=60)
    jittered = np.array(_schedule(ConstantLoadTimer(numeric.rate_schedule, 60.0), seed=3, limit=600))
    gaps = np.diff(jittered)
    assert float(gaps.std() / gaps.mean()) > 0.5
    assert list(jittered) != _schedule(ConstantLoadTimer(numeric.rate_schedule, 60.0), seed=4, limit=600)


# The request due at the end of the window is sent, and the count is what expected_requests says.
# rate "0.7" over 30s is 21 requests although 21 / 0.7 is a hair over 30 in floating point: the last
# goes out at exactly 30s, never past the window. "2 + 3" over 2s is 10 requests, the last at 2s.
def test_request_at_the_end_of_the_window_is_sent() -> None:
    for rate, duration, count in (("0.7", 30, 21), ("2 + 3", 2, 10), ("t", 10, 50), ("Min(5 + t/2, 40)", 120, 3575)):
        schedule = StandardLoadStage(rate=rate, duration=duration).rate_schedule
        times = list(ConstantLoadTimer(schedule, float(duration)).start_timer(initial=0.0))
        assert len(times) == schedule.expected_requests == count
        assert max(times) <= duration
    slow = StandardLoadStage(rate="0.7", duration=30).rate_schedule
    assert list(ConstantLoadTimer(slow, 30.0).start_timer(initial=0.0))[-1] == 30.0


# load.type cannot be set next to a rate expression. Stages of rate 4 and "t/10" under an explicit
# constant, and under an explicit poisson, are rejected naming the stage and saying to remove load.type.
@pytest.mark.parametrize("load_type", [LoadType.CONSTANT, LoadType.POISSON])
def test_load_type_rejected_next_to_a_rate_expression(load_type: LoadType) -> None:
    stages = [StandardLoadStage(rate=4, duration=60), StandardLoadStage(rate="t/10", duration=60)]
    message = f"Stage 1: load.type \\({load_type.value}\\) cannot be set alongside a rate expression. Remove load.type."
    with pytest.raises(ValidationError, match=message):
        LoadConfig(type=load_type, stages=stages)


# The rule reads what the config wrote, not the value: the same two stages with no load.type load
# as constant, the default, and all-numeric stages still load under an explicit constant or poisson.
def test_load_type_allowed_without_a_rate_expression() -> None:
    stages = [StandardLoadStage(rate=4, duration=60), StandardLoadStage(rate="t/10", duration=60)]
    assert LoadConfig(stages=stages).type == LoadType.CONSTANT
    for load_type in (LoadType.CONSTANT, LoadType.POISSON):
        assert LoadConfig(type=load_type, stages=[StandardLoadStage(rate=4, duration=60)]).type == load_type


# Writes a load block to a temporary YAML file with mock data and loads it through read_config,
# the path the CLI takes, with any CLI overrides on top.
def _read_load_config(load: dict[str, object], cli_overrides: dict[str, object] | None = None) -> LoadConfig:
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as tmp:
        yaml.dump({"data": {"type": "mock"}, "load": load}, tmp)
        tmp_path = tmp.name
    try:
        return read_config(tmp_path, cli_overrides).load
    finally:
        os.remove(tmp_path)


# A YAML file with rate "t" for 10s and no load.type loads through read_config: the defaults
# read_config merges in do not count as the user setting load.type. The stage sends 50 requests
# and the load type reads as constant.
def test_read_config_rate_expression_without_load_type() -> None:
    load = _read_load_config({"stages": [{"rate": "t", "duration": 10}]})
    stage = load.stages[0]
    assert isinstance(stage, StandardLoadStage)
    assert stage.expected_requests == 50
    assert load.type == LoadType.CONSTANT


# The same file with load.type: constant written out is rejected through read_config, and so is
# the file without it when --load.type poisson comes from the CLI. Both name load.type.
def test_read_config_rate_expression_rejects_a_written_load_type() -> None:
    stages = [{"rate": "t", "duration": 10}]
    with pytest.raises(ValueError, match="Remove load.type"):
        _read_load_config({"type": "constant", "stages": stages})
    with pytest.raises(ValueError, match="Remove load.type"):
        _read_load_config({"stages": stages}, {"load": {"type": "poisson"}})


# A file with a numeric rate keeps the merged-in default and any load.type it writes: rate 4
# with no load.type loads as constant with the default still merged in, and with
# load.type: poisson as poisson.
def test_read_config_numeric_rate_keeps_load_type() -> None:
    stages = [{"rate": 4, "duration": 10}]
    default = _read_load_config({"stages": stages})
    assert default.type == LoadType.CONSTANT
    assert "type" in default.model_fields_set
    assert _read_load_config({"type": "poisson", "stages": stages}).type == LoadType.POISSON


# The POISSON timer takes a constant rate only: handed the ramp rate = t over 20s it raises,
# since the config never pairs load.type with a rate expression.
def test_poisson_timer_refuses_a_varying_rate() -> None:
    with pytest.raises(ValueError, match="varies with t"):
        PoissonLoadTimer(StandardLoadStage(rate="t", duration=20).rate_schedule, 20.0)


# In-process run of a ramp rate = 4*t over 1s (2 expected requests) against the mock server:
# the stage completes, records 1 or 2 requests (the in-process loop can drop the last one at
# the window edge, a known pre-existing issue), and reports the mean rate 2.
async def test_in_process_run_reports_mean_rate() -> None:
    api_config = APIConfig(type=APIType.Chat)
    datagen = MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)
    stage = StandardLoadStage(rate="4*t", duration=1)
    loadgen = LoadGenerator(datagen, LoadConfig(stages=[stage], num_workers=0, worker_max_concurrency=4))
    collector = LocalRequestMetricCollector()
    await loadgen.run(MockModelServerClient(collector, api_config, mock_latency=0.01))
    info = loadgen.stage_runtime_info[0]
    assert info.status.name == "COMPLETED"
    assert info.rate == pytest.approx(2.0)
    assert len(collector.get_metrics()) in (1, 2)


# The config a report saves loads again. A file with rate "t" for 10s and rate 4 for 10s and no
# load.type is loaded, its report config is written out the way a run writes it, and that file is
# loaded: load.type is not in it, and the stages come back the same.
def test_report_config_with_a_rate_expression_loads_again(tmp_path: pathlib.Path) -> None:
    source = tmp_path / "config.yaml"
    source.write_text(
        yaml.dump({"data": {"type": "mock"}, "load": {"stages": [{"rate": "t", "duration": 10}, {"rate": 4, "duration": 10}]}})
    )
    config = read_config(str(source))
    saved = ReportGenerator(None, unittest.mock.Mock(), config).generate_config_report()
    assert "type" not in saved.get_contents()["load"]

    written = tmp_path / saved.get_filename()
    with open(written, "w") as f:
        yaml.dump(saved.get_contents(), f, sort_keys=False, default_flow_style=False)
    again = read_config(str(written))
    assert [stage.model_dump() for stage in again.load.stages] == [stage.model_dump() for stage in config.load.stages]
    assert again.load.type == LoadType.CONSTANT


# load.type stays in the saved config when it is legal there: a file with numeric rates keeps
# the merged-in default, constant, and one that writes load.type: poisson keeps poisson.
# So does a config built in code with a numeric rate and no load.type.
def test_report_config_keeps_a_legal_load_type(tmp_path: pathlib.Path) -> None:
    built = Config(load=LoadConfig(stages=[StandardLoadStage(rate=4, duration=10)]))
    assert (
        ReportGenerator(None, unittest.mock.Mock(), built).generate_config_report().get_contents()["load"]["type"]
        == "constant"
    )
    for load_type, saved_type in ((None, "constant"), ("poisson", "poisson")):
        load: dict[str, object] = {"stages": [{"rate": 4, "duration": 10}]}
        if load_type is not None:
            load["type"] = load_type
        source = tmp_path / "config.yaml"
        source.write_text(yaml.dump({"data": {"type": "mock"}, "load": load}))
        saved = ReportGenerator(None, unittest.mock.Mock(), read_config(str(source))).generate_config_report()
        assert saved.get_contents()["load"]["type"] == saved_type


# In-process run of the constant expression "4" for 1s against the mock server: requests are due at
# 0.25s, 0.5s, 0.75s and 1.0s, and all 4 are sent, the one at the end of the window included.
async def test_in_process_run_sends_the_request_at_the_end_of_the_window() -> None:
    api_config = APIConfig(type=APIType.Chat)
    datagen = MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)
    stage = StandardLoadStage(rate="4", duration=1)
    loadgen = LoadGenerator(datagen, LoadConfig(stages=[stage], num_workers=0, worker_max_concurrency=4))
    collector = LocalRequestMetricCollector()
    await loadgen.run(MockModelServerClient(collector, api_config, mock_latency=0.01))
    assert loadgen.stage_runtime_info[0].status.name == "COMPLETED"
    assert len(collector.get_metrics()) == 4
