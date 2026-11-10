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
"""The runtime metrics endpoint stays up for linger_seconds after the run, unless interrupted."""

import time
from argparse import ArgumentParser
from typing import List
from unittest.mock import MagicMock, patch

import pytest
from pydantic import ValidationError

from inference_perf.config import Config, RuntimeMetricsConfig, read_config
from inference_perf.main import stop_runtime_metrics
from inference_perf.observability.metrics import PrometheusMetricsServer
from inference_perf.observability.metrics.prometheus import LINGER_POLL_SECONDS, linger_until
from inference_perf.utils.cli_parser import add_global_args, add_pydantic_args, unflatten_dict


class _FakeClock:
    # A clock that only moves when slept on, so the wait runs instantly and exactly.
    # Records every sleep so a test can read how long the wait was and in what steps.
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: List[float] = []

    def __call__(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.sleeps.append(seconds)
        self.now += seconds


# Parses argv the way main_cli does, global args first and then every Config field
# as a flag, and returns the Config the run would use.
def _config_from_argv(argv: List[str]) -> Config:
    parser = ArgumentParser()
    base_args = add_global_args(parser)
    add_pydantic_args(parser, Config)
    args = parser.parse_args(argv)
    overrides = unflatten_dict({k: v for k, v in vars(args).items() if k not in base_args})
    return read_config(args.config_file, overrides)


# --- linger_until -----------------------------------------------------------


# A deadline of 5.0 when the clock already reads 5.0, and then 9.0.
# Expects a return with no sleep at all: a linger that is already over never waits.
def test_deadline_already_reached_does_not_wait() -> None:
    for now in (5.0, 9.0):
        clock = _FakeClock()
        clock.now = now
        linger_until(5.0, lambda: False, clock=clock, sleep=clock.sleep)
        assert clock.sleeps == []


# A deadline 1.0s away with the clock at 0.0.
# Expects the sleeps to add up to 1.0s, each no longer than the poll interval, and the clock to stop at the deadline.
def test_waits_out_the_remaining_time_in_poll_sized_steps() -> None:
    clock = _FakeClock()
    linger_until(1.0, lambda: False, clock=clock, sleep=clock.sleep)
    assert sum(clock.sleeps) == pytest.approx(1.0)
    assert max(clock.sleeps) <= LINGER_POLL_SECONDS
    assert clock.now == pytest.approx(1.0)


# A deadline 0.03s away, shorter than one poll interval.
# Expects a single sleep of 0.03s, not a full poll interval, so the wait never overshoots the deadline.
def test_last_sleep_is_clipped_to_the_deadline() -> None:
    clock = _FakeClock()
    linger_until(0.03, lambda: False, clock=clock, sleep=clock.sleep)
    assert clock.sleeps == [pytest.approx(0.03)]


# interrupted() already true on entry, with a deadline 100s away.
# Expects a return with no sleep: a signal that arrived before the wait skips it.
def test_interrupted_before_the_wait_skips_it() -> None:
    clock = _FakeClock()
    linger_until(100.0, lambda: True, clock=clock, sleep=clock.sleep)
    assert clock.sleeps == []


# interrupted() turns true after three sleeps, with a deadline 100s away.
# Expects the wait to stop after those three sleeps, far short of the deadline.
def test_interrupt_during_the_wait_ends_it_at_the_next_poll() -> None:
    clock = _FakeClock()
    linger_until(100.0, lambda: len(clock.sleeps) >= 3, clock=clock, sleep=clock.sleep)
    assert len(clock.sleeps) == 3
    assert clock.now < 1.0


# --- stop_runtime_metrics ---------------------------------------------------


# No metrics server (the block is unset, or the port would not bind), interrupted() false.
# Expects no linger and no error: a run without the endpoint never waits.
def test_no_server_means_no_wait() -> None:
    with patch("inference_perf.main.linger_until") as linger:
        stop_runtime_metrics(None, 30.0, time.monotonic(), lambda: False)
    linger.assert_not_called()


# A running server, linger_seconds 30, run_end 100.0, interrupted() false.
# Expects the wait to be called once with deadline 130.0, and only then the server to be stopped.
def test_server_is_stopped_after_the_linger_not_before() -> None:
    calls: List[str] = []
    server = MagicMock(spec=PrometheusMetricsServer)
    server.stop.side_effect = lambda: calls.append("stop")
    with patch("inference_perf.main.linger_until", side_effect=lambda *a, **k: calls.append("linger")) as linger:
        stop_runtime_metrics(server, 30.0, 100.0, lambda: False)
    assert calls == ["linger", "stop"]
    assert linger.call_args.args[0] == 130.0


# run_end 40s in the past with linger_seconds 30, as when reports took 40s.
# Expects the deadline to be run_end + 30 and so already over: reporting time counts toward the linger, never adds to it.
def test_time_spent_on_reports_counts_toward_the_linger() -> None:
    run_end = time.monotonic() - 40.0
    server = MagicMock(spec=PrometheusMetricsServer)
    start = time.monotonic()
    stop_runtime_metrics(server, 30.0, run_end, lambda: False)
    assert time.monotonic() - start < 1.0
    server.stop.assert_called_once()


# interrupted() true when the run ends, as after SIGINT or SIGTERM during the load.
# Expects no linger, and the server still stopped.
def test_interrupted_run_does_not_linger() -> None:
    server = MagicMock(spec=PrometheusMetricsServer)
    with patch("inference_perf.main.linger_until") as linger:
        stop_runtime_metrics(server, 30.0, time.monotonic(), lambda: True)
    linger.assert_not_called()
    server.stop.assert_called_once()


# linger_seconds 0 on a real server, interrupted() false.
# Expects the server stopped at once, with nothing left listening on its port.
def test_zero_linger_stops_the_server_immediately() -> None:
    from prometheus_client import CollectorRegistry

    server = PrometheusMetricsServer(CollectorRegistry(), port=0)
    server.start()
    stop_runtime_metrics(server, 0.0, time.monotonic(), lambda: False)
    assert server.bound_port is None


# --- config -----------------------------------------------------------------


class TestLingerConfig:
    # observability.metrics: {} in YAML, the shortest way to turn the endpoint on.
    # Expects linger_seconds to default to 30.
    def test_default_is_thirty_seconds(self) -> None:
        metrics = Config.model_validate({"observability": {"metrics": {}}}).observability.metrics
        assert metrics is not None
        assert metrics.linger_seconds == 30.0

    # linger_seconds 0, and then 2.5.
    # Expects both accepted as given: 0 turns the linger off, a fraction of a second is allowed.
    def test_zero_and_fractions_are_accepted(self) -> None:
        assert RuntimeMetricsConfig(linger_seconds=0).linger_seconds == 0.0
        assert RuntimeMetricsConfig(linger_seconds=2.5).linger_seconds == 2.5

    # linger_seconds -1, then inf, then nan.
    # Expects each rejected: a negative or unbounded wait is a typo, not a setting.
    def test_negative_and_non_finite_values_are_rejected(self) -> None:
        for value in (-1, float("inf"), float("nan")):
            with pytest.raises(ValidationError):
                RuntimeMetricsConfig(linger_seconds=value)

    # --observability.metrics.linger_seconds 5 with no config file.
    # Expects the flag to create the block with that linger, and the default port.
    def test_flag_sets_linger_seconds(self) -> None:
        metrics = _config_from_argv(["--observability.metrics.linger_seconds", "5"]).observability.metrics
        assert metrics is not None
        assert metrics.linger_seconds == 5.0
        assert metrics.port == 9464
