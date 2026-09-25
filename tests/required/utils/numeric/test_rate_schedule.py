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
"""RateSchedule: a rate expression integrated over a stage window, and its inverse."""

import numpy as np
import pytest

from inference_perf.utils.numeric.expression import Expression
from inference_perf.utils.numeric.rate_schedule import RateSchedule


# Builds a schedule from a rate string over a window of `duration` seconds.
def _schedule(rate: str, duration: float) -> RateSchedule:
    return RateSchedule(Expression(rate, allow_random=False, minimum=0, duration=duration), duration)


# Constant rate 4 over 60s: total 240, mean 4, and both directions are exact linear maps
# (cumulative(15) = 60, inverse(60) = 15), with no grid involved.
def test_constant_rate_is_exact() -> None:
    schedule = _schedule("4", 60)
    assert schedule.is_constant
    assert schedule.total == 240.0
    assert schedule.expected_requests == 240
    assert schedule.mean_rate == 4.0
    assert float(schedule.cumulative(15.0)) == 60.0
    assert float(schedule.inverse(60.0)) == 15.0


# Linear ramp rate = t over 10s: total = 10^2/2 = 50, mean 5, cumulative(t) = t^2/2,
# so inverse(12.5) = 5. The trapezoid rule is exact for a linear rate.
def test_linear_ramp_integrates_exactly() -> None:
    schedule = _schedule("t", 10)
    assert not schedule.is_constant
    assert schedule.total == pytest.approx(50.0)
    assert schedule.expected_requests == 50
    assert schedule.mean_rate == pytest.approx(5.0)
    assert float(schedule.cumulative(4.0)) == pytest.approx(8.0)
    assert float(schedule.inverse(12.5)) == pytest.approx(5.0)


# Sinusoidal rate 10 + 5*sin(t) over 2*pi seconds: the sine integrates to zero over a full
# period, so total = 20*pi (about 62.83), within the grid's error.
def test_sinusoid_total_matches_closed_form() -> None:
    schedule = _schedule("10 + 5*sin(t)", 2 * np.pi)
    assert schedule.total == pytest.approx(20 * np.pi, rel=1e-5)


# inverse undoes cumulative: for rate 2 + t/10 over 30s, mapping 7 times through
# cumulative and back returns the same 7 times.
def test_inverse_round_trips_cumulative() -> None:
    schedule = _schedule("2 + t/10", 30)
    times = np.linspace(0.5, 29.5, 7)
    np.testing.assert_allclose(schedule.inverse(schedule.cumulative(times)), times, atol=1e-6)


# Past the window the rate is held at the window's mean: a ramp 0 -> 10 over 10s has
# total 50 and mean 5, so 5 seconds past the end adds 25 (cumulative(15) = 75), and the
# inverse of 75 is 15. A ramp DOWN to zero still overruns at the mean instead of stalling.
def test_overrun_holds_the_mean_rate() -> None:
    up = _schedule("t", 10)
    assert float(up.cumulative(15.0)) == pytest.approx(75.0)
    assert float(up.inverse(75.0)) == pytest.approx(15.0)
    down = _schedule("10 - t", 10)
    assert float(down.inverse(down.total + 5.0)) == pytest.approx(11.0)


# A random rate is refused: the schedule has to be the same on every run.
def test_random_rate_rejected() -> None:
    with pytest.raises(ValueError, match="random"):
        RateSchedule(Expression("Uniform(1, 2)"), 10)


# A rate that goes negative inside the window is refused. Here the range can't be proven
# at construction (no minimum given), so the grid catches it: 5 - t is negative after t=5.
def test_negative_rate_rejected_on_the_grid() -> None:
    with pytest.raises(ValueError, match="negative"):
        RateSchedule(Expression("5 - t"), 10)
