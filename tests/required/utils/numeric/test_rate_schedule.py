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


# A time-varying schedule stops at its window: for the ramp rate = t over 10s (total 50), a time
# 5s past the end still reads 50 requests, and a count of 75 reads as the window end, 10s.
def test_schedule_stops_at_the_window() -> None:
    ramp = _schedule("t", 10)
    assert float(ramp.cumulative(15.0)) == pytest.approx(50.0)
    assert float(ramp.inverse(75.0)) == pytest.approx(10.0)


# A random rate, Uniform(1, 2) over 10s, is refused with "A rate is never random".
def test_random_rate_rejected() -> None:
    with pytest.raises(ValueError, match="A rate is never random"):
        RateSchedule(Expression("Uniform(1, 2)"), 10)


# A rate that goes negative inside the window is refused. Here the range can't be proven
# at construction (no minimum given), so the grid catches it: 5 - t is negative after t=5.
def test_negative_rate_rejected_on_the_grid() -> None:
    with pytest.raises(ValueError, match="negative"):
        RateSchedule(Expression("5 - t"), 10)


# send_times places request k where the area reaches k, one time per expected request. Constant
# rate 4 over 2s gives 0.25, 0.5, ... 2.0. The ramp rate = t over 10s gives sqrt(2k) for k = 1..50,
# ending at 10. A zero rate gives no times.
def test_send_times_are_where_the_area_reaches_each_request() -> None:
    assert _schedule("4", 2).send_times() == pytest.approx([0.25 * k for k in range(1, 9)])
    ramp = _schedule("t", 10).send_times()
    assert len(ramp) == 50
    assert ramp == pytest.approx([np.sqrt(2 * k) for k in range(1, 51)], abs=1e-3)
    assert float(ramp[-1]) == 10.0
    assert len(_schedule("0", 10).send_times()) == 0


# The last send time never passes the window: rate 0.7 over 30s is 21 requests, and 21 / 0.7 is
# 30.000000000000004 in floating point, so the last time is held to exactly 30.
def test_send_times_stay_inside_the_window() -> None:
    times = _schedule("0.7", 30).send_times()
    assert len(times) == 21
    assert float(times[-1]) == 30.0


# A schedule knows whether its rate was written as an expression: "4" is one, the number 4 is not.
def test_is_expression_reads_how_the_rate_was_written() -> None:
    assert _schedule("4", 10).is_expression
    assert not RateSchedule(Expression(4, allow_random=False, minimum=0, duration=10), 10).is_expression
