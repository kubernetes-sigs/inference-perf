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
"""RequestSchedule: request arrival offsets drawn from a request_interval expression."""

import unittest.mock

import numpy as np
from numpy.typing import NDArray
import pytest

from inference_perf.utils.numeric.request_schedule import RequestSchedule
from inference_perf.utils.numeric.expression import Expression


# Builds a schedule from a request_interval string over `duration` seconds, drawing with a seeded rng.
def _schedule(request_interval: str, duration: float, seed: int = 0) -> RequestSchedule:
    expression = Expression(request_interval, minimum=0, duration=duration, time_in_parameters=True)
    return RequestSchedule(expression, duration, rng=np.random.default_rng(seed))


# Recovers the gaps between consecutive arrivals, the first gap being measured from 0.
def _gaps(schedule: RequestSchedule) -> NDArray[np.float64]:
    return np.diff(np.concatenate(([0.0], schedule.offsets)))


# A fixed gap of 0.1s over 1s is exactly 10 requests at 0.1, 0.2, ... 1.0 (not 9, which is
# what summing 0.1 ten times in floating point would give), a mean of 10 per second.
# A gap of 0.3s over 1s fits 3 requests, at 0.3, 0.6 and 0.9.
def test_fixed_gap_is_evenly_spaced() -> None:
    tenth = _schedule("0.1", 1)
    assert tenth.expected_requests == 10
    assert tenth.offsets == pytest.approx([0.1 * k for k in range(1, 11)])
    assert tenth.mean_rate == 10.0
    assert _schedule("0.3", 1).offsets == pytest.approx([0.3, 0.6, 0.9])


# Exponential(10) gaps over 2000s (a Poisson process at 10 per second): about 20000
# requests, a mean gap near 0.1s, and a gap spread equal to the mean (ratio near 1).
# Offsets ascend, all fall inside the window, and the count matches the offsets.
def test_exponential_gaps_are_a_poisson_process() -> None:
    schedule = _schedule("Exponential(10)", 2000)
    gaps = _gaps(schedule)
    assert schedule.expected_requests == len(schedule.offsets)
    assert schedule.expected_requests == pytest.approx(20000, rel=0.05)
    assert schedule.mean_rate == pytest.approx(10, rel=0.05)
    assert float(gaps.mean()) == pytest.approx(0.1, rel=0.05)
    assert float(gaps.std() / gaps.mean()) == pytest.approx(1.0, abs=0.05)
    assert bool(np.all(gaps >= 0))
    assert 0 < float(schedule.offsets[0]) and float(schedule.offsets[-1]) <= 2000


# Gamma(0.25, 0.4) gaps have the same 0.1s mean as above but twice the spread (ratio 2):
# the same average load arriving in bursts. Uniform(0.05, 0.15) gaps never leave their bounds.
def test_gap_distribution_sets_the_burstiness() -> None:
    bursty = _gaps(_schedule("Gamma(0.25, 0.4)", 2000))
    assert float(bursty.mean()) == pytest.approx(0.1, rel=0.1)
    assert float(bursty.std() / bursty.mean()) == pytest.approx(2.0, abs=0.2)
    bounded = _gaps(_schedule("Uniform(0.05, 0.15)", 200))
    assert 0.05 <= float(bounded.min()) and float(bounded.max()) <= 0.15


# The same seed draws the same arrivals; a different seed draws different ones.
def test_draw_is_reproducible_from_the_seed() -> None:
    first = _schedule("Exponential(10)", 60, seed=3)
    assert np.array_equal(first.offsets, _schedule("Exponential(10)", 60, seed=3).offsets)
    assert not np.array_equal(first.offsets, _schedule("Exponential(10)", 60, seed=4).offsets)


# Rejected: gaps so small the stage would schedule over ten million requests
# (Exponential(1e9) over 60s), and a window that is not positive.
def test_unusable_schedules_are_rejected() -> None:
    with pytest.raises(ValueError, match="gaps are too small"):
        _schedule("Exponential(1e9)", 60)
    with pytest.raises(ValueError, match="duration must be positive"):
        _schedule("0.1", 0)


# A gap that is not random is the rate 1/gap. "1/t" over 10s is the rate t, whose area is t^2/2:
# 50 requests, request k where the area reaches k, so the first at sqrt(2) = 1.414s, the second
# at 2s and the last at 10s.
def test_gap_over_time_is_scheduled_as_its_rate() -> None:
    schedule = _schedule("1/t", 10)
    assert schedule.expected_requests == 50
    assert schedule.offsets == pytest.approx([np.sqrt(2 * k) for k in range(1, 51)], abs=1e-3)
    assert float(schedule.offsets[0]) == pytest.approx(1.4142, abs=1e-3)
    assert float(schedule.offsets[-1]) == pytest.approx(10.0)


# A varying gap is spaced along its curve, with none of the constant timer's jitter: for the gap
# 0.1 + t/100 over 10s each gap between arrivals is within 1% of the gap the expression gives
# at that time, and the gaps only grow.
def test_gap_over_time_is_evenly_spaced_along_the_curve() -> None:
    schedule = _schedule("0.1 + t/100", 10)
    gaps = _gaps(schedule)
    assert bool(np.all(np.diff(gaps) > 0))
    assert gaps[1:] == pytest.approx(0.1 + schedule.offsets[:-1] / 100, rel=0.01)


# An infinite gap is a zero rate and is legal. "oo" for the whole 10s sends nothing and says the gap
# is infinite. Infinite for the first 5s and 0.1s after it sends 50 requests, none before 5s.
def test_infinite_gap_is_a_zero_rate() -> None:
    never = _schedule("oo", 10)
    assert never.expected_requests == 0
    assert never.zero_requests_reason == "its request_interval 'oo' is infinite for the whole stage"
    pause = _schedule("Piecewise((oo, t < 5), (0.1, True))", 10)
    assert pause.expected_requests == 50
    assert float(pause.offsets[0]) > 5.0
    assert pause.zero_requests_reason is None
    assert _schedule("Piecewise((oo, t < 20), (0.1, True))", 10).zero_requests_reason == (
        "its request_interval 'Piecewise((oo, t < 20), (0.1, True))' is infinite for the whole stage"
    )


# A finite gap longer than the stage is not an error: a 100s gap over 60s sends nothing and says
# no gap fits. A stage that sends has no reason to give.
def test_gap_longer_than_the_stage_sends_nothing() -> None:
    schedule = _schedule("100", 60)
    assert schedule.expected_requests == 0
    assert schedule.zero_requests_reason == "no gap of its request_interval '100' fits in the stage"
    assert _schedule("0.1", 60).zero_requests_reason is None


# A gap that is zero or negative anywhere is rejected, naming where: "t" is zero at t=0, "Abs(t - 5)"
# is zero at t=5, and "t - 5" is negative before t=5.
@pytest.mark.parametrize(
    ("request_interval", "message"),
    [
        ("t", "is zero at t=0. A gap must be positive"),
        ("Abs(t - 5)", "is zero at t=5. A gap must be positive"),
        ("t - 5", "outside the permitted range"),
        ("Piecewise((-1, t < 5), (0.1, True))", "is negative at t=0. A gap must be positive"),
    ],
)
def test_zero_or_negative_gap_over_time_is_rejected(request_interval: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        _schedule(request_interval, 10)


# A random gap over t is used up along the curve, not fixed when it starts. A gap pinned to 0.1 + t (a uniform
# one nanosecond wide) sends where the area under 1/(0.1 + t) reaches 1, 2, 3: at 0.172, 0.639 and 1.909s, the
# times the gap "0.1 + t" with no randomness sends at. Fixing each gap at its start gave 0.1, 0.3, 0.7, 1.5, 3.1.
def test_random_gap_over_time_is_used_up_along_the_curve() -> None:
    schedule = _schedule("Uniform(0.1 + t, 0.1 + t + 1e-9)", 4)
    assert schedule.offsets == pytest.approx([0.171828, 0.638906, 1.908554], abs=1e-5)
    # the gap with no randomness is placed on the rate's 10ms grid, which is good to a few ms where the curve bends hard
    assert schedule.offsets == pytest.approx(_schedule("0.1 + t", 4).offsets, abs=5e-3)


# "Exponential(5 + t/10)" over 60s loads and follows its curve: the rate climbs from 5 to 11, so of
# about 480 requests (380 to 570 allowed) more land in the second half than in the first. The same
# seed draws the same arrivals and another seed different ones.
def test_random_gap_over_time_follows_the_curve() -> None:
    schedule = _schedule("Exponential(5 + t/10)", 60, seed=3)
    assert 380 <= schedule.expected_requests <= 570
    assert int(np.sum(schedule.offsets > 30)) > int(np.sum(schedule.offsets <= 30))
    assert bool(np.all(np.diff(schedule.offsets) >= 0)) and float(schedule.offsets[-1]) <= 60
    assert np.array_equal(schedule.offsets, _schedule("Exponential(5 + t/10)", 60, seed=3).offsets)
    assert not np.array_equal(schedule.offsets, _schedule("Exponential(5 + t/10)", 60, seed=4).offsets)


# A random gap with no t is still drawn in one batch: Exponential(10) over 60s with seed 3 gives
# exactly the running sum of one batch of 4096 seeded draws, cut off at 60s, and the schedule asks
# the expression for its draws once, 4096 at a time, not gap by gap.
def test_random_gap_without_time_is_still_drawn_in_a_batch() -> None:
    expression = Expression("Exponential(10)", minimum=0, duration=60)
    draws = np.cumsum(np.asarray(expression.sample(rng=np.random.default_rng(3), size=4096)))
    with unittest.mock.patch.object(expression, "sample", wraps=expression.sample) as sample:
        schedule = RequestSchedule(expression, 60, rng=np.random.default_rng(3))
    assert np.array_equal(schedule.offsets, draws[draws <= 60])
    assert [call.kwargs["size"] for call in sample.call_args_list] == [4096]


# A gap that is always zero where the stage stands never moves the stage on: "t*Exponential(5)" is
# zero at t=0, so it is rejected instead of drawing forever.
def test_gap_stuck_at_zero_is_rejected() -> None:
    with pytest.raises(ValueError, match="keeps drawing a zero gap at t=0"):
        _schedule("t*Exponential(5)", 10)
