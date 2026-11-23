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
"""A random request_interval over t: each gap's randomness is drawn once and used up along the curve.

Three kinds of check. Exact ones compare the arrivals with a closed form
built from the same draws. Statistical ones run many seeded schedules and
compare with what the theory of the process says, at tolerances a correct
implementation fails about once in thousands of seeds, and every seed here is
fixed. Structural ones hold for any input.
"""

import math
from functools import lru_cache
from typing import Callable

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from inference_perf.utils.numeric import request_schedule
from inference_perf.utils.numeric.expression import Expression
from inference_perf.utils.numeric.rate_schedule import RateSchedule
from inference_perf.utils.numeric.request_schedule import RequestSchedule, arrivals_along_the_curve

Floats = NDArray[np.float64]


# A request_interval expression that may take t in a distribution's parameters, checked over `duration` seconds.
# Kept per (raw, duration): the check at load is a proof, and the statistical tests ask for the same one 600 times.
@lru_cache(maxsize=None)
def _gap(raw: str, duration: float) -> Expression:
    expression = Expression(raw, minimum=0, duration=duration, time_in_parameters=True)
    expression.require_valid_draws()
    return expression


# The arrival offsets of `raw` over `duration` seconds from numpy's default_rng(seed).
def _arrivals(raw: str, duration: float, seed: int = 0) -> Floats:
    return arrivals_along_the_curve(_gap(raw, duration), duration, np.random.default_rng(seed))


# The running sum of the base draws the schedule itself makes from this seed: column `column`, first 4096 gaps.
def _running_base(raw: str, duration: float, seed: int, column: int = 0) -> Floats:
    return np.cumsum(_gap(raw, duration).sample_base(np.random.default_rng(seed), 4096)[:, column])


# The area under the rate a + b*t from 0 to t, and the time at which that area reaches `area`.
def _linear_area(a: float, b: float, t: Floats) -> Floats:
    return a * t + b * t * t / 2


# written without a division by b, so it holds for a flat or nearly flat rate. nan where the area is never reached
def _linear_time(a: float, b: float, area: Floats) -> Floats:
    with np.errstate(invalid="ignore"):
        return 2 * area / (a + np.sqrt(a * a + 2 * b * area))


# The largest gap between the empirical distribution of `sample` and the distribution function `cdf`.
def _ks_one(sample: Floats, cdf: Callable[[Floats], Floats]) -> float:
    ordered = np.sort(sample)
    n = len(ordered)
    theory = cdf(ordered)
    return float(max(np.max(np.arange(1, n + 1) / n - theory), np.max(theory - np.arange(n) / n)))


# The largest gap between the empirical distributions of two samples.
def _ks_two(first: Floats, second: Floats) -> float:
    points = np.sort(np.concatenate([first, second]))
    a = np.searchsorted(np.sort(first), points, side="right") / len(first)
    b = np.searchsorted(np.sort(second), points, side="right") / len(second)
    return float(np.max(np.abs(a - b)))


# ---------------------------------------------------------------------------
# Exact: the arrivals equal a closed form built from the same draws
# ---------------------------------------------------------------------------


# Exponential(a + b*t) is a Poisson process of rate a + b*t: arrival k is where the area a*t + b*t^2/2 equals
# the sum of the first k Exponential(1) draws. Rising from near zero, falling, and gently rising: every arrival
# matches that closed form to within a nanosecond, and none is missing or extra.
@pytest.mark.parametrize(("a", "b", "duration"), [(0.1, 1.0, 30.0), (20.0, -0.3, 60.0), (5.0, 0.1, 60.0), (50.0, 5.0, 10.0)])
@pytest.mark.parametrize("seed", [0, 7, 12345])
def test_exponential_over_time_is_exactly_a_poisson_process(a: float, b: float, duration: float, seed: int) -> None:
    raw = f"Exponential({a} + ({b})*t)"
    expected = _linear_time(a, b, _running_base(raw, duration, seed))
    expected = expected[np.isfinite(expected) & (expected <= duration)]
    got = _arrivals(raw, duration, seed)
    assert len(got) == len(expected) > 0
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-9)


# A rate that swings, 10 + 8*sin(2*pi*t/20) over 60s: arrival k is where the area under it, 10*t +
# (80/pi)*(1 - cos(2*pi*t/20)), equals the sum of the first k Exponential(1) draws. Checked by putting each
# arrival back into the area: it lands on the running sum to 1e-7.
def test_exponential_over_a_sine_is_exactly_a_poisson_process() -> None:
    raw = "Exponential(10 + 8*sin(2*pi*t/20))"
    got = _arrivals(raw, 60, seed=3)
    area = 10 * got + (80 / math.pi) * (1 - np.cos(2 * math.pi * got / 20))
    running = _running_base(raw, 60, seed=3)
    assert 500 < len(got) < 700
    np.testing.assert_allclose(area, running[: len(got)], rtol=0, atol=1e-7)
    # the next draw does not fit: its area lies past the end of the window
    assert running[len(got)] > 10 * 60 + (80 / math.pi) * (1 - math.cos(2 * math.pi * 60 / 20))


# A distribution whose size alone follows the curve is a renewal process on a clock that runs at the rate
# a + b*t = 4 + t/2: the gaps measured in area under the rate are the draws at a rate of 1. One row per family:
# Gamma by its scale, Weibull by its scale, Pareto by its minimum, LogNormal by its mean, Uniform by both ends,
# Normal by its mean and std together (floored at 0, as a gap must be). Every arrival matches the closed form to 1e-8.
@pytest.mark.parametrize(
    ("raw", "at_rate_one"),
    [
        ("Gamma(4, 0.25/(4 + t/2))", "Gamma(4, 0.25)"),
        ("Weibull(1/(4 + t/2), 2)", "Weibull(1, 2)"),
        ("Pareto(0.5/(4 + t/2), 3)", "Pareto(0.5, 3)"),
        ("LogNormal(-log(4 + t/2), 0.5)", "LogNormal(0, 0.5)"),
        ("Uniform(0.5/(4 + t/2), 1.5/(4 + t/2))", "Uniform(0.5, 1.5)"),
        ("Max(0, Normal(1/(4 + t/2), 0.1/(4 + t/2)))", "Max(0, Normal(1, 0.1))"),
    ],
)
def test_scaled_distribution_is_a_renewal_process_on_the_curves_clock(raw: str, at_rate_one: str) -> None:
    duration, seed = 40.0, 11
    moving, still = _gap(raw, duration), _gap(at_rate_one, duration)
    base = moving.sample_base(np.random.default_rng(seed), 4096)
    # the same base draws read as gaps at a rate of 1, i.e. the gap at time 0 times the rate at time 0
    unit_gaps = np.array([moving.value_given(row, 0.0) * 4.0 for row in base.tolist()])
    expected = _linear_time(4.0, 0.5, np.cumsum(unit_gaps))
    expected = expected[expected <= duration]
    got = arrivals_along_the_curve(moving, duration, np.random.default_rng(seed))
    assert len(got) == len(expected) > 100
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-8)
    # and those unit gaps are draws of the distribution at a rate of 1
    reference = np.asarray(still.sample(rng=np.random.default_rng(99), size=4096))
    assert _ks_two(unit_gaps, reference) < 1.95 * math.sqrt(2 / 4096)


# With constant parameters the rule changes nothing: the arrivals are the running sum of the draws. Run through
# the solver directly for Gamma(2, 0.05), Uniform(0.05, 0.15) and Exponential(10) over 30s, to 1e-9.
@pytest.mark.parametrize("raw", ["Gamma(2, 0.05)", "Uniform(0.05, 0.15)", "Exponential(10)", "0.02 + Exponential(20)"])
def test_constant_parameters_give_the_running_sum_of_the_draws(raw: str) -> None:
    expression = _gap(raw, 30)
    gaps = [expression.value_given(row, 0.0) for row in expression.sample_base(np.random.default_rng(5), 4096).tolist()]
    expected = np.cumsum(gaps)
    expected = expected[expected <= 30]
    got = arrivals_along_the_curve(expression, 30, np.random.default_rng(5))
    assert len(got) == len(expected) > 100
    np.testing.assert_allclose(got, expected, rtol=0, atol=1e-9)


# With no randomness the rule is how a rate is placed: request k where the area under 1/gap reaches k. The gaps
# 1/(2 + t), 0.1 + t/100 and 1/(10 + 5*sin(2*pi*t/60)) through the solver land within 1ms of the rate schedule's
# send times for the rates 2 + t, 1/(0.1 + t/100) and 10 + 5*sin(2*pi*t/60).
@pytest.mark.parametrize("raw", ["1/(2 + t)", "0.1 + t/100", "1/(10 + 5*sin(2*pi*t/60))"])
def test_no_randomness_is_placed_as_the_rate(raw: str) -> None:
    duration = 30.0
    got = arrivals_along_the_curve(Expression(raw, minimum=0, duration=duration), duration, np.random.default_rng(0))
    rate = RateSchedule(Expression(f"1/({raw})", allow_random=False, duration=duration), duration)
    expected = rate.send_times()
    # a last request due exactly as the window closes may fall either side of it
    assert abs(len(got) - len(expected)) <= 1
    shared = min(len(got), len(expected))
    np.testing.assert_allclose(got[:shared], expected[:shared], rtol=0, atol=1e-3)


# A gap pinned to 0.1 + t by a uniform one nanosecond wide: the first request is where the area under
# 1/(0.1 + t) reaches 1, t = 0.1*(e - 1) = 0.1718s, and each later one at (0.1 + previous)*e - 0.1. The old
# rule read the gap once at its start and sent at 0.1, 0.3, 0.7, 1.5.
def test_a_gap_is_used_up_along_the_curve_not_fixed_at_its_start() -> None:
    got = _arrivals("Uniform(0.1 + t, 0.100000001 + t)", 20, seed=1)
    expected = [0.1 * (math.e - 1)]
    while (0.1 + expected[-1]) * math.e - 0.1 <= 20:
        expected.append((0.1 + expected[-1]) * math.e - 0.1)
    assert len(got) == len(expected) == 5
    np.testing.assert_allclose(got, expected, rtol=1e-6, atol=0)


# Two random variables in one gap, one moving and one not: 0.01 + Exponential(20 + t) + Uniform(0, 0.01). Each
# gap's two draws are held together: putting arrival k and k+1 into the gap's own equation, the area under
# 1/(0.01 + e/(20 + t) + u) between them is 1, to 1e-6.
def test_two_draws_in_one_gap_are_held_together() -> None:
    raw = "0.01 + Exponential(20 + t) + Uniform(0, 0.01)"
    expression = _gap(raw, 20)
    base = expression.sample_base(np.random.default_rng(4), 4096)
    got = arrivals_along_the_curve(expression, 20, np.random.default_rng(4))
    assert base.shape == (4096, 2) and len(got) > 300
    starts = np.concatenate([[0.0], got[:-1]])
    for row, start, end in list(zip(base.tolist(), starts.tolist(), got.tolist(), strict=False))[:200]:
        times = np.linspace(start, end, 2001)
        rates = np.array([1 / expression.value_given(row, float(t)) for t in times])
        area = float(np.sum((rates[1:] + rates[:-1]) / 2 * np.diff(times)))
        assert area == pytest.approx(1.0, abs=1e-6)


# ---------------------------------------------------------------------------
# Statistical: many seeded schedules against the theory of the process
# ---------------------------------------------------------------------------


# The case that prompted the rule. Exponential(0.1 + t) over 30s, 600 seeded schedules. A Poisson process of
# that rate sends a Poisson number of requests with mean and variance 453 (the area), never sends nothing, and
# its first request has mean integral of exp(-(0.1*t + t^2/2)) = 1.16s. The old rule averaged 370 with a first
# request at 9.6s, and sent nothing in 5% of runs.
def test_exponential_from_near_zero_sends_the_area_on_average() -> None:
    runs = [_arrivals("Exponential(0.1 + t)", 30, seed) for seed in range(600)]
    counts = np.array([len(run) for run in runs], dtype=np.float64)
    area = 0.1 * 30 + 30 * 30 / 2
    assert area == 453
    assert abs(counts.mean() - area) <= 4 * math.sqrt(area / len(runs))
    assert 0.8 * area <= counts.var(ddof=1) <= 1.2 * area
    assert counts.min() > 0
    firsts = np.array([run[0] for run in runs])
    grid = np.linspace(0, 30, 300001)
    survival = np.exp(-(0.1 * grid + grid * grid / 2))
    mean_first = float(np.sum((survival[1:] + survival[:-1]) / 2 * np.diff(grid)))
    assert mean_first == pytest.approx(1.16, abs=5e-3)
    assert abs(firsts.mean() - mean_first) <= 4 * firsts.std(ddof=1) / math.sqrt(len(runs))
    # the chance the first request comes after 1s is exp(-0.6) = 0.549
    late = float(np.mean(firsts > 1.0))
    assert abs(late - math.exp(-0.6)) <= 4 * math.sqrt(math.exp(-0.6) * (1 - math.exp(-0.6)) / len(runs))


# The random twin of a rate sends the rate's count on average, whichever way the curve goes: rising from near
# zero, falling to near zero, falling, and swinging. 300 seeded schedules each, mean within 4 standard errors of
# the area under the rate. A rule that fixes the gap at its start runs slow on the first and fast on the next two.
@pytest.mark.parametrize(
    ("rate", "duration"),
    [("0.1 + t", 30.0), ("30.1 - t", 30.0), ("20.5 - t/3", 60.0), ("10 + 8*sin(2*pi*t/20)", 60.0)],
)
def test_random_twin_of_a_rate_sends_its_area_on_average(rate: str, duration: float) -> None:
    area = RateSchedule(Expression(rate, allow_random=False, minimum=0, duration=duration), duration).total
    counts = np.array([len(_arrivals(f"Exponential({rate})", duration, seed)) for seed in range(300)], dtype=np.float64)
    assert abs(counts.mean() - area) <= 4 * math.sqrt(area / len(counts))


# The time-rescaling theorem: for a Poisson process, the area under the rate between consecutive arrivals is a
# sequence of independent Exponential(1) draws. One schedule of Exponential(2 + 3*t) over 60s, about 5500 gaps:
# the areas pass a Kolmogorov-Smirnov test against Exponential(1) at the 0.1% level, have mean and variance 1,
# and neighbouring areas are uncorrelated.
def test_areas_between_arrivals_are_independent_unit_exponentials() -> None:
    got = _arrivals("Exponential(2 + 3*t)", 60, seed=21)
    areas = np.diff(_linear_area(2.0, 3.0, np.concatenate([[0.0], got])))
    n = len(areas)
    assert 5200 < n < 5800
    assert _ks_one(areas, lambda x: 1 - np.exp(-x)) < 1.95 / math.sqrt(n)
    assert abs(areas.mean() - 1) <= 4 / math.sqrt(n)
    assert abs(areas.var() - 1) <= 4 * math.sqrt(8 / n)
    lag_one = float(np.corrcoef(areas[:-1], areas[1:])[0, 1])
    assert abs(lag_one) <= 4 / math.sqrt(n)


# A Poisson process has independent counts in separate stretches of time, each Poisson with the stretch's area
# as its mean. Exponential(1 + t) over 30s, 600 schedules: the counts in [0, 10) and [20, 30) average their
# areas, 60 and 260, with variances to match, and are uncorrelated.
def test_counts_in_separate_stretches_are_independent_poisson() -> None:
    early, late = [], []
    for seed in range(600):
        got = _arrivals("Exponential(1 + t)", 30, seed)
        early.append(int(np.sum(got < 10)))
        late.append(int(np.sum(got >= 20)))
    first, second = np.array(early, dtype=np.float64), np.array(late, dtype=np.float64)
    for counts, area in ((first, 60.0), (second, 260.0)):
        assert abs(counts.mean() - area) <= 4 * math.sqrt(area / len(counts))
        assert 0.8 * area <= counts.var(ddof=1) <= 1.2 * area
    assert abs(float(np.corrcoef(first, second)[0, 1])) <= 4 / math.sqrt(len(first))


# Where a Poisson process puts its arrivals: given how many there are, they are independent draws from the
# rate's shape. For the rate 0.1 + t over 30s that is the distribution (0.1*t + t^2/2)/453. The arrivals of 40
# schedules pooled, about 18000 of them, pass a Kolmogorov-Smirnov test against it at the 0.1% level.
def test_arrival_times_follow_the_shape_of_the_rate() -> None:
    pooled = np.concatenate([_arrivals("Exponential(0.1 + t)", 30, seed) for seed in range(1000, 1040)])
    assert 17000 < len(pooled) < 19000
    assert _ks_one(pooled, lambda t: _linear_area(0.1, 1.0, t) / 453) < 1.95 / math.sqrt(len(pooled))


# A steadier distribution on the same curve. Gamma(4, 0.25/(2 + t/5)) has a mean gap of 1/(2 + t/5) and a
# quarter of an exponential's squared spread. Over 40s the area is 240, and 600 schedules send, on average,
# the area less 3/8 (renewal theory: (spread^2 - 1)/2 for a process that starts with a full gap), with a
# variance near a quarter of the area. A Poisson process on the same curve would have a variance of 240.
def test_gamma_on_a_curve_keeps_its_steadiness() -> None:
    counts = np.array([len(_arrivals("Gamma(4, 0.25/(2 + t/5))", 40, seed)) for seed in range(600)], dtype=np.float64)
    area = 2 * 40 + 40 * 40 / 10
    assert area == 240
    spread = counts.std(ddof=1)
    assert abs(counts.mean() - (area - 0.375)) <= 4 * spread / math.sqrt(len(counts))
    assert 0.2 * area <= counts.var(ddof=1) <= 0.3 * area


# The gaps of that Gamma process, measured in area under the rate, are Gamma(4, 0.25) draws: mean 1, variance
# 1/4, and the same distribution as 5000 direct draws (two-sample Kolmogorov-Smirnov at the 0.1% level).
def test_gamma_areas_between_arrivals_are_gamma_draws() -> None:
    got = np.concatenate([[0.0], _arrivals("Gamma(4, 0.25/(2 + t/5))", 400, seed=8)])
    areas = np.diff(_linear_area(2.0, 0.2, got))[:5000]
    assert len(areas) == 5000
    assert abs(areas.mean() - 1) <= 4 * 0.5 / math.sqrt(5000)
    assert abs(areas.var() - 0.25) <= 0.03
    direct = np.random.default_rng(77).gamma(4, 0.25, 5000)
    assert _ks_two(areas, direct) < 1.95 * math.sqrt(2 / 5000)


# A load that waits and then starts: the gap is infinite until t=5 and Exponential(10 + t) after. Nothing is sent
# before 5s in any of 200 schedules, and the count after it averages the area under 10 + t from 5 to 30, 687.5.
# The wait costs nothing: the first request comes about 1/15s after 5s, not a whole stale gap later.
# A config cannot write this today, since the check at load cannot bound a Piecewise with a draw in it, so the
# expression is built without that check.
def test_an_infinite_gap_pauses_the_stage_without_a_stale_gap() -> None:
    raw = "Piecewise((oo, t < 5), (0, True)) + Exponential(10 + t)"
    expression = Expression(raw, duration=30, time_in_parameters=True)
    runs = [arrivals_along_the_curve(expression, 30, np.random.default_rng(seed)) for seed in range(200)]
    assert min(float(run[0]) for run in runs) >= 5.0
    counts = np.array([len(run) for run in runs], dtype=np.float64)
    assert abs(counts.mean() - 687.5) <= 4 * math.sqrt(687.5 / len(runs))
    firsts = np.array([run[0] for run in runs]) - 5.0
    assert firsts.mean() < 0.12


# ---------------------------------------------------------------------------
# Structural: true of any input
# ---------------------------------------------------------------------------


# Any linear rate that stays positive, any window, any seed: the arrivals ascend strictly, lie inside the
# window, and equal the closed form built from the same draws to 1e-7. 60 generated cases.
@settings(max_examples=60, deadline=None)
@given(
    a=st.floats(min_value=0.05, max_value=40),
    slope=st.floats(min_value=-0.9, max_value=3),
    duration=st.floats(min_value=1, max_value=30),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_linear_rates_match_the_closed_form(a: float, slope: float, duration: float, seed: int) -> None:
    # the rate ends at a*(1 + slope) > 0 when slope > -1
    b = a * slope / duration
    raw = f"Exponential({a!r} + ({b!r})*t)"
    got = _arrivals(raw, duration, seed)
    assert bool(np.all(np.diff(got) > 0)) and bool(np.all((got > 0) & (got <= duration)))
    running = _running_base(raw, duration, seed)
    # a flat rate has no t left in it, and then the base draws are the gaps themselves, not areas
    expected = running if b == 0 else _linear_time(a, b, running)
    expected = expected[np.isfinite(expected) & (expected <= duration)]
    # an arrival within rounding of the window's end may fall either side of it
    assert abs(len(got) - len(expected)) <= 1
    shared = min(len(got), len(expected))
    np.testing.assert_allclose(got[:shared], expected[:shared], rtol=0, atol=1e-7)


# Any of the distributions that can follow a curve, on a rising or falling curve, any seed: the arrivals never
# go backwards, lie inside the window, and the same seed gives the same arrivals. 40 generated cases.
@settings(max_examples=40, deadline=None)
@given(
    raw=st.sampled_from(
        [
            "Exponential(3 + t)",
            "Gamma(2, 1/(6 + 2*t))",
            "Weibull(1/(3 + t), 1.5)",
            "Pareto(0.2/(3 + t), 2.5)",
            "LogNormal(-log(3 + t), 0.8)",
            "Uniform(0, 2/(3 + t))",
            "Max(0, Normal(1/(3 + t), 0.5/(3 + t)))",
            "Exponential(40 - 3*t)",
            "0.001 + Exponential(3 + t)*Uniform(0.5, 1.5)",
        ]
    ),
    seed=st.integers(min_value=0, max_value=2**32 - 1),
)
def test_arrivals_ascend_stay_in_the_window_and_repeat_from_a_seed(raw: str, seed: int) -> None:
    got = _arrivals(raw, 12, seed)
    assert bool(np.all(np.diff(got) >= 0)) and bool(np.all((got >= 0) & (got <= 12)))
    np.testing.assert_array_equal(got, _arrivals(raw, 12, seed))
    assert len(got) > 0


# Another seed gives other arrivals, and the schedule class takes this path for a random gap over t.
def test_another_seed_gives_other_arrivals() -> None:
    expression = _gap("Exponential(5 + t/10)", 60)
    first = RequestSchedule(expression, 60, np.random.default_rng(1)).offsets
    np.testing.assert_array_equal(first, arrivals_along_the_curve(expression, 60, np.random.default_rng(1)))
    assert not np.array_equal(first, RequestSchedule(expression, 60, np.random.default_rng(2)).offsets)


# The parameters are proved valid over the window only, so the gap is never read past its end.
# Exponential(60.5 - t) over 60s has a rate that goes negative half a second after the window: every read of
# the gap is at a t from 0 to 60.
def test_the_gap_is_never_read_past_the_window(monkeypatch: pytest.MonkeyPatch) -> None:
    expression = _gap("Exponential(60.5 - t)", 60)
    read: list[float] = []
    original = expression.value_given

    def recording(base: list[float], t: float) -> float:
        read.append(t)
        return original(base, t)

    monkeypatch.setattr(expression, "value_given", recording)
    got = arrivals_along_the_curve(expression, 60, np.random.default_rng(0))
    assert len(got) > 1500
    assert min(read) == 0.0 and max(read) <= 60.0


# A gap still open when the window closes ends the stage: Exponential(0.001 + t/1000) over 1s has a mean first
# gap of minutes, so nearly every schedule sends nothing, and none reads the gap past 1s or loops.
def test_a_gap_longer_than_the_window_sends_nothing() -> None:
    counts = [len(_arrivals("Exponential(0.001 + t/1000)", 1, seed)) for seed in range(50)]
    assert sum(counts) <= 2


# A zero gap is due at once: Max(0, Uniform(-1, 1))*(0.01 + t/1000) is zero for half its draws, so about half
# the requests share their time with the one before, and time never goes backwards.
def test_a_zero_gap_is_sent_at_once() -> None:
    got = _arrivals("Max(0, Uniform(-1, 1))*(0.01 + t/1000)", 2, seed=6)
    gaps = np.diff(got)
    assert len(got) > 100 and bool(np.all(gaps >= 0))
    assert 0.35 < float(np.mean(gaps == 0)) < 0.65


# A gap that is zero wherever the stage stands never moves it on: t*Exponential(5) is zero at t=0, so it is
# rejected instead of sending forever.
def test_a_gap_stuck_at_zero_is_rejected() -> None:
    with pytest.raises(ValueError, match="keeps drawing a zero gap at t=0"):
        _arrivals("t*Exponential(5)", 10)


# A gap that reaches zero partway through a step is found, not stepped over: Exponential(1)*Max(0, 1 - t) is
# zero from t=1 on, so a request not yet sent by then is due at t=1, and the stage is then stuck there.
def test_a_gap_that_falls_to_zero_is_due_when_it_does() -> None:
    expression = Expression("(0.5 + Exponential(1))*Max(0, 1 - t)", minimum=0, duration=10, time_in_parameters=True)
    with pytest.raises(ValueError, match="keeps drawing a zero gap at t=1"):
        arrivals_along_the_curve(expression, 10, np.random.default_rng(0))


# A gap that comes out negative or not a number where it is read is an error that names the time, not a silent
# step backwards. The proof at load rules this out, so the expression here skips it.
def test_a_negative_gap_is_an_error() -> None:
    expression = Expression("Normal(-1 - t, 0.001)", duration=10, time_in_parameters=True)
    with pytest.raises(ValueError, match="A gap must be nonnegative"):
        arrivals_along_the_curve(expression, 10, np.random.default_rng(0))


# Division by zero inside a gap reads as an infinite gap: Exponential(5)/t is infinite at t=0 and its rate is
# t/z after, so the first request is where t^2/(2*z) = 1, at sqrt(2*z) for the first draw z.
def test_division_by_zero_reads_as_an_infinite_gap() -> None:
    expression = _gap("Exponential(5)/t", 30)
    z = float(expression.sample_base(np.random.default_rng(2), 1)[0, 0])
    got = arrivals_along_the_curve(expression, 30, np.random.default_rng(2))
    assert got[0] == pytest.approx(math.sqrt(2 * z), abs=1e-6)


# The step solver on its own: the area under a rate that is 1, 2, 3 at the start, middle and end of a 1s step
# (the line 1 + 2x) reaches 0.75 at x = 0.5 and 2 at x = 1, and a flat rate of 4 reaches 1 at a quarter of the step.
def test_crossing_inside_a_step() -> None:
    assert request_schedule._cross_within(1.0, 1.0, 2.0, 3.0, 0.75) == pytest.approx(0.5, abs=1e-12)
    assert request_schedule._cross_within(1.0, 1.0, 2.0, 3.0, 2.0) == pytest.approx(1.0, abs=1e-12)
    assert request_schedule._cross_within(1.0, 4.0, 4.0, 4.0, 1.0) == pytest.approx(0.25, abs=1e-12)
    # a parabola: rate x^2 over a step of 3 has area x^3/3, which reaches 1 at the cube root of 3
    assert request_schedule._cross_within(3.0, 0.0, 2.25, 9.0, 1.0) == pytest.approx(3 ** (1 / 3), abs=1e-9)


# The single-gap solver against gaps with a known answer, starting at t=2 in a 100s window: a constant gap of
# 0.5 ends at 2.5; a gap of 1/t (rate t) ends where t^2/2 - 2 = 1, at sqrt(6); a gap of exp(-t) (rate exp(t))
# ends at log(exp(2) + 1); an infinite gap never ends; and a gap of 200 does not end inside the window.
def test_next_arrival_against_known_answers() -> None:
    def solve(gap_at: Callable[[float], float]) -> float | None:
        return request_schedule._next_arrival(gap_at, "test", 2.0, 100.0)

    assert solve(lambda t: 0.5) == pytest.approx(2.5, abs=1e-12)
    assert solve(lambda t: 1 / t) == pytest.approx(math.sqrt(6), abs=1e-9)
    assert solve(lambda t: math.exp(-t)) == pytest.approx(math.log(math.exp(2) + 1), abs=1e-9)
    assert solve(lambda t: math.inf) is None
    assert solve(lambda t: 200.0) is None
    # a rate that falls away: gap 2*exp(t - 2) has rate exp(2 - t)/2, whose area from 2 never passes 1/2
    assert solve(lambda t: 2 * math.exp(t - 2)) is None
    # a step up in the rate at t=3, from 0.5 to 4: half the gap is used by t=3 and the rest by 3.125
    assert solve(lambda t: 2.0 if t < 3 else 0.25) == pytest.approx(3.125, abs=1e-2)
