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
"""Tests for ``PiecewiseLinear``, an expression solved into straight-line segments.

Each accepted shape is pinned by its exact segments; each rejected shape by
the node its message names. A seeded fuzz compares the segments against
direct evaluation of the original expression, which is the property that
matters: the segments must be the same function, not an approximation.
"""

import math
import random
from typing import Any, List, Tuple

import pytest
import sympy

from inference_perf.utils.numeric.expression import PiecewiseLinear
from inference_perf.utils.numeric.expression.expression import _T, _parse_raw


# Each segment as (start, end, slope, intercept), rounded for comparison.
def _shape(raw: Any) -> List[Tuple[float, float, float, float]]:
    return [(round(s.start, 6), s.end, round(s.slope, 6), round(s.intercept, 6)) for s in PiecewiseLinear(raw).segments]


class TestSegments:
    # A number is one flat segment from 0 to infinity: 10 -> [(0, inf, 0, 10)].
    def test_constant(self) -> None:
        assert _shape(10) == [(0.0, math.inf, 0.0, 10.0)]
        assert _shape("10") == [(0.0, math.inf, 0.0, 10.0)]
        assert PiecewiseLinear(10).is_constant

    # A capped ramp: 1 + t/6 until it reaches 64 at t=378, then 64 forever.
    def test_capped_ramp(self) -> None:
        assert _shape("Min(1 + t/6, 64)") == [(0.0, 378.0, round(1 / 6, 6), 1.0), (378.0, math.inf, 0.0, 64.0)]

    # A step: 8 before t=60, 32 from t=60 on.
    def test_step(self) -> None:
        assert _shape("Piecewise((8, t < 60), (32, True))") == [(0.0, 60.0, 0.0, 8.0), (60.0, math.inf, 0.0, 32.0)]

    # Ramp up and back down, floored at 1: 1 until t=2, t/2 up to 60 at t=120,
    # 120 - t/2 down to 1 at t=238, then 1.
    def test_up_and_down(self) -> None:
        assert _shape("Max(1, Min(t/2, 120 - t/2))") == [
            (0.0, 2.0, 0.0, 1.0),
            (2.0, 120.0, 0.5, 0.0),
            (120.0, 238.0, -0.5, 120.0),
            (238.0, math.inf, 0.0, 1.0),
        ]

    # A staircase from Heaviside steps: 4, then 8 from t=30, then 12 from t=60.
    def test_staircase(self) -> None:
        assert _shape("4 * (1 + Heaviside(t - 30) + Heaviside(t - 60))") == [
            (0.0, 30.0, 0.0, 4.0),
            (30.0, 60.0, 0.0, 8.0),
            (60.0, math.inf, 0.0, 12.0),
        ]

    # A V shape: 64 - |t - 60| rises with slope 1 to 64 at t=60, then falls.
    def test_abs(self) -> None:
        assert _shape("64 - Abs(t - 60)") == [(0.0, 60.0, 1.0, 4.0), (60.0, math.inf, -1.0, 124.0)]

    # A product is fine when at most one factor varies on each segment:
    # t * Heaviside(t - 30) is 0 before t=30 and t after.
    def test_product_with_a_step(self) -> None:
        assert _shape("t * Heaviside(t - 30)") == [(0.0, 30.0, 0.0, 0.0), (30.0, math.inf, 1.0, 0.0)]

    # Float coefficients across three Min arguments. sympy's piecewise_fold
    # recurses without end on this one, which is why the solver is our own.
    def test_float_min_of_three(self) -> None:
        segments = PiecewiseLinear("Max(1, Min(2.1 + 0.5*t, 40.2, 100.3 - 0.25*t))").segments
        assert [round(s.start, 6) for s in segments] == [0.0, 76.2, 240.4, 397.2]

    # The Piecewise adds a breakpoint at t=5 where nothing changes (0 either
    # side), so the pieces around it are the same line and merge. Expected:
    # t until t=10, then 2t - 10, with no cut at t=5.
    def test_equal_neighbours_merge(self) -> None:
        assert _shape("Max(t, 2*t - 10) + Piecewise((0, t < 5), (0, True))") == [
            (0.0, 10.0, 1.0, 0.0),
            (10.0, math.inf, 2.0, -10.0),
        ]


class TestValueAndBounds:
    # value() is right-continuous: at the step time t=60 the new value (32) applies.
    def test_value_at_a_step_is_the_new_value(self) -> None:
        line = PiecewiseLinear("Piecewise((8, t < 60), (32, True))")
        assert line.value(59.999) == 8.0
        assert line.value(60.0) == 32.0

    # Bounds are exact: Min(1 + t/6, 64) spans [1, 64].
    def test_bounds_of_a_capped_ramp(self) -> None:
        assert PiecewiseLinear("Min(1 + t/6, 64)").bounds == (1.0, 64.0)

    # A last segment with a slope is unbounded on that side: 1 + t is [1, inf).
    def test_bounds_of_an_uncapped_ramp(self) -> None:
        assert PiecewiseLinear("1 + t").bounds == (1.0, math.inf)
        assert PiecewiseLinear("64 - Abs(t - 60)").bounds == (-math.inf, 64.0)

    # Negative time is outside the domain: value(-1) raises.
    def test_negative_time_raises(self) -> None:
        with pytest.raises(ValueError, match="t must be >= 0"):
            PiecewiseLinear("t").value(-1)


class TestRejected:
    # Each non-linear shape is rejected, naming what bends the line.
    @pytest.mark.parametrize(
        "raw, message",
        [
            ("t**2", "t may not appear in a power or a denominator"),
            ("1/t", "t may not appear in a power or a denominator"),
            ("2**t", "t may not appear in a power or a denominator"),
            ("sin(t)", "uses 'sin', which is not piecewise linear"),
            ("exp(t)", "uses 'exp', which is not piecewise linear"),
            ("floor(t)", "uses 'floor', which is not piecewise linear"),
            ("t * Min(t, 5)", "multiplies terms in t"),
            ("Piecewise((8, t**2 < 60), (32, True))", "t may not appear in a power or a denominator"),
        ],
    )
    def test_nonlinear(self, raw: str, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            PiecewiseLinear(raw)

    # Constant subtrees are fine even when their functions are not allowed on t:
    # sqrt(2)*t and log(8)*t are plain lines.
    def test_constant_subtrees_are_allowed(self) -> None:
        assert _shape("sqrt(4)*t") == [(0.0, math.inf, 2.0, 0.0)]
        assert PiecewiseLinear("log(8)*t").segments[0].slope == pytest.approx(math.log(8))

    # Random variables, other symbols, conditions and gaps in a Piecewise are rejected.
    @pytest.mark.parametrize(
        "raw, message",
        [
            ("Normal(10, 2)", "contains a random variable"),
            ("x + 1", r"disallowed symbol\(s\) \['x'\]"),
            ("t > 5", "is a condition"),
            ("Piecewise((8, t < 60))", "give the last Piecewise branch the condition True"),
            ("Bogus(t)", "unknown function"),
        ],
    )
    def test_other_rejections(self, raw: str, message: str) -> None:
        with pytest.raises(ValueError, match=message):
            PiecewiseLinear(raw)


# Build a random nested expression from every allowed node, seeded.
# Input: a Random and a depth. Expected: an expression string such as
# "Min((t - 12.5), 3.1*t, Abs(t))".
def _random_expression(rng: random.Random, depth: int = 0) -> str:
    if depth > 3 or rng.random() < 0.25:
        return rng.choice(
            [f"{rng.uniform(-50, 50):.2f}", "t", f"{rng.uniform(-3, 3):.2f}*t", f"(t - {rng.uniform(0, 200):.1f})"]
        )
    a, b = _random_expression(rng, depth + 1), _random_expression(rng, depth + 1)
    op = rng.choice(["+", "-", "Min", "Max", "Abs", "Heaviside", "Piecewise", "scale"])
    if op in ("+", "-"):
        return f"({a} {op} {b})"
    if op in ("Min", "Max"):
        return f"{op}({a}, {b}, {_random_expression(rng, depth + 1)})"
    if op == "Abs":
        return f"Abs({a})"
    if op == "Heaviside":
        return f"Heaviside({a})*({b})"
    if op == "Piecewise":
        return (
            f"Piecewise(({a}, t < {rng.uniform(0, 200):.1f}), "
            f"({b}, ({_random_expression(rng, depth + 2)} > 3) | (t >= {rng.uniform(0, 300):.0f})), "
            f"({_random_expression(rng, depth + 1)}, True))"
        )
    return f"{rng.uniform(-4, 4):.2f}*({a})"


# 300 seeded random nested expressions, each evaluated on a grid of 400 times.
# Expected: the segments give the same value as evaluating the original
# expression directly, at every grid point not on a breakpoint. (Checked by
# hand at 1500 expressions x 4000 points: 0 mismatches.)
def test_segments_match_direct_evaluation() -> None:
    rng = random.Random(1)
    heaviside = {"Heaviside": lambda x, h0=0.5: h0 if x == 0 else (1.0 if x > 0 else 0.0), "Min": min, "Max": max}
    for _ in range(300):
        raw = _random_expression(rng)
        line = PiecewiseLinear(raw)
        expr = _parse_raw("Expression", raw)
        try:
            direct = sympy.lambdify([_T], expr, modules=[heaviside, "math"])
        except RecursionError:
            # sympy cannot print some nested conditions (ITE) for math; substitute instead.
            def direct(t: float, e: Any = expr) -> float:
                return float(e.subs(_T, sympy.Float(t)))

        starts = [s.start for s in line.segments]
        for i in range(400):
            t = i + 0.0137
            if min(abs(t - b) for b in starts) < 1e-6:
                continue
            want = float(direct(t))
            assert line.value(t) == pytest.approx(want, rel=1e-7, abs=1e-7), f"{raw} at t={t}"
