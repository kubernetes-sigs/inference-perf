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
"""Tests for ``ConcurrencySchedule``: an integer level that follows an expression of t.

The level is the expression rounded down, it changes only at exact times, and
the expression must stay at least 1 and bounded for every t >= 0.
"""

import math
import random
from typing import List, Optional, Tuple, Union

import pytest

from inference_perf.utils.numeric.concurrency_schedule import ConcurrencySchedule


# Every (time, level) change from t=0 up to `until`, starting with (0, initial).
# Input: "Piecewise((8, t < 60), (32, True))", until 100.
# Expected: [(0, 8), (60, 32)]. Fails, rather than hangs, past 10,000 changes.
def _changes(raw: Union[int, str], until: float) -> List[Tuple[float, int]]:
    schedule = ConcurrencySchedule(raw)
    out = [(0.0, schedule.initial)]
    t: Optional[float] = 0.0
    for _ in range(10_000):
        t = schedule.next_change_after(out[-1][0])
        if t is None or t > until:
            return out
        out.append((t, schedule.level_at(t)))
    raise AssertionError(f"{raw}: more than 10,000 changes before t={until}")


class TestLevels:
    # An integer never changes: 8 -> initial 8, peak 8, no change ever.
    def test_integer(self) -> None:
        schedule = ConcurrencySchedule(8)
        assert (schedule.initial, schedule.peak) == (8, 8)
        assert schedule.is_constant
        assert schedule.next_change_after(0.0) is None

    # A step changes once, at exactly t=60: 8 then 32.
    def test_step(self) -> None:
        assert _changes("Piecewise((8, t < 60), (32, True))", 100) == [(0.0, 8), (60.0, 32)]

    # A ramp is rounded down: 1 + t/2 is level 1 until t=2, 2 until t=4, ...
    # and a cap of 4 means the last change is to 4 at t=6.
    def test_rising_ramp_rounds_down(self) -> None:
        assert _changes("Min(1 + t/2, 4)", 100) == [(0.0, 1), (2.0, 2), (4.0, 3), (6.0, 4)]

    # Falling, the level drops the moment the value goes below an integer:
    # 4.5 - t/2 is 4 until t=1, then 3 from t=1 (value 4 exactly, and below
    # straight after), 2 from t=3, 1 from t=5, and stays 1 (floored by Max).
    def test_falling_ramp_drops_as_the_value_passes_each_integer(self) -> None:
        assert _changes("Max(1, 4.5 - t/2)", 100) == [(0.0, 4), (1.0, 3), (3.0, 2), (5.0, 1)]

    # A non-integer cap: Min(1 + t, 2.5) peaks at level 2 (2.5 rounded down).
    def test_peak_is_rounded_down(self) -> None:
        assert ConcurrencySchedule("Min(1 + t, 2.5)").peak == 2

    # Float error must not cost a whole level: 0.7 + 0.1*t reaches 4 at t=33,
    # but in floats the crossing is t=32.99999999999999, where the value is
    # 3.999999999999999. (Max(1, ...) only keeps it valid before t=3.)
    # Expected: the change found there is to level 4, not a change that
    # leaves the level at 3.
    def test_float_error_does_not_lose_a_level(self) -> None:
        schedule = ConcurrencySchedule("Max(1, Min(0.7 + 0.1*t, 40))")
        change = schedule.next_change_after(32.5)
        assert change is not None and change == pytest.approx(33.0)
        assert schedule.level_at(change) == 4

    # A level can pass straight through integers at a jump: 1 then 10 at t=5
    # is a single change, not nine.
    def test_jump_is_one_change(self) -> None:
        assert _changes("1 + 9*Heaviside(t - 5)", 100) == [(0.0, 1), (5.0, 10)]


class TestRejected:
    # An uncapped ramp grows forever. Expected: rejected, suggesting a Min cap.
    def test_unbounded(self) -> None:
        with pytest.raises(ValueError, match=r"grows without limit as t increases; cap it, e.g. 'Min\(1 \+ t, 64\)'"):
            ConcurrencySchedule("1 + t")

    # Values below 1 are rejected, naming the first t it happens at.
    @pytest.mark.parametrize(
        "raw, at",
        [
            ("0", "t=0"),
            ("Min(t, 10)", "t=0"),
            ("Max(0.5, 10 - t)", "t=9"),
            ("Piecewise((4, t < 30), (0, True))", "t=30"),
        ],
    )
    def test_below_one(self, raw: str, at: str) -> None:
        with pytest.raises(ValueError, match=f"falls below 1 at {at};"):
            ConcurrencySchedule(raw)

    # Non-linear shapes are rejected by PiecewiseLinear with its own message.
    def test_nonlinear(self) -> None:
        with pytest.raises(ValueError, match="not piecewise linear"):
            ConcurrencySchedule("Min(1 + t**2, 64)")


# 200 seeded capped ramps (up, hold, down), each checked on a 0.05s grid.
# Expected: the level at every grid point equals the expression rounded down,
# and walking next_change_after visits every change with no spurious ones.
def test_levels_match_rounding_down_everywhere() -> None:
    rng = random.Random(0)
    for _ in range(200):
        # Rounded first, so the string and the reference use the same numbers.
        start, slope, cap = round(rng.uniform(1, 5), 3), round(rng.uniform(0.05, 3), 3), round(rng.uniform(10, 80), 3)
        top, down = round(cap + rng.uniform(5, 200), 3), round(slope / 2, 3)
        raw = f"Max(1, Min({start} + {slope}*t, {cap}, {top} - {down}*t))"
        schedule = ConcurrencySchedule(raw)
        changes = [t for t, _ in _changes(raw, 400)][1:]
        for i in range(8000):
            t = i * 0.05
            if any(abs(t - c) < 1e-6 for c in changes):
                continue
            value = max(1.0, min(start + slope * t, cap, top - down * t))
            # Between changes the level is the value rounded down.
            assert schedule.level_at(t) == math.floor(value + 1e-9), f"{raw} at t={t}"
        levels = [level for _, level in _changes(raw, 400)]
        assert all(a != b for a, b in zip(levels, levels[1:], strict=False)), f"{raw}: a change that changes nothing"
