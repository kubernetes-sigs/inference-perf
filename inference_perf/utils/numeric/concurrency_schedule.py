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
"""An integer concurrency level that follows a piecewise-linear expression of ``t``.

The expression says how it becomes a whole number: ``floor(...)`` or
``ceiling(...)`` around a piecewise-linear line, or a step function whose steps
are whole numbers already. ``floor(... + 1/2)`` rounds to the nearest. Because
the line is piecewise linear, every time the level changes is exact (a line
crosses an integer at one known ``t``), so the load generator can sleep until
the next change instead of polling.
"""

from __future__ import annotations

import math
from typing import Optional, Union

from inference_perf.utils.numeric.expression.piecewise_linear import PiecewiseLinear, Segment

# Relative slack when rounding down, so a value that is an integer up to float
# error (9.999999999 for 10) does not lose a whole unit of concurrency.
_TOLERANCE = 1e-9


def _slack(value: float) -> float:
    return _TOLERANCE * max(1.0, abs(value))


def _split_rounding(raw: Union[int, str]) -> tuple[Union[int, str], Optional[str]]:
    """The line under a top-level floor() or ceiling(), and which one; the whole value and None otherwise.

    Read from the text, not the parsed tree: sympy rewrites floor(1 + t) as
    floor(t) + 1, which would lose the wrapper.
    """
    if isinstance(raw, int):
        return raw, None
    text = raw.strip()
    for rounding in ("floor", "ceiling"):
        if text.startswith(f"{rounding}(") and text.endswith(")") and _closes_at_end(text, len(rounding)):
            return text[len(rounding) + 1 : -1].strip(), rounding
    return raw, None


# True when the parenthesis opened at text[open_at] is the one closed by the last character
def _closes_at_end(text: str, open_at: int) -> bool:
    depth = 0
    for i in range(open_at, len(text)):
        depth += {"(": 1, ")": -1}.get(text[i], 0)
        if depth == 0:
            return i == len(text) - 1
    return False


class ConcurrencySchedule:
    """A concurrency level over stage time ``t``, from a number or an expression.

    Args:
        raw: An integer, or an expression string such as ``"floor(Min(1 + t/6, 64))"``.
            The line under ``floor``/``ceiling`` must be piecewise linear in
            ``t`` (see :class:`PiecewiseLinear`); without one, the expression
            must be a step function of whole numbers. The level must be at
            least 1 at every ``t >= 0`` and bounded above. A concurrent stage
            has no fixed duration, so the bounds are proved over all
            ``t >= 0``; a ramp that would keep growing must be capped, e.g.
            with ``Min(..., 64)``.

    Raises:
        ValueError: If the expression is not piecewise linear, takes values
            that are not whole numbers without saying how to round them, can
            fall below 1, or is unbounded above.
    """

    def __init__(self, raw: Union[int, str]) -> None:
        self.raw = raw
        line, self._rounding = _split_rounding(raw)
        self._ceiling = self._rounding == "ceiling"
        self._line = PiecewiseLinear(line)
        if self._rounding is None and not all(s.slope == 0 and float(s.intercept).is_integer() for s in self._line.segments):
            raise ValueError(
                f"Concurrency {raw!r} takes values that are not whole numbers. Write 'floor({raw})' to round down,"
                f" 'floor({raw} + 1/2)' to round to the nearest, or 'ceiling({raw})' to round up."
            )
        lower, upper = self._line.bounds
        if upper == float("inf"):
            raise ValueError(
                f"Concurrency {raw!r} grows without limit as t increases; cap it, e.g. '{self._rounded(f'Min({line}, 64)')}'."
            )
        if self._level_of(lower) < 1:
            below = self._first_time_below_one()
            raise ValueError(
                f"Concurrency {raw!r} falls below 1 at t={below:g}; it must be at least 1 at every t,"
                f" e.g. '{self._rounded(f'Max(1, {line})')}'."
            )
        self.initial = self.level_at(0.0)
        self.peak = self._level_of(upper)

    def _rounded(self, line: str) -> str:
        return line if self._rounding is None else f"{self._rounding}({line})"

    def _level_of(self, value: float) -> int:
        if self._ceiling:
            return int(math.ceil(value - _slack(value)))
        return int(math.floor(value + _slack(value)))

    @property
    def is_constant(self) -> bool:
        """True when the level never changes."""
        return self.next_change_after(0.0) is None

    def level_at(self, t: float) -> int:
        """The level in effect from ``t`` until the next change.

        On a falling segment the value reaches an integer ``k`` at one instant
        and is below it straight after, so the level from that instant on is
        ``k - 1``. Taking the right-hand limit this way makes the level at a
        change time the level the change switches to.
        """
        segment = self._line.segment_at(t)
        value = segment.value(t)
        if self._ceiling:
            # mirror image of the floor case: rising off an integer k, the level is k + 1 straight after
            if segment.slope > 0:
                return int(math.floor(value + _slack(value))) + 1
            return int(math.ceil(value - _slack(value)))
        if segment.slope < 0:
            return int(math.ceil(value - _slack(value))) - 1
        return int(math.floor(value + _slack(value)))

    def next_change_after(self, t: float) -> Optional[float]:
        """The first time after ``t`` at which :meth:`level_at` changes, or ``None`` if it never does."""
        current = self.level_at(t)
        segment: Optional[Segment] = self._line.segment_at(t)
        while segment is not None:
            crossing = _crossing(segment, current, t, self._ceiling)
            if crossing is not None and crossing < segment.end:
                return crossing
            if segment.end == float("inf"):
                return None
            if self.level_at(segment.end) != current:
                return segment.end
            t = segment.end
            segment = self._line.segment_at(t)
        return None

    def _first_time_below_one(self) -> float:
        # the level is below 1 once the line is below 1 (floor) or at most 0 (ceiling)
        threshold = 0.0 if self._ceiling else 1.0
        for segment in self._line.segments:
            if self._level_of(segment.value(segment.start)) < 1:
                return segment.start
            if segment.slope < 0:
                crossing = (threshold - segment.intercept) / segment.slope
                if crossing < segment.end:
                    return crossing
        return 0.0

    def __repr__(self) -> str:
        return f"ConcurrencySchedule({self.raw!r})"


def _crossing(segment: Segment, level: int, after: float, ceiling: bool) -> Optional[float]:
    """When ``segment`` next moves the level off ``level``, strictly after ``after``."""
    if segment.slope > 0:
        # Rising: a floor becomes level + 1 when the value reaches it, a ceiling once the value passes level.
        target = level if ceiling else level + 1
    elif segment.slope < 0:
        # Falling: a floor becomes level - 1 once the value drops below level, a ceiling when it reaches level - 1.
        target = level - 1 if ceiling else level
    else:
        return None
    crossing = (target - segment.intercept) / segment.slope
    return crossing if crossing > after else None
