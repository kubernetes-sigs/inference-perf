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

The expression is rounded down: the level in effect at ``t`` is
``floor(value(t))``, so a stage never runs more requests at once than the
expression asks for. Because the expression is piecewise linear, every time
the level changes is exact (a line crosses an integer at one known ``t``), so
the load generator can sleep until the next change instead of polling.
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


class ConcurrencySchedule:
    """A concurrency level over stage time ``t``, from a number or an expression.

    Args:
        raw: An integer, or an expression string such as ``"Min(1 + t/6, 64)"``.
            An expression must be piecewise linear in ``t`` (see
            :class:`PiecewiseLinear`), at least 1 at every ``t >= 0``, and
            bounded above. A concurrent stage has no fixed duration, so the
            bounds are proved over all ``t >= 0``; a ramp that would keep
            growing must be capped, e.g. with ``Min(..., 64)``.

    Raises:
        ValueError: If the expression is not piecewise linear, can fall below
            1, or is unbounded above.
    """

    def __init__(self, raw: Union[int, str]) -> None:
        self.raw = raw
        self._line = PiecewiseLinear(raw)
        lower, upper = self._line.bounds
        if upper == float("inf"):
            raise ValueError(f"Concurrency {raw!r} grows without limit as t increases; cap it, e.g. 'Min({raw}, 64)'.")
        if lower + _slack(lower) < 1:
            below = self._first_time_below_one()
            raise ValueError(
                f"Concurrency {raw!r} falls below 1 at t={below:g}; it must be at least 1 at every t, e.g. 'Max(1, {raw})'."
            )
        self.initial = self.level_at(0.0)
        self.peak = int(math.floor(upper + _slack(upper)))

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
        if segment.slope < 0:
            return int(math.ceil(value - _slack(value))) - 1
        return int(math.floor(value + _slack(value)))

    def next_change_after(self, t: float) -> Optional[float]:
        """The first time after ``t`` at which :meth:`level_at` changes, or ``None`` if it never does."""
        current = self.level_at(t)
        segment: Optional[Segment] = self._line.segment_at(t)
        while segment is not None:
            crossing = _crossing(segment, current, t)
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
        for segment in self._line.segments:
            if segment.value(segment.start) < 1:
                return segment.start
            if segment.slope < 0:
                crossing = (1 - segment.intercept) / segment.slope
                if crossing < segment.end:
                    return crossing
        return 0.0

    def __repr__(self) -> str:
        return f"ConcurrencySchedule({self.raw!r})"


def _crossing(segment: Segment, level: int, after: float) -> Optional[float]:
    """When ``segment`` next moves the level off ``level``, strictly after ``after``."""
    if segment.slope > 0:
        # Rising: the level becomes level + 1 when the value reaches it.
        target = level + 1
    elif segment.slope < 0:
        # Falling: the level becomes level - 1 once the value drops below level.
        target = level
    else:
        return None
    crossing = (target - segment.intercept) / segment.slope
    return crossing if crossing > after else None
