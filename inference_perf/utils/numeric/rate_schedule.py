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
"""A request rate over stage time, integrated so load timers can schedule against it.

A stage's rate is an :class:`Expression` in stage time ``t``. The load timers
need two things from it: how many requests the stage expects (the integral of
the rate over the dispatch window) and when the k-th request is due (the
inverse of that integral). :class:`RateSchedule` precomputes both once per
stage. A constant rate takes an exact closed form, so constant stages schedule
exactly as they did before rates could vary.
"""

from __future__ import annotations

import math
from typing import Union

import numpy as np
from numpy.typing import NDArray

from inference_perf.utils.numeric.expression import Expression

# Grid spacing for integrating a time-varying rate, in seconds, and a cap on the
# number of grid points so a very long stage can't allocate without bound.
_GRID_STEP = 0.01
_MAX_GRID_POINTS = 1_000_000


class RateSchedule:
    """Cumulative request intensity of a rate expression over ``[0, duration]``.

    Args:
        rate: Requests per second as a function of stage time ``t``. Must be
            deterministic and nonnegative over the window.
        duration: Length of the dispatch window in seconds.

    Past ``duration`` (the Poisson timer runs over its window whenever its
    random per-second counts come in under the expected total), the rate is
    held at the window's mean rate. Holding the final value instead would hang
    a rate that ramps down to zero: the overrun would never reach the count.
    """

    def __init__(self, rate: Expression, duration: float) -> None:
        if rate.is_random:
            raise ValueError(f"Rate {rate.raw!r} is random; a rate must be a deterministic function of t.")
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}.")
        self.rate = rate
        self.duration = float(duration)
        self._constant: Union[float, None] = float(rate.evaluate(0.0)) if rate.is_constant else None
        if self._constant is not None:
            if self._constant < 0:
                raise ValueError(f"Rate {rate.raw!r} is negative.")
            self._total = self._constant * self.duration
            return

        points = min(max(math.ceil(self.duration / _GRID_STEP), 1) + 1, _MAX_GRID_POINTS)
        self._grid = np.linspace(0.0, self.duration, points)
        values = rate.evaluate(self._grid)
        if bool(np.any(values < 0)):
            index = int(np.argmax(values < 0))
            raise ValueError(f"Rate {rate.raw!r} is negative at t={float(self._grid[index]):g}.")
        # Cumulative trapezoid: requests expected by each grid time.
        steps = np.diff(self._grid) * (values[1:] + values[:-1]) / 2.0
        self._cumulative = np.concatenate(([0.0], np.cumsum(steps)))
        self._total = float(self._cumulative[-1])

    @property
    def is_constant(self) -> bool:
        return self._constant is not None

    @property
    def constant_rate(self) -> float:
        """The rate of a constant schedule. Raises for a time-varying one."""
        if self._constant is None:
            raise ValueError(f"Rate {self.rate.raw!r} varies with t.")
        return self._constant

    @property
    def total(self) -> float:
        """Expected requests over the window: the integral of the rate."""
        return self._total

    @property
    def expected_requests(self) -> int:
        """Requests a stage dispatches: the integral, truncated like ``int(rate * duration)``.

        A numerically integrated total that lands a hair under a whole number
        (49.99999999 for a ramp whose exact integral is 50) counts as that
        number. A constant rate keeps the plain truncation it always had.
        """
        if self._constant is not None:
            return int(self._total)
        return math.floor(self._total + 1e-9 * max(1.0, self._total))

    @property
    def mean_rate(self) -> float:
        """Average requests per second over the window, the figure reports carry."""
        return self._total / self.duration

    def cumulative(self, t: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Expected requests dispatched by stage time ``t``."""
        times = np.asarray(t, dtype=np.float64)
        if self._constant is not None:
            return times * self._constant
        inside = np.interp(np.minimum(times, self.duration), self._grid, self._cumulative)
        overrun = np.maximum(times - self.duration, 0.0) * self.mean_rate
        return np.asarray(inside + overrun, dtype=np.float64)

    def inverse(self, u: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Stage time by which ``u`` requests are expected: the inverse of :meth:`cumulative`."""
        counts = np.asarray(u, dtype=np.float64)
        if self._constant is not None:
            if self._constant == 0:
                raise ValueError("A zero rate never reaches any request count.")
            return counts / self._constant
        inside = np.interp(np.minimum(counts, self._total), self._cumulative, self._grid)
        beyond = np.maximum(counts - self._total, 0.0)
        if bool(np.any(beyond > 0)) and self._total <= 0:
            raise ValueError(f"Rate {self.rate.raw!r} is zero over the whole window, so it never reaches a request.")
        extra = beyond / self.mean_rate if self._total > 0 else np.zeros_like(beyond)
        return np.asarray(inside + extra, dtype=np.float64)
