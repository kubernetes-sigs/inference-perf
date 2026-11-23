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


def random_rate_error(raw: Union[str, int, float]) -> str:
    """The message for a rate that draws from a distribution."""
    return (
        f"Rate {raw!r} is random. A rate is never random. For random arrivals set request_interval instead, "
        'such as request_interval: "Exponential(10)".'
    )


# Grid spacing for integrating a time-varying rate, in seconds, and a cap on the
# number of grid points so a very long stage can't allocate without bound.
_GRID_STEP = 0.01
_MAX_GRID_POINTS = 1_000_000


def integration_grid(duration: float) -> NDArray[np.float64]:
    """The stage times a time-varying rate is evaluated at, from 0 to ``duration`` inclusive."""
    points = min(max(math.ceil(duration / _GRID_STEP), 1) + 1, _MAX_GRID_POINTS)
    return np.linspace(0.0, duration, points)


class RateSchedule:
    """Cumulative request intensity of a rate expression over ``[0, duration]``.

    Args:
        rate: Requests per second, which may vary with stage time ``t``. Never
            random, and nonnegative over the window.
        duration: Length of the dispatch window in seconds.

    A time-varying schedule covers the window and nothing past it: a time
    after ``duration`` reads as ``duration``, and a count above the total
    reads as the total.
    """

    def __init__(self, rate: Expression, duration: float) -> None:
        if rate.is_random:
            raise ValueError(random_rate_error(rate.raw))
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

        self._grid = integration_grid(self.duration)
        values = rate.evaluate(self._grid)
        if bool(np.any(values < 0)):
            index = int(np.argmax(values < 0))
            raise ValueError(f"Rate {rate.raw!r} is negative at t={float(self._grid[index]):g}.")
        # Cumulative trapezoid: requests expected by each grid time.
        steps = np.diff(self._grid) * (values[1:] + values[:-1]) / 2.0
        self._cumulative = np.concatenate(([0.0], np.cumsum(steps)))
        self._total = float(self._cumulative[-1])

    @property
    def zero_requests_reason(self) -> Union[str, None]:
        """Why the stage sends zero requests, for a warning, or None when it sends some."""
        if self.expected_requests > 0:
            return None
        if self._total <= 0:
            return f"its rate {self.rate.raw!r} is zero for the whole stage"
        return (
            f"its rate {self.rate.raw!r} adds up to {self._total:.3g} requests over {self.duration:g}s, "
            "which is less than one request"
        )

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

    @property
    def is_expression(self) -> bool:
        """True when the rate was written as an expression string, even a constant one such as ``"10"``."""
        return isinstance(self.rate.raw, str)

    def send_times(self) -> NDArray[np.float64]:
        """Stage times of the stage's requests with no jitter: request k where the area under the rate reaches k.

        One time per expected request. The last is held to the end of the
        window, where rounding could otherwise put it a hair past.
        """
        count = self.expected_requests
        if count == 0:
            return np.empty(0, dtype=np.float64)
        times = self.inverse(np.arange(1, count + 1, dtype=np.float64))
        return np.asarray(np.minimum(times, self.duration), dtype=np.float64)

    def cumulative(self, t: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Expected requests dispatched by stage time ``t``."""
        times = np.asarray(t, dtype=np.float64)
        if self._constant is not None:
            return times * self._constant
        return np.asarray(np.interp(times, self._grid, self._cumulative), dtype=np.float64)

    def inverse(self, u: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Stage time by which ``u`` requests are expected: the inverse of :meth:`cumulative`."""
        counts = np.asarray(u, dtype=np.float64)
        if self._constant is not None:
            if self._constant == 0:
                raise ValueError("A zero rate never reaches any request count.")
            return counts / self._constant
        return np.asarray(np.interp(counts, self._cumulative, self._grid), dtype=np.float64)
