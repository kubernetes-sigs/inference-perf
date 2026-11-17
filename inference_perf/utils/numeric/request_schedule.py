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
"""Request arrival times from a request_interval expression.

A stage's ``request_interval`` is an :class:`Expression` for the gap, in seconds,
between one request and the next. :class:`RequestSchedule` works the arrival
times out once, so the request count, the data sample count and the dispatch
times all come from the same schedule.

A gap that is not random is the rate ``1 / request_interval``, and its requests
go out where that rate's area reaches 1, 2, 3 and so on. A random gap is drawn:
the count is however many gaps fit in the dispatch window. A random gap that
varies with ``t`` is one rule covering both: the randomness of each gap is drawn
once, and the request goes out where the area under ``1 / gap`` reaches 1, with
the gap read from the distribution as it stands at each moment.
"""

from __future__ import annotations

import math
from typing import Callable, Optional

import numpy as np
from numpy.typing import NDArray

from inference_perf.utils.numeric.expression import Expression
from inference_perf.utils.numeric.rate_schedule import RateSchedule, integration_grid

# Gaps are drawn in batches of this size until they cover the window.
_BATCH = 4096
# A cap on arrivals in one stage, so a gap that is (almost) always zero fails
# with an error instead of allocating without bound.
_MAX_ARRIVALS = 10_000_000
# How many zero gaps in a row a time-varying draw may produce before it is
# taken to be stuck.
_MAX_ZERO_GAPS = 1000
# The longest stretch of stage time integrated in one step, in seconds. The
# rate grid's spacing, so a step in the curve is placed as finely here as there.
_MAX_STEP = 0.01
# Below this much stage time left, the window is over.
_WINDOW_END = 1e-12


def _progress_rate(gap: float, raw: object, t: float) -> float:
    """How fast a gap is used up, ``1 / gap``: 0 for an infinite gap, ``inf`` for a zero one."""
    if gap != gap or gap < 0:
        raise ValueError(f"request_interval {raw!r} is {gap} at t={t:g}. A gap must be nonnegative.")
    if gap == 0:
        return math.inf
    return 1.0 / gap


def _cross_within(step: float, r0: float, r1: float, r2: float, need: float) -> float:
    """How far into a step the area under the rate reaches ``need``.

    The rate is ``r0`` at the start of the step, ``r1`` halfway and ``r2`` at
    its end, and is taken to be the parabola through the three, the same one
    Simpson's rule integrates. The area over the whole step is at least
    ``need``. Newton's method, kept inside a bracket that halves when it
    would leave it.
    """
    b = (-3.0 * r0 + 4.0 * r1 - r2) / step
    c = 2.0 * (r0 - 2.0 * r1 + r2) / (step * step)
    whole = step * (r0 + 4.0 * r1 + r2) / 6.0
    low, high = 0.0, step
    x = step * need / whole if whole > 0 else step
    for _ in range(100):
        area = x * (r0 + x * (b / 2.0 + x * c / 3.0))
        if area > need:
            high = x
        else:
            low = x
        rate = r0 + x * (b + x * c)
        following = x - (area - need) / rate if rate > 0 else low
        if not low < following < high:
            following = (low + high) / 2.0
        if abs(following - x) <= 1e-15 * step:
            return following
        x = following
    return x


def _next_arrival(gap_at: Callable[[float], float], raw: object, start: float, end: float) -> Optional[float]:
    """The stage time at which the area under ``1 / gap_at`` from ``start`` reaches 1, or None if not by ``end``.

    Integrated with Simpson's rule in steps no longer than ``_MAX_STEP`` and no
    longer than the gap would last at the rate the step starts with. The gap
    is never read past ``end``, where its parameters are not known to be valid.
    """
    t = start
    area = 0.0
    r0 = _progress_rate(gap_at(t), raw, t)
    while True:
        if r0 == math.inf:
            # a zero gap: the request is due now
            return t
        left = end - t
        if left <= _WINDOW_END:
            return None
        need = 1.0 - area
        step = min(_MAX_STEP, left)
        if r0 > 0:
            step = min(step, need / r0)
        r1 = _progress_rate(gap_at(t + step / 2.0), raw, t + step / 2.0)
        r2 = _progress_rate(gap_at(t + step), raw, t + step)
        if r1 == math.inf or r2 == math.inf:
            # the gap reaches zero inside this step, so the area reaches 1 inside it too. close in on where
            if step <= _WINDOW_END:
                return t + step
            half = step / 2.0
            if r1 == math.inf:
                # between the start and halfway: look at the first half only
                r2, step = r1, half
                r1 = _progress_rate(gap_at(t + step / 2.0), raw, t + step / 2.0)
            while (r1 == math.inf or r2 == math.inf) and step > _WINDOW_END:
                step /= 2.0
                r1 = _progress_rate(gap_at(t + step / 2.0), raw, t + step / 2.0)
                r2 = _progress_rate(gap_at(t + step), raw, t + step)
            if r1 == math.inf or r2 == math.inf:
                return t + step
        gained = step * (r0 + 4.0 * r1 + r2) / 6.0
        if area + gained >= 1.0:
            return t + _cross_within(step, r0, r1, r2, need)
        area += gained
        t += step
        r0 = r2


def arrivals_along_the_curve(request_interval: Expression, duration: float, rng: np.random.Generator) -> NDArray[np.float64]:
    """Arrival offsets for a gap that is random and varies with stage time ``t``.

    The randomness of each gap is drawn once, with :meth:`Expression.sample_base`.
    The gap that draw stands for is then read at every moment from the
    distribution as it is at that moment, and the request goes out when the
    area under ``1 / gap`` since the request before it reaches 1. So a gap is
    used up faster while the rate is high and slower while it is low.

    What this is, case by case:

    * ``Exponential(r(t))``: a Poisson process of rate ``r(t)``. The area
      under ``r`` between two arrivals is an ``Exponential(1)`` draw, which
      is the definition.
    * a distribution whose mean alone moves, such as ``Gamma(4, s(t))``: a
      renewal process run on a clock that advances at ``1 / s(t)``.
    * constant parameters: the gaps are the draws themselves.
    * no randomness: request k where the area under ``1 / gap`` reaches k,
      which is how a rate expression is placed.

    The first gap starts when the window opens. A gap still open when the
    window closes ends the stage.
    """
    raw = request_interval.raw
    offsets: list[float] = []
    elapsed = 0.0
    stalled = 0
    while True:
        for base in request_interval.sample_base(rng, _BATCH).tolist():

            def gap_at(t: float, base: list[float] = base) -> float:
                return request_interval.value_given(base, t)

            arrival = _next_arrival(gap_at, raw, elapsed, duration)
            if arrival is None:
                return np.asarray(offsets, dtype=np.float64)
            # t only moves when a gap is above zero, so a gap that is always
            # zero at this t would never end
            stalled = stalled + 1 if arrival <= elapsed else 0
            if stalled > _MAX_ZERO_GAPS:
                raise ValueError(
                    f"request_interval {raw!r} keeps drawing a zero gap at t={elapsed:g}, "
                    "so the stage would never move on. A gap must be positive."
                )
            elapsed = arrival
            offsets.append(elapsed)
            if len(offsets) > _MAX_ARRIVALS:
                raise ValueError(
                    f"request_interval {raw!r} schedules more than {_MAX_ARRIVALS} requests "
                    f"in {duration:g}s; the gaps are too small."
                )


class RequestSchedule:
    """The arrival offsets of one stage, worked out from its request_interval expression.

    Args:
        request_interval: Seconds between consecutive requests. May be random and
            may vary with stage time ``t``; must be nonnegative.
        duration: Length of the dispatch window in seconds.
        rng: numpy Generator for the draws; a fresh one when omitted.

    The first request is due one gap after the window opens, and requests are
    kept while their offset is at most ``duration``. A constant gap takes a
    closed form (``duration / gap`` requests, truncated), so ``request_interval:
    0.1`` over 60s is exactly 600 evenly spaced requests. A gap that varies with
    ``t`` and is not random is scheduled as the rate ``1 / gap``: request k goes
    out when the area under that rate reaches k, so ``"1/t"`` sends as the rate
    ``t`` does. A random gap that varies with ``t`` is placed by
    :func:`arrivals_along_the_curve`.
    """

    def __init__(self, request_interval: Expression, duration: float, rng: Optional[np.random.Generator] = None) -> None:
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}.")
        self.request_interval = request_interval
        self.duration = float(duration)
        # set when the gap is infinite for the whole window, which is a zero rate
        self._never_sends = False
        if request_interval.is_random:
            generator = rng if rng is not None else np.random.default_rng()
            self._offsets = (
                self._draw_along_the_curve(generator) if request_interval.varies_with_time else self._draw(generator)
            )
        elif request_interval.is_constant:
            gap = float(request_interval.sample())
            if gap <= 0:
                raise ValueError(f"request_interval {request_interval.raw!r} must be positive.")
            self._never_sends = math.isinf(gap)
            # The same float tolerance as RateSchedule.expected_requests: 1 / 0.1
            # must count 10 gaps, not 9.
            count = math.floor(self.duration / gap + 1e-9)
            self._check_count(count)
            self._offsets = gap * np.arange(1, count + 1, dtype=np.float64)
        else:
            self._offsets = self._along_the_rate()

    def _along_the_rate(self) -> NDArray[np.float64]:
        """Offsets for a gap that varies with ``t`` and is not random: request k where the rate 1/gap has area k."""
        raw = self.request_interval.raw
        rate = Expression(f"1/({raw})", allow_random=False, duration=self.duration)
        # an infinite gap divides to a rate of zero, which is legal, so numpy's
        # warnings about it are not wanted
        with np.errstate(divide="ignore", invalid="ignore"):
            times = integration_grid(self.duration)
            values = rate.evaluate(times)
            # checked on the integration grid, so a gap that touches zero only
            # between two grid times is not seen
            for wrong, what in ((~np.isfinite(values), "zero"), (values < 0, "negative")):
                if bool(np.any(wrong)):
                    at = float(times[int(np.argmax(wrong))])
                    raise ValueError(
                        f"request_interval {raw!r} is {what} at t={at:g}. A gap must be positive: a zero gap is an infinite rate."
                    )
            schedule = RateSchedule(rate, self.duration)
        self._never_sends = schedule.total <= 0
        self._check_count(schedule.expected_requests)
        # the same placement a rate expression gets, so "1/t" sends when the rate "t" does
        return schedule.send_times()

    def _check_count(self, count: int) -> None:
        if count > _MAX_ARRIVALS:
            raise ValueError(
                f"request_interval {self.request_interval.raw!r} schedules more than {_MAX_ARRIVALS} requests "
                f"in {self.duration:g}s; the gaps are too small."
            )

    def _draw(self, rng: np.random.Generator) -> NDArray[np.float64]:
        batches: list[NDArray[np.float64]] = []
        elapsed = 0.0
        count = 0
        while elapsed <= self.duration:
            gaps = np.asarray(self.request_interval.sample(rng=rng, size=_BATCH), dtype=np.float64)
            batches.append(elapsed + np.cumsum(gaps))
            elapsed = float(batches[-1][-1])
            count += _BATCH
            self._check_count(count)
        offsets = np.concatenate(batches)
        return np.asarray(offsets[offsets <= self.duration], dtype=np.float64)

    def _draw_along_the_curve(self, rng: np.random.Generator) -> NDArray[np.float64]:
        offsets = arrivals_along_the_curve(self.request_interval, self.duration, rng)
        self._check_count(len(offsets))
        return offsets

    @property
    def zero_requests_reason(self) -> Optional[str]:
        """Why the stage sends zero requests, for a warning, or None when it sends some."""
        if len(self._offsets) > 0:
            return None
        if self._never_sends:
            return f"its request_interval {self.request_interval.raw!r} is infinite for the whole stage"
        return f"no gap of its request_interval {self.request_interval.raw!r} fits in the stage"

    @property
    def offsets(self) -> NDArray[np.float64]:
        """Seconds from the window's start at which each request is due, ascending."""
        return self._offsets

    @property
    def expected_requests(self) -> int:
        """Requests the stage dispatches: the arrivals that fit in the window."""
        return len(self._offsets)

    @property
    def mean_rate(self) -> float:
        """Requests per second this draw works out to over the window."""
        return len(self._offsets) / self.duration
