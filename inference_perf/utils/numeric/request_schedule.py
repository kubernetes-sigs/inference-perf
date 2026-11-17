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
the count is however many gaps fit in the dispatch window.
"""

from __future__ import annotations

import math
from typing import Optional

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
    ``t`` does. A random gap that varies with ``t`` is drawn one gap at a time,
    each from the distribution as it stands when the gap starts.
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
            self._offsets = self._draw_in_turn(generator) if request_interval.varies_with_time else self._draw(generator)
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

    def _draw_in_turn(self, rng: np.random.Generator) -> NDArray[np.float64]:
        """Offsets for a random gap that varies with ``t``: sample, wait, send.

        Each gap is drawn from the distribution as it stands at the stage time
        the gap starts, which is the previous arrival, or 0 for the first. The
        load therefore follows the curve one gap behind.
        """
        offsets: list[float] = []
        elapsed = 0.0
        stalled = 0
        while True:
            gap = float(self.request_interval.sample(elapsed, rng=rng))
            # t only moves when a gap is above zero, so a gap that is always
            # zero at this t would never end
            stalled = stalled + 1 if gap <= 0 else 0
            if stalled > _MAX_ZERO_GAPS:
                raise ValueError(
                    f"request_interval {self.request_interval.raw!r} keeps drawing a zero gap at t={elapsed:g}, "
                    "so the stage would never move on. A gap must be positive."
                )
            elapsed += gap
            if elapsed > self.duration:
                break
            offsets.append(elapsed)
            self._check_count(len(offsets))
        return np.asarray(offsets, dtype=np.float64)

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
