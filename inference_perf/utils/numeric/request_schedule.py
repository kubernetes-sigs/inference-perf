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
"""Request arrival times drawn from a request_interval expression.

A stage's ``request_interval`` is an :class:`Expression` for the gap, in seconds,
between one request and the next. Unlike a rate, the number of requests is not
known from the config: it is however many gaps fit in the dispatch window.
:class:`RequestSchedule` draws the gaps once, so the request count, the data
sample count and the dispatch times all come from the same draw.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np
from numpy.typing import NDArray

from inference_perf.utils.numeric.expression import Expression

# Gaps are drawn in batches of this size until they cover the window.
_BATCH = 4096
# A cap on arrivals in one stage, so a gap that is (almost) always zero fails
# with an error instead of allocating without bound.
_MAX_ARRIVALS = 10_000_000


class RequestSchedule:
    """The arrival offsets of one stage, drawn from its request_interval expression.

    Args:
        request_interval: Seconds between consecutive requests. May be random; must
            be nonnegative.
        duration: Length of the dispatch window in seconds.
        rng: numpy Generator for the draws; a fresh one when omitted.

    The first request is due one gap after the window opens, and requests are
    kept while their offset is at most ``duration``. A constant gap takes a
    closed form (``duration / gap`` requests, truncated), so ``request_interval:
    0.1`` over 60s is exactly 600 evenly spaced requests.
    """

    def __init__(self, request_interval: Expression, duration: float, rng: Optional[np.random.Generator] = None) -> None:
        if duration <= 0:
            raise ValueError(f"duration must be positive, got {duration}.")
        self.request_interval = request_interval
        self.duration = float(duration)
        if request_interval.is_random:
            self._offsets = self._draw(rng if rng is not None else np.random.default_rng())
        else:
            gap = float(request_interval.sample())
            if gap <= 0:
                raise ValueError(f"request_interval {request_interval.raw!r} must be positive.")
            # The same float tolerance as RateSchedule.expected_requests: 1 / 0.1
            # must count 10 gaps, not 9.
            count = math.floor(self.duration / gap + 1e-9)
            self._check_count(count)
            self._offsets = gap * np.arange(1, count + 1, dtype=np.float64)

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
