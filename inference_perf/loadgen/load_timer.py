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
import time
from abc import ABC, abstractmethod
from typing import Generator, Optional, Tuple, Union
import numpy as np
from inference_perf.utils.numeric.rate_schedule import RateSchedule
from inference_perf.utils.trace_reader import TraceReader
from pathlib import Path


class LoadTimer(ABC):
    """Abstract base class for load generators."""

    @abstractmethod
    def __init__(self, *args: Tuple[int, ...]) -> None:
        # TODO: Commmon functionallity
        pass

    @abstractmethod
    def start_timer(self, initial: Optional[float] = None) -> Generator[float, None, None]:
        """Yield the times at which requests should be made."""
        raise NotImplementedError


class ConstantLoadTimer(LoadTimer):
    """
    A load generator that generates requests at a constant rate.
    Introduces a small amount of random noise in timing.
    """

    def __init__(self, rate: Union[float, RateSchedule], duration: float) -> None:
        # A time-varying rate spreads the same jittered schedule over cumulative
        # intensity instead of time, then maps each point back to the time the
        # rate has accumulated that many requests. A constant rate keeps the
        # original time-space arithmetic exactly.
        self._schedule: Optional[RateSchedule] = None
        if isinstance(rate, RateSchedule):
            if rate.is_constant:
                rate = rate.constant_rate
            else:
                self._schedule = rate
                rate = rate.mean_rate
        self._rate = rate
        self._duration = duration
        # TODO: Make random state a global seed
        self._rand = np.random.default_rng()

    def start_timer(self, initial: Optional[float] = None) -> Generator[float, None, None]:
        if self._schedule is not None:
            yield from self._start_varying_timer(self._schedule, initial)
            return

        num_requests = int(self._rate * self._duration)
        if num_requests == 0:
            return

        # Generate random intervals
        intervals = self._rand.exponential(1 / self._rate, num_requests)

        # Normalize intervals to sum to the duration
        total_interval_time = np.sum(intervals)
        scale_factor = self._duration / total_interval_time
        normalized_intervals = intervals * scale_factor

        # Yield request times
        next_time = time.monotonic() if initial is None else initial
        for interval in normalized_intervals:
            next_time += interval
            yield next_time

    def _start_varying_timer(self, schedule: RateSchedule, initial: Optional[float]) -> Generator[float, None, None]:
        num_requests = schedule.expected_requests
        if num_requests == 0:
            return
        intervals = self._rand.exponential(1 / self._rate, num_requests)
        # Normalize so the last request lands where the rate has accumulated its
        # full total, which is the end of the window.
        counts = np.cumsum(intervals * (schedule.total / np.sum(intervals)))
        offsets = schedule.inverse(counts)
        start = time.monotonic() if initial is None else initial
        for offset in offsets:
            yield start + float(offset)


class PoissonLoadTimer(LoadTimer):
    """
    A load generator that generates requests based on a Poisson distribution.
    """

    def __init__(self, rate: Union[float, RateSchedule], duration: float) -> None:
        self._schedule: Optional[RateSchedule] = None
        if isinstance(rate, RateSchedule):
            if rate.is_constant:
                rate = rate.constant_rate
            else:
                self._schedule = rate
                rate = rate.mean_rate
        self._rate = rate
        self._duration = duration
        self._rand = np.random.default_rng()

    def start_timer(self, initial: Optional[float] = None) -> Generator[float, None, None]:
        if self._schedule is not None:
            yield from self._start_varying_timer(self._schedule, initial)
            return

        # Set start time
        next_time = time.perf_counter() if initial is None else initial

        # Given a rate, yield a time to wait before the next request
        while True:
            # How many requests in the next second
            req_count = self._rand.poisson(self._rate)

            # If no requests, wait for 1 second
            if req_count < 1:
                next_time += 1.0
                continue

            # Schedule the requests over the next second
            timer = ConstantLoadTimer(req_count, 1.0)
            time_generator = timer.start_timer(next_time)
            for _ in range(req_count):
                next_time = next(time_generator)
                yield next_time

    def _start_varying_timer(self, schedule: RateSchedule, initial: Optional[float]) -> Generator[float, None, None]:
        # The same second-by-second process as the constant case, except each
        # second's expected count is the rate integrated over that second, and
        # requests are spread within it by cumulative intensity rather than
        # uniformly in time. Unbounded, like the constant case: run_stage stops
        # pulling once the stage's request count is dispatched.
        start = time.perf_counter() if initial is None else initial
        second = 0
        while True:
            low, high = (float(v) for v in schedule.cumulative(np.array([second, second + 1], dtype=np.float64)))
            req_count = self._rand.poisson(high - low)
            if req_count >= 1:
                intervals = self._rand.exponential(1.0, req_count)
                counts = low + np.cumsum(intervals * ((high - low) / np.sum(intervals)))
                for offset in schedule.inverse(counts):
                    yield start + float(offset)
            second += 1


class TraceReplayLoadTimer(LoadTimer):
    def __init__(self, trace_reader: TraceReader, trace_file: Path) -> None:
        self._trace_reader = trace_reader
        self._trace_file = trace_file

    def start_timer(self, initial: Optional[float] = None) -> Generator[float, None, None]:
        start_time = time.monotonic() if initial is None else initial
        for timestamp, _, _ in self._trace_reader.load_traces(self._trace_file):
            yield start_time + timestamp
