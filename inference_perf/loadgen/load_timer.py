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
from typing import TYPE_CHECKING, Generator, List, Optional, Tuple
import numpy as np
from inference_perf.utils.distribution import sample_floats_from_distribution
from inference_perf.utils.trace_reader import TraceReader
from pathlib import Path

if TYPE_CHECKING:
    from inference_perf.config import Distribution


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

    def __init__(self, rate: float, duration: float) -> None:
        self._rate = rate
        self._duration = duration
        # TODO: Make random state a global seed
        self._rand = np.random.default_rng()

    def start_timer(self, initial: Optional[float] = None) -> Generator[float, None, None]:
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


class IntervalLoadTimer(LoadTimer):
    """
    A load generator that separates consecutive requests by gaps sampled
    from a Distribution (e.g. uniform 1-10 seconds).

    The schedule is precomputed so the number of requests that fit in the
    stage duration is known up front via num_requests.
    """

    def __init__(self, interval: "Distribution", duration: float, rng: Optional[np.random.Generator] = None) -> None:
        self._rand = rng if rng is not None else np.random.default_rng()
        offsets: List[float] = []
        elapsed = 0.0
        while elapsed <= duration:
            gaps = sample_floats_from_distribution(interval, 256, rng=self._rand)
            if np.sum(gaps) <= 0:
                raise ValueError(f"interval distribution produced non-positive gaps: {interval}")
            for gap in gaps:
                elapsed += float(gap)
                if elapsed > duration:
                    break
                offsets.append(elapsed)
            else:
                continue
            break
        self._offsets = offsets

    @property
    def num_requests(self) -> int:
        return len(self._offsets)

    def start_timer(self, initial: Optional[float] = None) -> Generator[float, None, None]:
        start_time = time.monotonic() if initial is None else initial
        for offset in self._offsets:
            yield start_time + offset


class PoissonLoadTimer(LoadTimer):
    """
    A load generator that generates requests based on a Poisson distribution.
    """

    def __init__(self, rate: float, duration: float) -> None:
        self._rate = rate
        self._duration = duration
        self._rand = np.random.default_rng()

    def start_timer(self, initial: Optional[float] = None) -> Generator[float, None, None]:
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


class TraceReplayLoadTimer(LoadTimer):
    def __init__(self, trace_reader: TraceReader, trace_file: Path) -> None:
        self._trace_reader = trace_reader
        self._trace_file = trace_file

    def start_timer(self, initial: Optional[float] = None) -> Generator[float, None, None]:
        start_time = time.monotonic() if initial is None else initial
        for timestamp, _, _ in self._trace_reader.load_traces(self._trace_file):
            yield start_time + timestamp
