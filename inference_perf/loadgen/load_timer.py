# Copyright 2025 The Kubernetes Authors.
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
from typing import Generator, Optional, Tuple
import numpy as np


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

    def __init__(self, rate: float) -> None:
        self._rate = rate
        # TODO: Make random state a global seed
        self._rand = np.random.default_rng()

    def start_timer(self, initial: Optional[float] = None) -> Generator[float, None, None]:
        # Set start time
        next_time = time.monotonic() if initial is None else initial

        # Given a rate, yield a time to wait before the next request
        while True:
            next_time += self._rand.exponential(1 / self._rate)
            yield next_time


class PoissonLoadTimer(LoadTimer):
    def __init__(self, rate: float) -> None:
        self._rate = rate
        self._rand = np.random.default_rng()

    def start_timer(self, initial: Optional[float] = None) -> Generator[float, None, None]:
        # Set start time
        next_time = time.monotonic() if initial is None else initial

        # Given a rate, yield a time to wait before the next request
        while True:
            # How many requests in the next second
            req_count = self._rand.poisson(self._rate)

            # If no requests, wait for 1 second
            if req_count < 1:
                yield next_time + 1.0
                continue

            # Schedule the requests over the next second
            timer = ConstantLoadTimer(req_count)
            for _ in range(req_count):
                next_time = next(timer.start_timer(next_time))
                yield next_time
