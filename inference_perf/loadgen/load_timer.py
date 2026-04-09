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
from inference_perf.utils.expressions import sample_distribution
import numpy as np
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


class ExpressionLoadTimer(LoadTimer):
    """
    A load generator that generates requests based on an expression for interval.
    """

    def __init__(self, interval: Union[float, str], duration: float) -> None:
        self._interval = interval
        self._duration = duration
        self._rand = np.random.default_rng()

    def start_timer(self, initial: Optional[float] = None) -> Generator[float, None, None]:
        start_time = time.perf_counter() if initial is None else initial
        next_time = start_time

        while True:
            t = next_time - start_time

            # Direct sampling of the interval!
            interval_val = sample_distribution(self._interval, t=t)

            if interval_val <= 0:
                # Avoid infinite loops or negative intervals
                interval_val = 1.0

            next_time += interval_val
            if next_time - start_time > self._duration:
                break
            yield next_time


class TraceReplayLoadTimer(LoadTimer):
    def __init__(self, trace_reader: TraceReader, trace_file: Path) -> None:
        self._trace_reader = trace_reader
        self._trace_file = trace_file

    def start_timer(self, initial: Optional[float] = None) -> Generator[float, None, None]:
        start_time = time.monotonic() if initial is None else initial
        for timestamp, _, _ in self._trace_reader.load_traces(self._trace_file):
            yield start_time + timestamp
