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
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, AsyncIterator
from contextlib import asynccontextmanager

from inference_perf.apis import RequestLifecycleMetric
from inference_perf.circuit_breaker import feed_breakers


class RequestMetricCollector(ABC):
    """
    Responsible for collecting request information
    """

    def __init__(self) -> None:
        self.metrics: List[RequestLifecycleMetric] = []
        # Callbacks invoked once per collected metric, in the process that
        # aggregates metrics (the parent process for multiprocess runs). Used by
        # runtime observability surfaces (e.g. Prometheus) to count requests live.
        self._observers: List[Callable[[RequestLifecycleMetric], None]] = []

    def add_observer(self, observer: Callable[[RequestLifecycleMetric], None]) -> None:
        self._observers.append(observer)

    def __getstate__(self) -> Dict[str, Any]:
        # Observers may hold unpicklable resources (sockets, locks) and only run
        # in the aggregating process; drop them when the collector is pickled to
        # load generator workers, which only enqueue metrics.
        state = self.__dict__.copy()
        state["_observers"] = []
        return state

    def _collect(self, metric: RequestLifecycleMetric) -> None:
        """Ingest one metric: store it, notify observers, feed circuit breakers.

        Call this exactly once per metric, and only in the process that
        aggregates metrics (never in load generator workers).
        """
        self.metrics.append(metric)
        for observer in self._observers:
            observer(metric)
        feed_breakers(metric)

    @abstractmethod
    def record_metric(self, metric: RequestLifecycleMetric) -> None:
        raise NotImplementedError

    def get_metrics(self) -> List[RequestLifecycleMetric]:
        return self.metrics

    @asynccontextmanager
    async def start(self) -> AsyncIterator[None]:
        yield
