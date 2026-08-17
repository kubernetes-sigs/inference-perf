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

RequestMetricObserver = Callable[[RequestLifecycleMetric], None]


class RequestMetricCollector(ABC):
    """
    Responsible for collecting request information
    """

    def __init__(self) -> None:
        # Invoked once per collected metric, in the process that aggregates
        # metrics (the parent for multiprocess runs; workers only enqueue).
        # Circuit breakers and the runtime metrics hub subscribe here.
        self._observers: List[RequestMetricObserver] = []

    def add_observer(self, observer: RequestMetricObserver) -> None:
        self._observers.append(observer)

    def _notify_observers(self, metric: RequestLifecycleMetric) -> None:
        for observer in self._observers:
            observer(metric)

    def __getstate__(self) -> Dict[str, Any]:
        # Observers may hold unpicklable resources (locks, sockets) and only run
        # in the aggregating process; drop them if the collector is pickled to
        # a load generator worker (forkserver/spawn start methods).
        state = self.__dict__.copy()
        state["_observers"] = []
        return state

    @abstractmethod
    def record_metric(self, metric: RequestLifecycleMetric) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_metrics(self) -> List[RequestLifecycleMetric]:
        raise NotImplementedError

    @asynccontextmanager
    async def start(self) -> AsyncIterator[None]:
        yield
