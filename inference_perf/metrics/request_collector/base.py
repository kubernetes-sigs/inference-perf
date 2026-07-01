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
from typing import List, AsyncIterator
from contextlib import asynccontextmanager

from inference_perf.apis import RequestLifecycleMetric
from inference_perf.metrics.request_counter import RequestSentCounter


class RequestMetricCollector(ABC):
    """
    Responsible for collecting request information
    """

    def __init__(self) -> None:
        # Running count of requests sent to the model server (see RequestSentCounter).
        self.request_sent_counter = RequestSentCounter()

    @abstractmethod
    def record_metric(self, metric: RequestLifecycleMetric) -> None:
        raise NotImplementedError

    def get_request_sent_counter(self) -> RequestSentCounter:
        return self.request_sent_counter

    @abstractmethod
    def get_metrics(self) -> List[RequestLifecycleMetric]:
        raise NotImplementedError

    @asynccontextmanager
    async def start(self) -> AsyncIterator[None]:
        yield
