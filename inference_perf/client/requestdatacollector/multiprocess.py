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

import multiprocessing as mp

from asyncio import get_event_loop, create_task
from contextlib import asynccontextmanager
from typing import AsyncIterator, Optional
from functools import partial
import logging
from inference_perf.client.requestdatacollector import RequestDataCollector
from inference_perf.apis import RequestLifecycleMetric
from inference_perf.circuit_breaker import feed_breakers

logger = logging.getLogger(__name__)


class MultiprocessRequestDataCollector(RequestDataCollector):
    """Responsible for accumulating client request metrics"""

    def __init__(self) -> None:
        self.queue: "mp.JoinableQueue[Optional[RequestLifecycleMetric]]" = mp.JoinableQueue()

    def record_metric(self, metric: RequestLifecycleMetric) -> None:
        self.queue.put(metric)

    async def collect_metrics(self) -> list[RequestLifecycleMetric]:
        metrics: list[RequestLifecycleMetric] = []
        event_loop = get_event_loop()
        # prevent get from blocking the executor for too long:
        get_queue = partial(self.queue.get, timeout=0.5)

        while True:
            try:
                item = await event_loop.run_in_executor(None, get_queue)
            except mp.queues.Empty:
                continue

            if item is None:
                self.queue.task_done()
                break

            metrics.append(item)
            feed_breakers(item)
            self.queue.task_done()

        return metrics

    @asynccontextmanager
    async def start(self) -> AsyncIterator[None]:
        collector_task = create_task(self.collect_metrics())

        yield

        self.queue.put(None)
        self.metrics = await collector_task
        logger.debug(f"Collector collected {len(self.metrics)} metrics")

    def get_metrics(self) -> list[RequestLifecycleMetric]:
        return self.metrics
