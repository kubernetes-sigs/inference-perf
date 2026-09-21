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

from asyncio import create_task, get_event_loop
from contextlib import asynccontextmanager
import logging
import multiprocessing as mp
from queue import Empty
from typing import AsyncIterator, List, Optional, Union

from inference_perf.apis import RequestLifecycleMetric
from inference_perf.circuit_breaker import feed_breakers
from inference_perf.metrics.request_collector import RequestMetricCollector

logger = logging.getLogger(__name__)


class MultiprocessRequestMetricCollector(RequestMetricCollector):
    """Responsible for accumulating client request metrics."""

    def __init__(self) -> None:
        self.queue: "mp.JoinableQueue[Optional[Union[RequestLifecycleMetric, List[RequestLifecycleMetric]]]]" = (
            mp.JoinableQueue()
        )

    def record_metric(self, metric: RequestLifecycleMetric) -> None:
        """Record a single metric directly to the multiprocessing queue."""
        self.queue.put(metric)

    async def collect_metrics(self) -> list[RequestLifecycleMetric]:
        metrics: list[RequestLifecycleMetric] = []
        event_loop = get_event_loop()

        def _drain_batch(max_batch_size: int = 4096) -> tuple[list[RequestLifecycleMetric], bool]:
            """Drain a batch of items from the queue in the executor thread."""
            batch: list[RequestLifecycleMetric] = []
            try:
                first_item = self.queue.get(timeout=0.5)
            except Empty:
                return batch, False

            if first_item is None:
                self.queue.task_done()
                return batch, True

            if isinstance(first_item, list):
                batch.extend(first_item)
            else:
                batch.append(first_item)
            self.queue.task_done()

            # Drain any remaining available items up to max_batch_size non-blockingly
            while len(batch) < max_batch_size:
                try:
                    item = self.queue.get_nowait()
                except Empty:
                    break

                if item is None:
                    self.queue.task_done()
                    return batch, True

                if isinstance(item, list):
                    batch.extend(item)
                else:
                    batch.append(item)
                self.queue.task_done()

            return batch, False

        while True:
            batch, done = await event_loop.run_in_executor(None, _drain_batch)
            if batch:
                metrics.extend(batch)
                for item in batch:
                    feed_breakers(item)

            if done:
                break

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
