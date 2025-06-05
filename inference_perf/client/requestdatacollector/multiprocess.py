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

from asyncio import create_task, sleep
from typing import List
import logging
from inference_perf.client.requestdatacollector import RequestDataCollector
from inference_perf.apis import RequestLifecycleMetric

logger = logging.getLogger(__name__)


class MultiprocessRequestDataCollector(RequestDataCollector):
    """Responsible for accumulating client request metrics"""

    def __init__(self) -> None:
        self.queue: mp.JoinableQueue[RequestLifecycleMetric] = mp.JoinableQueue()
        self.metrics: List[RequestLifecycleMetric] = []

    def record_metric(self, metric: RequestLifecycleMetric) -> None:
        self.queue.put(metric)

    async def collect_metrics(self) -> None:
        while True:
            try:
                item = self.queue.get_nowait()
                self.metrics.append(item)
                self.queue.task_done()
            except mp.queues.Empty:
                await sleep(1)

    def start(self) -> None:
        self.collection = create_task(self.collect_metrics())

    async def stop(self) -> None:
        # Ensure that the collection queue is empty before joining
        while self.queue.qsize() > 0:
            logger.debug(f"Collector waiting for empty request data queue, current size {self.queue.qsize()}")
            await sleep(1)
        self.queue.join()

    def get_metrics(self) -> List[RequestLifecycleMetric]:
        return self.metrics
