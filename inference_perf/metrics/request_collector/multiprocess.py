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

from asyncio import Task, create_task, get_event_loop, get_running_loop, sleep
from contextlib import asynccontextmanager
import ctypes
import logging
import multiprocessing as mp
from queue import Empty
from typing import TYPE_CHECKING, AsyncIterator, List, Optional, Union

from inference_perf.apis import RequestLifecycleMetric
from inference_perf.metrics.request_collector import RequestMetricCollector
from inference_perf.utils.mp_context import MP_CONTEXT

if TYPE_CHECKING:
    from multiprocessing.sharedctypes import SynchronizedArray

logger = logging.getLogger(__name__)

# How long wait_for_recorded keeps waiting while no new metric arrives. Metrics
# a worker had buffered when it was killed never arrive, so the wait gives up
# rather than holding the run.
_RECORDED_STALL_TIMEOUT_SECONDS = 10.0
_RECORDED_POLL_SECONDS = 0.01
# Recorded metrics are counted per stage, so metrics lost in one stage don't
# hold up the stages after it. Shared memory can't grow once workers have
# forked, so stage ids wrap around a fixed number of counters: stages that
# share one are this many stages apart.
_STAGE_COUNTERS = 1024


def _stage_counter(stage_id: Optional[int]) -> int:
    return (-1 if stage_id is None else stage_id) % _STAGE_COUNTERS


class MultiprocessRequestMetricCollector(RequestMetricCollector):
    """Responsible for accumulating client request metrics."""

    def __init__(self) -> None:
        super().__init__()
        self.queue: "mp.JoinableQueue[Optional[Union[RequestLifecycleMetric, List[RequestLifecycleMetric]]]]" = (
            MP_CONTEXT.JoinableQueue()
        )
        self.metrics: list[RequestLifecycleMetric] = []
        # Metrics put on the queue so far per stage, counted by the process
        # that puts them. A worker puts a request's metric before it
        # acknowledges the end of the stage, but this process drains the queue
        # asynchronously; wait_for_recorded compares these counts with the
        # metrics received to know when a finished stage's have all arrived.
        self._recorded: "SynchronizedArray[int]" = MP_CONTEXT.Array(ctypes.c_uint64, _STAGE_COUNTERS)
        self._received = [0] * _STAGE_COUNTERS
        self._received_counted = 0
        self._collector_task: Optional["Task[list[RequestLifecycleMetric]]"] = None

    def __getstate__(self) -> dict[str, object]:
        # Workers get this collector through their client, which spawn and
        # forkserver pickle into every new worker, including the ones respawned
        # mid-run. Workers only put on the queue, so leave out the metrics this
        # process has collected so far and its drain task.
        state: dict[str, object] = dict(super().__getstate__())
        state["metrics"] = []
        state["_collector_task"] = None
        return state

    def record_metric(self, metric: RequestLifecycleMetric) -> None:
        """Record a single metric directly to the multiprocessing queue."""
        self.queue.put(metric)
        with self._recorded.get_lock():
            self._recorded[_stage_counter(metric.stage_id)] += 1

    async def collect_metrics(self) -> list[RequestLifecycleMetric]:
        # Accumulate onto self.metrics (not a local) so snapshot() can serve
        # partial results to mid-run readers like the stage-end reports.
        metrics = self.metrics
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
                    self._notify_observers(item)

            if done:
                break

        return metrics

    @asynccontextmanager
    async def start(self) -> AsyncIterator[None]:
        collector_task = create_task(self.collect_metrics())
        self._collector_task = collector_task

        yield

        self.queue.put(None)
        self.metrics = await collector_task
        logger.debug(f"Collector collected {len(self.metrics)} metrics")

    async def wait_for_recorded(self, stage_id: int, stall_timeout: float = _RECORDED_STALL_TIMEOUT_SECONDS) -> bool:
        counter = _stage_counter(stage_id)
        # Read without the lock: a worker terminated while holding it would
        # otherwise hang this read, and with it the run.
        recorded = int(self._recorded.get_obj()[counter])
        collector_task = self._collector_task
        loop = get_running_loop()
        collected = len(self.metrics)
        last_progress = loop.time()
        # Without a running drain nothing more can arrive.
        while self._count_received(counter) < recorded and collector_task is not None and not collector_task.done():
            if len(self.metrics) > collected:
                collected = len(self.metrics)
                last_progress = loop.time()
            elif loop.time() - last_progress >= stall_timeout:
                logger.warning(
                    "Stage %d: %d of %d recorded request metrics have not reached the collector after %.1fs without progress",
                    stage_id,
                    recorded - self._received[counter],
                    recorded,
                    stall_timeout,
                )
                return False
            await sleep(_RECORDED_POLL_SECONDS)
        return self._received[counter] >= recorded

    def _count_received(self, counter: int) -> int:
        for metric in self.metrics[self._received_counted :]:
            self._received[_stage_counter(metric.stage_id)] += 1
        self._received_counted = len(self.metrics)
        return self._received[counter]

    def get_metrics(self) -> list[RequestLifecycleMetric]:
        return self.metrics

    def snapshot(self) -> list[RequestLifecycleMetric]:
        return list(self.metrics)
