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
import logging
import multiprocessing as mp
from queue import Empty
from typing import Generic, List, TypeVar

from inference_perf.utils.mp_context import MP_CONTEXT

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RequestQueue(Generic[T]):
    """Multi-channel request queue between the main process and workers.

    Plain mp.Queue channels, deliberately without JoinableQueue's
    task_done/join accounting: stage completion is tracked through counters
    and the stage rendezvous in the load generator, and task_done acquires a
    shared condition lock that a worker killed mid-call would strand,
    hanging every other process that touches it.
    """

    def __init__(self, num_channels: int = 1):
        """
        initialize request queue based on number of channels, when num_channels is 1, there is only one global channel for all consumers.

        Args:
            num_channels (int, optional): number of channels. Defaults to 1.
        """
        self.num_channels: int = num_channels
        self.queues: List["mp.Queue[T]"] = [MP_CONTEXT.Queue() for _ in range(num_channels)]

    def get_channel(self, channel_id: int) -> "mp.Queue[T]":
        return self.queues[channel_id % self.num_channels]

    def replace_channel(self, channel_id: int) -> "mp.Queue[T]":
        """Replace a channel with a fresh queue and return it.

        Used when the channel's sole consumer died abruptly: the dead process
        may have stranded the queue's internal locks, which would starve any
        replacement consumer. Only valid for per-consumer channels; a shared
        channel must not be replaced while other consumers hold the old one.
        """
        queue: "mp.Queue[T]" = MP_CONTEXT.Queue()
        self.queues[channel_id % self.num_channels] = queue
        return queue

    def drain(self, channel_id: int = -1) -> int:
        """
        drain the specific queue by giving channel id, when id is -1, drain all queues.

        Args:
            channel_id (int, optional): the id of the queue to drain. Defaults to -1 (all queues).

        Returns:
            int: number of items removed.
        """
        drained = 0
        queues_to_drain = self.queues if channel_id == -1 else [self.get_channel(channel_id)]
        for queue in queues_to_drain:
            while True:
                try:
                    _ = queue.get_nowait()
                    drained += 1
                except Empty:
                    # No qsize() check: it raises NotImplementedError on macOS.
                    # Empty from get_nowait() is sufficient since producers are stopped before drain.
                    logger.debug("Drain finished")
                    break
        return drained

    def put(self, item: T, channel_id: int = -1) -> None:
        """
        put item into the specific queue by giving channel id, when channel id is -1, put into all queues.

        Args:
            item (object): the item to put into the queue.
            channel_id (int, optional): the id of the queue to put into. Defaults to -1 (all queues).
        """
        queues_to_put = self.queues if channel_id == -1 else [self.get_channel(channel_id)]
        for queue in queues_to_put:
            queue.put(item)
