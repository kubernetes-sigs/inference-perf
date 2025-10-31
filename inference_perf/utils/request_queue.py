import logging
import multiprocessing as mp
from typing import Generic, List, TypeVar

logger = logging.getLogger(__name__)

T = TypeVar("T")


class RequestQueue(Generic[T]):
    def __init__(self, num_channels: int = 1):
        """
        initialize request queue based on number of channels, when num_channels is 1, there is only one global channel for all consumers.

        Args:
            num_channels (int, optional): number of channels. Defaults to 1.
        """
        self.num_channels: int = num_channels
        self.queues: List[mp.JoinableQueue[T]] = [mp.JoinableQueue() for _ in range(num_channels)]

    def get_channel(self, channel_id: int) -> "mp.JoinableQueue[T]":
        return self.queues[channel_id % self.num_channels]

    def drain(self, channel_id: int = -1) -> None:
        """
        drain the specific queue by giving channel id, when id is -1, drain all queues.

        Args:
            channel_id (int, optional): the id of the queue to drain. Defaults to -1 (all queues).
        """
        queues_to_drain = self.queues if channel_id == -1 else [self.get_channel(channel_id)]
        for queue in queues_to_drain:
            while True:
                try:
                    _ = queue.get_nowait()
                    queue.task_done()
                except mp.queues.Empty:
                    if queue.qsize() == 0:
                        logger.debug("Drain finished")
                        break

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

    def join(self, channel_id: int = -1) -> None:
        """
        join the queue by giving channel id, when channel id is -1, join all queues.

        Args:
            channel_id (int, optional): the id of the queue to join. Defaults to -1 (all queues).
        """
        queues_to_join = self.queues if channel_id == -1 else [self.get_channel(channel_id)]
        for queue in queues_to_join:
            queue.join()
