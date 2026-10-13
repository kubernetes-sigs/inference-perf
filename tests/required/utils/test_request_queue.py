import logging
import multiprocessing as mp
import threading
import time
from collections.abc import Callable, Iterator
from typing import Optional

import pytest

from inference_perf.utils import request_queue
from inference_perf.utils.request_queue import RequestQueue

# More than a pipe holds, so most items are still in the feeder buffer when drain() starts.
ITEMS = 2000
PAYLOAD = "x" * 1024

MakeQueue = Callable[..., RequestQueue[str]]


def exits_cleanly(channel: "mp.Queue[Optional[str]]") -> bool:
    """Whether process exit's join of the channel's feeder thread returns."""
    channel.close()
    joined = threading.Thread(target=channel.join_thread, daemon=True)
    joined.start()
    joined.join(timeout=10)
    return not joined.is_alive()


@pytest.fixture
def make_queue() -> Iterator[MakeQueue]:
    """Skips each channel's exit-time join, so leftover items fail a test instead of hanging it."""
    queues: list[RequestQueue[str]] = []

    def make(num_channels: int = 1) -> RequestQueue[str]:
        queue: RequestQueue[str] = RequestQueue(num_channels=num_channels)
        queues.append(queue)
        return queue

    yield make
    for queue in queues:
        for channel in queue.queues:
            channel.cancel_join_thread()


@pytest.mark.parametrize("num_channels", [1, 3])
def test_drain_removes_items_the_feeder_has_not_flushed_yet(make_queue: MakeQueue, num_channels: int) -> None:
    queue = make_queue(num_channels)
    for _ in range(5):
        for i in range(ITEMS):
            queue.put(PAYLOAD, channel_id=i)

        assert queue.drain() == ITEMS
        assert queue.drain() == 0


def test_a_drained_queue_lets_its_process_exit(make_queue: MakeQueue) -> None:
    """Process exit joins the feeder thread, which blocks while it still holds items."""
    queue = make_queue()
    for _ in range(ITEMS):
        queue.put(PAYLOAD)
    queue.drain()

    assert exits_cleanly(queue.get_channel(0))


def test_drain_gives_up_on_a_channel_a_dead_worker_left_locked(
    make_queue: MakeQueue, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(request_queue, "_DRAIN_TIMEOUT_SECONDS", 0.2)
    queue = make_queue(2)
    queue.put(PAYLOAD, channel_id=0)
    for _ in range(ITEMS):
        queue.put(PAYLOAD, channel_id=1)
    # What a worker killed inside get() leaves behind.
    queue.get_channel(1)._rlock.acquire()  # type: ignore[attr-defined]

    started = time.monotonic()
    with caplog.at_level(logging.WARNING):
        drained = queue.drain()

    assert drained == 1
    assert time.monotonic() - started < 2
    assert "Drain of channel 1 stopped after 0 item(s)" in caplog.text
    assert exits_cleanly(queue.get_channel(1))
