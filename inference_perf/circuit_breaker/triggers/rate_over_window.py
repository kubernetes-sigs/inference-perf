from __future__ import annotations
import collections
import datetime
from typing import Deque
from .base import Trigger, HitSample
from .config import TriggerRateOverWindow


class RateOverWindow(Trigger, spec_cls=TriggerRateOverWindow):
    """Open when hit rate over a time window crosses threshold."""

    def __init__(self, window_sec: float, threshold: float, min_samples: int = 0):
        self.size = datetime.timedelta(seconds=window_sec)
        self.th = threshold
        self.min = min_samples
        self.buf: Deque[HitSample] = collections.deque()
        self._fired = False

    def update(self, s: HitSample) -> None:
        now = s.ts
        self.buf.append(s)
        cutoff = now - self.size
        while self.buf and self.buf[0].ts < cutoff:
            self.buf.popleft()
        total = len(self.buf)
        if total >= self.min:
            hits = sum(x.hit for x in self.buf)
            rate = hits / total if total else 0.0
            if rate >= self.th:
                self._fired = True

    def fired(self) -> bool:
        return self._fired

    def reset(self) -> None:
        self.buf.clear()
        self._fired = False
