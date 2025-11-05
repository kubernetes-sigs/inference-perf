from __future__ import annotations
from .config import TriggerConsecutive
from .base import Trigger, HitSample


class Consecutive(Trigger, spec_cls=TriggerConsecutive):
    """Open when N consecutive hits occur."""

    def __init__(self, threshold: int):
        self.n = threshold
        self.c = 0
        self._fired = False

    def update(self, s: HitSample) -> None:
        self.c = self.c + 1 if s.hit else 0
        if self.c >= self.n:
            self._fired = True

    def fired(self) -> bool:
        return self._fired

    def reset(self) -> None:
        self.c = 0
        self._fired = False
