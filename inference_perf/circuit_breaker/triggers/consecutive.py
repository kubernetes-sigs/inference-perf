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
