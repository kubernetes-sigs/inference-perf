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
"""Bounded-distinct payload pool for synthetic audio."""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Callable, List, Optional

from inference_perf.mediagen.synthesis import generate_wav_bytes


@dataclass(frozen=True)
class AudioPoolEntry:
    """One pre-rendered audio clip — duration plus the encoded WAV bytes."""

    duration: float
    raw_bytes: bytes
    b64_data: str


class AudioPool:
    """Eagerly-materialized list of :class:`AudioPoolEntry` for one process.

    WAV synthesis is deterministic by duration (silent samples), so two
    entries sampled with the same duration are byte-identical. Pool size
    still caps the distinct-count from above; equal durations just collapse
    the effective distinct count below ``size``. That's by design — users
    who want N distinct items should also vary ``durations``.
    """

    def __init__(self, size: int, duration_sampler: Callable[[], float]) -> None:
        if size < 1:
            raise ValueError(f"AudioPool size must be >= 1, got {size}")
        self._entries: List[AudioPoolEntry] = []
        for _ in range(size):
            duration = duration_sampler()
            wav = generate_wav_bytes(duration)
            b64 = base64.b64encode(wav).decode("ascii")
            self._entries.append(AudioPoolEntry(duration=duration, raw_bytes=wav, b64_data=b64))

    @property
    def size(self) -> int:
        return len(self._entries)

    def get(self, index: int) -> AudioPoolEntry:
        return self._entries[index]


_pool: Optional[AudioPool] = None


def set_pool(pool: Optional[AudioPool]) -> None:
    global _pool
    _pool = pool


def get_pool() -> Optional[AudioPool]:
    return _pool


def reset_pool() -> None:
    set_pool(None)


__all__ = ["AudioPool", "AudioPoolEntry", "get_pool", "set_pool", "reset_pool"]
