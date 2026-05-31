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
"""Bounded-distinct payload pool for synthetic images.

When ``image.pool.size`` is configured, datagen pre-renders that many
distinct images at config time by drawing dimensions from
:class:`ImageDatagenConfig.resolutions`. At request-build time
:class:`SyntheticImageSpec` instances carry a ``pool_index`` and the
materializer fetches the pre-rendered bytes from the per-process pool
instead of re-encoding. Prefix-side specs leave ``pool_index`` ``None``
and bypass the pool (their bytes are deterministic-by-seed).
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from inference_perf.mediagen.synthesis import generate_jpeg_bytes, generate_png_bytes

from .spec import ImageRepresentation


@dataclass(frozen=True)
class ImagePoolEntry:
    """One pre-rendered image — dimensions plus the encoded bytes / data URL."""

    width: int
    height: int
    raw_bytes: bytes
    data_url: str


class ImagePool:
    """Eagerly-materialized list of :class:`ImagePoolEntry` for one process."""

    def __init__(
        self,
        size: int,
        resolution_sampler: Callable[[], Tuple[int, int]],
        representation: ImageRepresentation,
        rng: np.random.Generator,
    ) -> None:
        if size < 1:
            raise ValueError(f"ImagePool size must be >= 1, got {size}")
        self._entries: List[ImagePoolEntry] = []
        for _ in range(size):
            w, h = resolution_sampler()
            if representation == ImageRepresentation.JPEG:
                raw = generate_jpeg_bytes(w, h, rng)
                url = f"data:image/jpeg;base64,{base64.b64encode(raw).decode('ascii')}"
            else:
                raw = generate_png_bytes(w, h, rng)
                url = f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"
            self._entries.append(ImagePoolEntry(width=w, height=h, raw_bytes=raw, data_url=url))

    @property
    def size(self) -> int:
        return len(self._entries)

    def get(self, index: int) -> ImagePoolEntry:
        return self._entries[index]


# Per-process singleton. Each loadgen worker process forks before datagen
# runs and constructs its own pool on its first use — no shared state, no
# IPC. ``set_pool`` is called from datagen init when ``image.pool`` is
# configured; otherwise this stays ``None`` and the materializer falls
# through to its on-the-fly encoding path.
_pool: Optional[ImagePool] = None


def set_pool(pool: Optional[ImagePool]) -> None:
    global _pool
    _pool = pool


def get_pool() -> Optional[ImagePool]:
    return _pool


def reset_pool() -> None:
    """Discard the per-process pool. Intended for tests."""
    set_pool(None)


__all__ = ["ImagePool", "ImagePoolEntry", "get_pool", "set_pool", "reset_pool"]
