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
"""Per-process pool of pre-encoded media blobs.

Video encoding (MP4) is the expensive part of synthesizing multimodal payloads,
so we keep a small bucket of distinct encodings per profile and sample from it
instead of re-encoding on every request. Different blobs in the bucket give
the model server's mm cache something to chew on (so all requests don't
collapse to a single cache hit) while keeping per-request cost near zero.

Image (PNG) and audio (WAV) generation is cheap enough that we don't pool them.
"""

from typing import Optional

import numpy as np

from inference_perf.mediagen.synthesis import generate_mp4_bytes


class VideoBytesPool:
    """Per-profile cache of pre-encoded MP4 blobs keyed by ``(width, height, frames)``.

    Each profile holds up to ``pool_size`` distinct encodings; ``get`` lazily
    fills the bucket and samples from it on subsequent calls.
    """

    def __init__(self, pool_size: int = 4, rng: Optional[np.random.Generator] = None) -> None:
        self._pool: dict[tuple[int, int, int], list[bytes]] = {}
        self._pool_size = pool_size
        self._rng = rng or np.random.default_rng()

    def get(self, width: int, height: int, frames: int) -> bytes:
        key = (width, height, frames)
        bucket = self._pool.setdefault(key, [])
        if len(bucket) < self._pool_size:
            bucket.append(generate_mp4_bytes(width, height, frames, self._rng))
        return bucket[int(self._rng.integers(0, len(bucket)))]


_video_pool: Optional[VideoBytesPool] = None


def get_video_pool() -> VideoBytesPool:
    """Return the per-process video pool, lazily initializing on first call.

    Lazy initialization matters for multi-worker loadgen: the parent process
    forks workers before any datagen runs, so each worker inherits ``None``
    and creates its own pool on first access — no shared state, no IPC.
    """
    global _video_pool
    if _video_pool is None:
        _video_pool = VideoBytesPool()
    return _video_pool


def reset_video_pool() -> None:
    """Reset the module-level pool. Intended for tests."""
    global _video_pool
    _video_pool = None
