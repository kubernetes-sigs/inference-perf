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
"""Singleton + lazy-fill invariants for the per-process video pool."""

from typing import Generator

import pytest

from inference_perf.mediagen import pool as pool_module
from inference_perf.mediagen.pool import VideoBytesPool, get_video_pool, reset_video_pool


@pytest.fixture(autouse=True)
def _reset_pool() -> Generator[None, None, None]:
    """Each test starts with a fresh module-level pool."""
    reset_video_pool()
    yield
    reset_video_pool()


def test_get_video_pool_returns_same_instance() -> None:
    """Two calls within a process yield the same lazily-initialized pool."""
    p1 = get_video_pool()
    p2 = get_video_pool()
    assert p1 is p2


def test_reset_video_pool_clears_module_state() -> None:
    """``reset_video_pool`` discards the existing instance so the next ``get_video_pool``
    constructs a new one."""
    p1 = get_video_pool()
    reset_video_pool()
    assert pool_module._video_pool is None
    p2 = get_video_pool()
    assert p1 is not p2


def test_pool_grows_to_pool_size_then_stops() -> None:
    """The bucket grows by one encoding per call until ``pool_size``; further
    calls reuse what's already there instead of re-encoding."""
    pool = VideoBytesPool(pool_size=3)
    key = (64, 64, 4)

    # Fill the bucket: each call appends a new blob.
    pool.get(*key)
    assert len(pool._pool[key]) == 1
    pool.get(*key)
    assert len(pool._pool[key]) == 2
    pool.get(*key)
    assert len(pool._pool[key]) == 3

    # Bucket is full — subsequent calls don't grow it; they sample from existing.
    for _ in range(10):
        result = pool.get(*key)
        assert result in pool._pool[key]
    assert len(pool._pool[key]) == 3


def test_pool_buckets_are_per_profile() -> None:
    """Different ``(width, height, frames)`` profiles maintain independent buckets."""
    pool = VideoBytesPool(pool_size=2)
    pool.get(64, 64, 4)
    pool.get(64, 64, 8)  # different frame count → different bucket
    assert (64, 64, 4) in pool._pool
    assert (64, 64, 8) in pool._pool
    assert len(pool._pool[(64, 64, 4)]) == 1
    assert len(pool._pool[(64, 64, 8)]) == 1
