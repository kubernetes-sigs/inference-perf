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
"""Bounded-distinct payload pool for synthetic videos.

Holds either MP4 blobs (one per entry) or per-frame image blobs depending
on the configured :class:`VideoRepresentation`. Distinct from the legacy
:class:`~inference_perf.mediagen.pool.VideoBytesPool`, which is a lazy
per-profile bucket used when no ``video.pool`` is configured.
"""

from __future__ import annotations

import base64
from dataclasses import dataclass
from typing import Callable, List, Optional, Tuple

import numpy as np

from inference_perf.mediagen.synthesis import generate_jpeg_bytes, generate_mp4_bytes, generate_png_bytes

from ..image import ImageRepresentation
from .spec import VideoRepresentation


@dataclass(frozen=True)
class VideoPoolEntry:
    """One pre-rendered video — geometry plus per-frame (or whole-MP4) blobs."""

    width: int
    height: int
    frames: int
    representation: VideoRepresentation
    # For MP4: a single element holding the encoded MP4 blob.
    # For PNG_FRAMES / JPEG_FRAMES: one element per frame.
    frame_blobs: Tuple[bytes, ...]
    # Parallel to ``frame_blobs`` — pre-built data URLs ready for the wire.
    frame_urls: Tuple[str, ...]


class VideoPool:
    """Eagerly-materialized list of :class:`VideoPoolEntry` for one process."""

    def __init__(
        self,
        size: int,
        profile_sampler: Callable[[], Tuple[int, int, int]],
        representation: VideoRepresentation,
        rng: np.random.Generator,
    ) -> None:
        if size < 1:
            raise ValueError(f"VideoPool size must be >= 1, got {size}")
        self._entries: List[VideoPoolEntry] = []
        for _ in range(size):
            w, h, frames = profile_sampler()
            if representation == VideoRepresentation.MP4:
                mp4 = generate_mp4_bytes(w, h, frames, rng)
                url = f"data:video/mp4;base64,{base64.b64encode(mp4).decode('ascii')}"
                blobs: Tuple[bytes, ...] = (mp4,)
                urls: Tuple[str, ...] = (url,)
            else:
                blob_list: List[bytes] = []
                url_list: List[str] = []
                for _f in range(frames):
                    if representation == VideoRepresentation.JPEG_FRAMES:
                        raw = generate_jpeg_bytes(w, h, rng)
                        u = f"data:image/jpeg;base64,{base64.b64encode(raw).decode('ascii')}"
                    else:
                        raw = generate_png_bytes(w, h, rng)
                        u = f"data:image/png;base64,{base64.b64encode(raw).decode('ascii')}"
                    blob_list.append(raw)
                    url_list.append(u)
                blobs = tuple(blob_list)
                urls = tuple(url_list)
            self._entries.append(
                VideoPoolEntry(
                    width=w,
                    height=h,
                    frames=frames,
                    representation=representation,
                    frame_blobs=blobs,
                    frame_urls=urls,
                )
            )

    @property
    def size(self) -> int:
        return len(self._entries)

    def get(self, index: int) -> VideoPoolEntry:
        return self._entries[index]


_pool: Optional[VideoPool] = None


def set_pool(pool: Optional[VideoPool]) -> None:
    global _pool
    _pool = pool


def get_pool() -> Optional[VideoPool]:
    return _pool


def reset_pool() -> None:
    set_pool(None)


# Frame format helper — small enough to live here rather than in synthesis.
def _frame_image_representation(video_rep: VideoRepresentation) -> ImageRepresentation:
    return ImageRepresentation.JPEG if video_rep == VideoRepresentation.JPEG_FRAMES else ImageRepresentation.PNG


__all__ = ["VideoPool", "VideoPoolEntry", "get_pool", "set_pool", "reset_pool"]
