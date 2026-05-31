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
"""Synthetic image spec — bytes generated at materialization time."""

from typing import Literal, Optional

from ..metrics import Image
from .base import ImageSpec


class SyntheticImageSpec(ImageSpec):
    """Image whose bytes are synthesized from ``(width, height, representation)``."""

    kind: Literal["synthetic"] = "synthetic"
    # When set, the materializer fetches pre-rendered bytes from the
    # per-process :class:`~inference_perf.payloads.image.pool.ImagePool` at
    # this index instead of encoding fresh bytes. ``width`` / ``height``
    # mirror the pool entry's geometry so :meth:`get_metrics` stays an
    # exact measurement without a pool round-trip. Only meaningful for
    # payload-side specs — prefix-side specs always leave this ``None``
    # so prefix-cache benchmarks keep their deterministic-by-seed bytes.
    pool_index: Optional[int] = None

    def get_metrics(self, wire_bytes: int) -> Image:
        # We generated the bytes from our own declared geometry — every
        # field is an exact measurement.
        return Image(
            pixels=self.width * self.height,
            bytes=wire_bytes,
            aspect_ratio=self.width / self.height if self.height > 0 else 0.0,
        )


__all__ = ["SyntheticImageSpec"]
