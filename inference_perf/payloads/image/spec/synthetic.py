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

from typing import Literal

from ..metrics import Image
from .base import ImageSpec


class SyntheticImageSpec(ImageSpec):
    """Image whose bytes are synthesized from ``(width, height, representation)``."""

    kind: Literal["synthetic"] = "synthetic"

    def get_metrics(self, wire_bytes: int) -> Image:
        # We generated the bytes from our own declared geometry — every
        # field is an exact measurement.
        return Image(
            pixels=self.width * self.height,
            bytes=wire_bytes,
            aspect_ratio=self.width / self.height if self.height > 0 else 0.0,
        )


__all__ = ["SyntheticImageSpec"]
