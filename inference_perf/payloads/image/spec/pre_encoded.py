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
"""Pre-encoded image spec — bytes already encoded by an upstream loader.

Stub: in :data:`ImageSpecUnion` (construction works) but not yet wired into
the chat.py materializer — passing one through ``to_request_body`` raises
``TypeError`` until that branch lands. A future dataset loader populates
``image_bytes`` and the materializer will base64-wrap the bytes using
``representation`` to pick the data-URL mime type.
"""

from typing import Literal

from ..metrics import Image
from .base import ImageSpec


class PreEncodedImageSpec(ImageSpec):
    """Image whose bytes are already encoded (PNG or JPEG)."""

    kind: Literal["pre_encoded"] = "pre_encoded"
    image_bytes: bytes

    def get_metrics(self, wire_bytes: int) -> Image:
        # Bytes are the loader-supplied blob (also == wire_bytes, since the
        # materializer just base64-wraps them). Pixels come from the spec's
        # declared geometry, which the loader is responsible for populating
        # from manifest metadata — trusted as authoritative.
        return Image(
            pixels=self.width * self.height,
            bytes=wire_bytes,
            aspect_ratio=self.width / self.height if self.height > 0 else 0.0,
        )


__all__ = ["PreEncodedImageSpec"]
