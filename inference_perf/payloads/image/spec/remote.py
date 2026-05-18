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
"""Remote-URL image spec — server fetches bytes from a URL.

Stub: in :data:`ImageSpecUnion` (construction works) but not yet wired into
the chat.py materializer — passing one through ``to_request_body`` raises
``TypeError`` until that branch lands. The materializer will emit the URL
verbatim in an ``image_url`` wire block and
record ``bytes=0`` on the realized metric (we send a URL, not pixels; the
server does the fetch). Private buckets are expected to use signed URLs —
no auth-headers field on the spec.
"""

from typing import Literal

from ..metrics import Image
from .base import ImageSpec


class RemoteImageSpec(ImageSpec):
    """Image referenced by URL; the server fetches the bytes."""

    kind: Literal["remote"] = "remote"
    url: str

    def get_metrics(self, wire_bytes: int) -> Image:
        # We sent a URL string, not the image. Bytes-on-the-wire is the URL
        # length — meaningless as an image size — and we can't claim pixels
        # because we never saw the content. ``width``/``height`` on the spec
        # are unverified caller hints, deliberately not surfaced as
        # measurements. Returning zeros makes "remote means no measurement"
        # visible in aggregate reports rather than silently faking signal.
        return Image()


__all__ = ["RemoteImageSpec"]
