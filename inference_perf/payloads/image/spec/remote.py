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

Stub: defined for the layout, not yet wired into the chat.py materializer or
into a discriminated union under :class:`MultimodalSpec.images`. The
materializer will emit the URL verbatim in an ``image_url`` wire block and
record ``bytes=0`` on the realized metric (we send a URL, not pixels; the
server does the fetch). Private buckets are expected to use signed URLs —
no auth-headers field on the spec.
"""

from typing import Literal

from .base import ImageSpec


class RemoteImageSpec(ImageSpec):
    """Image referenced by URL; the server fetches the bytes."""

    kind: Literal["remote"] = "remote"
    url: str


__all__ = ["RemoteImageSpec"]
