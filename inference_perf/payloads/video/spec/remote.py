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
"""Remote-URL video spec — server fetches bytes from a URL.

Stub: defined for the layout, not yet wired into the chat.py materializer or
added to :data:`VideoSpecUnion`. The materializer will emit the URL verbatim
in a single ``video_url`` wire block (whatever the URL points to — MP4 is the
usual case) and record ``bytes=0`` on the realized metric. Per-frame remote
URLs are intentionally not supported; that use case fits
:class:`PreEncodedFramesVideoSpec` after a local fetch. Private buckets are
expected to use signed URLs — no auth-headers field on the spec.
"""

from typing import Literal

from .base import VideoSpec


class RemoteVideoSpec(VideoSpec):
    """Video referenced by URL; the server fetches the bytes.

    Carries a ``kind`` discriminator because :class:`VideoSpec` variants already
    live in a discriminated union; image and audio remotes will get one when
    they're wired into their unions.
    """

    kind: Literal["remote"] = "remote"
    url: str


__all__ = ["RemoteVideoSpec"]
