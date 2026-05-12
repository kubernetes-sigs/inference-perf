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
"""Local-file video spec — bytes read from disk at materialization time.

Stub: defined for the layout, not yet wired into the chat.py materializer or
added to :data:`VideoSpecUnion`. The materializer will open ``path``,
base64-wrap the contents, and emit a single ``video_url`` data URL (MP4 is
the usual case). Realized ``bytes`` reflects the file size.

This stub only covers the single-file (MP4) case, mirroring
:class:`RemoteVideoSpec`. A future ``LocalFileFramesVideoSpec`` could carry
``List[str]`` paths and emit N ``image_url`` blocks if frame-sequence local
files become a use case.
"""

from typing import Literal

from .base import VideoSpec


class LocalFileVideoSpec(VideoSpec):
    """Video read from a local file path at materialization time.

    Carries a ``kind`` discriminator because :class:`VideoSpec` variants already
    live in a discriminated union; image and audio local-file stubs will get
    one when they're wired into their unions.
    """

    kind: Literal["local_file"] = "local_file"
    path: str


__all__ = ["LocalFileVideoSpec"]
