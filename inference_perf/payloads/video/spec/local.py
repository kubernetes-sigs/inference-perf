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

Stub: in :data:`VideoSpecUnion` (construction works) but not yet wired into
the chat.py materializer — passing one through ``to_request_body`` raises
``TypeError`` until that branch lands. The materializer will open ``path``,
base64-wrap the contents, and emit a single ``video_url`` data URL (MP4 is
the usual case). Realized ``bytes`` reflects the file size.

This stub only covers the single-file (MP4) case, mirroring
:class:`RemoteVideoSpec`. A future ``LocalFileFramesVideoSpec`` could carry
``List[str]`` paths and emit N ``image_url`` blocks if frame-sequence local
files become a use case.
"""

from typing import Literal

from ..metrics import Video
from .base import VideoSpec


class LocalFileVideoSpec(VideoSpec):
    """Video read from a local file path at materialization time."""

    kind: Literal["local_file"] = "local_file"
    path: str

    def get_metrics(self, wire_bytes: int) -> Video:
        # ``wire_bytes`` is the on-disk file size. Geometry/frames come
        # from the spec — the loader is responsible for populating them from
        # container metadata, since we don't decode locally.
        return Video(
            pixels=self.width * self.height,
            bytes=wire_bytes,
            aspect_ratio=self.width / self.height if self.height > 0 else 0.0,
            frames=self.frames,
        )


__all__ = ["LocalFileVideoSpec"]
