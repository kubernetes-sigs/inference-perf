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
"""Video spec base + the user-facing wire-format enum.

:class:`VideoSpec` holds fields shared by every video subtype regardless of
provenance or wire encoding (geometry, frame count, insertion point). The
concrete provenance- and container-specific subclasses live in sibling
modules and are discriminated by their ``kind`` literal.

:class:`VideoRepresentation` is the user-facing taxonomy used by config
(``VideoDatagenConfig.representation``, ``ShareGPT4VideoConfig.representation``)
to pick a wire-format strategy. Datagen translates a chosen representation
into the appropriate Spec subclass at sample time; Spec classes themselves
do not carry this enum.
"""

from enum import Enum

from pydantic import BaseModel, Field


class VideoSpec(BaseModel):
    """Fields common to every video on the wire."""

    width: int
    height: int
    frames: int
    insertion_point: float = Field(ge=0.0, le=1.0)


class VideoRepresentation(str, Enum):
    """User-facing wire-format strategy for video payloads.

    - ``MP4`` — single ``video_url`` block carrying an MP4 blob; measures the
      full pipeline including the server's MP4 decode cost.
    - ``PNG_FRAMES`` — ``frames`` × PNG ``image_url`` blocks at one insertion
      point. No server-side video-decode dependency; useful for prefix-cache
      benchmarks and servers that don't accept ``video_url``.
    - ``JPEG_FRAMES`` — same as ``PNG_FRAMES`` but with JPEG-encoded frames
      (smaller wire payload, matches client pipelines that pre-extract and
      JPEG-compress frames before sending).
    """

    MP4 = "mp4"
    PNG_FRAMES = "png_frames"
    JPEG_FRAMES = "jpeg_frames"


__all__ = ["VideoSpec", "VideoRepresentation"]
