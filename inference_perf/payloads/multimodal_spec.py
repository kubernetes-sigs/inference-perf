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
"""Specs for multimodal request payloads.

Datagen samples concrete dimensions, profiles, and insertion points and
attaches a ``MultimodalSpec`` to the API data object. The API
implementation materializes bytes from the spec at request-build time and
records realized per-instance metrics on the same data object so that
``process_response`` can build the final ``RequestMetrics``.
"""

from enum import Enum
from typing import List

from pydantic import BaseModel, Field


class VideoRepresentation(str, Enum):
    """How a video instance is materialized on the wire.

    - ``MP4`` emits a single ``video_url`` block carrying an MP4-encoded blob;
      measures the full pipeline including the server's MP4 decode cost.
    - ``PNG_FRAMES`` emits ``frames`` × PNG ``image_url`` blocks at one
      insertion point. No server-side video-decode dependency; useful for
      prefix-cache benchmarks and servers that don't accept ``video_url``.
    - ``JPEG_FRAMES`` is the same as ``PNG_FRAMES`` but with JPEG-encoded
      frames — smaller wire payload, matches client pipelines that pre-extract
      and JPEG-compress frames before sending to the model server.
    """

    MP4 = "mp4"
    PNG_FRAMES = "png_frames"
    JPEG_FRAMES = "jpeg_frames"


class ImageRepresentation(str, Enum):
    """Wire encoding for standalone image payloads.

    Some VLMs accept either; some prefer JPEG for size, some PNG for fidelity.
    For per-frame encoding inside video specs, use the corresponding
    ``VideoRepresentation`` (``PNG_FRAMES`` / ``JPEG_FRAMES``) instead.
    """

    PNG = "png"
    JPEG = "jpeg"


class ImageInstanceSpec(BaseModel):
    width: int
    height: int
    insertion_point: float = Field(ge=0.0, le=1.0)
    representation: ImageRepresentation = ImageRepresentation.PNG


class VideoInstanceSpec(BaseModel):
    width: int
    height: int
    frames: int
    insertion_point: float = Field(ge=0.0, le=1.0)
    representation: VideoRepresentation = VideoRepresentation.MP4


class AudioInstanceSpec(BaseModel):
    duration: float
    insertion_point: float = Field(ge=0.0, le=1.0)


class MultimodalSpec(BaseModel):
    images: List[ImageInstanceSpec] = []
    videos: List[VideoInstanceSpec] = []
    audios: List[AudioInstanceSpec] = []
