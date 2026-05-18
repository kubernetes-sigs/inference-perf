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
"""Modality data contracts: pre-flight specs and post-flight metric records.

Modality is the primary organizing axis — each per-modality subpackage
co-locates the spec types (the *plan*: sampled by datagen, consumed by the
materializer) and the metric types (the *measurement*: recorded by the
materializer, consumed by reportgen) for that modality. Inside each modality
package:

- ``spec/`` — pre-flight request specs, factored modality-primary /
  provenance-secondary (synthetic / pre_encoded / remote / local). The video
  variants are unioned into :data:`VideoSpecUnion`.
- ``metrics.py`` — post-flight measurement records for that modality.

The text modality has no spec types — text-side request fields live on
:class:`InferenceAPIData` subclasses — and only contributes :class:`Text` to
the metric aggregator.

This module holds the two cross-modality aggregators that mirror each other
field-for-field:

- :class:`MultimodalSpec` — request-scoped container of per-modality specs.
- :class:`RequestMetrics` — request-scoped container of per-modality metrics
  plus :class:`Text`.

:class:`VideoRepresentation` lives in ``video/spec/base.py`` as the
user-facing wire-format enum; Spec classes themselves don't carry it.
"""

from typing import Any, List, Optional

from pydantic import BaseModel

from .audio import (
    Audio,
    Audios,
    AudioSpec,
    AudioSpecUnion,
    LocalFileAudioSpec,
    PreEncodedAudioSpec,
    RemoteAudioSpec,
    SyntheticAudioSpec,
)
from .image import (
    Image,
    ImageRepresentation,
    Images,
    ImageSpec,
    ImageSpecUnion,
    LocalFileImageSpec,
    PreEncodedImageSpec,
    RemoteImageSpec,
    SyntheticImageSpec,
)
from .text import Text
from .video import (
    LocalFileVideoSpec,
    PreEncodedFramesVideoSpec,
    RemoteVideoSpec,
    SyntheticFramesVideoSpec,
    SyntheticMp4VideoSpec,
    Video,
    VideoRepresentation,
    Videos,
    VideoSpec,
    VideoSpecUnion,
)


class MultimodalSpec(BaseModel):
    """Aggregate container of every modality's pre-flight specs on one request."""

    images: List[ImageSpecUnion] = []
    videos: List[VideoSpecUnion] = []
    audios: List[AudioSpecUnion] = []


class RequestMetrics(BaseModel):
    """Aggregate container of every modality's post-flight metrics on one request."""

    text: Text
    image: Optional[Images] = None
    video: Optional[Videos] = None
    audio: Optional[Audios] = None


# In-flight wire body returned by ``InferenceAPIData.to_request_body`` and
# serialized to HTTP. Intentionally a plain dict — modeling it would mean
# materialized image/video/audio bytes living on the lifecycle metric.
RequestBody = dict[str, Any]


__all__ = [
    # Aggregates
    "MultimodalSpec",
    "RequestMetrics",
    # Text
    "Text",
    # Image
    "ImageSpec",
    "ImageSpecUnion",
    "SyntheticImageSpec",
    "PreEncodedImageSpec",
    "RemoteImageSpec",
    "LocalFileImageSpec",
    "ImageRepresentation",
    "Image",
    "Images",
    # Audio
    "AudioSpec",
    "AudioSpecUnion",
    "SyntheticAudioSpec",
    "PreEncodedAudioSpec",
    "RemoteAudioSpec",
    "LocalFileAudioSpec",
    "Audio",
    "Audios",
    # Video
    "VideoSpec",
    "VideoSpecUnion",
    "VideoRepresentation",
    "SyntheticMp4VideoSpec",
    "SyntheticFramesVideoSpec",
    "PreEncodedFramesVideoSpec",
    "RemoteVideoSpec",
    "LocalFileVideoSpec",
    "Video",
    "Videos",
    # In-flight wire body alias
    "RequestBody",
]
