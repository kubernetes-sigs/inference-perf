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
"""Video pre-flight request specs.

Layout follows the modality-primary / provenance-secondary scheme:

- :class:`VideoSpec` (``base.py``) — shared fields for every video variant,
  plus :class:`VideoRepresentation` (the user-facing wire-format enum used by
  config, not by Spec classes).
- :class:`SyntheticMp4VideoSpec` (``synthetic_mp4.py``) — single ``video_url``
  MP4 blob synthesized at materialization time.
- :class:`SyntheticFramesVideoSpec` (``synthetic_frames.py``) — N
  ``image_url`` blocks, each frame synthesized at materialization time.
- :class:`PreEncodedFramesVideoSpec` (``pre_encoded_frames.py``) — N
  ``image_url`` blocks, frame bytes pre-supplied by an upstream dataset
  loader.
- :class:`RemoteVideoSpec` (``remote.py``) — stub for URL-referenced video
  fetched server-side; not yet in :data:`VideoSpecUnion` or the materializer.
- :class:`LocalFileVideoSpec` (``local.py``) — stub for an on-disk video file
  read at materialization time (single-file / MP4 case); not yet in
  :data:`VideoSpecUnion` or the materializer.

Use :data:`VideoSpecUnion` when typing a list/field that holds any concrete
subclass — Pydantic discriminates on the ``kind`` literal.
"""

from typing import Annotated, Union

from pydantic import Field

from .base import VideoRepresentation, VideoSpec
from .local import LocalFileVideoSpec
from .pre_encoded_frames import PreEncodedFramesVideoSpec
from .remote import RemoteVideoSpec
from .synthetic_frames import SyntheticFramesVideoSpec
from .synthetic_mp4 import SyntheticMp4VideoSpec

VideoSpecUnion = Annotated[
    Union[SyntheticMp4VideoSpec, SyntheticFramesVideoSpec, PreEncodedFramesVideoSpec],
    Field(discriminator="kind"),
]

__all__ = [
    "VideoSpec",
    "VideoSpecUnion",
    "VideoRepresentation",
    "SyntheticMp4VideoSpec",
    "SyntheticFramesVideoSpec",
    "PreEncodedFramesVideoSpec",
    "RemoteVideoSpec",
    "LocalFileVideoSpec",
]
