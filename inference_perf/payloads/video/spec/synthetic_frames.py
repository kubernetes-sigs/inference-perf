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
"""Synthetic frame-sequence video spec — N ``image_url`` blocks, generated per frame."""

from typing import Literal

from ...image import ImageRepresentation
from .base import VideoSpec


class SyntheticFramesVideoSpec(VideoSpec):
    """Frames-format video with each frame synthesized at materialization time.

    The materializer generates ``frames`` images sized ``width × height`` encoded
    as ``frame_representation`` (PNG or JPEG) and emits one ``image_url`` block
    per frame at the same insertion point.
    """

    kind: Literal["synthetic_frames"] = "synthetic_frames"
    frame_representation: ImageRepresentation = ImageRepresentation.PNG


__all__ = ["SyntheticFramesVideoSpec"]
