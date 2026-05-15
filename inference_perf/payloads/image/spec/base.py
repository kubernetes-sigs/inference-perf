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
"""Image spec base — shared fields across every provenance variant."""

from enum import Enum

from pydantic import BaseModel, Field


class ImageRepresentation(str, Enum):
    """Wire encoding for image bytes (standalone images and per-frame video).

    Some VLMs accept either; some prefer JPEG for size, some PNG for fidelity.
    """

    PNG = "png"
    JPEG = "jpeg"


class ImageSpec(BaseModel):
    """Fields shared by every image on the wire, regardless of provenance."""

    width: int
    height: int
    insertion_point: float = Field(ge=0.0, le=1.0)
    representation: ImageRepresentation = ImageRepresentation.PNG


__all__ = ["ImageRepresentation", "ImageSpec"]
