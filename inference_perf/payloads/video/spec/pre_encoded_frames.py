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
"""Pre-encoded frame-sequence video spec — frame bytes supplied by an upstream loader."""

from typing import Any, List, Literal

from pydantic import model_validator

from ...image import ImageRepresentation
from .base import VideoSpec


class PreEncodedFramesVideoSpec(VideoSpec):
    """Frames-format video with frame bytes already encoded (PNG or JPEG).

    Used by dataset loaders that extract real frames from on-disk videos. The
    materializer base64-wraps each entry of ``frames_bytes`` directly — no
    synthesis. ``frames`` is auto-derived from ``len(frames_bytes)`` if not
    supplied; if supplied, it must match.
    """

    kind: Literal["pre_encoded_frames"] = "pre_encoded_frames"
    frame_representation: ImageRepresentation = ImageRepresentation.JPEG
    frames_bytes: List[bytes]

    @model_validator(mode="before")
    @classmethod
    def _derive_frames(cls, data: Any) -> Any:
        # Allow callers to omit `frames`; we fill it in from frames_bytes so the
        # inherited required `frames: int` field is satisfied without making it
        # Optional on the subclass.
        if isinstance(data, dict):
            fb = data.get("frames_bytes")
            if fb is not None and data.get("frames") is None:
                data["frames"] = len(fb)
        return data

    @model_validator(mode="after")
    def _validate_frame_count(self) -> "PreEncodedFramesVideoSpec":
        if self.frames != len(self.frames_bytes):
            raise ValueError(f"frames ({self.frames}) does not match len(frames_bytes) ({len(self.frames_bytes)}).")
        return self


__all__ = ["PreEncodedFramesVideoSpec"]
