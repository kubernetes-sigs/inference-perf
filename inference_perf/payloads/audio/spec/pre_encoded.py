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
"""Pre-encoded audio spec — bytes already encoded by an upstream loader.

Stub: in :data:`AudioSpecUnion` (construction works) but not yet wired into
the chat.py materializer — passing one through ``to_request_body`` raises
``TypeError`` until that branch lands. A future dataset loader populates
``audio_bytes`` and the materializer will base64-wrap the bytes for the wire
``input_audio`` block. A wire-format selector (WAV vs MP3 vs ...) will be
added alongside the materializer wire-up.
"""

from typing import Literal

from ..metrics import Audio
from .base import AudioSpec


class PreEncodedAudioSpec(AudioSpec):
    """Audio whose bytes are already encoded."""

    kind: Literal["pre_encoded"] = "pre_encoded"
    audio_bytes: bytes

    def get_metrics(self, wire_bytes: int) -> Audio:
        # Bytes are the loader-supplied blob. ``duration`` is the loader's
        # declared length — trusted as authoritative.
        return Audio(bytes=wire_bytes, seconds=self.duration)


__all__ = ["PreEncodedAudioSpec"]
