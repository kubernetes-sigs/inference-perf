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

Stub: defined for the layout, not yet wired into the chat.py materializer or
into a discriminated union under :class:`MultimodalSpec.audios`. A future
dataset loader populates ``audio_bytes`` and the materializer will base64-wrap
the bytes for the wire ``input_audio`` block. A wire-format selector (WAV vs
MP3 vs ...) will be added alongside the materializer wire-up.
"""

from typing import Literal

from .base import AudioSpec


class PreEncodedAudioSpec(AudioSpec):
    """Audio whose bytes are already encoded."""

    kind: Literal["pre_encoded"] = "pre_encoded"
    audio_bytes: bytes


__all__ = ["PreEncodedAudioSpec"]
