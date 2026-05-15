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
"""Audio modality.

Both halves of one audio clip's lifecycle live here:

- ``spec/`` — pre-flight request specs (modality-primary / provenance-secondary).
- ``metrics.py`` — post-flight measurement records.
"""

from .metrics import Audio, Audios
from .spec import (
    AudioSpec,
    AudioSpecUnion,
    LocalFileAudioSpec,
    PreEncodedAudioSpec,
    RemoteAudioSpec,
    SyntheticAudioSpec,
)

__all__ = [
    # Specs
    "AudioSpec",
    "AudioSpecUnion",
    "LocalFileAudioSpec",
    "PreEncodedAudioSpec",
    "RemoteAudioSpec",
    "SyntheticAudioSpec",
    # Metrics
    "Audio",
    "Audios",
]
