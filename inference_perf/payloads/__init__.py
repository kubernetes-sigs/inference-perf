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
from typing import Any

from .multimodal_spec import (
    AudioInstanceSpec,
    ImageRepresentation,
    ImageInstanceSpec,
    MultimodalSpec,
    VideoInstanceSpec,
    VideoRepresentation,
)
from .payload import (
    Audio,
    Audios,
    Image,
    Images,
    RequestMetrics,
    Text,
    Video,
    Videos,
)

# In-flight wire body returned by ``InferenceAPIData.to_request_body`` and
# serialized to HTTP. Intentionally a plain dict — modeling it would mean
# materialized image/video/audio bytes living on the lifecycle metric.
RequestBody = dict[str, Any]

__all__ = [
    # Post-flight measurement records (payload.py)
    "Text",
    "Image",
    "Images",
    "Video",
    "Videos",
    "Audio",
    "Audios",
    "RequestMetrics",
    # Pre-flight request specs (multimodal_spec.py)
    "ImageInstanceSpec",
    "VideoInstanceSpec",
    "AudioInstanceSpec",
    "MultimodalSpec",
    "VideoRepresentation",
    "ImageRepresentation",
    # In-flight wire body alias
    "RequestBody",
]
