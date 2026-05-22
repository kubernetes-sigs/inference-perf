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
from enum import Enum
from typing import List, Optional, Union

from pydantic import BaseModel, Field

from inference_perf.config.common import Distribution
from inference_perf.payloads import ImageRepresentation, VideoRepresentation


# --- Base Utility Types ---
class ResolutionPreset(str, Enum):
    """Standard resolution shortcuts for image and video specs.

    Mappings:
      - ``"4k"``    → 3840 × 2160
      - ``"1080p"`` → 1920 × 1080
      - ``"720p"``  → 1280 × 720
      - ``"360p"``  → 640 × 360
    """

    P4K = "4k"
    P1080 = "1080p"
    P720 = "720p"
    P360 = "360p"


class Resolution(BaseModel):
    height: int = Field(description="Pixel height (e.g. 1080).")
    width: int = Field(description="Pixel width (e.g. 1920).")


AnyResolution = Union[ResolutionPreset, Resolution]


class WeightedResolution(BaseModel):
    resolution: AnyResolution = Field(description='A ``ResolutionPreset`` (e.g. ``"1080p"``) or explicit ``Resolution``.')
    weight: float = Field(default=1.0, description="Relative frequency of this resolution being selected from the list.")


class VideoProfile(BaseModel):
    resolution: AnyResolution = Field(description="Frame resolution. Preset string or explicit ``Resolution``.")
    frames: int = Field(description="Number of frames in the video. Required.")


class WeightedVideoProfile(BaseModel):
    profile: VideoProfile
    weight: float = Field(default=1.0, description="Relative frequency of this exact video profile being selected.")


class WeightedDuration(BaseModel):
    duration: float = Field(description="The length of the audio clip in seconds.")
    weight: float = Field(default=1.0, description="Relative frequency of this duration being selected.")


# --- Modality-Specific Request Configs ---
class MediaDatagenConfig(BaseModel):
    count: Optional[Distribution] = Field(
        default=None, description="Distribution of the number of media items to generate per request."
    )
    insertion_point: Optional[Union[float, Distribution]] = Field(
        default=None,
        description="Placement of media within the text prompt. Float in range [0.0, 1.0] (0=start, 1=end), or a Distribution to sample from.",
    )


class ImageDatagenConfig(MediaDatagenConfig):
    resolutions: Optional[Union[AnyResolution, List[WeightedResolution]]] = Field(
        default=None, description="Resolution or list of weighted resolutions for generated images."
    )
    representation: ImageRepresentation = Field(
        default=ImageRepresentation.PNG,
        description=(
            "Wire encoding for emitted image bytes: ``png`` (default, lossless) or ``jpeg`` "
            "(lossy, smaller payload). Some VLMs prefer one or the other; consult the model's spec sheet."
        ),
    )


class VideoDatagenConfig(MediaDatagenConfig):
    profiles: Optional[Union[VideoProfile, List[WeightedVideoProfile]]] = Field(
        default=None, description="Video profile or list of weighted video profiles for generated videos."
    )
    representation: VideoRepresentation = Field(
        default=VideoRepresentation.MP4,
        description=(
            "Wire-format strategy. ``mp4`` sends one ``video_url`` block carrying an MP4 blob "
            "(measures full pipeline including server-side decode). ``png_frames`` and "
            "``jpeg_frames`` send ``frames`` × ``image_url`` blocks at one insertion point in "
            "the named encoding (no decode dependency, useful for prefix-cache benchmarks "
            "and servers that don't accept ``video_url``)."
        ),
    )


class AudioDatagenConfig(MediaDatagenConfig):
    durations: Optional[Union[float, List[WeightedDuration]]] = Field(
        default=None, description="Duration or list of weighted durations for generated audio clips."
    )


# --- Main Wrapper Models ---
class SyntheticMultimodalDatagenConfig(BaseModel):
    """Configuration for standard multimodal data generation.

    Caveat: resolutions, video profiles, and audio durations specified here are
    sent to the model server as-is. There is no model-aware validation — real
    VLMs each impose their own limits (per-request media count via
    --limit-mm-per-prompt, vision-encoder pixel caps, video frame budgets,
    audio duration caps, max-model-len). Out-of-range payloads typically fail
    at the wire (counted in `failures`) or get silently downsized server-side,
    which makes per-modality byte/pixel throughput numbers reflect what was
    sent rather than what the model processed. Consult your model's spec sheet
    when picking values. See docs/config.md ("Multimodal Data Generation").
    """

    image: Optional[ImageDatagenConfig] = None
    video: Optional[VideoDatagenConfig] = None
    audio: Optional[AudioDatagenConfig] = None
