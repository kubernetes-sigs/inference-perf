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
"""Synthetic MP4 video spec — one ``video_url`` block carrying an MP4 blob."""

from typing import Literal

from .base import VideoSpec


class SyntheticMp4VideoSpec(VideoSpec):
    """MP4-container video synthesized at materialization time from geometry."""

    kind: Literal["synthetic_mp4"] = "synthetic_mp4"


__all__ = ["SyntheticMp4VideoSpec"]
