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
"""Post-flight measurement records for video instances on a request."""

from typing import List, Optional

from pydantic import BaseModel


class Video(BaseModel):
    """Realized stats of one video instance (per-instance, recorded post-flight).

    None means not measured, not zero (see :class:`~inference_perf.payloads.Image`).
    ``pixels`` is per frame.
    """

    pixels: Optional[int] = None
    bytes: Optional[int] = None
    aspect_ratio: Optional[float] = None
    frames: Optional[int] = None
    # Duration, which frames alone do not give without a frame rate. No input
    # spec declares one, so only responses fill it.
    seconds: Optional[float] = None


class Videos(BaseModel):
    """Request-scoped container of all :class:`Video` records on one request."""

    count: int = 0
    instances: List[Video] = []


__all__ = ["Video", "Videos"]
