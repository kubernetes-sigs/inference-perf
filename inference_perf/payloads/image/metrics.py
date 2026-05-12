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
"""Post-flight measurement records for image instances on a request."""

from typing import List

from pydantic import BaseModel


class Image(BaseModel):
    """Realized stats of one image instance (per-instance, recorded post-flight)."""

    pixels: int = 0
    bytes: int = 0
    aspect_ratio: float = 0.0


class Images(BaseModel):
    """Request-scoped container of all :class:`Image` records on one request."""

    count: int = 0
    instances: List[Image] = []


__all__ = ["Image", "Images"]
