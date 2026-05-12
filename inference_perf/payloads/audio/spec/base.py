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
"""Audio spec base — shared fields across every provenance variant."""

from pydantic import BaseModel, Field


class AudioSpec(BaseModel):
    """Fields shared by every audio clip on the wire, regardless of provenance."""

    duration: float
    insertion_point: float = Field(ge=0.0, le=1.0)


__all__ = ["AudioSpec"]
