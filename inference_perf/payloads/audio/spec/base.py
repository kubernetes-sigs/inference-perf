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
"""Audio spec base — shared fields across every provenance variant.

Itself abstract: the inherited :meth:`MediaSpec.get_metrics` is left
unimplemented, so concrete provenance subclasses must override it.
"""

from ...media_base import MediaSpec
from ..metrics import Audio


class AudioSpec(MediaSpec[Audio]):
    """Fields shared by every audio clip on the wire, regardless of provenance."""

    duration: float


__all__ = ["AudioSpec"]
