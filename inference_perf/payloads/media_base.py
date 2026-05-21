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
"""Shared abstract base for every modality's pre-flight request spec.

Each modality (image / video / audio / ...) binds :class:`MediaSpec` to its
own post-flight metric class via ``MediaSpec[Image]`` / ``MediaSpec[Video]``
/ ``MediaSpec[Audio]``. Each concrete provenance subclass under a modality
must implement :meth:`MediaSpec.get_metrics`, returning that modality's
metric type populated according to what the provenance can actually claim
about the bytes on the wire.

The abstract method is the contract that pushes per-provenance measurement
rules out of the materializer's ``isinstance`` chain and into the spec
class itself — adding a new provenance becomes a single self-contained file
that declares both *what bytes it puts on the wire* (the materializer
branch) and *what those bytes mean for measurement* (``get_metrics``).

Static-check coverage:

- mypy / pyright flag any concrete provenance subclass that forgets to
  override ``get_metrics``.
- The generic ``T_Metric`` binding flags any subclass that returns the
  wrong modality's metric (e.g. a ``VideoSpec`` subclass returning
  ``Image``).
- Runtime: instantiating a subclass that left ``get_metrics`` abstract
  raises ``TypeError`` from the standard ABC machinery.
"""

from abc import abstractmethod
from typing import Generic, TypeVar

from pydantic import BaseModel, Field

T_Metric = TypeVar("T_Metric", bound=BaseModel)


class MediaSpec(BaseModel, Generic[T_Metric]):
    """Abstract base for every modality's spec. Subclasses bind ``T_Metric``."""

    insertion_point: float = Field(ge=0.0, le=1.0)

    @abstractmethod
    def get_metrics(self, wire_bytes: int) -> T_Metric:
        """Build the post-flight metric record for this instance.

        ``wire_bytes`` is the count of bytes the materializer actually put
        on the wire for this instance (the encoded payload for synthetic /
        pre-encoded variants; ~0 for remote variants where we sent only a
        URL). Each provenance decides how to translate that into the
        modality's pixel / duration / frame fields based on what it can
        honestly claim — for example, :class:`RemoteImageSpec` returns
        zero pixels because we never saw the bytes; :class:`SyntheticImageSpec`
        returns ``width * height`` because it generated them.
        """
        ...


__all__ = ["MediaSpec", "T_Metric"]
