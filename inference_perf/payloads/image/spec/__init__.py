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
"""Image pre-flight request specs.

Layout follows the modality-primary / provenance-secondary scheme:

- :class:`ImageSpec` (``base.py``) — shared fields for every image variant.
- :class:`SyntheticImageSpec` (``synthetic.py``) — image generated at
  materialization time.
- :class:`PreEncodedImageSpec` (``pre_encoded.py``) — stub for dataset-loader-
  supplied bytes; not yet wired into the materializer.
- :class:`RemoteImageSpec` (``remote.py``) — stub for URL-referenced media
  fetched server-side; not yet wired into the materializer.
- :class:`LocalFileImageSpec` (``local.py``) — stub for on-disk file paths
  read at materialization time; not yet wired into the materializer.

All four concrete variants carry a ``kind`` literal discriminator and are
unioned into :data:`ImageSpecUnion` so :class:`MultimodalSpec.images` can hold
any of them. Only :class:`SyntheticImageSpec` is materializer-wired today;
the others are typed stubs awaiting their materializer branches.
"""

from typing import Annotated, Union

from pydantic import Field

from .base import ImageRepresentation, ImageSpec
from .local import LocalFileImageSpec
from .pre_encoded import PreEncodedImageSpec
from .remote import RemoteImageSpec
from .synthetic import SyntheticImageSpec

ImageSpecUnion = Annotated[
    Union[SyntheticImageSpec, PreEncodedImageSpec, RemoteImageSpec, LocalFileImageSpec],
    Field(discriminator="kind"),
]

__all__ = [
    "ImageRepresentation",
    "ImageSpec",
    "ImageSpecUnion",
    "LocalFileImageSpec",
    "PreEncodedImageSpec",
    "RemoteImageSpec",
    "SyntheticImageSpec",
]
