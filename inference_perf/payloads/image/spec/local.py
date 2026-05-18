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
"""Local-file image spec — bytes read from disk at materialization time.

Stub: in :data:`ImageSpecUnion` (construction works) but not yet wired into
the chat.py materializer — passing one through ``to_request_body`` raises
``TypeError`` until that branch lands. The materializer will open ``path``,
base64-wrap the contents, and emit an
``image_url`` data URL using ``representation`` for the mime type — same wire
shape as :class:`PreEncodedImageSpec`, just with deferred disk reads instead
of in-memory bytes. Realized ``bytes`` reflects the file size.
"""

from typing import Literal

from ..metrics import Image
from .base import ImageSpec


class LocalFileImageSpec(ImageSpec):
    """Image read from a local file path at materialization time."""

    kind: Literal["local_file"] = "local_file"
    path: str

    def get_metrics(self, wire_bytes: int) -> Image:
        # ``wire_bytes`` is the on-disk file size measured by the
        # materializer when it opened ``path``. Pixels come from the spec's
        # declared geometry — the loader is responsible for populating
        # ``width``/``height`` from manifest metadata, since we don't decode
        # locally. Aspect derives from those.
        return Image(
            pixels=self.width * self.height,
            bytes=wire_bytes,
            aspect_ratio=self.width / self.height if self.height > 0 else 0.0,
        )


__all__ = ["LocalFileImageSpec"]
