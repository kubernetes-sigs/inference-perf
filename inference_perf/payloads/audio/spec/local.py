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
"""Local-file audio spec — bytes read from disk at materialization time.

Stub: in :data:`AudioSpecUnion` (construction works) but not yet wired into
the chat.py materializer — passing one through ``to_request_body`` raises
``TypeError`` until that branch lands. The materializer will open ``path``,
base64-wrap the contents, and emit an
``input_audio`` block — same wire shape as :class:`PreEncodedAudioSpec`, just
with deferred disk reads. Realized ``bytes`` reflects the file size. A
wire-format selector (WAV / MP3 / ...) will be added alongside the
materializer wire-up.
"""

from typing import Literal

from ..metrics import Audio
from .base import AudioSpec


class LocalFileAudioSpec(AudioSpec):
    """Audio read from a local file path at materialization time."""

    kind: Literal["local_file"] = "local_file"
    path: str

    def get_metrics(self, wire_bytes: int) -> Audio:
        # ``wire_bytes`` is the on-disk file size. ``duration`` is the
        # loader's declared length from manifest metadata — trusted, since
        # we don't decode locally.
        return Audio(bytes=wire_bytes, seconds=self.duration)


__all__ = ["LocalFileAudioSpec"]
