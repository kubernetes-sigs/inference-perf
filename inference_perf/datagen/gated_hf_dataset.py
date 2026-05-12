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
"""Shared base for loaders backed by gated HuggingFace datasets.

Gated datasets require a HuggingFace access token to download. This base
resolves the token from explicit config first, then ``HF_TOKEN`` /
``HUGGING_FACE_HUB_TOKEN`` env vars. Subclasses call ``self._load_dataset``
to stream rows with the token wired in.
"""

from __future__ import annotations

import os
from typing import Any, Optional

from datasets import load_dataset


class GatedHFDataset:
    """Mixin for data generators backed by gated HuggingFace datasets."""

    def __init__(self, token: Optional[str] = None) -> None:
        resolved = token or os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN")
        if not resolved:
            raise ValueError(
                f"{type(self).__name__} requires a HuggingFace access token. "
                "Provide one via the loader's `token` config field or set the HF_TOKEN environment variable."
            )
        self._hf_token: str = resolved

    def _load_dataset(self, path: str, **kwargs: Any) -> Any:
        return load_dataset(path, token=self._hf_token, **kwargs)
