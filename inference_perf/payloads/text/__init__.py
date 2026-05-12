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
"""Text modality.

Text has no pre-flight spec types — the text-side request fields (messages,
prompt) live on :class:`InferenceAPIData` subclasses, not on
:class:`MultimodalSpec`. The directory exists for structural consistency
with the other modalities and currently holds only the post-flight
:class:`Text` metric record.
"""

from .metrics import Text

__all__ = ["Text"]
