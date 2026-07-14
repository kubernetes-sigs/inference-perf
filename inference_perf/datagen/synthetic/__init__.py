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
"""Generators that fabricate requests from distributions rather than data:
random/synthetic token streams, shared-prefix, multimodal, and mock."""

from .mock_datagen import MockDataGenerator
from .multimodal_datagen import MultimodalDataGenerator
from .random_datagen import RandomDataGenerator
from .shared_prefix_datagen import SharedPrefixDataGenerator
from .synthetic_datagen import SyntheticDataGenerator

__all__ = [
    "MockDataGenerator",
    "MultimodalDataGenerator",
    "RandomDataGenerator",
    "SharedPrefixDataGenerator",
    "SyntheticDataGenerator",
]
