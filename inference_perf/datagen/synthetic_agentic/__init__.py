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
"""Procedural synthetic multi-agent session generator: builds replay graphs
from config + theme without any recorded traffic."""

from .synthetic_agentic_datagen import SyntheticAgenticDataGenerator
from .synthetic_themes import GENERIC_THEME, Theme, load_theme

__all__ = [
    "SyntheticAgenticDataGenerator",
    "GENERIC_THEME",
    "Theme",
    "load_theme",
]
