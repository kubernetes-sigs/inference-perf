# Copyright 2025 The Kubernetes Authors.
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
from .base import build_trigger, Trigger, HitSample
from .config import TriggerSpec

from pkgutil import walk_packages
from importlib import import_module

for module_info in walk_packages(__path__):
    import_module(f"{__name__}.{module_info.name}")

__all__ = [
    "build_trigger",
    "HitSample",
    "Trigger",
    "TriggerSpec",
]
