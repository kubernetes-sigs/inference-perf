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
from typing import Callable

__all__ = ["main_cli"]


# Resolved on first use so that importing a subpackage, the record types for
# one, does not load the whole application with it.
def __getattr__(name: str) -> Callable[[], None]:
    if name == "main_cli":
        from .main import main_cli

        return main_cli
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
