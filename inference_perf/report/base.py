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

import json
import os
from typing import Any, Optional


class ReportFile:
    name: str
    contents: Any
    path: Optional[str] = None

    def __init__(self, name: str, contents: Any):
        self.name = f"{name}.json"
        self.contents = contents
        self._store_locally()

    def _store_locally(self) -> None:
        filename = self.get_filename()
        contents = self.get_contents()
        with open(filename, "w", encoding="utf-8") as f:
            f.write(json.dumps(contents, indent=2))
            self.path = os.path.abspath(filename)

    def get_filename(self) -> str:
        return self.name

    def get_contents(self) -> Any:
        return self.contents
