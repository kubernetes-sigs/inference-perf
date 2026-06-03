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
from typing import List
from pydantic import BaseModel

from ..base import Metric


class RequestsResult(BaseModel):
    total: float = 0.0
    per_second: float = 0.0


class RequestsMetric(Metric[RequestsResult]):
    def __init__(self, metric_name: str, filters: List[str], target_field: str = "requests") -> None:
        self.target_field = target_field
        self.metric_name = metric_name
        self.filters = ",".join(filters)

    def _selector(self) -> str:
        m = self.metric_name
        if m.startswith("{") and m.endswith("}"):
            return f"{{{m[1:-1]},{self.filters}}}" if self.filters else m
        return f"{m}{{{self.filters}}}"

    def get_queries(self, duration: float) -> List[str]:
        selector = self._selector()
        return [
            f"sum(increase({selector}[{duration:.0f}s]))",
            f"sum(rate({selector}[{duration:.0f}s]))",
        ]

    def parse(self, results: List[float]) -> RequestsResult:
        return RequestsResult(total=results[0], per_second=results[1])
