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

from ..base import Metric
from ..counter.base import CounterResult


class CustomMetric(Metric[CounterResult]):
    def __init__(self, target_field: str, metric_name: str, op: str, type: str, filters: List[str]) -> None:
        self.target_field = target_field
        self.metric_name = metric_name
        self.op = op
        self.type = type
        self.filters = ",".join(filters)

    def _selector(self) -> str:
        # If the metric name is wrapped in `{...}` (e.g. `{__name__=~"foo(_total)?"}`),
        # merge the filters into the selector instead of appending a second `{...}` group.
        m = self.metric_name
        if m.startswith("{") and m.endswith("}"):
            return f"{{{m[1:-1]},{self.filters}}}" if self.filters else m
        return f"{m}{{{self.filters}}}"

    def get_queries(self, duration: float) -> List[str]:
        if self.type != "counter":
            return []
        selector = self._selector()
        if self.op == "rate":
            return [f"sum(rate({selector}[{duration:.0f}s]))"]
        if self.op == "increase":
            return [f"sum(increase({selector}[{duration:.0f}s]))"]
        return []

    def parse(self, results: List[float]) -> CounterResult:
        return CounterResult(value=results[0]) if results else CounterResult()
