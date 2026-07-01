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


class CounterResult(BaseModel):
    """Result of a counter query: the windowed total (increase), the average per-second rate
    over the window, and the overall per-second rate."""

    total: float = 0.0
    avg: float = 0.0
    per_second: float = 0.0


class CounterMetric(Metric[CounterResult]):
    """avg is the average per-second rate over the window (avg_over_time of the rate), matching the
    pre-refactor counter "mean" semantics rather than the window total."""

    def __init__(self, metric_name: str) -> None:
        self.metric_name = metric_name

    def _selector(self, filters: str) -> str:
        # A counter name may be a plain metric or a `{__name__=~"foo(_total)?"}` selector;
        # merge the filters inside the braces for the latter instead of appending a second group.
        m = self.metric_name
        if m.startswith("{") and m.endswith("}"):
            return f"{{{m[1:-1]},{filters}}}" if filters else m
        return f"{m}{{{filters}}}"

    def get_queries(self, duration: float, filters: str) -> List[str]:
        s = self._selector(filters)
        return [
            f"sum(increase({s}[{duration:.0f}s]))",
            f"avg_over_time(rate({s}[{duration:.0f}s])[{duration:.0f}s:{duration:.0f}s])",
            f"sum(rate({s}[{duration:.0f}s]))",
        ]

    def parse(self, results: List[float]) -> CounterResult:
        return CounterResult(total=results[0], avg=results[1], per_second=results[2])
