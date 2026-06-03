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
from typing import Dict, List
from pydantic import BaseModel

from ..base import Metric


class GaugeResult(BaseModel):
    avg: float = 0.0
    median: float = 0.0
    p90: float = 0.0
    p99: float = 0.0

    def as_summary(self) -> Dict[str, float]:
        """Project to the report's mean/median/p90/p99 shape.

        HistogramResult inherits this and is intentionally narrowed to the same
        four keys (its per_second field is not part of the per-metric summary).
        """
        return {"mean": self.avg, "median": self.median, "p90": self.p90, "p99": self.p99}


class GaugeMetric(Metric[GaugeResult]):
    def __init__(self, metric_name: str) -> None:
        self.metric_name = metric_name

    def get_queries(self, duration: float, filters: str) -> List[str]:
        f, m = filters, self.metric_name
        return [
            f"avg_over_time({m}{{{f}}}[{duration:.0f}s])",
            f"quantile_over_time(0.5, {m}{{{f}}}[{duration:.0f}s])",
            f"quantile_over_time(0.9, {m}{{{f}}}[{duration:.0f}s])",
            f"quantile_over_time(0.99, {m}{{{f}}}[{duration:.0f}s])",
        ]

    def parse(self, results: List[float]) -> GaugeResult:
        return GaugeResult(avg=results[0], median=results[1], p90=results[2], p99=results[3])
