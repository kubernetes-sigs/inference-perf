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
from ..gauge.base import GaugeResult


class HistogramResult(GaugeResult):
    per_second: float = 0.0


class HistogramMetric(Metric[HistogramResult]):
    def __init__(self, target_field: str, metric_name: str, filters: List[str]) -> None:
        self.target_field = target_field
        self.metric_name = metric_name
        self.filters = ",".join(filters)

    def get_queries(self, duration: float) -> List[str]:
        f, m = self.filters, self.metric_name
        return [
            f"sum(rate({m}_sum{{{f}}}[{duration:.0f}s])) / (sum(rate({m}_count{{{f}}}[{duration:.0f}s])) > 0)",
            f"histogram_quantile(0.5, sum(rate({m}_bucket{{{f}}}[{duration:.0f}s])) by (le))",
            f"histogram_quantile(0.9, sum(rate({m}_bucket{{{f}}}[{duration:.0f}s])) by (le))",
            f"histogram_quantile(0.99, sum(rate({m}_bucket{{{f}}}[{duration:.0f}s])) by (le))",
            f"sum(rate({m}_sum{{{f}}}[{duration:.0f}s]))",
        ]

    def parse(self, results: List[float]) -> HistogramResult:
        return HistogramResult(avg=results[0], median=results[1], p90=results[2], p99=results[3], per_second=results[4])
