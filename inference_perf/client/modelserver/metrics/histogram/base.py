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
    """Distribution (avg/median/p90/p99 + as_summary, from GaugeResult) plus a per-second rate."""

    per_second: float = 0.0


class HistogramMetric(Metric[HistogramResult]):
    def __init__(self, metric_name: str) -> None:
        self.metric_name = metric_name

    def get_queries(self, duration: float, filters: str) -> List[str]:
        f, m = filters, self.metric_name
        return [
            f"sum(rate({m}_sum{{{f}}}[{duration:.0f}s])) / (sum(rate({m}_count{{{f}}}[{duration:.0f}s])) > 0)",
            f"histogram_quantile(0.5, sum(rate({m}_bucket{{{f}}}[{duration:.0f}s])) by (le))",
            f"histogram_quantile(0.9, sum(rate({m}_bucket{{{f}}}[{duration:.0f}s])) by (le))",
            f"histogram_quantile(0.99, sum(rate({m}_bucket{{{f}}}[{duration:.0f}s])) by (le))",
            f"sum(rate({m}_count{{{f}}}[{duration:.0f}s]))",
        ]

    def parse(self, results: List[float]) -> HistogramResult:
        return HistogramResult(avg=results[0], median=results[1], p90=results[2], p99=results[3], per_second=results[4])
