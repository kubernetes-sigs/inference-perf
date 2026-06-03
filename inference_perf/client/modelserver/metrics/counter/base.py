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
from ..histogram.base import HistogramResult


class CounterResult(BaseModel):
    value: float = 0.0


class CounterMetric(Metric[HistogramResult]):
    """Returns HistogramResult (not CounterResult) because target fields like prompt_tokens may
    be populated by either CounterMetric (vllm/sglang) or HistogramMetric (tgi). Counter fills
    only avg+per_second; percentile fields default to 0."""

    def __init__(self, target_field: str, metric_name: str, filters: List[str]) -> None:
        self.target_field = target_field
        self.metric_name = metric_name
        self.filters = ",".join(filters)

    def get_queries(self, duration: float) -> List[str]:
        f, m = self.filters, self.metric_name
        return [
            f"sum(increase({m}{{{f}}}[{duration:.0f}s]))",
            f"sum(rate({m}{{{f}}}[{duration:.0f}s]))",
        ]

    def parse(self, results: List[float]) -> HistogramResult:
        return HistogramResult(avg=results[0], per_second=results[1])
