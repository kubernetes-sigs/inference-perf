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
from .base import ReportGenerator, RequestMetric, MetricsSummary
from typing import List
import statistics


class MockReportGenerator(ReportGenerator):
    def __init__(self) -> None:
        self.metrics: List[RequestMetric] = []

    def collect_request_metrics(self, metric: RequestMetric) -> None:
        self.metrics.append(metric)

    async def generate_report(self) -> None:
        print("\n\nGenerating Report ..")
        summary = MetricsSummary(
            total_requests=len(self.metrics),
            avg_prompt_tokens=statistics.mean([x.prompt_tokens for x in self.metrics]),
            avg_completion_tokens=statistics.mean([x.completion_tokens for x in self.metrics]),
            avg_latency=statistics.mean([x.time_taken for x in self.metrics]),
        )

        print(summary.model_dump())
