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

from collections import defaultdict
from typing import List


from inference_perf.collector_reporters import MetricsCollectorReporter
from inference_perf.config import RequestLifecycleMetricsReportConfig
from inference_perf.prompts.base import PromptLifecycleMetric
from inference_perf.report import ReportFile


class PromptLifecycleMetricsCollectorReporter(MetricsCollectorReporter):
    """Responsible for accumulating client request metrics and generating corresponding reports"""

    def __init__(self) -> None:
        self.metrics: List[PromptLifecycleMetric] = []

    def record_metric(self, metric: PromptLifecycleMetric) -> None:
        self.metrics.append(metric)

    async def to_reports(self, report_config: RequestLifecycleMetricsReportConfig) -> List[ReportFile]:
        reports: List[ReportFile] = []
        if report_config.summary:
            print("Generating a summary report of request lifecycle metrics")
            request_metrics = self.metrics
            if len(self.metrics) != 0:
                reports.append(
                    ReportFile(
                        name="summary", contents=request_metrics[0].request.summarize_requests(request_metrics).model_dump()
                    )
                )

        if report_config.per_stage:
            print("Generating a per stage report of request lifecycle metrics")
            stage_buckets: dict[int, List[PromptLifecycleMetric]] = defaultdict(list)
            for metric in self.metrics:
                if metric.stage_id is not None:
                    stage_buckets[metric.stage_id].append(metric)
            for stage_id, metrics in stage_buckets.items():
                reports.append(
                    ReportFile(name=f"stage_{stage_id}", contents=metrics[0].request.summarize_requests(metrics).model_dump())
                )

        if report_config.per_request:
            print("Generating a per request report of request lifecycle metrics")
            reports.append(ReportFile(name="per_request", contents=[metric.model_dump() for metric in self.metrics]))

        return reports
