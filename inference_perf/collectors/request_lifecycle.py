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

from typing import List


from inference_perf.collectors.base import MetricsCollector
from inference_perf.config import RequestLifecycleMetricsReportConfig
from inference_perf.prompts.base import PromptLifecycleMetric
from inference_perf.report.base import ReportFile


class PromptLifecycleMetricsCollector(MetricsCollector[PromptLifecycleMetric]):
    """Responsible for accumulating client request metrics and generating corresponding reports"""

    def __init__(self) -> None:
        self.metrics: List[PromptLifecycleMetric] = []

    def record_metric(self, metric: PromptLifecycleMetric) -> None:
        self.metrics.append(metric)

    async def to_reports(self, report_config: RequestLifecycleMetricsReportConfig) -> List[ReportFile]:
        reports: List[ReportFile] = []
        if report_config.summary:
            request_metrics = self.metrics
            if len(self.metrics) != 0:
                reports.append(
                    ReportFile(
                        name="summary", contents=request_metrics[0].request.summarize_requests(request_metrics).model_dump()
                    )
                )
        if report_config.per_request:
            reports.append(ReportFile(name="per_request", contents=[metric.model_dump() for metric in self.metrics]))
        return reports
