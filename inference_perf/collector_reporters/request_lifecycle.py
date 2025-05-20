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


class RequestLifecycleMetricsCollectorReporter(MetricsCollectorReporter):
    """Responsible for accumulating client request metrics and generating corresponding reports"""

    def __init__(self) -> None:
        self.metrics: List[PromptLifecycleMetric] = []

    def record_metric(self, metric: PromptLifecycleMetric) -> None:
        self.metrics.append(metric)

    async def reports(self, report_config: RequestLifecycleMetricsReportConfig) -> List[ReportFile]:
        reports: List[ReportFile] = []
        if report_config.summary:
            request_metrics = self.metrics
            if len(self.metrics) != 0:
                report_file = ReportFile(
                    name="summary_lifecycle_metrics",
                    contents=request_metrics[0].request.summarize_requests(request_metrics).model_dump(),
                )
                reports.append(report_file)
                if report_file.path is not None:
                    print(f"Successfully saved summary report of request lifecycle metrics to {report_file.path}")

        if report_config.per_stage:
            stage_buckets: dict[int, List[PromptLifecycleMetric]] = defaultdict(list)
            for metric in self.metrics:
                if metric.stage_id is not None:
                    stage_buckets[metric.stage_id].append(metric)
            for stage_id, metrics in stage_buckets.items():
                report_file = ReportFile(
                    name=f"stage_{stage_id}_lifecycle_metrics",
                    contents=metrics[0].request.summarize_requests(metrics).model_dump(),
                )
                reports.append(report_file)
                if report_file is not None:
                    print(f"Successfully saved per stage report of request lifecycle metrics to {report_file.path}")

        if report_config.per_request:
            report_file = ReportFile(
                name="per_request_lifecycle_metrics",
                contents=[
                    {
                        "start_time": metric.start_time,
                        "end_time": metric.end_time,
                        "request": metric.request.model_dump(),
                        "response": metric.response.model_dump(),
                    }
                    for metric in self.metrics
                ],
            )
            reports.append(report_file)
            if report_file is not None:
                print(f"Successfully saved per request report of request lifecycle metrics to {report_file.path}")
        return reports
