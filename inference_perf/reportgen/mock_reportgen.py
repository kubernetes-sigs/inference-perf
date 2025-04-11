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
from inference_perf.metrics.base import PerfRuntimeParameters
from .base import ReportGenerator, RequestMetric
from typing import List
import statistics
from inference_perf.metrics import MetricsClient, MetricsSummary


class MockReportGenerator(ReportGenerator):
    def __init__(self, metrics_client: MetricsClient) -> None:
        self.metrics_client = metrics_client
        self.metrics: List[RequestMetric] = []

    def collect_request_metrics(self, metric: RequestMetric) -> None:
        self.metrics.append(metric)

    async def generate_report(self, runtime_parameters: PerfRuntimeParameters) -> None:
        print("\n\nGenerating Report ..")
        summary = self.metrics_client.collect_metrics_summary(runtime_parameters)
        if summary is not None:
            for field_name, value in summary:
                print(f"{field_name}: {value}")

        elif summary is None and len(self.metrics) > 0:
            summary = MetricsSummary(
                total_requests=len(self.metrics),
                avg_prompt_tokens=int(statistics.mean([x.prompt_tokens for x in self.metrics])),
                avg_output_tokens=int(statistics.mean([x.output_tokens for x in self.metrics])),
                avg_request_latency=statistics.mean([x.time_per_request for x in self.metrics]),
                avg_time_to_first_token=0.0,
                avg_time_per_output_token=0.0,
                avg_queue_length=0,
            )
            for field_name, value in summary:
                print(f"{field_name}: {value}")
        else:
            print("Report generation failed - no metrics collected")
