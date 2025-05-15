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
import statistics
from typing import List, Optional

from inference_perf.collectors.request_lifecycle import PromptLifecycleMetricsCollector
from inference_perf.config import RequestLifecycleMetricsReportConfig
from inference_perf.metrics import MetricsClient
from inference_perf.client.modelserver.base import ModelServerMetrics
from inference_perf.metrics.base import PerfRuntimeParameters
from inference_perf.report import ReportFile


class ReportGenerator:
    def __init__(
        self, lifecycle_metrics_collector: PromptLifecycleMetricsCollector, metrics_client: Optional[MetricsClient]
    ) -> None:
        self.lifecycle_metrics_collector = lifecycle_metrics_collector
        self.metrics_client = metrics_client

    async def generate_reports(
        self, request_lifecycle_metrics_config: RequestLifecycleMetricsReportConfig, runtime_parameters: PerfRuntimeParameters
    ) -> List[ReportFile]:
        print("\n\nGenerating Reports ..")
        lifecycle_reports = await self.lifecycle_metrics_collector.to_reports(report_config=request_lifecycle_metrics_config)
        metrics_client_summary = self.report_metrics_summary(runtime_parameters)
        return [
            *lifecycle_reports,
            ReportFile(name="metrics_client_report", contents=metrics_client_summary.model_dump()),
        ]

    def report_request_summary(self, runtime_parameters: PerfRuntimeParameters) -> ModelServerMetrics:
        """
        report request metrics collected by the model server client during the run.
        Args:
            runtime_parameters (PerfRuntimeParameters): The runtime parameters containing the model server client, query eval time in the metrics db, duration.
        """
        request_metrics = runtime_parameters.model_server_client.get_request_metrics()
        request_summary = ModelServerMetrics()
        if len(request_metrics) > 0:
            total_prompt_tokens = sum([x.prompt_tokens for x in request_metrics])
            total_output_tokens = sum([x.output_tokens for x in request_metrics])
            if runtime_parameters.duration > 0:
                prompt_tokens_per_second = total_prompt_tokens / runtime_parameters.duration
                output_tokens_per_second = total_output_tokens / runtime_parameters.duration
                requests_per_second = len(request_metrics) / runtime_parameters.duration
            else:
                prompt_tokens_per_second = 0.0
                output_tokens_per_second = 0.0
                requests_per_second = 0.0

            request_summary = ModelServerMetrics(
                total_requests=len(request_metrics),
                requests_per_second=requests_per_second,
                prompt_tokens_per_second=prompt_tokens_per_second,
                output_tokens_per_second=output_tokens_per_second,
                avg_prompt_tokens=int(statistics.mean([x.prompt_tokens for x in request_metrics])),
                avg_output_tokens=int(statistics.mean([x.output_tokens for x in request_metrics])),
                avg_request_latency=statistics.mean([x.time_per_request for x in request_metrics]),
                median_request_latency=statistics.median([x.time_per_request for x in request_metrics]),
                p90_request_latency=statistics.quantiles([x.time_per_request for x in request_metrics], n=10)[8],
                p99_request_latency=statistics.quantiles([x.time_per_request for x in request_metrics], n=100)[98],
                avg_time_to_first_token=0.0,
                median_time_to_first_token=0.0,
                p90_time_to_first_token=0.0,
                p99_time_to_first_token=0.0,
                median_time_per_output_token=0.0,
                p90_time_per_output_token=0.0,
                p99_time_per_output_token=0.0,
                avg_time_per_output_token=0.0,
                avg_queue_length=0,
            )
            print("-" * 50)
            print("Request Summary")
            print("-" * 50)
            for field_name, value in request_summary:
                print(f"{field_name}: {value}")
        else:
            print("Report generation failed - no request metrics collected")
        return request_summary

    def report_metrics_summary(self, runtime_parameters: PerfRuntimeParameters) -> ModelServerMetrics:
        """
        Report summary of the metrics collected by the metrics client during the run.
        Args:
            runtime_parameters (PerfRuntimeParameters): The runtime parameters containing the model server client, query eval time in the metrics db, duration.
        """
        metrics_client_summary = ModelServerMetrics()
        if self.metrics_client is not None:
            print("-" * 50)
            print("Metrics Client Summary")
            print("-" * 50)
            collected_metrics = self.metrics_client.collect_model_server_metrics(runtime_parameters)
            if collected_metrics is not None:
                metrics_client_summary = collected_metrics
                for field_name, value in metrics_client_summary:
                    print(f"{field_name}: {value}")
            else:
                print("Report generation failed - no metrics collected by metrics client")
        return metrics_client_summary
