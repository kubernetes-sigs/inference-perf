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
from .base import ReportGenerator
import statistics
from inference_perf.metrics import MetricsClient, MetricsSummary


class MockReportGenerator(ReportGenerator):
    def __init__(self, metrics_client: MetricsClient) -> None:
        self.metrics_client = metrics_client

    async def generate_report(self, runtime_parameters: PerfRuntimeParameters) -> None:
        print("\n\nGenerating Report ..")
        summary: MetricsSummary | None = None
        # check metrics_client has generated the summary, if not, use the request metrics from model server client
        if self.metrics_client is None:
            print("No metrics client provided, falling back to model server request metrics...")
        else:
            print("Using metrics client to generate report...")
            summary = self.metrics_client.collect_metrics_summary(runtime_parameters)

        if summary is not None:
            for field_name, value in summary:
                print(f"{field_name}: {value}")
        else:
            request_metrics = runtime_parameters.model_server_client.get_request_metrics()
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

                summary = MetricsSummary(
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
                for field_name, value in summary:
                    print(f"{field_name}: {value}")
            else:
                print("Report generation failed - no metrics collected")
