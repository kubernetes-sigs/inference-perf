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
from typing import List, Optional, Any
from pydantic import BaseModel
from collections import defaultdict
from inference_perf.config import RequestLifecycleMetricsReportConfig
from inference_perf.client.metricsclient import MetricsClient, PerfRuntimeParameters, ModelServerMetrics
from inference_perf.utils import ReportFile
from inference_perf.client.requestdatacollector import LocalRequestDataCollector, RequestDataCollector
from inference_perf.apis import RequestLifecycleMetric
import numpy as np


def safe_float(value: Any) -> float:
    """NOTE: Only for use in summarize_requests after validating safe access"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0


def summarize(items: List[float]) -> Optional[dict[str, float]]:
    return (
        {
            "mean": float(np.mean(items)),
            "min": float(np.min(items)),
            "p10": float(np.percentile(items, 10)),
            "p50": float(np.percentile(items, 50)),
            "p90": float(np.percentile(items, 90)),
            "max": float(np.max(items)),
        }
        if len(items) != 0
        else None
    )


class ResponsesSummary(BaseModel):
    load_summary: dict[str, Any]
    successes: dict[str, Any]
    failures: dict[str, Any]


def summarize_requests(metrics: List[RequestLifecycleMetric]) -> ResponsesSummary:
    all_successful: List[RequestLifecycleMetric] = [x for x in metrics if x.error is None]
    all_failed: List[RequestLifecycleMetric] = [x for x in metrics if x.error is not None]

    total_time = max(x.end_time for x in metrics) - min(x.start_time for x in metrics)

    return ResponsesSummary(
        load_summary={
            "count": len(metrics),
        },
        successes={
            "count": len(all_successful),
            "throughput": {
                "input_tokens_per_sec": sum(x.info.input_tokens for x in all_successful) / total_time,
                "output_tokens_per_sec": sum(x.info.output_tokens for x in all_successful) / total_time,
                "total_tokens_per_sec": sum((x.info.input_tokens + x.info.output_tokens) for x in all_successful) / total_time,
                "requests_per_sec": len(all_successful) / total_time,
            },
            "request_latency": summarize([(successful.end_time - successful.start_time) for successful in all_successful]),
            "prompt_len": summarize([safe_float(success.info.input_tokens) for success in all_successful]),
            "output_len": summarize([float(v) for success in all_successful if (v := success.info.output_tokens) is not None]),
            "normalized_time_per_output_token": summarize(
                [
                    ((metric.end_time - metric.start_time) / output_len) if output_len and output_len != 0 else 0
                    for metric in all_successful
                    for output_len in [safe_float(metric.info.output_tokens)]
                ]
            ),
        },
        failures={
            "count": len(all_failed),
            "request_latency": summarize([(failed.end_time - failed.start_time) for failed in all_failed]),
        },
    )


class ReportGenerator:
    def __init__(
        self,
        metrics_client: Optional[MetricsClient],
    ) -> None:
        self.metrics_collector = LocalRequestDataCollector()
        self.metrics_client = metrics_client

    def get_metrics_collector(self) -> RequestDataCollector:
        """
        Returns the metrics collector.
        """
        return self.metrics_collector

    async def generate_reports(
        self, report_config: RequestLifecycleMetricsReportConfig, runtime_parameters: PerfRuntimeParameters
    ) -> List[ReportFile]:
        print("\n\nGenerating Reports ..")
        lifecycle_reports = []
        request_metrics = self.metrics_collector.get_metrics()
        if report_config.summary:
            if len(request_metrics) != 0:
                report_file = ReportFile(
                    name="summary_lifecycle_metrics",
                    contents=summarize_requests(request_metrics).model_dump(),
                )
                lifecycle_reports.append(report_file)
                if report_file.path is not None:
                    print(f"Successfully saved summary report of request lifecycle metrics to {report_file.path}")

        if report_config.per_stage:
            stage_buckets: dict[int, List[RequestLifecycleMetric]] = defaultdict(list)
            for metric in request_metrics:
                if metric.stage_id is not None:
                    stage_buckets[metric.stage_id].append(metric)
            for stage_id, metrics in stage_buckets.items():
                report_file = ReportFile(
                    name=f"stage_{stage_id}_lifecycle_metrics",
                    contents=summarize_requests(metrics).model_dump(),
                )
                lifecycle_reports.append(report_file)
                if report_file is not None:
                    print(f"Successfully saved stage {stage_id} report of request lifecycle metrics to {report_file.path}")

        if report_config.per_request:
            report_file = ReportFile(
                name="per_request_lifecycle_metrics",
                contents=[
                    {
                        "start_time": metric.start_time,
                        "end_time": metric.end_time,
                        "request": metric.request_data,
                        "response": metric.response_data,
                    }
                    for metric in request_metrics
                ],
            )
            lifecycle_reports.append(report_file)
            if report_file is not None:
                print(f"Successfully saved per request report of request lifecycle metrics to {report_file.path}")

        if self.metrics_client is not None:
            metrics_client_summary = self.report_metrics_summary(runtime_parameters)
            lifecycle_reports.append(ReportFile(name="metrics_client_report", contents=metrics_client_summary.model_dump()))

        return lifecycle_reports

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
