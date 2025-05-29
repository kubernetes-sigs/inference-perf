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
from inference_perf.client.metricsclient.base import ModelServerMetrics
from inference_perf.client.metricsclient.prometheus_client import PrometheusMetricsClient
from inference_perf.config import ReportConfig, PrometheusMetricsReportConfig
from inference_perf.client.metricsclient import MetricsClient, PerfRuntimeParameters
from inference_perf.utils import ReportFile
from inference_perf.client.requestdatacollector import RequestDataCollector
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


def summarize_prometheus_metrics(metrics: ModelServerMetrics) -> ResponsesSummary:
    return ResponsesSummary(
        load_summary={},  # model server doesn't report failed requests
        failures={},
        successes={
            "count": metrics.total_requests,
            "rate": metrics.requests_per_second,
            "prompt_len": {
                "mean": metrics.avg_prompt_tokens,
                "rate": metrics.prompt_tokens_per_second,
            },
            "output_len": {
                "mean": metrics.avg_output_tokens,
                "rate": metrics.output_tokens_per_second,
            },
            "queue_len": {
                "mean": metrics.avg_queue_length,
            },
            "request_latency": {
                "mean": metrics.avg_request_latency,
                "p50": metrics.median_request_latency,
                "p90": metrics.p90_request_latency,
                "p99": metrics.p99_request_latency,
            },
            "time_to_first_token": {
                "mean": metrics.avg_time_to_first_token,
                "p50": metrics.median_time_to_first_token,
                "p90": metrics.p90_time_to_first_token,
                "p99": metrics.p99_time_to_first_token,
            },
            "time_per_output_token": {
                "mean": metrics.avg_time_per_output_token,
                "p50": metrics.median_time_per_output_token,
                "p90": metrics.p90_time_per_output_token,
                "p99": metrics.p99_time_per_output_token,
            },
        },
    )


def summarize_requests(metrics: List[RequestLifecycleMetric]) -> ResponsesSummary:
    all_successful: List[RequestLifecycleMetric] = [x for x in metrics if x.error is None]
    all_failed: List[RequestLifecycleMetric] = [x for x in metrics if x.error is not None]

    return ResponsesSummary(
        load_summary={
            "count": len(metrics),
        },
        successes={
            "count": len(all_successful),
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
        metrics_collector: RequestDataCollector,
    ) -> None:
        self.metrics_collector = metrics_collector
        self.metrics_client = metrics_client

    def get_metrics_collector(self) -> RequestDataCollector:
        """
        Returns the metrics collector.
        """
        return self.metrics_collector

    async def generate_reports(
        self, report_config: ReportConfig, runtime_parameters: PerfRuntimeParameters
    ) -> List[ReportFile]:
        print("\n\nGenerating Reports ..")
        lifecycle_reports = []
        request_metrics = self.metrics_collector.get_metrics()
        if report_config.request_lifecycle.summary:
            if len(request_metrics) != 0:
                report_file = ReportFile(
                    name="summary_lifecycle_metrics",
                    contents=summarize_requests(request_metrics).model_dump(),
                )
                lifecycle_reports.append(report_file)
                if report_file.path is not None:
                    print(f"Successfully saved summary report of request lifecycle metrics to {report_file.path}")

        if report_config.request_lifecycle.per_stage:
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

        if report_config.request_lifecycle.per_request:
            report_file = ReportFile(
                name="per_request_lifecycle_metrics",
                contents=[
                    {
                        "start_time": metric.start_time,
                        "end_time": metric.end_time,
                        "request": metric.request_data,
                        "response": metric.response_data,
                        "error": metric.error.model_dump() if metric.error else None,
                    }
                    for metric in request_metrics
                ],
            )
            lifecycle_reports.append(report_file)
            if report_file is not None:
                print(f"Successfully saved per request report of request lifecycle metrics to {report_file.path}")

        lifecycle_reports.extend(self.generate_prometheus_metrics_report(runtime_parameters, report_config.prometheus))
        return lifecycle_reports

    def generate_prometheus_metrics_report(
        self, runtime_parameters: PerfRuntimeParameters, report_config: PrometheusMetricsReportConfig
    ) -> List[ReportFile]:
        """
        Report summary of the metrics collected by the metrics client during the run.
        Args:
            runtime_parameters (PerfRuntimeParameters): The runtime parameters containing the model server client, query eval time in the metrics db, duration.
        """
        prometheus_metrics_reports: List[ReportFile] = []

        if self.metrics_client is None or not isinstance(self.metrics_client, PrometheusMetricsClient):
            print("Prometheus Metrics Client is not configured or not of type PrometheusMetricsClient")
            return prometheus_metrics_reports

        # Wait for Prometheus to collect metrics for the last stage
        self.metrics_client.wait()

        if report_config.summary:
            collected_metrics = self.metrics_client.collect_metrics_summary(runtime_parameters)
            if collected_metrics is not None:
                report_file = ReportFile(
                    name="summary_prometheus_metrics",
                    contents=summarize_prometheus_metrics(collected_metrics).model_dump(),
                )
                if report_file is not None:
                    print(f"Successfully saved summary report of prometheus metrics to {report_file.path}")
                prometheus_metrics_reports.append(report_file)
            else:
                print("Report generation failed - no metrics collected by metrics client")

        if report_config.per_stage:
            for stage_id, _stage_info in runtime_parameters.stages.items():
                collected_metrics = self.metrics_client.collect_metrics_for_stage(runtime_parameters, stage_id)
                if collected_metrics is not None:
                    report_file = ReportFile(
                        name=f"stage_{stage_id}_prometheus_metrics",
                        contents=summarize_prometheus_metrics(collected_metrics).model_dump(),
                    )
                    if report_file is not None:
                        print(f"Successfully saved stage {stage_id} report of prometheus metrics to {report_file.path}")
                    prometheus_metrics_reports.append(report_file)
                else:
                    print(f"No metrics collected for Stage {stage_id}")

        return prometheus_metrics_reports
