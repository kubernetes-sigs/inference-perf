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
from inference_perf.config import ReportConfig, PrometheusMetricsReportConfig, Config
from inference_perf.client.metricsclient import MetricsClient, PerfRuntimeParameters
from inference_perf.utils import ReportFile
from inference_perf.client.requestdatacollector import RequestDataCollector
from inference_perf.apis import RequestLifecycleMetric
import numpy as np
import logging

logger = logging.getLogger(__name__)


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
            "p0.1": float(np.percentile(items, 0.1)),
            "p1": float(np.percentile(items, 1)),
            "p5": float(np.percentile(items, 5)),
            "p10": float(np.percentile(items, 10)),
            "p25": float(np.percentile(items, 25)),
            "median": float(np.percentile(items, 50)),
            "p75": float(np.percentile(items, 75)),
            "p90": float(np.percentile(items, 90)),
            "p95": float(np.percentile(items, 95)),
            "p99": float(np.percentile(items, 99)),
            "p99.9": float(np.percentile(items, 99.9)),
            "max": float(np.max(items)),
        }
        if len(items) != 0
        else None
    )


class ResponsesSummary(BaseModel):
    load_summary: dict[str, Any]
    successes: dict[str, Any]
    failures: dict[str, Any]

def calculate_slo_metrics(metrics: List[RequestLifecycleMetric]) -> Optional[dict[str, Any]]:
    """
    Calculate SLO attainment statistics from request metrics.
    
    Returns None if no SLO thresholds were set, otherwise returns dict with SLO metrics.
    """
    # Filter out errors
    valid_metrics = [m for m in metrics if m.error is None]
    
    if not valid_metrics:
        return None
    
    # Check if any metrics have SLO fields populated
    has_ttft_slo = any(m.ttft_slo_met is not None for m in valid_metrics)
    has_tpot_slo = any(m.tpot_slo_met is not None for m in valid_metrics)

    ttft_slo = max([m.ttft_slo for m in valid_metrics if m.ttft_slo is not None], default=None)
    tpot_slo = max([m.tpot_slo for m in valid_metrics if m.tpot_slo is not None], default=None)
    
    # calculate sum of input and output tokens of all valid metrics and with slo met
   
    if not has_ttft_slo and not has_tpot_slo:
        return None  # No SLO tracking enabled
    
    slo_metrics: dict[str, Any] = {}
    total = len(valid_metrics)
    
    total_benchmark_time = max(m.end_time for m in valid_metrics) - min(m.start_time for m in valid_metrics)
    total_goodput_tokens = sum(m.info.input_tokens + m.info.output_tokens for m in valid_metrics if m.info and m.info.input_tokens is not None and m.info.output_tokens is not None and m.ttft_slo_met)
    goodput_rate = total_goodput_tokens / total_benchmark_time if total_benchmark_time > 0 else 0.0
    
    # TTFT SLO metrics
    if has_ttft_slo:
        ttft_met = sum(1 for m in valid_metrics if m.ttft_slo_met is True)
        ttft_failed = sum(1 for m in valid_metrics if m.ttft_slo_met is False)
        
        # Get threshold from first metric that has it
        ttft_threshold = None
        for m in valid_metrics:
            if m.ttft is not None:
                # Infer threshold from whether it met SLO
                # We'll store the actual threshold in a better way later
                ttft_threshold = "configured"
                break
        
        slo_metrics["ttft_slo"] = {
            "attainment_pct": (ttft_met / total * 100) if total > 0 else 0,
            "requests_met": ttft_met,
            "requests_failed": ttft_failed,
            "total_requests": total,
            "slo": ttft_slo
            
        }
    
    # TPOT SLO metrics
    if has_tpot_slo:
        tpot_met = sum(1 for m in valid_metrics if m.tpot_slo_met is True)
        tpot_failed = sum(1 for m in valid_metrics if m.tpot_slo_met is False)
        
        slo_metrics["tpot_slo"] = {
            "attainment_pct": (tpot_met / total * 100) if total > 0 else 0,
            "requests_met": tpot_met,
            "requests_failed": tpot_failed,
            "total_requests": total,
            "slo": tpot_slo
        }
    
    # Combined SLO (both must be met)
    if has_ttft_slo and has_tpot_slo:
        combined_met = sum(
            1 for m in valid_metrics 
            if m.ttft_slo_met is True and m.tpot_slo_met is True
        )
        
        slo_metrics["combined_slo"] = {
            "attainment_pct": (combined_met / total * 100) if total > 0 else 0,
            "requests_met": combined_met,
            "requests_failed": total - combined_met,
            "total_requests": total,
            "ttft_slo": ttft_slo,
            "tpot_slo": tpot_slo,
            "goodput_rate": goodput_rate
        }
    
    return slo_metrics

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
                "median": metrics.median_request_latency,
                "p90": metrics.p90_request_latency,
                "p99": metrics.p99_request_latency,
            },
            "time_to_first_token": {
                "mean": metrics.avg_time_to_first_token,
                "median": metrics.median_time_to_first_token,
                "p90": metrics.p90_time_to_first_token,
                "p99": metrics.p99_time_to_first_token,
            },
            "time_per_output_token": {
                "mean": metrics.avg_time_per_output_token,
                "median": metrics.median_time_per_output_token,
                "p90": metrics.p90_time_per_output_token,
                "p99": metrics.p99_time_per_output_token,
            },
            "kv_cache_usage_percentage": {
                "mean": metrics.avg_kv_cache_usage,
                "median": metrics.median_kv_cache_usage,
                "p90": metrics.p90_kv_cache_usage,
                "p99": metrics.p99_kv_cache_usage,
            },
            "num_requests_swapped": {
                "mean": metrics.num_requests_swapped,
            },
            "num_preemptions_total": {"mean": metrics.num_preemptions_total},
            "prefix_cache_hit_percent": {
                "mean": (metrics.prefix_cache_hits / metrics.prefix_cache_queries) * 100.0
                if metrics.prefix_cache_queries > 0
                else 0.0
            },
        },
    )


def summarize_requests(
    metrics: List[RequestLifecycleMetric], stage_rate: Optional[float] = None, stage_concurrency: Optional[int] = None
) -> ResponsesSummary:
    all_successful: List[RequestLifecycleMetric] = [x for x in metrics if x.error is None]
    all_failed: List[RequestLifecycleMetric] = [x for x in metrics if x.error is not None]

    total_time = max(x.end_time for x in metrics) - min(x.start_time for x in metrics)
    streamable = [x for x in all_successful if x.info.output_token_times and len(x.info.output_token_times) > 1]

    schedule_deltas = [x.start_time - x.scheduled_time for x in metrics]
    send_duration = max([x.start_time for x in metrics]) - min([x.start_time for x in metrics])

    load_summary: dict[Any, Any] = {
        "count": len(metrics),
        "schedule_delay": summarize(schedule_deltas),
    }

    if stage_rate is not None:
        load_summary = {
            "count": len(metrics),
            "schedule_delay": summarize(schedule_deltas),
            "send_duration": send_duration,
            "requested_rate": stage_rate,
            "achieved_rate": len(metrics) / send_duration,
        }
        if stage_concurrency is not None:
            load_summary["concurrency"] = stage_concurrency
    # NEW: Calculate SLO metrics
    slo_metrics = calculate_slo_metrics(metrics)
    successes_dict = {
            "count": len(all_successful),
            "latency": {
                "request_latency": summarize([(successful.end_time - successful.start_time) for successful in all_successful]),
                "normalized_time_per_output_token": summarize(
                    [
                        ((metric.end_time - metric.start_time) / output_len) if output_len and output_len != 0 else 0
                        for metric in all_successful
                        for output_len in [safe_float(metric.info.output_tokens)]
                    ]
                ),
                "time_per_output_token": summarize(
                    [
                        (x.info.output_token_times[-1] - x.info.output_token_times[0]) / (len(x.info.output_token_times) - 1)
                        for x in streamable
                        if len(x.info.output_token_times) > 1
                    ]
                ),
                "time_to_first_token": summarize([x.info.output_token_times[0] - x.start_time for x in streamable]),
                "inter_token_latency": summarize(
                    [
                        t2 - t1
                        for x in streamable
                        for t1, t2 in zip(x.info.output_token_times, x.info.output_token_times[1:], strict=False)
                    ]
                ),
            },
            "throughput": {
                "input_tokens_per_sec": sum(x.info.input_tokens for x in all_successful) / total_time,
                "output_tokens_per_sec": sum(x.info.output_tokens for x in all_successful) / total_time,
                "total_tokens_per_sec": sum((x.info.input_tokens + x.info.output_tokens) for x in all_successful) / total_time,
                "requests_per_sec": len(all_successful) / total_time,
            },
            "prompt_len": summarize([safe_float(success.info.input_tokens) for success in all_successful]),
            "output_len": summarize([float(v) for success in all_successful if (v := success.info.output_tokens) is not None]),
        }
    if slo_metrics:
        successes_dict["slo_metrics"] = slo_metrics
    return ResponsesSummary(
        load_summary=load_summary,
        successes=successes_dict,
        failures={
            "count": len(all_failed),
            "request_latency": summarize([(failed.end_time - failed.start_time) for failed in all_failed]),
            "prompt_len": summarize([safe_float(failed.info.input_tokens) for failed in all_failed]),
        },
    )




class ReportGenerator:
    def __init__(
        self,
        metrics_client: Optional[MetricsClient],
        metrics_collector: RequestDataCollector,
        config: "Config",
    ) -> None:
        self.metrics_collector = metrics_collector
        self.metrics_client = metrics_client
        self.config = config

    def get_metrics_collector(self) -> RequestDataCollector:
        """
        Returns the metrics collector.
        """
        return self.metrics_collector

    def generate_config_report(self) -> ReportFile:
        """
        Generates a report file containing the config.
        """
        return ReportFile(
            name="config",
            file_type="yaml",
            contents=self.config.model_dump(mode="json"),
        )

    async def generate_reports(
        self, report_config: ReportConfig, runtime_parameters: PerfRuntimeParameters
    ) -> List[ReportFile]:
        logger.info("Generating Reports...")
        lifecycle_reports = []

        # Filter out the preprocessing stage -1
        request_metrics = [
            metric for metric in self.metrics_collector.get_metrics() if metric.stage_id is not None and metric.stage_id >= 0
        ]

        if report_config.request_lifecycle.summary:
            if len(request_metrics) != 0:
                report_file = ReportFile(
                    name="summary_lifecycle_metrics",
                    contents=summarize_requests(request_metrics).model_dump(),
                )
                lifecycle_reports.append(report_file)

        if report_config.request_lifecycle.per_stage:
            stage_buckets: dict[int, List[RequestLifecycleMetric]] = defaultdict(list)
            for metric in request_metrics:
                if metric.stage_id is not None:
                    stage_buckets[metric.stage_id].append(metric)
            for stage_id, metrics in stage_buckets.items():
                stage_rate = runtime_parameters.stages[stage_id].rate
                concurrency_level = runtime_parameters.stages[stage_id].concurrency_level
                if concurrency_level is not None:
                    report_file = ReportFile(
                        name=f"stage_{stage_id}_lifecycle_metrics",
                        contents=summarize_requests(metrics, stage_rate, concurrency_level).model_dump(),
                    )
                else:
                    report_file = ReportFile(
                        name=f"stage_{stage_id}_lifecycle_metrics",
                        contents=summarize_requests(metrics, stage_rate).model_dump(),
                    )
                lifecycle_reports.append(report_file)

        if report_config.request_lifecycle.per_request:
            report_file = ReportFile(
                name="per_request_lifecycle_metrics",
                contents=[
                    {
                        "start_time": metric.start_time,
                        "end_time": metric.end_time,
                        "request": metric.request_data,
                        "response": metric.response_data,
                        "info": metric.info.model_dump() if metric.info else None,
                        "error": metric.error.model_dump() if metric.error else None,
                    }
                    for metric in request_metrics
                ],
            )
            lifecycle_reports.append(report_file)

        if report_config.prometheus:
            lifecycle_reports.extend(self.generate_prometheus_metrics_report(runtime_parameters, report_config.prometheus))

        lifecycle_reports.append(self.generate_config_report())
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
            logger.warning("Prometheus Metrics Client is not configured or not of type PrometheusMetricsClient")
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
                prometheus_metrics_reports.append(report_file)
            else:
                logger.warning("Report generation failed - no metrics collected by metrics client")

        if report_config.per_stage:
            for stage_id, _stage_info in runtime_parameters.stages.items():
                collected_metrics = self.metrics_client.collect_metrics_for_stage(runtime_parameters, stage_id)
                if collected_metrics is not None:
                    report_file = ReportFile(
                        name=f"stage_{stage_id}_prometheus_metrics",
                        contents=summarize_prometheus_metrics(collected_metrics).model_dump(),
                    )
                    prometheus_metrics_reports.append(report_file)
                else:
                    logger.warning("No metrics collected for Stage %d", stage_id)

        return prometheus_metrics_reports
