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
import logging
from collections import defaultdict
from typing import Any, List, Optional


import numpy as np
from pydantic import BaseModel

from inference_perf.apis import RequestLifecycleMetric
from inference_perf.client.metricsclient import MetricsClient, PerfRuntimeParameters
from inference_perf.client.metricsclient.base import ModelServerMetrics
from inference_perf.client.metricsclient.prometheus_client import PrometheusMetricsClient
from inference_perf.client.requestdatacollector import RequestDataCollector
from inference_perf.config import Config, PrometheusMetricsReportConfig, ReportConfig
from inference_perf.utils import ReportFile

logger = logging.getLogger(__name__)

def safe_float(value: Any) -> float:
    """NOTE: Only for use in summarize_requests after validating safe access"""
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0
    
def summarize(items: List[float], percentiles: List[float]) -> Optional[dict[str, float]]:
    if len(items) == 0:
        return None
    result = {
        "mean": float(np.mean(items)),
        "min": float(np.min(items)),
        "max": float(np.max(items)),
    }
    for p in percentiles:
        key = "median" if p == 50 else f"p{p:g}"
        result[key] = float(np.percentile(items, p))
    return result


class ResponsesSummary(BaseModel):
    load_summary: dict[str, Any]
    successes: dict[str, Any]
    failures: dict[str, Any]

def calculate_slo_metrics(
    metrics: List[RequestLifecycleMetric],
    ttft_values: List[Optional[float]],
    tpot_values: List[Optional[float]]
) -> Optional[dict[str, Any]]:
    """
    Calculate SLO attainment statistics from request metrics and pre-calculated latencies.
    
    Args:
        metrics: List of successful RequestLifecycleMetric objects.
        ttft_values: List of calculated TTFT values corresponding to metrics (same index). None if not streamable.
        tpot_values: List of calculated TPOT values corresponding to metrics (same index). None if not streamable.
    
    Returns None if no SLO thresholds were set, otherwise returns dict with SLO metrics.
    """
    if not metrics:
        return None
    
    # Check if any metrics have SLO fields populated
    has_ttft_slo = any(m.ttft_slo_sec is not None for m in metrics)
    has_tpot_slo = any(m.tpot_slo_sec is not None for m in metrics)

    ttft_slo_limit = max([m.ttft_slo_sec for m in metrics if m.ttft_slo_sec is not None], default=None)
    tpot_slo_limit = max([m.tpot_slo_sec for m in metrics if m.tpot_slo_sec is not None], default=None)
   
    if not has_ttft_slo and not has_tpot_slo:
        return None  # No SLO tracking enabled
    
    slo_metrics: dict[str, Any] = {}
    total = len(metrics)
    
    total_benchmark_time = max(m.end_time for m in metrics) - min(m.start_time for m in metrics)
    goodput_token_counts = []
    
    ttft_results = []
    tpot_results = []
    
    # Iterate through metrics and their corresponding calculated values.
    # Using zip(..., strict=True) ensures that all sequences have the same length
    # and prevents misalignment or IndexError if they do not.
    for m, ttft, tpot in zip(metrics, ttft_values, tpot_values, strict=True):
        
        # Check TTFT SLO (Only if streamable / ttft exists)
        ttft_met = None
        if m.ttft_slo_sec is not None:
            if ttft is not None:
                ttft_met = (ttft <= m.ttft_slo_sec)
            else:
                # If it's not streamable, we can't measure TTFT, so strictly speaking it's not "met"
                # OR we treat non-streamable as failing the streaming SLO.
                # Standard practice: if we can't measure it, it counts as failed or ignored.
                # Here assuming failed if data is missing but SLO required.
                ttft_met = False
        ttft_results.append(ttft_met)
        
        # Check TPOT SLO (Only if streamable / tpot exists)
        tpot_met = None
        if m.tpot_slo_sec is not None:
            if tpot is not None:
                tpot_met = (tpot <= m.tpot_slo_sec)
            else:
                tpot_met = False
        tpot_results.append(tpot_met)

        # Goodput Calculation Logic
        ttft_ok = (ttft_met is True) if has_ttft_slo else True
        tpot_ok = (tpot_met is True) if has_tpot_slo else True
        
        if ttft_ok and tpot_ok:
            if m.info.input_tokens is not None and m.info.output_tokens is not None:
                goodput_token_counts.append(m.info.input_tokens + m.info.output_tokens)

    total_goodput_tokens = sum(goodput_token_counts)
    goodput_rate = total_goodput_tokens / total_benchmark_time if total_benchmark_time > 0 else 0.0
    
    # TTFT SLO metrics
    if has_ttft_slo:
        ttft_met_count = sum(1 for res in ttft_results if res is True)
        ttft_failed_count = sum(1 for res in ttft_results if res is False)
        
        slo_metrics["ttft_slo"] = {
            "attainment_pct": (ttft_met_count / total * 100) if total > 0 else 0,
            "requests_met": ttft_met_count,
            "requests_failed": ttft_failed_count,
            "total_requests": total,
            "slo": ttft_slo_limit
            
        }
    
    # TPOT SLO metrics
    if has_tpot_slo:
        tpot_met_count = sum(1 for res in tpot_results if res is True)
        tpot_failed_count = sum(1 for res in tpot_results if res is False)
        
        slo_metrics["tpot_slo"] = {
            "attainment_pct": (tpot_met_count / total * 100) if total > 0 else 0,
            "requests_met": tpot_met_count,
            "requests_failed": tpot_failed_count,
            "total_requests": total,
            "slo": tpot_slo_limit
        }
    
    # Combined SLO (both must be met)
    if has_ttft_slo and has_tpot_slo:
        combined_met = sum(
            1 for t_res, p_res in zip(ttft_results, tpot_results, strict=True)
            if t_res is True and p_res is True
        )
        
        slo_metrics["combined_slo"] = {
            "attainment_pct": (combined_met / total * 100) if total > 0 else 0,
            "requests_met": combined_met,
            "requests_failed": total - combined_met,
            "total_requests": total,
            "ttft_slo": ttft_slo_limit,
            "tpot_slo": tpot_slo_limit,
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
    metrics: List[RequestLifecycleMetric],
    percentiles: List[float],
    stage_rate: Optional[float] = None,
    stage_concurrency: Optional[int] = None,
) -> ResponsesSummary:
    all_successful: List[RequestLifecycleMetric] = [x for x in metrics if x.error is None]
    all_failed: List[RequestLifecycleMetric] = [x for x in metrics if x.error is not None]

    total_time = max(x.end_time for x in metrics) - min(x.start_time for x in metrics)
    
    schedule_deltas = [x.start_time - x.scheduled_time for x in metrics]
    send_duration = max(x.start_time for x in metrics) - min(x.start_time for x in metrics)

    load_summary: dict[Any, Any] = {
        "count": len(metrics),
        "schedule_delay": summarize(schedule_deltas, percentiles),
    }

    if stage_rate is not None:
        # Guard against zero send_duration to avoid ZeroDivisionError when all
        # requests have identical start times or there is only a single request.
        achieved_rate = len(metrics) / send_duration if send_duration > 0 else 0.0
        load_summary = {
            "count": len(metrics),
            "schedule_delay": summarize(schedule_deltas, percentiles),
            "send_duration": send_duration,
            "requested_rate": stage_rate,
            "achieved_rate": achieved_rate,
        }
        if stage_concurrency is not None:
            load_summary["concurrency"] = stage_concurrency

    # --- Pre-calculate Metrics for all successful requests ---
    # We maintain 1:1 mapping with 'all_successful' to pass to SLO calculator
    
    ntpot_values: List[float] = []
    tpot_values: List[Optional[float]] = [] # Optional: None if not streamable
    ttft_values: List[Optional[float]] = [] # Optional: None if not streamable
    inter_token_latencies: List[float] = []

    for m in all_successful:
        # NTPOT: (End - Start) / Output Tokens (Calculated for ALL successful requests)
        if m.info.output_tokens and m.info.output_tokens > 0:
            ntpot_values.append((m.end_time - m.start_time) / m.info.output_tokens)
        else:
            ntpot_values.append(0.0)
        
        # Check if streamable: Must have more than 1 output token timestamp
        is_streamable = m.info.output_token_times and len(m.info.output_token_times) > 1
        
        if is_streamable:
            # TTFT: First Token Time - Start Time
            ttft = m.info.output_token_times[0] - m.start_time
            ttft_values.append(ttft)

            # TPOT: (Last Token Time - First Token Time) / (Num Output Tokens - 1)
            duration = m.info.output_token_times[-1] - m.info.output_token_times[0]
            count = len(m.info.output_token_times)
            tpot = duration / (count - 1)
            tpot_values.append(tpot)
            
            # Add inter-token deltas
            for t1, t2 in zip(m.info.output_token_times, m.info.output_token_times[1:], strict=False):
                inter_token_latencies.append(t2 - t1)
        else:
            # Not streamable, so TTFT and TPOT are undefined
            ttft_values.append(None)
            tpot_values.append(None)

    # --- Pass pre-calculated lists to SLO Calculator ---
    slo_metrics = calculate_slo_metrics(all_successful, ttft_values, tpot_values)

    # --- Filter lists for summarization (remove Nones) ---
    valid_tpot = [v for v in tpot_values if v is not None]
    valid_ttft = [v for v in ttft_values if v is not None]

    successes_dict = {
            "count": len(all_successful),
            "latency": {
                "request_latency": summarize(
                    [(successful.end_time - successful.start_time) for successful in all_successful], percentiles
                ),
                "normalized_time_per_output_token": summarize(ntpot_values, percentiles),
                "time_per_output_token": summarize(valid_tpot, percentiles),
                "time_to_first_token": summarize(valid_ttft, percentiles),
                "inter_token_latency": summarize(inter_token_latencies, percentiles),
            },
            "throughput": {
                "input_tokens_per_sec": (
                    sum(safe_float(x.info.input_tokens) for x in all_successful) / total_time
                    if total_time > 0
                    else 0.0
                ),
                "output_tokens_per_sec": (
                    sum(safe_float(x.info.output_tokens) for x in all_successful) / total_time
                    if total_time > 0
                    else 0.0
                ),
                "total_tokens_per_sec": (
                    sum(
                        safe_float(x.info.input_tokens) + safe_float(x.info.output_tokens)
                        for x in all_successful
                    ) / total_time
                    if total_time > 0
                    else 0.0
                ),
                "requests_per_sec": (
                    len(all_successful) / total_time
                    if total_time > 0
                    else 0.0
                ),
            },
            "prompt_len": summarize([safe_float(success.info.input_tokens) for success in all_successful], percentiles),
            "output_len": summarize([float(v) for success in all_successful if (v := success.info.output_tokens) is not None], percentiles),
        }
    if slo_metrics:
        successes_dict["slo_metrics"] = slo_metrics
    return ResponsesSummary(
        load_summary=load_summary,
        successes=successes_dict,
        failures={
            "count": len(all_failed),
            "request_latency": summarize([(failed.end_time - failed.start_time) for failed in all_failed], percentiles),
            "prompt_len": summarize([safe_float(failed.info.input_tokens) for failed in all_failed], percentiles),
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
            contents=self.config.model_dump(mode="json", by_alias=True),
        )

    async def generate_reports(
        self, report_config: ReportConfig, runtime_parameters: PerfRuntimeParameters
    ) -> List[ReportFile]:
        logger.info("Generating Reports...")
        lifecycle_reports = []
        percentiles = report_config.request_lifecycle.percentiles

        # Filter out the preprocessing stage -1
        request_metrics = [
            metric for metric in self.metrics_collector.get_metrics() if metric.stage_id is not None and metric.stage_id >= 0
        ]

        if report_config.request_lifecycle.summary:
            if len(request_metrics) != 0:
                report_file = ReportFile(
                    name="summary_lifecycle_metrics",
                    contents=summarize_requests(request_metrics, percentiles).model_dump(),
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
                        contents=summarize_requests(metrics, percentiles, stage_rate, concurrency_level).model_dump(),
                    )
                else:
                    report_file = ReportFile(
                        name=f"stage_{stage_id}_lifecycle_metrics",
                        contents=summarize_requests(metrics, percentiles, stage_rate).model_dump(),
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

        if report_config.request_lifecycle.per_adapter:
            adapter_buckets: dict[Optional[str], List[RequestLifecycleMetric]] = defaultdict(list)
            for metric in request_metrics:
                if metric.info.lora_adapter is not None:
                    adapter_buckets[metric.info.lora_adapter].append(metric)
            for adapter, metrics in adapter_buckets.items():
                report_file = ReportFile(
                    name=f"adapter_{adapter}_lifecycle_metrics", contents=summarize_requests(metrics, percentiles).model_dump()
                )
                lifecycle_reports.append(report_file)

        if report_config.request_lifecycle.per_adapter_stage:
            # Group by (adapter, stage_id) tuple
            adapter_stage_buckets: dict[tuple[Optional[str], int], List[RequestLifecycleMetric]] = defaultdict(list)
            for metric in request_metrics:
                if metric.stage_id is not None and metric.info.lora_adapter is not None:
                    adapter_stage_buckets[(metric.info.lora_adapter, metric.stage_id)].append(metric)
            for (adapter, stage_id), metrics in adapter_stage_buckets.items():
                stage_rate = runtime_parameters.stages[stage_id].rate
                report_file = ReportFile(
                    name=f"adapter_{adapter}_stage_{stage_id}_lifecycle_metrics",
                    contents=summarize_requests(metrics, percentiles, stage_rate).model_dump(),
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
            for stage_id in runtime_parameters.stages:
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
