# Copyright 2026 The Kubernetes Authors.
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
from typing import Any, Dict, List, Optional, Union, TYPE_CHECKING

if TYPE_CHECKING:
    from inference_perf.datagen import DataGenerator, SessionGenerator


import numpy as np
from pydantic import BaseModel

from inference_perf.apis import RequestLifecycleMetric, SessionLifecycleMetric
from inference_perf.client.metricsclient import MetricsClient, PerfRuntimeParameters
from inference_perf.client.metricsclient.base import ModelServerMetrics
from inference_perf.client.metricsclient.prometheus_client import PrometheusMetricsClient
from inference_perf.client.requestdatacollector import RequestDataCollector
from inference_perf.config import (
    Config,
    PrometheusMetricsReportConfig,
    ReportConfig,
    SessionLifecycleReportConfig,
    GoodputConfig,
)
from inference_perf.metrics import SessionMetricsCollector
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
    benchmark_time_seconds: float
    load_summary: dict[str, Any]
    successes: dict[str, Any]
    failures: dict[str, Any]


def calculate_goodput_metrics(
    metrics: List[RequestLifecycleMetric],
    goodput_config: Optional[GoodputConfig],
    ttft_values: List[Optional[float]],
    tpot_values: List[Optional[float]],
    ntpot_values: List[float],
    request_latency_values: List[float],
    itl_values: List[Optional[float]],
) -> Optional[dict[str, Any]]:
    has_constraints = False
    if goodput_config and goodput_config.constraints:
        has_constraints = True

    if not has_constraints and not any(m.ttft_slo_sec is not None or m.tpot_slo_sec is not None for m in metrics):
        return None

    total = len(metrics)
    if total == 0:
        return None

    total_benchmark_time = max(m.end_time for m in metrics) - min(m.start_time for m in metrics)

    good_requests_count = 0
    good_total_tokens = 0

    attainment_counts: defaultdict[str, int] = defaultdict(int)
    total_applicable_counts: defaultdict[str, int] = defaultdict(int)

    for i, m in enumerate(metrics):
        is_good = True

        effective_ttft_slo = (
            m.ttft_slo_sec
            if m.ttft_slo_sec is not None
            else (goodput_config.constraints.get("ttft") if goodput_config else None)
        )
        effective_tpot_slo = (
            m.tpot_slo_sec
            if m.tpot_slo_sec is not None
            else (goodput_config.constraints.get("tpot") if goodput_config else None)
        )

        effective_itl_slo = goodput_config.constraints.get("itl") if goodput_config else None
        effective_ntpot_slo = goodput_config.constraints.get("ntpot") if goodput_config else None
        effective_latency_slo = goodput_config.constraints.get("request_latency") if goodput_config else None

        if effective_ttft_slo is not None:
            total_applicable_counts["ttft"] += 1
            val = ttft_values[i]
            if val is not None and val <= effective_ttft_slo:
                attainment_counts["ttft"] += 1
            else:
                is_good = False

        if effective_tpot_slo is not None:
            total_applicable_counts["tpot"] += 1
            val = tpot_values[i]
            if val is not None and val <= effective_tpot_slo:
                attainment_counts["tpot"] += 1
            else:
                is_good = False

        if effective_itl_slo is not None:
            total_applicable_counts["itl"] += 1
            val = itl_values[i]
            if val is not None and val <= effective_itl_slo:
                attainment_counts["itl"] += 1
            else:
                is_good = False

        if effective_ntpot_slo is not None:
            total_applicable_counts["ntpot"] += 1
            val = ntpot_values[i]
            if val is not None and val <= effective_ntpot_slo:
                attainment_counts["ntpot"] += 1
            else:
                is_good = False

        if effective_latency_slo is not None:
            total_applicable_counts["request_latency"] += 1
            val = request_latency_values[i]
            if val is not None and val <= effective_latency_slo:
                attainment_counts["request_latency"] += 1
            else:
                is_good = False

        if is_good:
            good_requests_count += 1
            in_tokens = m.info.input_tokens if m.info.input_tokens is not None else 0
            out_tokens = m.info.output_tokens if m.info.output_tokens is not None else 0
            good_total_tokens += in_tokens + out_tokens

    goodput_percentage = (good_requests_count / total * 100) if total > 0 else 0.0
    request_goodput = good_requests_count / total_benchmark_time if total_benchmark_time > 0 else 0.0
    token_goodput = good_total_tokens / total_benchmark_time if total_benchmark_time > 0 else 0.0

    result = {
        "goodput_percentage": goodput_percentage,
        "request_goodput": request_goodput,
        "token_goodput": token_goodput,
        "good_requests": good_requests_count,
        "total_requests": total,
    }

    for k in total_applicable_counts:
        if total_applicable_counts[k] > 0:
            result[f"{k}_attainment_percentage"] = attainment_counts[k] / total_applicable_counts[k] * 100

    return result


def summarize_prometheus_metrics(metrics: ModelServerMetrics) -> ResponsesSummary:
    return ResponsesSummary(
        benchmark_time_seconds=0.0,
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
            "inter_token_latency": {
                "mean": metrics.avg_inter_token_latency,
                "median": metrics.median_inter_token_latency,
                "p90": metrics.p90_inter_token_latency,
                "p99": metrics.p99_inter_token_latency,
            },
            "num_requests_running": {
                "mean": metrics.avg_num_requests_running,
            },
            "request_queue_time": {
                "mean": metrics.avg_request_queue_time,
                "median": metrics.median_request_queue_time,
                "p90": metrics.p90_request_queue_time,
                "p99": metrics.p99_request_queue_time,
            },
            "request_inference_time": {
                "mean": metrics.avg_request_inference_time,
                "median": metrics.median_request_inference_time,
                "p90": metrics.p90_request_inference_time,
                "p99": metrics.p99_request_inference_time,
            },
            "request_prefill_time": {
                "mean": metrics.avg_request_prefill_time,
                "median": metrics.median_request_prefill_time,
                "p90": metrics.p90_request_prefill_time,
                "p99": metrics.p99_request_prefill_time,
            },
            "request_decode_time": {
                "mean": metrics.avg_request_decode_time,
                "median": metrics.median_request_decode_time,
                "p90": metrics.p90_request_decode_time,
                "p99": metrics.p99_request_decode_time,
            },
            "request_prompt_tokens": {
                "mean": metrics.avg_request_prompt_tokens,
                "median": metrics.median_request_prompt_tokens,
                "p90": metrics.p90_request_prompt_tokens,
                "p99": metrics.p99_request_prompt_tokens,
            },
            "request_generation_tokens": {
                "mean": metrics.avg_request_generation_tokens,
                "median": metrics.median_request_generation_tokens,
                "p90": metrics.p90_request_generation_tokens,
                "p99": metrics.p99_request_generation_tokens,
            },
            "request_max_num_generation_tokens": {
                "mean": metrics.avg_request_max_num_generation_tokens,
                "median": metrics.median_request_max_num_generation_tokens,
                "p90": metrics.p90_request_max_num_generation_tokens,
                "p99": metrics.p99_request_max_num_generation_tokens,
            },
            "request_params_n": {
                "mean": metrics.avg_request_params_n,
                "median": metrics.median_request_params_n,
                "p90": metrics.p90_request_params_n,
                "p99": metrics.p99_request_params_n,
            },
            "request_params_max_tokens": {
                "mean": metrics.avg_request_params_max_tokens,
                "median": metrics.median_request_params_max_tokens,
                "p90": metrics.p90_request_params_max_tokens,
                "p99": metrics.p99_request_params_max_tokens,
            },
            "request_success_count": metrics.request_success_count,
            "iteration_tokens": {
                "mean": metrics.avg_iteration_tokens,
                "median": metrics.median_iteration_tokens,
                "p90": metrics.p90_iteration_tokens,
                "p99": metrics.p99_iteration_tokens,
            },
            "prompt_tokens_cached": metrics.prompt_tokens_cached,
            "prompt_tokens_recomputed": metrics.prompt_tokens_recomputed,
            "external_prefix_cache_hit_percent": {
                "mean": (metrics.external_prefix_cache_hits / metrics.external_prefix_cache_queries) * 100.0
                if metrics.external_prefix_cache_queries > 0
                else 0.0
            },
            "mm_cache_hit_percent": {
                "mean": (metrics.mm_cache_hits / metrics.mm_cache_queries) * 100.0 if metrics.mm_cache_queries > 0 else 0.0
            },
            "corrupted_requests": metrics.corrupted_requests,
            "request_prefill_kv_computed_tokens": {
                "mean": metrics.avg_request_prefill_kv_computed_tokens,
                "median": metrics.median_request_prefill_kv_computed_tokens,
                "p90": metrics.p90_request_prefill_kv_computed_tokens,
                "p99": metrics.p99_request_prefill_kv_computed_tokens,
            },
            "kv_block_idle_before_evict": {
                "mean": metrics.avg_kv_block_idle_before_evict,
                "median": metrics.median_kv_block_idle_before_evict,
                "p90": metrics.p90_kv_block_idle_before_evict,
                "p99": metrics.p99_kv_block_idle_before_evict,
            },
            "kv_block_lifetime": {
                "mean": metrics.avg_kv_block_lifetime,
                "median": metrics.median_kv_block_lifetime,
                "p90": metrics.p90_kv_block_lifetime,
                "p99": metrics.p99_kv_block_lifetime,
            },
            "kv_block_reuse_gap": {
                "mean": metrics.avg_kv_block_reuse_gap,
                "median": metrics.median_kv_block_reuse_gap,
                "p90": metrics.p90_kv_block_reuse_gap,
                "p99": metrics.p99_kv_block_reuse_gap,
            },
        },
    )


def summarize_requests(
    metrics: List[RequestLifecycleMetric],
    percentiles: List[float],
    stage_rate: Optional[float] = None,
    stage_concurrency: Optional[int] = None,
    goodput_config: Optional[GoodputConfig] = None,
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
    tpot_values: List[Optional[float]] = []  # Optional: None if not streamable
    ttft_values: List[Optional[float]] = []  # Optional: None if not streamable
    request_latency_values: List[float] = []
    itl_values: List[Optional[float]] = []
    inter_token_latencies: List[float] = []

    for m in all_successful:
        request_latency_values.append(m.end_time - m.start_time)

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
            request_itl = []
            for t1, t2 in zip(m.info.output_token_times, m.info.output_token_times[1:], strict=False):
                inter_token_latencies.append(t2 - t1)
                request_itl.append(t2 - t1)

            if request_itl:
                itl_values.append(sum(request_itl) / len(request_itl))
            else:
                itl_values.append(None)
        else:
            # Not streamable, so TTFT and TPOT are undefined
            ttft_values.append(None)
            tpot_values.append(None)
            itl_values.append(None)

    # --- Calculate Goodput Metrics ---
    goodput_metrics = calculate_goodput_metrics(
        all_successful, goodput_config, ttft_values, tpot_values, ntpot_values, request_latency_values, itl_values
    )

    # --- Filter lists for summarization (remove Nones) ---
    valid_tpot = [v for v in tpot_values if v is not None]
    valid_ttft = [v for v in ttft_values if v is not None]

    successes_dict = {
        "count": len(all_successful),
        "latency": {
            "request_latency": summarize(request_latency_values, percentiles),
            "normalized_time_per_output_token": summarize(ntpot_values, percentiles),
            "time_per_output_token": summarize(valid_tpot, percentiles),
            "time_to_first_token": summarize(valid_ttft, percentiles),
            "inter_token_latency": summarize(inter_token_latencies, percentiles),
        },
        "throughput": {
            "input_tokens_per_sec": (
                sum(safe_float(x.info.input_tokens) for x in all_successful) / total_time if total_time > 0 else 0.0
            ),
            "output_tokens_per_sec": (
                sum(safe_float(x.info.output_tokens) for x in all_successful) / total_time if total_time > 0 else 0.0
            ),
            "total_tokens_per_sec": (
                sum(safe_float(x.info.input_tokens) + safe_float(x.info.output_tokens) for x in all_successful) / total_time
                if total_time > 0
                else 0.0
            ),
            "requests_per_sec": (len(all_successful) / total_time if total_time > 0 else 0.0),
        },
        "prompt_len": summarize([safe_float(success.info.input_tokens) for success in all_successful], percentiles),
        "output_len": summarize(
            [float(v) for success in all_successful if (v := success.info.output_tokens) is not None], percentiles
        ),
    }
    if goodput_metrics:
        successes_dict["goodput_metrics"] = goodput_metrics

    return ResponsesSummary(
        benchmark_time_seconds=total_time,
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
        datagen: Optional[Union["DataGenerator", "SessionGenerator"]] = None,
    ) -> None:
        self.metrics_collector = metrics_collector
        self.metrics_client = metrics_client
        self.config = config
        self.datagen = datagen
        self.session_metrics_collector: Optional[SessionMetricsCollector] = None

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
                    contents=summarize_requests(
                        request_metrics, percentiles, goodput_config=report_config.goodput
                    ).model_dump(),
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
                        contents=summarize_requests(
                            metrics, percentiles, stage_rate, concurrency_level, goodput_config=report_config.goodput
                        ).model_dump(),
                    )
                else:
                    report_file = ReportFile(
                        name=f"stage_{stage_id}_lifecycle_metrics",
                        contents=summarize_requests(
                            metrics, percentiles, stage_rate, goodput_config=report_config.goodput
                        ).model_dump(),
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
                    name=f"adapter_{adapter}_lifecycle_metrics",
                    contents=summarize_requests(metrics, percentiles, goodput_config=report_config.goodput).model_dump(),
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
                    contents=summarize_requests(
                        metrics, percentiles, stage_rate, goodput_config=report_config.goodput
                    ).model_dump(),
                )
                lifecycle_reports.append(report_file)

        if report_config.prometheus:
            lifecycle_reports.extend(self.generate_prometheus_metrics_report(runtime_parameters, report_config.prometheus))

        # Session-level reports (OTel agentic workloads only)
        if self.session_metrics_collector and report_config.session_lifecycle:
            session_reports = self.generate_session_reports(
                self.session_metrics_collector.get_metrics(),
                report_config.session_lifecycle,
                percentiles,
            )
            lifecycle_reports.extend(session_reports)

        lifecycle_reports.append(self.generate_config_report())
        return lifecycle_reports

    def summarize_sessions(self, metrics: List[SessionLifecycleMetric], percentiles: List[float]) -> Dict[str, Any]:
        """Compute aggregated stats across a list of session lifecycle metrics."""
        num_sessions = len(metrics)
        num_succeeded = sum(1 for m in metrics if m.success is True)
        num_failed = sum(1 for m in metrics if m.success is False)
        total_events = sum(m.num_events for m in metrics)
        total_events_completed = sum(m.num_events_completed for m in metrics)
        total_events_cancelled = sum(m.num_events_cancelled for m in metrics if m.num_events_cancelled is not None)

        sessions_per_second = 0.0
        if num_sessions > 0:
            total_span = max(m.end_time for m in metrics) - min(m.start_time for m in metrics)
            if total_span > 0:
                sessions_per_second = num_sessions / total_span

        return {
            "num_sessions": num_sessions,
            "num_sessions_succeeded": num_succeeded,
            "num_sessions_failed": num_failed,
            "total_events": total_events,
            "total_events_completed": total_events_completed,
            "total_events_cancelled": total_events_cancelled,
            "sessions_per_second": sessions_per_second,
            "session_duration_sec": summarize([m.duration_sec for m in metrics], percentiles),
            "num_events": summarize([float(m.num_events) for m in metrics], percentiles),
            "num_events_cancelled": summarize(
                [float(m.num_events_cancelled) for m in metrics if m.num_events_cancelled is not None], percentiles
            ),
            "total_input_tokens": summarize(
                [float(m.total_input_tokens) for m in metrics if m.total_input_tokens is not None], percentiles
            ),
            "total_output_tokens": summarize(
                [float(m.total_output_tokens) for m in metrics if m.total_output_tokens is not None], percentiles
            ),
        }

    def generate_session_reports(
        self,
        session_metrics: List[SessionLifecycleMetric],
        report_config: SessionLifecycleReportConfig,
        percentiles: List[float],
    ) -> List[ReportFile]:
        """Generate session-level lifecycle reports.

        Note: Session metrics should be enriched (via SessionMetricsCollector.enrich_metrics())
        before calling this method.
        """
        reports: List[ReportFile] = []

        if not session_metrics:
            return reports

        if report_config.summary:
            reports.append(
                ReportFile(
                    name="summary_session_lifecycle_metrics",
                    contents=self.summarize_sessions(session_metrics, percentiles),
                )
            )

        if report_config.per_stage:
            stage_buckets: dict[int, List[SessionLifecycleMetric]] = defaultdict(list)
            for m in session_metrics:
                stage_buckets[m.stage_id].append(m)
            for stage_id, stage_metrics in stage_buckets.items():
                reports.append(
                    ReportFile(
                        name=f"stage_{stage_id}_session_lifecycle_metrics",
                        contents=self.summarize_sessions(stage_metrics, percentiles),
                    )
                )

        if report_config.per_session:
            reports.append(
                ReportFile(
                    name="per_session_lifecycle_metrics",
                    contents=[m.model_dump() for m in session_metrics],
                )
            )

        return reports

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
