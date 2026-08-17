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
"""Metrics exported on every run, regardless of config.

Run and stage state, request outcomes and token volume: the signals that
answer "is the run alive, which stage is it in, is it making progress" from a
scrape alone. Latency histograms live in latency.py.

Metric naming conventions are still being settled in
kubernetes-sigs/inference-perf#628; keep new names under the
inference_perf_ prefix and consistent with these until then. Labels are
bounded (stage by the stage count, error_type by the client's error
classes and HTTP status codes); nothing per-request goes on a label.
"""

import time
from typing import Any, Optional

from prometheus_client import Counter, Gauge

from inference_perf.apis.base import RequestLifecycleMetric, ResponseMetrics
from inference_perf.observability.metrics.registry import MetricSpec, RunContext


def stage_label(metric: RequestLifecycleMetric) -> str:
    return "" if metric.stage_id is None else str(metric.stage_id)


def output_tokens(response_metrics: Optional[ResponseMetrics]) -> int:
    """Output token count for runtime metrics: the server's own count when it
    reported one, else the client-side count. Prompt tokens already resolve
    this way at construction (#676); the report keeps a separate opt-in for
    output tokens for back-compat, but a new surface has no such constraint."""
    if response_metrics is None:
        return 0
    if response_metrics.server_usage:
        completion_tokens = response_metrics.server_usage.get("completion_tokens")
        if completion_tokens:
            return int(completion_tokens)
    return response_metrics.output_tokens


def _mark_run_start(gauge: Gauge, context: RunContext) -> None:
    start = time.monotonic()
    gauge.set_function(lambda: time.monotonic() - start)


def _set_stage_count(gauge: Gauge, context: RunContext) -> None:
    gauge.set(len(context.config.load.stages))


def _bind_in_flight(gauge: Gauge, context: RunContext) -> None:
    gauge.set_function(context.in_flight_requests)


def _stage_running(gauge: Gauge, stage_id: int) -> None:
    gauge.labels(str(stage_id)).set(1)


def _stage_done(gauge: Gauge, stage_id: int) -> None:
    gauge.labels(str(stage_id)).set(0)


def _stamp_stage(gauge: Gauge, stage_id: int) -> None:
    gauge.labels(str(stage_id)).set(time.time())


def _count_request(counter: Counter, metric: RequestLifecycleMetric) -> None:
    status = "failure" if metric.error is not None else "success"
    counter.labels(stage_label(metric), status).inc()


def _count_error(counter: Counter, metric: RequestLifecycleMetric) -> None:
    if metric.error is not None:
        counter.labels(stage_label(metric), metric.error.error_type).inc()


def _count_prompt_tokens(counter: Counter, metric: RequestLifecycleMetric) -> None:
    if metric.error is None:
        counter.labels(stage_label(metric)).inc(metric.info.request_metrics.text.input_tokens)


def _count_output_tokens(counter: Counter, metric: RequestLifecycleMetric) -> None:
    if metric.error is None:
        counter.labels(stage_label(metric)).inc(output_tokens(metric.info.response_metrics))


CORE_SPECS: tuple[MetricSpec[Any], ...] = (
    MetricSpec(
        name="inference_perf_run_elapsed_seconds",
        documentation="Wall-clock seconds elapsed since the benchmark run started; 0 until the run starts.",
        metric_type=Gauge,
        on_run_start=_mark_run_start,
    ),
    MetricSpec(
        name="inference_perf_stages",
        documentation="Number of load stages configured for the run.",
        metric_type=Gauge,
        on_run_start=_set_stage_count,
    ),
    MetricSpec(
        name="inference_perf_stage_running",
        documentation="1 while the stage is executing, 0 once it has ended. A stage that has not started has no series.",
        metric_type=Gauge,
        labelnames=("stage",),
        on_stage_start=_stage_running,
        on_stage_end=_stage_done,
    ),
    MetricSpec(
        name="inference_perf_stage_start_timestamp_seconds",
        documentation="Unix time at which the stage started.",
        metric_type=Gauge,
        labelnames=("stage",),
        on_stage_start=_stamp_stage,
    ),
    MetricSpec(
        name="inference_perf_stage_end_timestamp_seconds",
        documentation="Unix time at which the stage ended, whether it completed or was cut short.",
        metric_type=Gauge,
        labelnames=("stage",),
        on_stage_end=_stamp_stage,
    ),
    MetricSpec(
        name="inference_perf_requests_in_flight",
        documentation="Requests sent to the server and not yet finished, sampled at scrape time.",
        metric_type=Gauge,
        on_run_start=_bind_in_flight,
    ),
    MetricSpec(
        name="inference_perf_requests",
        documentation=(
            "Request attempts that have completed, by stage and final status. "
            "Incremented when the attempt finishes or fails, not when it is sent."
        ),
        metric_type=Counter,
        labelnames=("stage", "status"),
        on_request=_count_request,
    ),
    MetricSpec(
        name="inference_perf_request_errors",
        documentation=(
            "Failed request attempts by stage and error class (the client's exception class or 'HTTP Error <status>')."
        ),
        metric_type=Counter,
        labelnames=("stage", "error_type"),
        on_request=_count_error,
    ),
    MetricSpec(
        name="inference_perf_prompt_tokens",
        documentation="Prompt tokens of successful requests by stage; rate() gives input throughput.",
        metric_type=Counter,
        labelnames=("stage",),
        on_request=_count_prompt_tokens,
    ),
    MetricSpec(
        name="inference_perf_output_tokens",
        documentation=(
            "Output tokens of successful requests by stage; rate() gives output throughput. "
            "Uses the server's usage.completion_tokens when reported, else the client-side count."
        ),
        metric_type=Counter,
        labelnames=("stage",),
        on_request=_count_output_tokens,
    ),
)
