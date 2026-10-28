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
from typing import Any

from prometheus_client import Counter, Gauge  # noqa: TID251 (declares each spec's metric_type)

from inference_perf.apis.base import RequestLifecycleMetric
from inference_perf.observability.context import RunContext, StageContext
from inference_perf.observability.metrics.registry import MetricSpec, ObservedRequest


def stage_label(metric: RequestLifecycleMetric) -> str:
    return "" if metric.stage_id is None else str(metric.stage_id)


def _mark_run_start(gauge: Gauge, context: RunContext) -> None:
    start = time.monotonic()
    gauge.set_function(lambda: time.monotonic() - start)


def _bind_stage_count(gauge: Gauge, context: RunContext) -> None:
    if context.stage_count is not None:
        gauge.set_function(context.stage_count)
    else:
        gauge.set(len(context.config.load.stages))


def _bind_in_flight(gauge: Gauge, context: RunContext) -> None:
    gauge.set_function(context.in_flight_requests)


def _stage_running(gauge: Gauge, context: StageContext) -> None:
    gauge.labels(context.stage_label).set(1)


def _stage_done(gauge: Gauge, context: StageContext) -> None:
    gauge.labels(context.stage_label).set(0)


def _stamp_stage(gauge: Gauge, context: StageContext) -> None:
    gauge.labels(context.stage_label).set(time.time())


def _count_request(counter: Counter, request: ObservedRequest) -> None:
    status = "failure" if request.metric.error is not None else "success"
    counter.labels(stage_label(request.metric), status).inc()


def _count_error(counter: Counter, request: ObservedRequest) -> None:
    if request.metric.error is not None:
        counter.labels(stage_label(request.metric), request.metric.error.error_type).inc()


def _count_prompt_tokens(counter: Counter, request: ObservedRequest) -> None:
    if request.metric.error is None:
        counter.labels(stage_label(request.metric)).inc(request.metric.info.request_metrics.text.input_tokens)


def _count_output_tokens(counter: Counter, request: ObservedRequest) -> None:
    if request.metric.error is None:
        counter.labels(stage_label(request.metric)).inc(request.output_tokens)


def _stamp_last_completion(gauge: Gauge, request: ObservedRequest) -> None:
    # end_time is on the perf_counter clock the workers share. Shifting it by
    # that clock's gap to Unix time now gives the Unix time the request finished.
    finished = time.time() - (time.perf_counter() - request.metric.end_time)
    gauge.labels(stage_label(request.metric)).set(finished)


def _count_workers_lost(counter: Counter, context: StageContext) -> None:
    for cause in context.workers_lost():
        counter.labels(context.stage_label, cause).inc()


STAGES = MetricSpec[Gauge](
    name="inference_perf_stages",
    documentation=(
        "Number of load stages the run executes. On a sweep run this is 0 until the saturation probe has generated the stages."
    ),
    metric_type=Gauge,
    on_run_start=_bind_stage_count,
)

CORE_SPECS: tuple[MetricSpec[Any], ...] = (
    MetricSpec(
        name="inference_perf_run_elapsed_seconds",
        documentation="Wall-clock seconds elapsed since the benchmark run started; 0 until the run starts.",
        metric_type=Gauge,
        on_run_start=_mark_run_start,
    ),
    STAGES,
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
        signal="in_flight",
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
        series="error",
        signal="completed",
        on_request=_count_request,
    ),
    MetricSpec(
        name="inference_perf_last_completion_timestamp_seconds",
        documentation="Unix time at which a request of the stage last finished, success or failure.",
        metric_type=Gauge,
        labelnames=("stage",),
        signal="since_last_completion",
        on_request=_stamp_last_completion,
    ),
    MetricSpec(
        name="inference_perf_workers_lost",
        documentation=(
            "Worker processes that died during the stage, by stage and cause "
            "(the exception class the worker reported, else the signal that killed it, else its exit code). "
            "The run replaces a dead worker and carries on, so a non-zero value marks a stage that did not "
            "offer the load it was configured for: its results are not comparable with a stage that kept "
            "every worker."
        ),
        metric_type=Counter,
        labelnames=("stage", "cause"),
        on_stage_end=_count_workers_lost,
    ),
    MetricSpec(
        name="inference_perf_request_errors",
        documentation=(
            "Failed request attempts by stage and error class (the client's exception class or 'HTTP Error <status>')."
        ),
        metric_type=Counter,
        labelnames=("stage", "error_type"),
        signal="errors",
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
            "Counted as the report's output_tokens_per_sec counts them: the client-side count, or the server's "
            "reported count (usage.completion_tokens, or usage.output_tokens on the Anthropic Messages API) "
            "under report.request_lifecycle.use_server_output_tokens."
        ),
        metric_type=Counter,
        labelnames=("stage",),
        on_request=_count_output_tokens,
    ),
)
