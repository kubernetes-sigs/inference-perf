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
"""Per-request histograms, one per expression series.

Each takes its value from series_values in reportgen/base.py, the report's own
formulas over the same RequestLifecycleMetric, so a runtime value is the number
the report will show. TPOT and NTPOT divide by the count the report uses under
report.request_lifecycle.use_server_output_tokens. Inter-token latency is
deliberately absent: it needs a timestamp per token, which only the report
computes, after the run (#564). A runtime ITL over raw chunk gaps would be a
second, different number under the same name.
"""

from typing import Any, Callable

from prometheus_client import Histogram  # noqa: TID251 (declares each spec's metric_type)

from inference_perf.config import Config
from inference_perf.observability.metrics.registry import MetricSpec, ObservedRequest, always

from .core import stage_label

# Seconds. Wide enough for interactive chat through minute-long batch requests.
TTFT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
TPOT_BUCKETS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
# NTPOT folds TTFT in, so a short response runs well past any TPOT.
NTPOT_BUCKETS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0)
REQUEST_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)
# A healthy run sends within a millisecond or two of schedule. The upper buckets show a client that cannot keep up.
SCHEDULE_DELAY_BUCKETS = (0.0005, 0.001, 0.0025, 0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0)


def streaming_only(config: Config) -> bool:
    """Streaming runs only (api.streaming); unary responses have no token timeline."""
    return bool(config.api.streaming)


def _observe(series: str) -> Callable[[Histogram, ObservedRequest], None]:
    def observe(histogram: Histogram, request: ObservedRequest) -> None:
        value = request.series.get(series)
        if value is not None:
            histogram.labels(stage_label(request.metric)).observe(value)

    return observe


# Takes the series once for both the declaration and the hook, so a histogram cannot
# declare one series while observing another.
def _series_histogram(
    name: str,
    documentation: str,
    series: str,
    buckets: tuple[float, ...],
    enabled: Callable[[Config], bool] = always,
) -> MetricSpec[Histogram]:
    return MetricSpec(
        name=name,
        documentation=documentation,
        metric_type=Histogram,
        labelnames=("stage",),
        buckets=buckets,
        enabled=enabled,
        series=series,
        on_request=_observe(series),
    )


LATENCY_SPECS: tuple[MetricSpec[Any], ...] = (
    _series_histogram(
        name="inference_perf_request_latency_seconds",
        documentation="End-to-end latency of successful requests by stage.",
        series="latency",
        buckets=REQUEST_LATENCY_BUCKETS,
    ),
    _series_histogram(
        name="inference_perf_time_to_first_token_seconds",
        documentation="Time to first token of successful streaming requests by stage: first generated token minus request start.",
        series="ttft",
        buckets=TTFT_BUCKETS,
        enabled=streaming_only,
    ),
    _series_histogram(
        name="inference_perf_time_per_output_token_seconds",
        documentation=(
            "Time per output token of successful streaming requests by stage: "
            "(last chunk - first chunk) / (output tokens - 1), for requests with more than one output token. "
            "Output tokens are counted as the report counts them: the client-side count, or the server's "
            "under report.request_lifecycle.use_server_output_tokens."
        ),
        series="tpot",
        buckets=TPOT_BUCKETS,
        enabled=streaming_only,
    ),
    _series_histogram(
        name="inference_perf_normalized_time_per_output_token_seconds",
        documentation=(
            "Request latency over output tokens of successful requests with output tokens, by stage. "
            "Output tokens are counted as for time per output token."
        ),
        series="ntpot",
        buckets=NTPOT_BUCKETS,
    ),
    _series_histogram(
        name="inference_perf_schedule_delay_seconds",
        documentation="How late each request was sent against its schedule, by stage: request start minus scheduled time. Every request, successes and failures.",
        series="schedule_delay",
        buckets=SCHEDULE_DELAY_BUCKETS,
    ),
)
