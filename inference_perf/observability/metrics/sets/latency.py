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
"""Per-request latency histograms, observed once per successful request.

Runtime values are derived exactly as the report derives them from the same
RequestLifecycleMetric (see reportgen/base.py): TTFT is the first
content-bearing chunk minus request start, TPOT is (last chunk - first chunk)
divided by (output tokens - 1). Inter-token latency is deliberately absent:
the report expands chunk timestamps to per-token timestamps at report time
(#564), and a runtime ITL over raw chunk gaps would be a second, different
number under the same name.
"""

from typing import Any

from prometheus_client import Histogram

from inference_perf.apis.base import RequestLifecycleMetric, StreamedResponseMetrics
from inference_perf.config import Config
from inference_perf.observability.metrics.registry import MetricSpec

from .core import output_tokens, stage_label

# Seconds. Wide enough for interactive chat through minute-long batch requests.
TTFT_BUCKETS = (0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0)
TPOT_BUCKETS = (0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.25, 0.5, 1.0)
REQUEST_LATENCY_BUCKETS = (0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0, 10.0, 30.0, 60.0, 120.0, 300.0, 600.0)


def streaming_only(config: Config) -> bool:
    """Streaming runs only (api.streaming); unary responses have no token timeline."""
    return bool(config.api.streaming)


def _observe_request_latency(histogram: Histogram, metric: RequestLifecycleMetric) -> None:
    if metric.error is None:
        histogram.labels(stage_label(metric)).observe(metric.end_time - metric.start_time)


def _observe_ttft(histogram: Histogram, metric: RequestLifecycleMetric) -> None:
    response = metric.info.response_metrics
    if metric.error is None and isinstance(response, StreamedResponseMetrics) and response.output_token_times:
        histogram.labels(stage_label(metric)).observe(response.output_token_times[0] - metric.start_time)


def _observe_tpot(histogram: Histogram, metric: RequestLifecycleMetric) -> None:
    response = metric.info.response_metrics
    if metric.error is not None or not isinstance(response, StreamedResponseMetrics):
        return
    times = response.output_token_times
    tokens = output_tokens(response)
    if len(times) > 1 and tokens > 1:
        histogram.labels(stage_label(metric)).observe((times[-1] - times[0]) / (tokens - 1))


LATENCY_SPECS: tuple[MetricSpec[Any], ...] = (
    MetricSpec(
        name="inference_perf_request_latency_seconds",
        documentation="End-to-end latency of successful requests by stage.",
        metric_type=Histogram,
        labelnames=("stage",),
        buckets=REQUEST_LATENCY_BUCKETS,
        on_request=_observe_request_latency,
    ),
    MetricSpec(
        name="inference_perf_time_to_first_token_seconds",
        documentation="Time to first token of successful streaming requests by stage: first content chunk minus request start.",
        metric_type=Histogram,
        labelnames=("stage",),
        buckets=TTFT_BUCKETS,
        enabled=streaming_only,
        on_request=_observe_ttft,
    ),
    MetricSpec(
        name="inference_perf_time_per_output_token_seconds",
        documentation=(
            "Time per output token of successful streaming requests by stage: "
            "(last chunk - first chunk) / (output tokens - 1), for requests with more than one output token."
        ),
        metric_type=Histogram,
        labelnames=("stage",),
        buckets=TPOT_BUCKETS,
        enabled=streaming_only,
        on_request=_observe_tpot,
    ),
)
