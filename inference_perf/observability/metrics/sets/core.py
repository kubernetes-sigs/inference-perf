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

Metric naming conventions are still being settled in
kubernetes-sigs/inference-perf#489; keep new names under the
``inference_perf_`` prefix and consistent with these until then.
"""

import time
from typing import Any

from prometheus_client import Counter, Gauge

from inference_perf.apis.base import RequestLifecycleMetric
from inference_perf.observability.metrics.registry import MetricSpec


def _mark_run_start(gauge: Gauge) -> None:
    start = time.monotonic()
    gauge.set_function(lambda: time.monotonic() - start)


def _count_request(counter: Counter, metric: RequestLifecycleMetric) -> None:
    stage = "" if metric.stage_id is None else str(metric.stage_id)
    status = "failure" if metric.error is not None else "success"
    counter.labels(stage, status).inc()


CORE_SPECS: tuple[MetricSpec[Any], ...] = (
    MetricSpec(
        name="inference_perf_run_elapsed_seconds",
        documentation="Wall-clock seconds elapsed since the benchmark run started; 0 until the run starts.",
        metric_type=Gauge,
        on_run_start=_mark_run_start,
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
)
