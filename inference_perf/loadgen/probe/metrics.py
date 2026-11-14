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
"""Adapter from request lifecycle metrics to probe rung measurements.

This is the probe package's single inward-facing module: it consumes
`RequestLifecycleMetric` records produced by the model server clients and
turns them into `RungResult`s for the estimator. Timestamps on lifecycle
metrics come from `time.perf_counter()`, so window bounds passed here must
be captured with the same clock.

Phase classification compares a rung's median TTFT and inter-token latency
against the unloaded baseline rung. It needs token-level timestamps, which
only streaming APIs record; without them rungs keep `SaturationSignal.NONE`
and the ladder still works on throughput alone.
"""

from dataclasses import dataclass
from typing import List, Optional, Sequence

import numpy as np

from inference_perf.apis import RequestLifecycleMetric, StreamedResponseMetrics

from .estimator import make_rung
from .result import RungResult, SaturationSignal, classify_saturation

# The generator is only a suspect once its own loop is late by a readable
# margin; below this the lag is timer resolution, not back-pressure. The
# baseline floor keeps a run whose unloaded lag is already at the noise floor
# from reading a two-fold ratio out of timer jitter.
CLIENT_LAG_FLOOR_SEC = 0.01
CLIENT_LAG_BASELINE_FLOOR_SEC = 0.001


@dataclass(frozen=True)
class LatencyProfile:
    """Median per-request TTFT and inter-token latency over one measurement window."""

    ttft: float
    itl: float


def latency_profile(metrics: Sequence[RequestLifecycleMetric]) -> Optional[LatencyProfile]:
    """Median TTFT and inter-token latency of the streamed requests in `metrics`.

    Returns None when no request carries at least two token-level timestamps
    (non-streaming APIs, or single-token responses), in which case phase
    classification is unavailable.
    """
    ttfts: List[float] = []
    itls: List[float] = []
    for metric in metrics:
        response_metrics = metric.info.response_metrics
        if not isinstance(response_metrics, StreamedResponseMetrics) or len(response_metrics.output_token_times) < 2:
            continue
        token_times = np.asarray(response_metrics.output_token_times, dtype=np.float64)
        ttfts.append(float(token_times[0] - metric.start_time))
        itls.append(float(np.mean(np.diff(token_times))))
    if not ttfts:
        return None
    ttft = float(np.median(np.asarray(ttfts)))
    itl = float(np.median(np.asarray(itls)))
    if ttft <= 0 or itl <= 0:
        return None
    return LatencyProfile(ttft=ttft, itl=itl)


def rung_from_metrics(
    concurrency: int,
    metrics: Sequence[RequestLifecycleMetric],
    window_start: float,
    window_end: float,
    baseline: Optional[LatencyProfile] = None,
    baseline_client_lag: Optional[float] = None,
    client_lag: float = 0.0,
    inflation_threshold: float = 2.0,
    num_batches: int = 8,
) -> RungResult:
    """Build a `RungResult` for one closed-loop rung from its lifecycle metrics.

    Only successful requests count toward throughput, and the failed fraction
    is recorded as `error_rate` so a backend that fails fast is visible as
    such rather than silently measuring a subset of the traffic. When
    `baseline` (the unloaded rung's latency profile), `baseline_client_lag` (the
    unloaded rung's loop lag) and streamed timestamps are available, the rung
    is classified prefill-, decode-, or client-bound from TTFT, inter-token,
    and generator-loop-lag inflation; otherwise the signal stays NONE.

    Raises ValueError when no successful request completed inside the window,
    which is a measurement failure, not a zero rate.
    """
    successes = [metric for metric in metrics if metric.error is None]
    completion_times = [metric.end_time for metric in successes]
    latencies = [metric.end_time - metric.start_time for metric in successes]
    error_rate = 0.0 if not metrics else 1.0 - len(successes) / len(metrics)

    ttft_inflation = 1.0
    itl_inflation = 1.0
    if baseline is not None:
        in_window = [metric for metric in successes if window_start <= metric.end_time < window_end]
        profile = latency_profile(in_window)
        if profile is not None:
            ttft_inflation = profile.ttft / baseline.ttft
            itl_inflation = profile.itl / baseline.itl

    client_lag_inflation = 1.0
    if baseline_client_lag is not None and client_lag >= CLIENT_LAG_FLOOR_SEC:
        client_lag_inflation = client_lag / max(baseline_client_lag, CLIENT_LAG_BASELINE_FLOOR_SEC)

    signal = SaturationSignal.NONE
    if baseline is not None or baseline_client_lag is not None:
        signal = classify_saturation(
            ttft_inflation=ttft_inflation,
            itl_inflation=itl_inflation,
            client_lag_inflation=client_lag_inflation,
            threshold=inflation_threshold,
        )
    return make_rung(
        concurrency,
        completion_times,
        latencies,
        window_start,
        window_end,
        num_batches=num_batches,
        signal=signal,
        error_rate=error_rate,
        client_lag=client_lag,
    )
