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
"""Evaluate a stage's stop condition on the results the load generator observes.

A condition over ``t`` alone compiles to a deadline and needs nothing here. One
that reads measured symbols is re-evaluated on every result the request metric
collector hands to its observers in the main process, the path the circuit
breaker is fed by, so it reacts when a measurement changes and never on a timer.
"""

from __future__ import annotations

from collections import deque
from typing import Deque, Dict, Optional

import numpy as np

from inference_perf.apis import RequestLifecycleMetric, StreamedResponseMetrics
from inference_perf.utils.numeric.expression import Predicate

_COUNTERS = ("completed", "errors")


def per_result_values(metric: RequestLifecycleMetric) -> Dict[str, float]:
    """The latency, ttft and tpot of one successful result, by the report's definitions.

    Same formulas as ``compute_request_latency_metrics`` in the report, pinned
    equal by test; kept here so the load generator does not import the report.
    A failed result has none, and ttft and tpot need a streamed response.
    """
    if metric.error is not None:
        return {}
    values = {"latency": metric.end_time - metric.start_time}
    response = metric.info.response_metrics
    if isinstance(response, StreamedResponseMetrics) and len(response.output_token_times) > 1:
        first, last = response.output_token_times[0], response.output_token_times[-1]
        values["ttft"] = first - metric.start_time
        if response.output_tokens > 1:
            values["tpot"] = (last - first) / (response.output_tokens - 1)
    return values


class StopConditionMonitor:
    """One stage's measured stop condition, evaluated as its results are observed.

    ``completed`` and ``errors`` are stage totals. ``latency``, ``ttft`` and
    ``tpot`` aggregates are over the last ``window`` results and have no value
    until the window is full, so a comparison on them is false until then. A
    counter cannot flap, so with ``hold`` 1 it ends the stage on the exact
    result that flipped it; a windowed percentile can, so ``hold`` asks for the
    condition to hold on that many consecutive results first. Only the running
    stage's results count: a result from an earlier stage drained late is
    ignored.
    """

    def __init__(self, stage_id: int, predicate: Predicate, window: int, hold: int) -> None:
        self.stage_id = stage_id
        self.predicate = predicate
        self.window = window
        self.hold = hold
        self.completed = 0
        self.errors = 0
        self.met_at: Optional[float] = None
        self.values_when_met: Dict[str, float] = {}
        self._started_at: Optional[float] = None
        self._consecutive = 0
        families = {name.rsplit("_", 1)[0] for name in predicate.measured_symbols if name not in _COUNTERS}
        self._samples: Dict[str, Deque[float]] = {family: deque(maxlen=window) for family in families}

    def start(self, started_at: float) -> None:
        """Mark the stage's dispatch start on the clock ``observe`` reads, so ``t`` is stage time."""
        self._started_at = started_at

    def observe(self, metric: RequestLifecycleMetric, now: float) -> None:
        """Account one observed result and re-evaluate the condition at stage time ``now - start``."""
        if self._started_at is None or metric.stage_id != self.stage_id or self.met_at is not None:
            return
        self.completed += 1
        if metric.error is not None:
            self.errors += 1
        for family, value in per_result_values(metric).items():
            if family in self._samples:
                self._samples[family].append(value)
        t = now - self._started_at
        values = self.values()
        if not self.predicate.holds(t, values):
            self._consecutive = 0
            return
        self._consecutive += 1
        if self._consecutive >= self.hold:
            self.met_at = t
            self.values_when_met = values

    def values(self) -> Dict[str, float]:
        """Current value of each measured symbol the condition reads; an aggregate whose window is not full is absent."""
        out: Dict[str, float] = {}
        for name in sorted(self.predicate.measured_symbols):
            if name == "completed":
                out[name] = float(self.completed)
            elif name == "errors":
                out[name] = float(self.errors)
            else:
                family, stat = name.rsplit("_", 1)
                samples = self._samples[family]
                if len(samples) < self.window:
                    continue
                arr = np.fromiter(samples, dtype=np.float64, count=len(samples))
                out[name] = float(arr.mean()) if stat == "mean" else float(np.percentile(arr, int(stat[1:])))
        return out
