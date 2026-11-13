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
"""Feed a stage's finished requests to its stop condition and say when it has held.

A condition over ``t`` alone compiles to a deadline and needs nothing here. One
that reads measured terms is evaluated on every result the request metric
collector hands to its observers in the main process, the path the circuit
breakers are fed by, and once a second while no result arrives. The second
kind of evaluation is what lets a stall (``since_last_completion``) or a time
range that closes during a quiet spell end the stage.

Each result is valued by ``series_values``, the report's own formulas, so a
term reads the number the report and the runtime metrics carry.
"""

from __future__ import annotations

from collections import defaultdict, deque
from typing import Deque, DefaultDict, Dict, Mapping, Optional

from inference_perf.apis import RequestLifecycleMetric
from inference_perf.apis.base import is_stage_teardown_cancelled
from inference_perf.reportgen.base import series_values
from inference_perf.utils.numeric.expression import MeasuredState, Observation, Predicate

# Seconds without an evaluation after which the stage loop's tick evaluates the condition.
TICK_SECONDS = 1.0


class StopConditionMonitor:
    """One stage's measured stop condition, evaluated as its results are observed.

    ``hold`` asks for the condition to hold on that many consecutive
    evaluations before it ends the stage, so a windowed quantile that flaps
    above a threshold for one result does not. Only the running stage's results
    count: a result from an earlier stage drained late is ignored.

    ``phases`` is the load generator's ``requests_by_phase()`` at the moment of
    the evaluation. ``in_flight`` and ``queued`` have no value without it.
    """

    def __init__(self, stage_id: int, predicate: Predicate, hold: int, use_server_output_tokens: bool = False) -> None:
        self.stage_id = stage_id
        self.predicate = predicate
        self.hold = hold
        self.use_server_output_tokens = use_server_output_tokens
        self.state = MeasuredState(predicate.terms, predicate.signals)
        self.met_at: Optional[float] = None
        self.values_when_met: Dict[str, float] = {}
        self._started_at: Optional[float] = None
        self._consecutive = 0
        self._last_evaluation = 0.0
        # Dispatch order, for _over_sent terms: each scheduled time maps to the
        # positions it was dispatched at, recorded only up to the last one read.
        self._sent_needed = max((t.sent_stop or 0 for t in predicate.terms.values()), default=0)
        self._dispatched = 0
        self._sent_positions: DefaultDict[float, Deque[int]] = defaultdict(deque)

    def start(self, started_at: float) -> None:
        """Mark the stage's dispatch start on the clock results and ticks read, so ``t`` is stage time."""
        self._started_at = started_at
        # The first tick lands a second into the stage, never before it starts.
        self._last_evaluation = started_at

    def dispatched(self, scheduled_time: float) -> None:
        """Record the next request's scheduled time, in dispatch order, so its result can be placed."""
        if self._dispatched < self._sent_needed:
            self._sent_positions[scheduled_time].append(self._dispatched)
        self._dispatched += 1

    def observe(self, metric: RequestLifecycleMetric, now: float, phases: Optional[Mapping[str, int]] = None) -> None:
        """Account one observed result and evaluate the condition at clock ``now``."""
        if self._started_at is None or metric.stage_id != self.stage_id or self.met_at is not None:
            return
        positions = self._sent_positions.get(metric.scheduled_time)
        self.state.observe(
            Observation(
                t=metric.end_time - self._started_at,
                failed=metric.error is not None,
                values=series_values(metric, self.use_server_output_tokens),
                sent_index=positions.popleft() if positions else None,
                # the load generator cut it off at teardown: neither a success nor a failure
                cancelled=is_stage_teardown_cancelled(metric.error),
            ),
            now - self._started_at,
        )
        self._evaluate(now, phases)

    def tick(self, now: float, phases: Optional[Mapping[str, int]] = None) -> None:
        """Evaluate the condition at clock ``now`` if nothing has for ``TICK_SECONDS``."""
        if self._started_at is None or self.met_at is not None or now - self._last_evaluation < TICK_SECONDS:
            return
        self._evaluate(now, phases)

    def _evaluate(self, now: float, phases: Optional[Mapping[str, int]]) -> None:
        assert self._started_at is not None
        self._last_evaluation = now
        t = now - self._started_at
        if not self.predicate.holds(t, self.state.values(t, phases=phases)):
            self._consecutive = 0
            return
        self._consecutive += 1
        if self._consecutive >= self.hold:
            self.met_at = t
            self.values_when_met = self.state.readings(t, phases=phases)
