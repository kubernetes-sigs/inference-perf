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
"""How far through its planned work each stage is: the liveness family.

These answer "how much is left" rather than "how did it go". They are
deliberately separate from the outcome metrics in ``core.py``, which count
requests that produced a result, because the two populations differ: work
abandoned before dispatch is finished as far as the stage is concerned but
never produces a lifecycle metric. Conflating the two is what makes a
progress bar built on outcome counters stall short of its total on any run
that skips work.

Every metric here is sampled from the load generator's own counters at scrape
time, through the probes on :class:`StageContext`, so none of it travels the
worker-to-parent metric queue and none of it lags the run. That is also why
they are Gauges rather than Counters despite only climbing within a stage:
the value is sampled, not accumulated, and ``rate()`` over a progress metric
is not a question anyone asks. The pairs are meant to be read as ratios, and
`progress.py` builds the CLI's progress bars out of exactly these.
"""

from typing import Any, Callable

from prometheus_client import Gauge

from inference_perf.observability.context import RunContext, StageContext
from inference_perf.observability.metrics.registry import MetricSpec


def _freeze(gauge: Gauge, label: str, value: int) -> None:
    """Pin a stage's series to its final value.

    Rebinding ``set_function`` is the only way to do this. Once a labelled
    child has a ``set_function`` bound, every other method on it (``set``
    included) is a documented no-op, so ``gauge.labels(label).set(value)``
    here would silently leave the live probe in place. The probes read
    counters the load generator resets for the next stage, so a child left
    bound would go on to report the *next* stage's numbers under this
    stage's label.
    """

    def pinned() -> float:
        return float(value)

    gauge.labels(label).set_function(pinned)


def _bind(gauge: Gauge, label: str, probe: Callable[[], int]) -> None:
    gauge.labels(label).set_function(probe)


def _set_planned_requests(gauge: Gauge, context: StageContext) -> None:
    if context.planned_requests is not None:
        gauge.labels(context.stage_label).set(context.planned_requests)


def _set_planned_sessions(gauge: Gauge, context: StageContext) -> None:
    if context.planned_sessions is not None:
        gauge.labels(context.stage_label).set(context.planned_sessions)


def _bind_requests_finished(gauge: Gauge, context: StageContext) -> None:
    if context.planned_requests is not None:
        _bind(gauge, context.stage_label, context.requests_finished)


def _freeze_requests_finished(gauge: Gauge, context: StageContext) -> None:
    if context.planned_requests is not None:
        _freeze(gauge, context.stage_label, context.requests_finished())


def _bind_sessions_finished(gauge: Gauge, context: StageContext) -> None:
    if context.planned_sessions is not None:
        _bind(gauge, context.stage_label, context.sessions_finished)


def _freeze_sessions_finished(gauge: Gauge, context: StageContext) -> None:
    if context.planned_sessions is not None:
        _freeze(gauge, context.stage_label, context.sessions_finished())


def _bind_requests_skipped(gauge: Gauge, context: StageContext) -> None:
    if context.planned_requests is not None:
        _bind(gauge, context.stage_label, context.requests_skipped)


def _freeze_requests_skipped(gauge: Gauge, context: StageContext) -> None:
    if context.planned_requests is not None:
        _freeze(gauge, context.stage_label, context.requests_skipped())


def _bind_sessions_skipped(gauge: Gauge, context: StageContext) -> None:
    if context.planned_sessions is not None:
        _bind(gauge, context.stage_label, context.sessions_skipped)


def _freeze_sessions_skipped(gauge: Gauge, context: StageContext) -> None:
    if context.planned_sessions is not None:
        _freeze(gauge, context.stage_label, context.sessions_skipped())


def _zero_stages_completed(gauge: Gauge, context: RunContext) -> None:
    gauge.set(0)


def _advance_stages_completed(gauge: Gauge, context: StageContext) -> None:
    gauge.inc()


STAGE_REQUESTS_PLANNED = MetricSpec[Gauge](
    name="inference_perf_stage_requests_planned",
    documentation="Requests the stage set out to issue. Absent for stages whose request count is not known up front.",
    metric_type=Gauge,
    labelnames=("stage",),
    on_stage_start=_set_planned_requests,
)

STAGE_REQUESTS_FINISHED = MetricSpec[Gauge](
    name="inference_perf_stage_requests_finished",
    documentation=(
        "Requests the stage is done with, whatever became of them, including any abandoned before dispatch. "
        "Read against inference_perf_stage_requests_planned for stage completion; "
        "this is the same count the stage's own termination check reads."
    ),
    metric_type=Gauge,
    labelnames=("stage",),
    on_stage_start=_bind_requests_finished,
    on_stage_end=_freeze_requests_finished,
)

STAGE_REQUESTS_SKIPPED = MetricSpec[Gauge](
    name="inference_perf_stage_requests_skipped",
    documentation=(
        "Requests counted as finished that were never sent, because the session had already failed or the "
        "request could not be built. These produce no lifecycle metric, so they appear in no outcome counter "
        "and in no report."
    ),
    metric_type=Gauge,
    labelnames=("stage",),
    on_stage_start=_bind_requests_skipped,
    on_stage_end=_freeze_requests_skipped,
)

STAGE_SESSIONS_PLANNED = MetricSpec[Gauge](
    name="inference_perf_stage_sessions_planned",
    documentation="Sessions the stage set out to run. Only present for session-based stages.",
    metric_type=Gauge,
    labelnames=("stage",),
    on_stage_start=_set_planned_sessions,
)

STAGE_SESSIONS_FINISHED = MetricSpec[Gauge](
    name="inference_perf_stage_sessions_finished",
    documentation=(
        "Sessions the stage is done with, including any skipped because their graph could not be built. "
        "Read against inference_perf_stage_sessions_planned for stage completion."
    ),
    metric_type=Gauge,
    labelnames=("stage",),
    on_stage_start=_bind_sessions_finished,
    on_stage_end=_freeze_sessions_finished,
)

STAGE_SESSIONS_SKIPPED = MetricSpec[Gauge](
    name="inference_perf_stage_sessions_skipped",
    documentation=(
        "Sessions counted as finished whose graph could not be built, so they dispatched nothing and produced "
        "no session lifecycle metric."
    ),
    metric_type=Gauge,
    labelnames=("stage",),
    on_stage_start=_bind_sessions_skipped,
    on_stage_end=_freeze_sessions_skipped,
)

STAGES_COMPLETED = MetricSpec[Gauge](
    name="inference_perf_stages_completed",
    documentation="Stages that have ended, whether they completed or were cut short. Read against inference_perf_stages.",
    metric_type=Gauge,
    on_run_start=_zero_stages_completed,
    on_stage_end=_advance_stages_completed,
)

PROGRESS_SPECS: tuple[MetricSpec[Any], ...] = (
    STAGE_REQUESTS_PLANNED,
    STAGE_REQUESTS_FINISHED,
    STAGE_REQUESTS_SKIPPED,
    STAGE_SESSIONS_PLANNED,
    STAGE_SESSIONS_FINISHED,
    STAGE_SESSIONS_SKIPPED,
    STAGES_COMPLETED,
)
