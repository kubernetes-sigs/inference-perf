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
"""The CLI's progress bars, rendered from the metrics the run exports.

Every bar is a view of two exported metrics: a numerator and a denominator,
named by :class:`BarSpec` and read out of the run's ``CollectorRegistry``.
Nothing here accepts a count, so there is no way to draw a bar from a local
variable; adding a bar means exporting the metrics it reads first. That is
the point. A progress bar and a metric that disagree are worse than either
alone, and the only way they cannot disagree is if there is one number.

This is the only module allowed to import ``rich.progress`` (enforced by the
``TID251`` banned-api rule in ``pyproject.toml``, so a second import site
fails ``pdm run lint`` rather than review). Callers take a
:class:`ProgressBars` and can only ask it to open a declared bar.

Reading is by snapshot. ``CollectorRegistry.get_sample_value`` runs a full
collect internally, so it costs about as much for one value as a whole
snapshot does for all of them; taking one snapshot per refresh is both
cheaper and the reason every bar on screen shows the same instant.
"""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

from prometheus_client import CollectorRegistry
from rich.console import Console
from rich.progress import (  # noqa: TID251
    BarColumn,
    MofNCompleteColumn,
    Progress,
    TaskID,
    TaskProgressColumn,
    TextColumn,
    TimeElapsedColumn,
    TimeRemainingColumn,
)

from inference_perf.observability.metrics.registry import MetricSpec, always, exposition_name
from inference_perf.observability.metrics.sets import ALL_SPECS
from inference_perf.observability.metrics.sets.core import STAGES
from inference_perf.observability.metrics.sets.progress import (
    STAGE_REQUESTS_FINISHED,
    STAGE_REQUESTS_PLANNED,
    STAGE_SESSIONS_FINISHED,
    STAGE_SESSIONS_PLANNED,
    STAGES_COMPLETED,
)

# Matches rich's own default refresh rate: the screen cannot show more than
# this, so sampling faster only costs collects.
DEFAULT_REFRESH_HZ = 10.0

SampleKey = Tuple[str, Tuple[Tuple[str, str], ...]]


@dataclass(frozen=True)
class BarSpec:
    """One progress bar, declared as a ratio of two exported metrics.

    ``description`` may reference the labels the bar is opened with, as in
    ``"Stage {stage} Requests"``. ``completed`` and ``total`` are the specs
    themselves rather than metric names so that a typo is an import error and
    so the sample name (including the ``_total`` suffix
    ``prometheus_client`` adds to counters) is resolved in one place.
    """

    name: str
    description: str
    completed: MetricSpec[Any]
    total: MetricSpec[Any]


OVERALL_BAR = BarSpec(
    name="overall",
    description="Overall Progress",
    completed=STAGES_COMPLETED,
    total=STAGES,
)

STAGE_REQUESTS_BAR = BarSpec(
    name="stage_requests",
    description="Stage {stage} Requests",
    completed=STAGE_REQUESTS_FINISHED,
    total=STAGE_REQUESTS_PLANNED,
)

STAGE_SESSIONS_BAR = BarSpec(
    name="stage_sessions",
    description="Stage {stage} Sessions",
    completed=STAGE_SESSIONS_FINISHED,
    total=STAGE_SESSIONS_PLANNED,
)

BAR_SPECS: Tuple[BarSpec, ...] = (OVERALL_BAR, STAGE_REQUESTS_BAR, STAGE_SESSIONS_BAR)


def _sample_key(name: str, labels: Dict[str, str]) -> SampleKey:
    return (name, tuple(sorted(labels.items())))


def snapshot(registry: CollectorRegistry) -> Dict[SampleKey, float]:
    """Every sample in the registry, read in one collect."""
    values: Dict[SampleKey, float] = {}
    for metric_family in registry.collect():
        for sample in metric_family.samples:
            values[_sample_key(sample.name, dict(sample.labels))] = sample.value
    return values


def validate_bar_specs(bar_specs: Sequence[BarSpec], metric_specs: Sequence[MetricSpec[Any]]) -> None:
    """Check every bar can actually be rendered from the exported metrics.

    Two ways a bar silently breaks, both caught here rather than on screen:
    the metric it names is not exported at all, or it is exported only under
    some configs, in which case the bar would be empty on every other config.
    """
    exported = {spec.name: spec for spec in metric_specs}
    for bar in bar_specs:
        for role, spec in (("completed", bar.completed), ("total", bar.total)):
            if exported.get(spec.name) is not spec:
                raise ValueError(
                    f"progress bar {bar.name!r} reads {spec.name!r} as its {role} metric, "
                    "but that spec is not in ALL_SPECS; a bar can only read an exported metric"
                )
            if spec.enabled is not always:
                raise ValueError(
                    f"progress bar {bar.name!r} reads {spec.name!r} as its {role} metric, but that metric is "
                    "config-conditional; a bar-backing metric must be exported on every run or the bar has no "
                    "source when the condition is off"
                )


class BarHandle:
    """A bar that has been opened. Closing it removes it from the display."""

    def __init__(self, bars: "ProgressBars", task_id: TaskID) -> None:
        self._bars = bars
        self._task_id = task_id

    def close(self) -> None:
        self._bars._close(self._task_id)


class ProgressBars:
    """Owns the terminal progress display and refreshes it from the registry.

    Use as a context manager around the run. The only way to add a bar is
    :meth:`open` with a declared :class:`BarSpec`, and the only way a bar's
    value changes is :meth:`refresh`, which re-reads the registry. There is
    deliberately no method that takes a count.
    """

    def __init__(
        self,
        registry: Optional[CollectorRegistry],
        console: Optional[Console] = None,
        refresh_hz: float = DEFAULT_REFRESH_HZ,
    ) -> None:
        # No registry means no metrics, and a bar with no metric behind it is
        # exactly what this module exists to prevent, so the display is simply
        # not shown rather than falling back to counting locally.
        validate_bar_specs(BAR_SPECS, ALL_SPECS)
        self._registry = registry
        self._min_interval = 1.0 / refresh_hz if refresh_hz > 0 else 0.0
        self._last_refresh = 0.0
        self._open: List[Tuple[TaskID, BarSpec, Dict[str, str]]] = []
        self._progress = Progress(  # noqa: TID251
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            MofNCompleteColumn(),
            TimeRemainingColumn(),
            TimeElapsedColumn(),
            console=console,
            redirect_stdout=True,
            redirect_stderr=True,
        )

    def __enter__(self) -> "ProgressBars":
        if self._registry is not None:
            self._progress.__enter__()
        return self

    def __exit__(self, *exc_info: Any) -> None:
        if self._registry is not None:
            self._progress.__exit__(*exc_info)

    def open(self, spec: BarSpec, **labels: str) -> BarHandle:
        """Add a bar reading ``spec``'s metrics at ``labels``.

        The bar renders at whatever the metrics say right now, so a bar opened
        before its stage has recorded anything starts empty rather than at a
        stale value.
        """
        if self._registry is None:
            return BarHandle(self, TaskID(-1))
        task_id = self._progress.add_task(description=spec.description.format(**labels), total=None)
        self._open.append((task_id, spec, dict(labels)))
        self._refresh_now(self._snapshot())
        return BarHandle(self, task_id)

    def refresh(self, force: bool = False) -> None:
        """Re-read the registry and update every open bar.

        Throttled to the configured rate: callers may drive this from a hot
        loop without paying a collect per iteration.
        """
        if self._registry is None or not self._open:
            return
        now = time.monotonic()
        if not force and (now - self._last_refresh) < self._min_interval:
            return
        self._last_refresh = now
        self._refresh_now(self._snapshot())

    def _snapshot(self) -> Dict[SampleKey, float]:
        assert self._registry is not None
        return snapshot(self._registry)

    def _refresh_now(self, values: Dict[SampleKey, float]) -> None:
        for task_id, spec, labels in self._open:
            completed = values.get(_sample_key(exposition_name(spec.completed), labels))
            total = values.get(_sample_key(exposition_name(spec.total), labels))
            self._progress.update(
                task_id,
                completed=0.0 if completed is None else completed,
                total=total,
            )

    def _close(self, task_id: TaskID) -> None:
        if self._registry is None:
            return
        self._open = [entry for entry in self._open if entry[0] != task_id]
        self._progress.remove_task(task_id)
