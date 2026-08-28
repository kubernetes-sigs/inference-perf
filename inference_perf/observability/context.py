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

"""What the run tells its observers when a run or a stage begins.

These carry live probes, not snapshots: a gauge bound to
:attr:`StageContext.requests_finished` reads the workers' shared counter at
scrape time, so it never lags behind the run the way a value copied at stage
start would. Kept in a leaf module with no inference-perf imports so the load
generator can build a context without importing the metrics package.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from inference_perf.config import Config


def _no_requests_in_flight() -> int:
    return 0


def _zero() -> int:
    return 0


@dataclass(frozen=True)
class RunContext:
    """What ``on_run_start`` hooks may read: the static run config plus live
    probes into the load generator that gauges can sample on every scrape."""

    config: "Config"
    in_flight_requests: Callable[[], int] = _no_requests_in_flight


@dataclass(frozen=True)
class StageContext:
    """What ``on_stage_start`` / ``on_stage_end`` hooks may read about one stage.

    The ``planned_*`` counts are the denominators the stage is working towards
    and are ``None`` when the stage does not have one (a session stage has no
    request count up front, a request stage has no session count at all). A
    hook must not create a series for a count that is ``None``.

    The ``*_finished`` probes are read live, and they count every unit the
    stage is done with, including work abandoned before dispatch. That is what
    makes them the right denominator-mates for a progress bar: they are the
    same numbers the stage's own termination check reads.

    The ``*_skipped`` probes are the subset of finished work that was
    abandoned before it reached the server and so produced no lifecycle
    metric. They are what makes the two metric families reconcilable: at stage
    end, ``finished`` equals ``skipped`` plus the outcome counters.
    """

    stage_id: int
    planned_requests: Optional[int] = None
    planned_sessions: Optional[int] = None
    requests_finished: Callable[[], int] = _zero
    sessions_finished: Callable[[], int] = _zero
    requests_skipped: Callable[[], int] = _zero
    sessions_skipped: Callable[[], int] = _zero

    @property
    def stage_label(self) -> str:
        return str(self.stage_id)
