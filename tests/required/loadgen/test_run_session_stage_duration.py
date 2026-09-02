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
"""`duration` on a trace_session_replay stage: stopping on time, and accounting for it.

A stage could only be bounded by session count before this; the nearest time-based
option was ``timeout``, which is a failure mechanism, so an intended stop was recorded
as FAILED and the sessions in flight at the cutoff produced no lifecycle metric at all.
These tests pin the two halves of the fix: the stage stops on its own deadline and still
reports COMPLETED, and the sessions it cut short are recorded as truncated rather than
dropped.

``run_session_stage`` is driven directly with a scripted SessionGenerator and mock IPC —
no worker processes and no server. ``sleep`` is patched out so the dispatch loop spins
freely, which is how the existing timeout coverage keeps a wall-clock deadline fast; the
assertions are about ordering and status, never about exact timings, so a slow machine
makes these slower rather than flaky.
"""

import multiprocessing as mp
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from inference_perf.apis import LazyLoadInferenceAPIData, SessionLifecycleMetric
from inference_perf.client.server_metrics.base import StageStatus
from inference_perf.config import APIConfig, APIType, DataConfig, LoadConfig, LoadType, TraceSessionReplayLoadStage
from inference_perf.datagen import SessionGenerator
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.metrics import SessionMetricsCollector


class ScriptedSessionGenerator(SessionGenerator):
    """A corpus of trivial sessions whose completion the test controls.

    Each session holds one event and completes after ``polls_to_complete`` checks, so a
    stage makes steady progress without any real request being sent. ``activated`` records
    dispatch order, which is what the admission assertions read.
    """

    def __init__(self, num_sessions: int, polls_to_complete: Optional[int] = 2) -> None:
        """``polls_to_complete=None`` means no session ever finishes on its own.

        That is how a stage is held open until its deadline without depending on how fast
        the loop happens to spin: the pool fills to ``concurrent_sessions`` and stays full,
        so admission is capped by the pool rather than by throughput.
        """
        super().__init__(APIConfig(type=APIType.Chat), DataConfig(), None)
        self._num_sessions = num_sessions
        self._polls_to_complete = polls_to_complete
        self.activated: List[str] = []
        self.cleaned_up: List[str] = []
        self._polls: Dict[str, int] = {}
        self._never_completes: set[str] = set()

    def never_complete(self, session_id: str) -> None:
        """Keep one session active for the whole stage, so it is still running at the cutoff."""
        self._never_completes.add(session_id)

    # --- SessionGenerator surface ---------------------------------------

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Chat]

    def get_session_count(self) -> int:
        return self._num_sessions

    def get_session_info(self, session_index: int) -> Dict[str, Any]:
        return {
            "session_id": f"s{session_index}",
            "file_path": f"s{session_index}.json",
            "source_id": f"s{session_index}.json",
            "session_index": session_index,
            "num_events": 1,
        }

    def get_session_event_indices(self, session_index: int) -> List[int]:
        return [0]

    def get_session_events(self, session_index: int) -> List[LazyLoadInferenceAPIData]:
        return [LazyLoadInferenceAPIData(data_index=session_index, preferred_worker_id=-1)]

    def activate_session(self, session_id: str) -> None:
        self.activated.append(session_id)
        self._polls[session_id] = 0

    def check_session_completed(self, session_id: str) -> bool:
        if self._polls_to_complete is None or session_id in self._never_completes:
            return False
        self._polls[session_id] = self._polls.get(session_id, 0) + 1
        return self._polls[session_id] >= self._polls_to_complete

    def get_session_state(self, session_id: str) -> Any:
        return None

    def build_session_metric(
        self, session_id: str, stage_id: int, start_time: float, end_time: float
    ) -> SessionLifecycleMetric:
        completed = session_id not in self._never_completes
        return SessionLifecycleMetric(
            session_id=session_id,
            stage_id=stage_id,
            file_path=f"{session_id}.json",
            start_time=start_time,
            end_time=end_time,
            duration_sec=end_time - start_time,
            num_events=1,
            num_events_completed=1 if completed else 0,
        )

    def cleanup_session(self, session_id: str) -> None:
        self.cleaned_up.append(session_id)


def _load_generator(datagen: SessionGenerator) -> tuple[LoadGenerator, SessionMetricsCollector]:
    collector = SessionMetricsCollector()
    config = LoadConfig(type=LoadType.TRACE_SESSION_REPLAY, stages=[], num_workers=1)
    return LoadGenerator(datagen, config, collector), collector


async def _run_stage(
    loadgen: LoadGenerator,
    stage: TraceSessionReplayLoadStage,
    stage_id: int = 0,
) -> MagicMock:
    """Drive one session stage with mock IPC. Returns the request queue for inspection."""
    request_queue = MagicMock()
    # Real shared counters: run_session_stage takes their locks.
    finished = mp.Value("i", 0)
    active = mp.Value("i", 0)
    request_phase = mp.Event()

    # `sleep` patched so the dispatch loop spins instead of yielding for real time; the
    # stage's own deadline still comes from the wall clock.
    with patch("inference_perf.loadgen.load_generator.sleep", new_callable=AsyncMock):
        await loadgen.run_session_stage(
            stage_id,
            stage,
            request_queue,
            active,
            finished,
            request_phase,
            cancel_signal=None,
            progress_ctx=None,
        )
    return request_queue


@pytest.mark.asyncio
async def test_duration_bounded_stage_completes_rather_than_fails(caplog: pytest.LogCaptureFixture) -> None:
    """A stage that stops on its own deadline is COMPLETED, not FAILED.

    This is the whole point of the field: `timeout` already stopped a stage on time, but
    recorded it as a failure, so an intended stop was indistinguishable from a fault.
    """
    datagen = ScriptedSessionGenerator(num_sessions=5000, polls_to_complete=None)
    loadgen, _ = _load_generator(datagen)
    stage = TraceSessionReplayLoadStage(concurrent_sessions=4, duration=0.2)

    with caplog.at_level("INFO"):
        await _run_stage(loadgen, stage)

    # Assert which exit fired, so the test cannot pass by the stage merely running out of
    # sessions before the deadline.
    assert "duration 0.2s reached" in caplog.text
    info = loadgen.stage_runtime_info[0]
    assert info.status == StageStatus.COMPLETED
    assert info.duration == 0.2


@pytest.mark.asyncio
async def test_no_session_is_admitted_after_the_deadline() -> None:
    """Nothing new starts once the deadline passes, so the stage stops offering load.

    Dispatch runs earlier in the loop than the exit check, so without the admission guard
    a session could be started just after the deadline only to be cut short at once.
    """
    # No session finishes on its own, so the pool fills to 4 and stays full: admission is
    # capped by the pool, and the stage can only end on its deadline.
    datagen = ScriptedSessionGenerator(num_sessions=5000, polls_to_complete=None)
    loadgen, _ = _load_generator(datagen)
    stage = TraceSessionReplayLoadStage(concurrent_sessions=4, duration=0.2)

    await _run_stage(loadgen, stage)

    assert datagen.activated == ["s0", "s1", "s2", "s3"]
    assert loadgen.stage_runtime_info[0].status == StageStatus.COMPLETED
    # Every session that started was accounted for and released.
    assert len(datagen.cleaned_up) == len(datagen.activated)


@pytest.mark.asyncio
async def test_sessions_running_at_the_deadline_are_recorded_as_truncated() -> None:
    """A session still running at the cutoff produces a metric marked truncated.

    Before this, session metrics were only built on completion, so the sessions in flight
    at the boundary produced nothing at all — and because a long session is likelier to be
    mid-flight at any instant, the ones lost were disproportionately the long ones.
    """
    datagen = ScriptedSessionGenerator(num_sessions=8)
    datagen.never_complete("s0")  # still running when the deadline lands
    loadgen, collector = _load_generator(datagen)
    stage = TraceSessionReplayLoadStage(concurrent_sessions=4, duration=0.2)

    await _run_stage(loadgen, stage)

    metrics = {m.session_id: m for m in collector.get_metrics()}
    assert "s0" in metrics, "a session cut short must still be reported"
    assert metrics["s0"].truncated is True
    # Sessions that finished normally are untouched.
    assert all(not m.truncated for sid, m in metrics.items() if sid != "s0")


@pytest.mark.asyncio
async def test_boundary_rows_carry_the_tfut_dispatch_anchor() -> None:
    """Rows recorded at the boundary get the same TFUT anchor as the in-loop path.

    `_compute_tfut` needs `dispatch_perf_counter`; without it the session reports
    `no_dispatch_anchor` and drops out of `tfut_sec`. That loses a valid observation,
    since a session's first user-facing event usually completes long before the deadline
    falls, and first-token latency is much of what a duration-bounded run measures.
    """
    datagen = ScriptedSessionGenerator(num_sessions=5000, polls_to_complete=None)
    loadgen, collector = _load_generator(datagen)
    stage = TraceSessionReplayLoadStage(concurrent_sessions=4, duration=0.2)

    await _run_stage(loadgen, stage)

    metrics = collector.get_metrics()
    assert metrics, "the boundary rows must be recorded at all"
    assert all(m.dispatch_perf_counter is not None for m in metrics)


@pytest.mark.asyncio
async def test_boundary_rows_end_at_the_stage_boundary_not_after_teardown() -> None:
    """Boundary rows are stamped with the end of the load window, not the current time.

    Teardown is excluded from the metrics window, so reading the clock while recording
    these rows would pull up to the whole teardown grace into two published numbers: the
    `session_duration_sec` percentiles (via a wind-down completion, which is not marked
    truncated) and the span `summarize_sessions` divides by for `sessions_per_second`.
    """
    datagen = ScriptedSessionGenerator(num_sessions=5000, polls_to_complete=None)
    loadgen, collector = _load_generator(datagen)
    stage = TraceSessionReplayLoadStage(concurrent_sessions=4, duration=0.2)

    await _run_stage(loadgen, stage)

    stage_end = loadgen.stage_runtime_info[0].end_time
    metrics = collector.get_metrics()
    assert metrics
    # Exactly the stage boundary, not merely close to it: both come from the same stamp.
    assert all(m.end_time == stage_end for m in metrics)


@pytest.mark.asyncio
async def test_completed_sessions_are_not_marked_truncated() -> None:
    """A stage that runs its corpus out before the deadline truncates nothing."""
    datagen = ScriptedSessionGenerator(num_sessions=4)
    loadgen, collector = _load_generator(datagen)
    # Generous window: all four sessions finish well inside it.
    stage = TraceSessionReplayLoadStage(concurrent_sessions=4, duration=30)

    await _run_stage(loadgen, stage)

    metrics = collector.get_metrics()
    assert len(metrics) == 4
    assert all(not m.truncated for m in metrics)
    assert loadgen.stage_runtime_info[0].status == StageStatus.COMPLETED


@pytest.mark.asyncio
async def test_cursor_advances_only_by_sessions_actually_consumed() -> None:
    """The session cursor must not skip sessions the stage never started.

    The cursor used to advance by the planned slice up front, which was right when a stage
    always ran its whole slice. A duration-bounded stage stops early, so pre-advancing
    would silently burn the remainder and the next stage would start too far in.
    """
    datagen = ScriptedSessionGenerator(num_sessions=5000, polls_to_complete=None)
    loadgen, _ = _load_generator(datagen)
    stage = TraceSessionReplayLoadStage(concurrent_sessions=4, duration=0.2)

    await _run_stage(loadgen, stage)

    # Only the 4 sessions the pool admitted are consumed; the other 4996 remain for the
    # next stage. Pre-advancing would have left the cursor at 5000.
    assert loadgen._session_cursor == 4
    assert loadgen._session_cursor == len(datagen.activated)


@pytest.mark.asyncio
async def test_cursor_still_covers_the_whole_slice_when_the_stage_runs_out() -> None:
    """Without duration, the cursor lands exactly where it did before this change."""
    datagen = ScriptedSessionGenerator(num_sessions=5)
    loadgen, _ = _load_generator(datagen)
    stage = TraceSessionReplayLoadStage(concurrent_sessions=2, num_sessions=5)

    await _run_stage(loadgen, stage)

    assert loadgen._session_cursor == 5


@pytest.mark.asyncio
async def test_short_corpus_warns_that_the_window_was_not_met(caplog: pytest.LogCaptureFixture) -> None:
    """Running out of corpus before the deadline is reported, not silently accepted.

    The stage still succeeded — nothing failed — but it measured a shorter window than was
    asked for, which would otherwise be invisible.
    """
    datagen = ScriptedSessionGenerator(num_sessions=3)
    loadgen, _ = _load_generator(datagen)
    stage = TraceSessionReplayLoadStage(concurrent_sessions=2, duration=60)

    with caplog.at_level("WARNING"):
        await _run_stage(loadgen, stage)

    assert "corpus exhausted" in caplog.text
    assert loadgen.stage_runtime_info[0].status == StageStatus.COMPLETED


@pytest.mark.asyncio
async def test_stage_without_duration_is_unchanged() -> None:
    """A count-bounded stage behaves exactly as before: no duration, no truncation."""
    datagen = ScriptedSessionGenerator(num_sessions=6)
    loadgen, collector = _load_generator(datagen)
    stage = TraceSessionReplayLoadStage(concurrent_sessions=3, num_sessions=6)

    await _run_stage(loadgen, stage)

    info = loadgen.stage_runtime_info[0]
    assert info.status == StageStatus.COMPLETED
    assert info.duration is None
    assert len(collector.get_metrics()) == 6
    assert all(not m.truncated for m in collector.get_metrics())
