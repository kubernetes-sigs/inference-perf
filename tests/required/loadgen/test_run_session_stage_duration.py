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


class _IdTable:
    """`datagen._session_ids[i]`, resolved on demand so a replay index is always valid."""

    def __init__(self, gen: "ScriptedSessionGenerator") -> None:
        self._gen = gen

    def __getitem__(self, session_index: int) -> str:
        return self._gen._session_id(session_index)


class ScriptedSessionGenerator(SessionGenerator):
    """A corpus of trivial sessions whose completion the test controls.

    Each session holds one event and completes after ``polls_to_complete`` checks, so a
    stage makes steady progress without any real request being sent. ``activated`` records
    dispatch order, which is what the admission assertions read.
    """

    def __init__(
        self,
        num_sessions: int,
        polls_to_complete: Optional[int] = 2,
        cycling: bool = False,
        unbuildable_slots: Optional[set[int]] = None,
    ) -> None:
        """``polls_to_complete=None`` means no session ever finishes on its own.

        That is how a stage is held open until its deadline without depending on how fast
        the loop happens to spin: the pool fills to ``concurrent_sessions`` and stays full,
        so admission is capped by the pool rather than by throughput.

        ``cycling`` mirrors what a real replay generator does for a duration-bounded stage:
        an index past the corpus resolves to a further play of ``index % num_sessions``,
        named with the ``_dup{play}`` suffix the real generators use.
        """
        super().__init__(APIConfig(type=APIType.Chat), DataConfig(), None)
        self._num_sessions = num_sessions
        self._polls_to_complete = polls_to_complete
        self._cycling = cycling
        self.activated: List[str] = []
        self.cleaned_up: List[str] = []
        self._polls: Dict[str, int] = {}
        self._never_completes: set[str] = set()
        # Corpus slots whose graph fails to build, as a malformed trace does. The load
        # generator asks before dispatching and reads _session_ids on the skip path.
        self._unbuildable_slots = unbuildable_slots or set()
        self._session_ids = _IdTable(self)

    def is_session_buildable(self, session_index: int) -> bool:
        return (session_index % self._num_sessions) not in self._unbuildable_slots

    def supports_corpus_cycling(self) -> bool:
        return self._cycling

    def _session_id(self, session_index: int) -> str:
        if session_index < self._num_sessions:
            return f"s{session_index}"
        slot, play = session_index % self._num_sessions, session_index // self._num_sessions
        return f"s{slot}_dup{play}"

    def never_complete(self, session_id: str) -> None:
        """Keep one session active for the whole stage, so it is still running at the cutoff."""
        self._never_completes.add(session_id)

    # --- SessionGenerator surface ---------------------------------------

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Chat]

    def get_session_count(self) -> int:
        return self._num_sessions

    def get_session_info(self, session_index: int) -> Dict[str, Any]:
        session_id = self._session_id(session_index)
        return {
            "session_id": session_id,
            "file_path": f"{session_id}.json",
            "source_id": f"{session_id}.json",
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
    progress_ctx: Any = None,
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
            progress_ctx=progress_ctx,
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
async def test_cycling_runs_the_full_window_on_a_corpus_too_small_for_it() -> None:
    """A generator that can cycle keeps the stage running past the end of its corpus.

    Without this, `duration` is only honest when the operator has sized the corpus by hand:
    a three-session corpus would end the stage in milliseconds and still report COMPLETED
    for a window it never ran.
    """
    datagen = ScriptedSessionGenerator(num_sessions=3, cycling=True)
    loadgen, _ = _load_generator(datagen)
    stage = TraceSessionReplayLoadStage(concurrent_sessions=2, duration=0.2)

    await _run_stage(loadgen, stage)

    # More sessions dispatched than the corpus holds, so the window was filled by replaying.
    assert len(datagen.activated) > 3
    assert loadgen.stage_runtime_info[0].status == StageStatus.COMPLETED


@pytest.mark.asyncio
async def test_cycled_sessions_get_distinct_replay_ids() -> None:
    """Each play of a source session runs under its own id.

    Session state, completion tracking and the prefix-cache marker are all keyed by id, so
    two plays sharing one would collide and the second would read as a cache hit on the
    first.
    """
    datagen = ScriptedSessionGenerator(num_sessions=2, cycling=True)
    loadgen, _ = _load_generator(datagen)
    stage = TraceSessionReplayLoadStage(concurrent_sessions=2, duration=0.2)

    await _run_stage(loadgen, stage)

    assert len(datagen.activated) == len(set(datagen.activated)), "replay ids must not repeat"
    assert datagen.activated[:2] == ["s0", "s1"]
    # Later plays carry the _dup suffix the real generators use to trigger the marker.
    assert any("_dup" in sid for sid in datagen.activated)


@pytest.mark.asyncio
async def test_a_generator_that_cannot_cycle_still_ends_when_its_corpus_does() -> None:
    """Cycling is opt-in: a generator that does not support it behaves exactly as before."""
    datagen = ScriptedSessionGenerator(num_sessions=3, cycling=False)
    loadgen, _ = _load_generator(datagen)
    stage = TraceSessionReplayLoadStage(concurrent_sessions=2, duration=60)

    await _run_stage(loadgen, stage)

    assert len(datagen.activated) == 3
    assert loadgen.stage_runtime_info[0].status == StageStatus.COMPLETED


@pytest.mark.asyncio
async def test_skip_message_says_why_a_replayable_corpus_still_ran_out(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """A skipped stage must not blame the trace files when the files are fine.

    Only a duration-bounded stage replays the corpus, so a stage bounded by num_sessions
    stops at the end of it even on a generator that could carry on. The old wording -- "no
    sessions remaining in trace files" -- sends the reader to check their corpus, when what
    ran out is the count this stage was given.
    """
    datagen = ScriptedSessionGenerator(num_sessions=3, cycling=True)
    loadgen, _ = _load_generator(datagen)

    await _run_stage(loadgen, TraceSessionReplayLoadStage(concurrent_sessions=2, num_sessions=3), stage_id=0)
    assert loadgen._session_cursor == 3, "precondition: the first stage consumed the corpus"

    with caplog.at_level("WARNING"):
        await _run_stage(loadgen, TraceSessionReplayLoadStage(concurrent_sessions=2, num_sessions=1), stage_id=1)

    assert "this generator can replay its corpus, but only a duration-bounded stage does" in caplog.text
    assert "trace files" not in caplog.text


@pytest.mark.asyncio
async def test_non_cycling_generator_keeps_the_plain_skip_message(caplog: pytest.LogCaptureFixture) -> None:
    """Guard: with nothing left to replay, running out of corpus is the honest reason."""
    datagen = ScriptedSessionGenerator(num_sessions=3, cycling=False)
    loadgen, _ = _load_generator(datagen)

    await _run_stage(loadgen, TraceSessionReplayLoadStage(concurrent_sessions=2, num_sessions=3), stage_id=0)

    with caplog.at_level("WARNING"):
        await _run_stage(loadgen, TraceSessionReplayLoadStage(concurrent_sessions=2, num_sessions=1), stage_id=1)

    assert "no sessions remaining in trace files" in caplog.text


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


@pytest.mark.asyncio
async def test_a_corpus_where_nothing_builds_fails_instead_of_spinning(caplog: pytest.LogCaptureFixture) -> None:
    """A corpus with no buildable session must stop the stage, not replay its way to the deadline.

    A session that fails to build never enters the pool, so it never counts against
    concurrent_sessions -- which means it does not slow admission down at all. With a fixed
    corpus that is harmless: the stage skips its slice and ends. While cycling there is
    always another index, so the loop skips as fast as the process can go for the whole
    window, growing the generator's index tables the entire time and admitting nothing.
    Measured at ~115k indices in 0.05s, which is ~4 billion over a 30-minute stage.
    """
    datagen = ScriptedSessionGenerator(num_sessions=3, cycling=True, unbuildable_slots={0, 1, 2})
    loadgen, _ = _load_generator(datagen)

    with caplog.at_level("ERROR"):
        await _run_stage(loadgen, TraceSessionReplayLoadStage(concurrent_sessions=2, duration=30))

    assert datagen.activated == [], "nothing was buildable, so nothing should have started"
    # One pass over the corpus is enough to know; the bound is the corpus, not the deadline.
    assert loadgen._session_cursor <= 3 * 2
    assert loadgen.stage_runtime_info[0].status == StageStatus.FAILED
    assert "no session in the corpus could be built" in caplog.text


@pytest.mark.asyncio
async def test_one_buildable_session_is_enough_to_keep_cycling() -> None:
    """Guard: a partly broken corpus still runs. Only a wholly unbuildable one is fatal."""
    datagen = ScriptedSessionGenerator(num_sessions=3, polls_to_complete=None, cycling=True, unbuildable_slots={1, 2})
    loadgen, _ = _load_generator(datagen)

    await _run_stage(loadgen, TraceSessionReplayLoadStage(concurrent_sessions=2, duration=0.2))

    assert datagen.activated, "the buildable slot must still be replayed"
    assert all("s0" in sid for sid in datagen.activated)
    assert loadgen.stage_runtime_info[0].status == StageStatus.COMPLETED


@pytest.mark.asyncio
async def test_a_count_bounded_stage_that_times_out_also_reports_its_open_sessions() -> None:
    """Truncation reporting is not duration-only: any early exit strands sessions.

    A count-bounded stage with a ``timeout`` has always been able to stop with sessions in
    flight, and before this change those sessions produced no metric and no closed span —
    the run simply lost them. Recording them as truncated is a change to that existing
    behaviour, so it is pinned here rather than left to the duration tests.

    The status stays exceptional -- TIMED_OUT, as #786 made it -- because hitting a cap is a
    fault, unlike reaching a planned duration, which reports COMPLETED. What changed is only
    that the sessions cut off by it are now accounted for.
    """
    datagen = ScriptedSessionGenerator(num_sessions=8)
    datagen.never_complete("s0")
    loadgen, collector = _load_generator(datagen)
    stage = TraceSessionReplayLoadStage(concurrent_sessions=4, num_sessions=8, timeout=0.2)

    await _run_stage(loadgen, stage)

    info = loadgen.stage_runtime_info[0]
    assert info.status == StageStatus.TIMED_OUT, "a timeout is still a fault, not a planned stop"
    assert info.duration is None, "no duration was configured"
    assert info.sessions_not_completed_active == 1, "the stranded session is counted once, as it is recorded once"

    metrics = {m.session_id: m for m in collector.get_metrics()}
    assert "s0" in metrics, "a session stranded by the timeout must still be reported"
    assert metrics["s0"].truncated is True
    assert metrics["s0"].success is None, "truncated sessions are neither a success nor a failure"


# --- progress reporting ---------------------------------------------------
#
# A cycling stage has no session count to count towards, so the session-count progress
# bar has nothing to divide by. What it does have is a deadline, and that is what the
# stage is actually progressing towards.


class _RecordingProgress:
    """Captures what the stage asked the progress bar to display."""

    def __init__(self) -> None:
        self.added: list[dict[str, object]] = []
        self.updates: list[dict[str, Any]] = []

    def add_task(self, description: str, total: object) -> str:
        self.added.append({"description": description, "total": total})
        return "task-0"

    def update(self, task: str, **kwargs: Any) -> None:
        self.updates.append(kwargs)

    def remove_task(self, task: str) -> None:
        pass


@pytest.mark.asyncio
async def test_duration_bounded_stage_shows_progress_towards_its_deadline() -> None:
    """The bar must measure time, because a cycling stage has no session total.

    Handing the bar a total of 0 - which is what "unbounded session count" reduces to -
    leaves it stuck at 0% printing "18/0" for the whole window: the completed count is
    real but there is nothing to divide it by, so the bar conveys no progress and can
    offer no ETA.
    """
    datagen = ScriptedSessionGenerator(num_sessions=3, cycling=True)
    loadgen, _ = _load_generator(datagen)
    progress = _RecordingProgress()

    await _run_stage(loadgen, TraceSessionReplayLoadStage(concurrent_sessions=2, duration=0.3), progress_ctx=progress)

    assert len(progress.added) == 1
    task = progress.added[0]
    assert task["total"] == 0.3, "the deadline is the total a duration-bounded stage progresses towards"

    completions = [u["completed"] for u in progress.updates if "completed" in u]
    assert completions, "the bar was never advanced"
    # Elapsed seconds, never past the deadline - not a session count.
    assert all(isinstance(c, float) and 0.0 <= c <= 0.3 for c in completions), completions
    assert completions == sorted(completions), "elapsed time must not go backwards"


@pytest.mark.asyncio
async def test_count_bounded_stage_still_shows_progress_by_session_count() -> None:
    """Guard: a stage with a real session total keeps counting sessions, as before."""
    datagen = ScriptedSessionGenerator(num_sessions=4)
    loadgen, _ = _load_generator(datagen)
    progress = _RecordingProgress()

    await _run_stage(loadgen, TraceSessionReplayLoadStage(concurrent_sessions=2, num_sessions=4), progress_ctx=progress)

    assert progress.added[0]["total"] == 4
    completions = [u["completed"] for u in progress.updates if "completed" in u]
    assert completions[-1] == 4, "all four sessions finished, so the bar must read full"
    assert all(isinstance(c, int) for c in completions), "session counts, not seconds"


class _CompletesDuringTeardown(ScriptedSessionGenerator):
    """No session finishes while the stage loop runs; all of them finish during teardown.

    This is the ordinary wind-down, not an edge case: a session's last request is in
    flight at the deadline and lands inside the teardown grace, so the check after
    teardown finds it complete.
    """

    def __init__(self, num_sessions: int) -> None:
        super().__init__(num_sessions=num_sessions, polls_to_complete=None)
        self.teardown_done = False

    def check_session_completed(self, session_id: str) -> bool:
        return self.teardown_done


@pytest.mark.asyncio
async def test_sessions_finishing_during_teardown_are_not_counted_as_stranded() -> None:
    """The stranded-active count must agree with the truncated rows, which settle later.

    The count is taken while sessions are still winding down, but a session that finishes
    in the grace period gets a normal lifecycle row, not a truncated one. Reading the pool
    size pre-teardown therefore reports sessions as stranded that the report shows as
    completed, and `num_sessions` is derived from that count.
    """
    datagen = _CompletesDuringTeardown(num_sessions=5000)
    loadgen, collector = _load_generator(datagen)

    real_teardown = loadgen._teardown_stage

    async def _teardown(*args: Any, **kwargs: Any) -> Any:
        result = await real_teardown(*args, **kwargs)
        datagen.teardown_done = True  # the in-flight requests landed inside the grace
        return result

    loadgen._teardown_stage = _teardown  # type: ignore[method-assign]
    await _run_stage(loadgen, TraceSessionReplayLoadStage(concurrent_sessions=4, duration=0.2))

    recorded = collector.get_metrics()
    assert len(recorded) == 4, "all four sessions in the pool were recorded"
    assert [m.truncated for m in recorded] == [False] * 4, "none was cut short"
    assert loadgen.stage_runtime_info[0].sessions_not_completed_active == 0
