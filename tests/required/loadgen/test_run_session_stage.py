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
"""Tests for run_session_stage: session pool management, the cross-stage
session cursor, rate limiting, timeout, and dispatch stamping — driven by a
scripted SessionGenerator, no worker processes."""

import multiprocessing as mp
import unittest
from typing import Any, Dict, List, Optional
from unittest.mock import AsyncMock, MagicMock, patch

from inference_perf.apis import LazyLoadInferenceAPIData
from inference_perf.config import APIType, LoadConfig, LoadType, TraceSessionReplayLoadStage
from inference_perf.datagen.base import SessionGenerator
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.metrics import SessionMetricsCollector
from inference_perf.utils.request_queue import RequestQueue


class FakeSessionGenerator(SessionGenerator):
    """Scripted in-memory session corpus.

    A session reports completed on its ``completes_after_checks``-th
    check_session_completed poll after activation, which lets tests drive the
    pool through deterministic refill cycles without running workers.
    """

    def __init__(
        self,
        num_sessions: int,
        events_per_session: int = 2,
        completes_after_checks: int = 1,
        preferred_worker_ids: Optional[List[int]] = None,
        unbuildable_indices: Optional[List[int]] = None,
    ) -> None:
        # Deliberately no super().__init__: the base requires api/data configs
        # that session-pool management never touches.
        self._session_ids = [f"s{i}" for i in range(num_sessions)]
        self.events_per_session = events_per_session
        self.completes_after_checks = completes_after_checks
        self.preferred_worker_ids = preferred_worker_ids
        self.unbuildable = set(unbuildable_indices or [])
        self.activated: List[str] = []
        self.cleaned: List[str] = []
        self.events: Dict[str, List[LazyLoadInferenceAPIData]] = {}
        self._check_counts: Dict[str, int] = {}
        self._live: set[str] = set()
        self.max_live = 0

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion]

    def get_session_count(self) -> int:
        return len(self._session_ids)

    def get_session_info(self, session_index: int) -> Dict[str, Any]:
        return {"session_id": self._session_ids[session_index]}

    def get_session_event_indices(self, session_index: int) -> List[int]:
        return list(range(self.events_per_session))

    def is_session_buildable(self, session_index: int) -> bool:
        return session_index not in self.unbuildable

    def get_session_events(self, session_index: int) -> List[LazyLoadInferenceAPIData]:
        session_id = self._session_ids[session_index]
        events = []
        for event_index in range(self.events_per_session):
            preferred = self.preferred_worker_ids[event_index] if self.preferred_worker_ids else -1
            events.append(
                LazyLoadInferenceAPIData(data_index=session_index * 100 + event_index, preferred_worker_id=preferred)
            )
        self.events[session_id] = events
        return events

    def activate_session(self, session_id: str) -> None:
        self.activated.append(session_id)
        self._live.add(session_id)
        self.max_live = max(self.max_live, len(self._live))

    def check_session_completed(self, session_id: str) -> bool:
        if session_id not in self._live:
            return False
        self._check_counts[session_id] = self._check_counts.get(session_id, 0) + 1
        if self._check_counts[session_id] >= self.completes_after_checks:
            self._live.discard(session_id)
            return True
        return False

    def build_session_metric(self, session_id: str, stage_id: int, start_time: float, end_time: float) -> Any:
        return {"session_id": session_id, "stage_id": stage_id}

    def cleanup_session(self, session_id: str) -> None:
        self.cleaned.append(session_id)

    def get_session_state(self, session_id: str) -> Any:
        return None


def _make_load_generator(
    datagen: FakeSessionGenerator, stages: List[TraceSessionReplayLoadStage], num_workers: int = 2
) -> LoadGenerator:
    load_config = LoadConfig(
        type=LoadType.TRACE_SESSION_REPLAY,
        num_workers=num_workers,
        stages=stages,
        base_seed=42,
    )
    with patch("inference_perf.loadgen.load_generator.get_circuit_breaker"):
        return LoadGenerator(datagen, load_config)


class TestRunSessionStage(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.collector = MagicMock(spec=SessionMetricsCollector)
        self.active_counter = mp.Value("i", 0)
        self.finished_counter = mp.Value("i", 0)
        self.request_phase = mp.Event()
        self.cancel_signal = mp.Event()

    async def _run(
        self,
        load_generator: LoadGenerator,
        stage: TraceSessionReplayLoadStage,
        stage_id: int = 0,
        request_queue: Optional[Any] = None,
    ) -> None:
        if request_queue is None:
            # No worker consumes the queue in these tests, so a real
            # JoinableQueue would hang in the teardown join(): items put()
            # reach the pipe through a feeder thread, and with sleep patched
            # the drain can run before the feeder flushes them.
            request_queue = MagicMock(spec=RequestQueue)
        with patch("inference_perf.loadgen.load_generator.sleep", new_callable=AsyncMock):
            await load_generator.run_session_stage(
                stage_id,
                stage,
                request_queue,
                self.active_counter,
                self.finished_counter,
                self.request_phase,
                self.cancel_signal,
            )

    async def test_pool_refills_up_to_concurrent_sessions(self) -> None:
        datagen = FakeSessionGenerator(num_sessions=4, completes_after_checks=2)
        stage = TraceSessionReplayLoadStage(concurrent_sessions=2)
        load_generator = _make_load_generator(datagen, [stage])
        load_generator.session_metrics_collector = self.collector

        await self._run(load_generator, stage)

        self.assertEqual(datagen.activated, ["s0", "s1", "s2", "s3"], "pool refills in corpus order")
        self.assertLessEqual(datagen.max_live, 2, "active sessions must never exceed concurrent_sessions")
        self.assertEqual(self.collector.record_metric.call_count, 4)
        self.assertEqual(sorted(datagen.cleaned), ["s0", "s1", "s2", "s3"])
        info = load_generator.stage_runtime_info[0]
        self.assertEqual(info.status.name, "COMPLETED")
        self.assertEqual(info.concurrency_level, 2)

    async def test_session_cursor_slices_corpus_across_stages(self) -> None:
        datagen = FakeSessionGenerator(num_sessions=5)
        stages = [TraceSessionReplayLoadStage(concurrent_sessions=0, num_sessions=2) for _ in range(4)]
        load_generator = _make_load_generator(datagen, stages)
        load_generator.session_metrics_collector = self.collector

        await self._run(load_generator, stages[0], stage_id=0)
        self.assertEqual(datagen.activated, ["s0", "s1"])

        await self._run(load_generator, stages[1], stage_id=1)
        self.assertEqual(datagen.activated, ["s0", "s1", "s2", "s3"])

        # Only one session remains; num_sessions=2 must clamp to it.
        await self._run(load_generator, stages[2], stage_id=2)
        self.assertEqual(datagen.activated, ["s0", "s1", "s2", "s3", "s4"])
        self.assertEqual(load_generator._session_cursor, 5)

        # Corpus exhausted: the stage skips without recording runtime info.
        await self._run(load_generator, stages[3], stage_id=3)
        self.assertEqual(len(datagen.activated), 5)
        self.assertNotIn(3, load_generator.stage_runtime_info)

    async def test_stage_timeout_fails_stage_and_respects_rate_limit(self) -> None:
        # A rate limit far below 1/timeout means only the first session is
        # ever dispatched; the stage must exit FAILED at the timeout instead
        # of waiting on the remaining pending sessions.
        datagen = FakeSessionGenerator(num_sessions=3)
        stage = TraceSessionReplayLoadStage(concurrent_sessions=0, session_rate=0.001, timeout=0.3)
        load_generator = _make_load_generator(datagen, [stage])
        load_generator.session_metrics_collector = self.collector

        await self._run(load_generator, stage)

        self.assertEqual(datagen.activated, ["s0"], "rate limit must gate later dispatches")
        self.assertEqual(self.collector.record_metric.call_count, 1)
        info = load_generator.stage_runtime_info[0]
        self.assertEqual(info.status.name, "FAILED")
        self.assertEqual(info.rate, 0.001)
        self.assertEqual(info.timeout, 0.3)

    async def test_dispatch_stamps_events_and_remaps_affinity(self) -> None:
        datagen = FakeSessionGenerator(num_sessions=1, events_per_session=4, preferred_worker_ids=[0, 1, 2, -1])
        stage = TraceSessionReplayLoadStage(concurrent_sessions=0)
        load_generator = _make_load_generator(datagen, [stage], num_workers=3)
        load_generator.session_metrics_collector = self.collector
        request_queue = MagicMock(spec=RequestQueue)

        await self._run(load_generator, stage, stage_id=9, request_queue=request_queue)

        channels = [call.args[1] for call in request_queue.put.call_args_list]
        self.assertEqual(channels, [0, 1, 2, -1], "preferred workers wrap modulo num_workers; unaffined stay broadcast")
        for event in datagen.events["s0"]:
            self.assertEqual(event.session_id, "s0")
            self.assertEqual(event.stage_id, 9)
        request_queue.drain.assert_called_once()
        request_queue.join.assert_called_once()

    async def test_unbuildable_session_counts_complete_but_goes_unreported(self) -> None:
        # Pins the documented behavior: a session whose graph cannot be built
        # is counted complete (so the stage can finish) but produces no
        # session lifecycle metric and dispatches no events.
        datagen = FakeSessionGenerator(num_sessions=2, unbuildable_indices=[0])
        stage = TraceSessionReplayLoadStage(concurrent_sessions=0)
        load_generator = _make_load_generator(datagen, [stage])
        load_generator.session_metrics_collector = self.collector
        request_queue = MagicMock(spec=RequestQueue)

        await self._run(load_generator, stage, request_queue=request_queue)

        self.assertEqual(datagen.activated, ["s1"], "unbuildable session is never activated")
        self.assertEqual(len(request_queue.put.call_args_list), datagen.events_per_session)
        self.assertEqual(self.collector.record_metric.call_count, 1)
        self.assertEqual(self.collector.record_metric.call_args.args[0]["session_id"], "s1")
        self.assertEqual(load_generator.stage_runtime_info[0].status.name, "COMPLETED")


if __name__ == "__main__":
    unittest.main()
