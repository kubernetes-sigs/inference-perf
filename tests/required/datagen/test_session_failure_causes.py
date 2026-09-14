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
"""Unit tests for session failure cause capture and propagation.

The worker records why a session failed; the main process reads those payloads
off the completion queue. These tests cover the path between the two, where the
cause was previously either never set or overwritten with None.
"""

import queue
from unittest.mock import Mock

from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType
from inference_perf.datagen.replay.replay_graph_session_datagen import (
    ReplayGraphSessionGeneratorBase,
    ReplaySessionState,
    SessionFailureCause,
    WorkerSessionTracker,
)
from inference_perf.datagen.replay.replay_graph_types import ReplayGraph


class TestWorkerSessionTrackerFailureCauses:
    def test_cause_and_reason_are_recorded(self) -> None:
        tracker = WorkerSessionTracker()
        tracker.mark_session_failed("s1", SessionFailureCause.PREDECESSOR_FAILED, "predecessor failed")

        assert tracker.is_session_failed("s1")
        assert tracker.get_session_failure("s1") == ("predecessor_failed", "predecessor failed")

    def test_first_cause_wins(self) -> None:
        """A failed event cascades to its successors, so later causes are consequences.

        Keeping the first keeps the reported cause pointing at the thing that
        actually broke rather than at the resulting cascade.
        """
        tracker = WorkerSessionTracker()
        tracker.mark_session_failed("s1", SessionFailureCause.REQUEST_FAILED, "TimeoutError: boom")
        tracker.mark_session_failed("s1", SessionFailureCause.SESSION_ALREADY_FAILED, "session already failed")

        assert tracker.get_session_failure("s1") == ("request_failed", "TimeoutError: boom")

    def test_mark_without_cause_still_flags_failure(self) -> None:
        tracker = WorkerSessionTracker()
        tracker.mark_session_failed("s1")

        assert tracker.is_session_failed("s1")
        assert tracker.get_session_failure("s1") is None

    def test_forget_session_clears_recorded_cause(self) -> None:
        tracker = WorkerSessionTracker()
        tracker.mark_session_failed("s1", SessionFailureCause.PREDECESSOR_FAILED, "predecessor failed")
        tracker.forget_session("s1")

        assert not tracker.is_session_failed("s1")
        assert tracker.get_session_failure("s1") is None


class TestCompletionQueueFailurePropagation:
    """The main process must not degrade a cause it has already been told."""

    def _make_generator(self) -> ReplayGraphSessionGeneratorBase:
        gen = ReplayGraphSessionGeneratorBase(
            api_config=APIConfig(type=APIType.Chat),
            config=DataConfig(type=DataGenType.Random),
            tokenizer=None,
        )
        graph = ReplayGraph(events={"evt_1": Mock()}, root_event_ids=["evt_1"], source_file="t.json")
        gen.session_graph_state["s1"] = ReplaySessionState(
            session_id="s1",
            graph=graph,
            ready_events=set(),
            dispatched_events=set(),
            completed_events=set(),
            event_completion_times={},
        )
        gen.session_completion_queue = queue.Queue()
        return gen

    def test_cause_and_reason_land_on_state(self) -> None:
        gen = self._make_generator()
        gen.session_completion_queue.put(
            {
                "session_id": "s1",
                "completion_time": 1.0,
                "failed": True,
                "failure_cause": SessionFailureCause.PREDECESSOR_FAILED.value,
                "failure_reason": "predecessor failed",
                "event_completion_times": {},
            }
        )
        gen._process_completion_queue()

        state = gen.session_graph_state["s1"]
        assert state.failed is True
        assert state.failure_cause == "predecessor_failed"
        assert state.failure_reason == "predecessor failed"

    def test_later_payload_without_reason_does_not_blank_it(self) -> None:
        """The regression: a second payload for an already-failed session.

        The skip-failure push carries the cause; the all-events-completed push
        that follows it used to assign failure_reason unconditionally, replacing
        a real cause with None and making the failure unattributable.
        """
        gen = self._make_generator()
        q = gen.session_completion_queue
        q.put(
            {
                "session_id": "s1",
                "completion_time": 1.0,
                "failed": True,
                "failure_cause": SessionFailureCause.RECORDED_FALLBACK_MALFORMED.value,
                "failure_reason": "recorded fallback for evt_0 is also malformed",
                "event_completion_times": {},
            }
        )
        q.put({"session_id": "s1", "completion_time": 2.0, "failed": True, "event_completion_times": {}})
        gen._process_completion_queue()

        state = gen.session_graph_state["s1"]
        assert state.failure_cause == "recorded_fallback_malformed"
        assert state.failure_reason == "recorded fallback for evt_0 is also malformed"

    def test_failed_flag_is_sticky(self) -> None:
        """A later payload reporting failed=False must not clear a recorded failure."""
        gen = self._make_generator()
        q = gen.session_completion_queue
        q.put(
            {
                "session_id": "s1",
                "completion_time": 1.0,
                "failed": True,
                "failure_cause": SessionFailureCause.REQUEST_FAILED.value,
                "failure_reason": "TimeoutError: boom",
                "event_completion_times": {},
            }
        )
        q.put({"session_id": "s1", "completion_time": 2.0, "failed": False, "event_completion_times": {}})
        gen._process_completion_queue()

        assert gen.session_graph_state["s1"].failed is True

    def test_clean_completion_records_no_failure(self) -> None:
        gen = self._make_generator()
        gen.session_completion_queue.put(
            {"session_id": "s1", "completion_time": 1.0, "failed": False, "event_completion_times": {"evt_1": 1.0}}
        )
        gen._process_completion_queue()

        state = gen.session_graph_state["s1"]
        assert state.failed is False
        assert state.failure_reason is None
        assert state.failure_cause is None
