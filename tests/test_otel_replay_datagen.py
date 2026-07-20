#!/usr/bin/env python3

# Copyright 2025 The Kubernetes Authors.
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

"""
Tests for OTelTraceReplayDataGenerator output-aware replay.

Tests the OTEL trace replay architecture:
- EventOutputRegistry: intra-worker output coordination via asyncio.Event
- WorkerSessionTracker: per-worker session state tracking
- SessionChatCompletionAPIData: output substitution and completion notification
- Session completion via mp.Queue (event-driven, not polling)

Key architectural principles tested:
1. Session-to-worker affinity: all events of a session run on same worker
2. Zero-thread waiting: asyncio.Event suspension, no OS threads
3. Event-driven completion: queue notifications, not polling
4. Nested dict structure: efficient O(1) lookups without string operations
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any, List, Optional
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic import ValidationError

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference_perf.datagen.otel_trace_replay_datagen import OTelTraceReplayDataGenerator
from inference_perf.datagen.replay_graph_session_datagen import (
    EventFailedError,
    EventOutputRegistry,
    ReplayGraphSessionGeneratorBase,
    ReplaySession,
    ReplaySessionEvent,
    ReplaySessionState,
    SessionChatCompletionAPIData,
    SessionInferenceInfo,
    WorkerSessionTracker,
)
from inference_perf.config.datagen.replay import BadToolCallHandling, OTelTraceReplayConfig
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.datagen.otel_trace_to_replay_graph import (
    build_graph,
    build_raw_calls,
)
from inference_perf.datagen.replay_graph_types import GraphCall, GraphEvent, InputSegment, ReplayGraph
from inference_perf.apis.chat import ChatMessage
from inference_perf.config import APIConfig, APIType, SessionReplayConfig


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

SIMPLE_CHAIN_JSON = Path(__file__).parent.parent / "examples/otel/test_traces/simple/simple_chain.json"


def make_api_config() -> APIConfig:
    return APIConfig(type=APIType.Chat, streaming=False)


def make_mock_tokenizer() -> MagicMock:
    """Return a mock tokenizer that counts words as tokens."""
    tok = MagicMock()
    tok.count_tokens = lambda text: max(1, len((text or "").split()))
    return tok


def make_mock_response(content: str) -> MagicMock:
    """Return a mock aiohttp ClientResponse for non-streaming mode."""
    response = MagicMock()
    response.json = AsyncMock(return_value={"choices": [{"message": {"role": "assistant", "content": content}}]})
    return response


def make_mock_api_config_streaming(streaming: bool = False) -> MagicMock:
    cfg = MagicMock()
    cfg.streaming = streaming
    return cfg


# ---------------------------------------------------------------------------
# EventOutputRegistry tests
# ---------------------------------------------------------------------------


class TestEventOutputRegistry:
    """Test intra-worker output coordination via asyncio.Event."""

    def test_record_and_get(self) -> None:
        """Basic record/retrieve functionality."""
        reg = EventOutputRegistry()
        reg.record("event_001", "Hello world", [])
        assert reg.get_output_by_event_id("event_001") == "Hello world"

    def test_get_missing_returns_none(self) -> None:
        """Missing events return None."""
        reg = EventOutputRegistry()
        assert reg.get_output_by_event_id("nonexistent") is None

    def test_require_async_already_registered(self) -> None:
        """Fast path: output already present before require_async is called."""
        reg = EventOutputRegistry()
        reg.record("event_001", "Some output", [])
        result = asyncio.run(reg.require_async("event_001"))
        assert result == "Some output"

    def test_require_async_waits_for_record(self) -> None:
        """require_async suspends via asyncio.Event until record() is called."""
        reg = EventOutputRegistry()

        async def _run() -> str:
            async def producer() -> None:
                await asyncio.sleep(0.05)
                reg.record("event_001", "delayed output", [])

            asyncio.create_task(producer())
            return str(await reg.require_async("event_001", timeout_sec=2.0))

        result = asyncio.run(_run())
        assert result == "delayed output"

    def test_require_async_timeout(self) -> None:
        """require_async raises TimeoutError when output never arrives."""
        reg = EventOutputRegistry()
        with pytest.raises(TimeoutError):
            asyncio.run(reg.require_async("event_001", timeout_sec=0.1))

    def test_double_record_raises_error(self) -> None:
        """Recording the same event twice should raise ValueError."""
        reg = EventOutputRegistry()
        reg.record("event_001", "first output", [])

        # Attempting to record the same event again should raise ValueError
        with pytest.raises(ValueError, match="Event event_001 has already been recorded"):
            reg.record("event_001", "second output", [])

    def test_get_messages_by_event_id(self) -> None:
        """get_messages_by_event_id returns input messages."""
        reg = EventOutputRegistry()
        messages = [ChatMessage(role="user", content="hello")]
        reg.record("event_001", "output", messages)
        retrieved = reg.get_messages_by_event_id("event_001")
        assert retrieved is not None
        assert len(retrieved) == 1
        assert retrieved[0].role == "user"
        assert retrieved[0].content == "hello"

    def test_record_with_empty_messages(self) -> None:
        """Recording with empty messages list should still store the messages (as empty list)."""
        reg = EventOutputRegistry()
        reg.record("event_001", "output", [])

        # Should return empty list, not None
        retrieved = reg.get_messages_by_event_id("event_001")
        assert retrieved is not None
        assert retrieved == []

    def test_record_failure_fast_path(self) -> None:
        """require_async raises EventFailedError immediately for an already-failed event."""
        reg = EventOutputRegistry()
        reg.record_failure("event_001")
        assert reg.is_event_failed("event_001")

        with pytest.raises(EventFailedError) as exc_info:
            asyncio.run(reg.require_async("event_001"))
        assert exc_info.value.event_id == "event_001"

    def test_record_failure_wakes_waiter_with_error(self) -> None:
        """record_failure wakes a coroutine blocked in require_async with EventFailedError."""
        reg = EventOutputRegistry()

        async def _run() -> None:
            async def fail_producer() -> None:
                await asyncio.sleep(0.05)
                reg.record_failure("event_001")

            asyncio.create_task(fail_producer())
            await reg.require_async("event_001", timeout_sec=2.0)

        with pytest.raises(EventFailedError):
            asyncio.run(_run())

    def test_record_failure_idempotent(self) -> None:
        """Calling record_failure multiple times for the same event is safe."""
        reg = EventOutputRegistry()
        reg.record_failure("event_001")
        reg.record_failure("event_001")  # should not raise
        assert reg.is_event_failed("event_001")


# ---------------------------------------------------------------------------
# WorkerSessionTracker tests
# ---------------------------------------------------------------------------


class TestWorkerSessionTracker:
    """Test per-worker session state tracking with nested dict structure."""

    def test_record_and_check_event_completion(self) -> None:
        """Basic event completion tracking."""
        tracker = WorkerSessionTracker()
        tracker.record_event_completed("session_1", "event_0", 1.0)
        assert tracker.is_event_completed("session_1", "event_0")
        assert not tracker.is_event_completed("session_1", "event_1")

    def test_get_event_completion_time(self) -> None:
        """Retrieve completion time for an event."""
        tracker = WorkerSessionTracker()
        tracker.record_event_completed("session_1", "event_0", 123.456)
        assert tracker.get_event_completion_time("session_1", "event_0") == 123.456
        assert tracker.get_event_completion_time("session_1", "event_1") is None

    def test_mark_and_check_session_failed(self) -> None:
        """Session failure tracking."""
        tracker = WorkerSessionTracker()
        assert not tracker.is_session_failed("session_1")
        tracker.mark_session_failed("session_1")
        assert tracker.is_session_failed("session_1")

    def test_get_session_event_count(self) -> None:
        """Count completed events per session."""
        tracker = WorkerSessionTracker()
        tracker.record_event_completed("session_1", "event_0", 1.0)
        tracker.record_event_completed("session_1", "event_1", 2.0)
        tracker.record_event_completed("session_2", "event_0", 3.0)
        assert tracker.get_session_event_count("session_1") == 2
        assert tracker.get_session_event_count("session_2") == 1
        assert tracker.get_session_event_count("session_3") == 0

    def test_get_session_completion_times(self) -> None:
        """Retrieve all completion times for a session."""
        tracker = WorkerSessionTracker()
        tracker.record_event_completed("session_1", "event_0", 1.0)
        tracker.record_event_completed("session_1", "event_1", 2.0)
        times = tracker.get_session_completion_times("session_1")
        assert times == {"event_0": 1.0, "event_1": 2.0}


# ---------------------------------------------------------------------------
# SessionChatCompletionAPIData tests
# ---------------------------------------------------------------------------


class TestSessionChatCompletionAPIData:
    """Test output capture, substitution, and completion notification."""

    def _make_api_data(
        self,
        event_id: str,
        registry: EventOutputRegistry,
        worker_tracker: WorkerSessionTracker,
        completion_queue: Optional[mp.Queue[Any]] = None,
        total_events: int = 1,
        predecessor_event_ids: Optional[List[str]] = None,
    ) -> SessionChatCompletionAPIData:
        return SessionChatCompletionAPIData(
            messages=[ChatMessage(role="user", content="Hello")],
            max_tokens=50,
            event_id=event_id,
            registry=registry,
            worker_tracker=worker_tracker,
            completion_queue=completion_queue,
            total_events_in_session=total_events,
            predecessor_event_ids=predecessor_event_ids or [],
        )

    @pytest.mark.asyncio
    async def test_non_streaming_captures_output(self) -> None:
        """process_response captures output and registers it."""
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()
        api_data = self._make_api_data("session_1:event_0", registry, tracker)
        response = make_mock_response("Paris is the capital of France.")
        config = make_mock_api_config_streaming(streaming=False)
        tokenizer = make_mock_tokenizer()

        info = await api_data.process_response(response, config, tokenizer)

        # Output should be registered
        assert registry.get_output_by_event_id("session_1:event_0") == "Paris is the capital of France."
        # Returns SessionInferenceInfo
        assert isinstance(info, SessionInferenceInfo)
        assert info.output_text == "Paris is the capital of France."

    @pytest.mark.asyncio
    async def test_on_completion_records_in_tracker(self) -> None:
        """on_completion records event completion in WorkerSessionTracker."""
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()
        api_data = self._make_api_data("session_1:event_0", registry, tracker, total_events=2)

        info = SessionInferenceInfo(
            output_text="test output",
            request_metrics=RequestMetrics(text=Text(input_tokens=10)),
        )

        api_data.on_completion(info)

        # Should be recorded in tracker with unqualified event_id
        assert tracker.is_event_completed("session_1", "event_0")
        assert tracker.get_event_completion_time("session_1", "event_0") is not None

    @pytest.mark.asyncio
    async def test_on_completion_pushes_to_queue_when_session_complete(self) -> None:
        """on_completion pushes to queue when last event completes."""
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()

        # Use a mock queue that captures put_nowait calls
        completion_notifications = []
        mock_queue = MagicMock()
        mock_queue.put_nowait = lambda data: completion_notifications.append(data)

        # Session has 2 events total
        # First complete event_0
        api_data_0 = self._make_api_data("session_1:event_0", registry, tracker, mock_queue, total_events=2)
        info_0 = SessionInferenceInfo(
            output_text="first",
            request_metrics=RequestMetrics(text=Text(input_tokens=5)),
        )
        api_data_0.on_completion(info_0)

        # No notification yet (only 1 of 2 events complete)
        assert len(completion_notifications) == 0

        # Now complete event_1 (the last event)
        api_data_1 = self._make_api_data("session_1:event_1", registry, tracker, mock_queue, total_events=2)
        info_1 = SessionInferenceInfo(
            output_text="done",
            request_metrics=RequestMetrics(text=Text(input_tokens=5)),
        )
        api_data_1.on_completion(info_1)

        # Should now have completion notification
        assert len(completion_notifications) == 1
        completion_data = completion_notifications[0]
        assert completion_data["session_id"] == "session_1"
        assert completion_data["failed"] is False
        assert "event_completion_times" in completion_data
        assert len(completion_data["event_completion_times"]) == 2

    @pytest.mark.asyncio
    async def test_wait_for_predecessors_skips_on_session_failure(self) -> None:
        """wait_for_predecessors_and_substitute skips if session already failed."""
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()
        tracker.mark_session_failed("session_1")

        api_data = self._make_api_data(
            "session_1:event_1",
            registry,
            tracker,
            predecessor_event_ids=["session_1:event_0"],
        )

        await api_data.wait_for_predecessors_and_substitute()

        # Should have set skip_request flag
        assert api_data.skip_request is True
        # Registry should record failure (not empty output)
        assert registry.is_event_failed("session_1:event_1")
        assert registry.get_output_by_event_id("session_1:event_1") is None

    @pytest.mark.asyncio
    async def test_wait_for_predecessors_waits_and_substitutes(self) -> None:
        """wait_for_predecessors_and_substitute waits for predecessors and substitutes output."""
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()

        # Set up predecessor output
        registry.record("session_1:event_0", "Predecessor output", [])

        # Create event with output segment
        messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "RECORDED"},
        ]
        segments = [
            InputSegment(type="unique", message_count=1, token_count=5),
            InputSegment(type="output", message_count=1, token_count=10, source_event_id="session_1:event_0"),
        ]

        api_data = SessionChatCompletionAPIData(
            messages=[ChatMessage(role="user", content="Question"), ChatMessage(role="assistant", content="RECORDED")],
            max_tokens=50,
            event_id="session_1:event_1",
            registry=registry,
            worker_tracker=tracker,
            completion_queue=None,
            total_events_in_session=2,
            predecessor_event_ids=["session_1:event_0"],
            input_segments=segments,
            original_messages=messages,
        )

        await api_data.wait_for_predecessors_and_substitute()

        # Messages should be substituted
        assert len(api_data.messages) == 2
        assert api_data.messages[1].content == "Predecessor output"

    @pytest.mark.asyncio
    async def test_disable_output_substitution_keeps_recorded(self) -> None:
        """With disable_output_substitution=True, predecessors are still awaited
        but the recorded assistant message is sent as-is (no live substitution)."""
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()

        # Predecessor produced a *different* live output than what was recorded.
        registry.record("session_1:event_0", "Predecessor output", [])

        messages = [
            {"role": "user", "content": "Question"},
            {"role": "assistant", "content": "RECORDED"},
        ]
        segments = [
            InputSegment(type="unique", message_count=1, token_count=5),
            InputSegment(type="output", message_count=1, token_count=10, source_event_id="session_1:event_0"),
        ]

        api_data = SessionChatCompletionAPIData(
            messages=[ChatMessage(role="user", content="Question"), ChatMessage(role="assistant", content="RECORDED")],
            max_tokens=50,
            event_id="session_1:event_1",
            registry=registry,
            worker_tracker=tracker,
            completion_queue=None,
            total_events_in_session=2,
            predecessor_event_ids=["session_1:event_0"],
            input_segments=segments,
            original_messages=messages,
            disable_output_substitution=True,
        )

        await api_data.wait_for_predecessors_and_substitute()

        # Predecessor was awaited (no failure / skip) ...
        assert api_data.skip_request is False
        assert tracker.is_session_failed("session_1") is False
        # ... but the recorded assistant content is preserved, NOT substituted
        # with the live "Predecessor output".
        assert api_data.messages[1].content == "RECORDED"


# ---------------------------------------------------------------------------
# OTelTraceReplayDataGenerator tests - Critical queue processing logic
# ---------------------------------------------------------------------------


class TestOTelTraceReplayDataGenerator:
    """Test the main generator class, focusing on queue processing and session lifecycle."""

    def test_process_completion_queue_updates_session_state(self) -> None:
        """_process_completion_queue drains queue and updates session_graph_state."""
        # Create generator with mock queue
        gen = object.__new__(OTelTraceReplayDataGenerator)

        # Set up session state

        mock_graph = MagicMock()
        mock_graph.events = {"event_0": MagicMock(), "event_1": MagicMock()}

        gen.session_graph_state = {
            "session_1": ReplaySessionState(
                session_id="session_1",
                graph=mock_graph,
                ready_events=set(),
                dispatched_events=set(),
                completed_events=set(),
                event_completion_times={},
                is_active=True,
                is_complete=False,
            )
        }

        # Create queue with completion notification
        mock_queue = MagicMock()

        # Simulate queue with one item, then empty
        call_count = [0]

        def get_nowait_side_effect() -> Any:
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "session_id": "session_1",
                    "completion_time": 123.456,
                    "failed": False,
                    "event_completion_times": {"event_0": 100.0, "event_1": 120.0},
                }
            else:
                raise Exception("Queue empty")

        mock_queue.get_nowait = get_nowait_side_effect
        gen.session_completion_queue = mock_queue

        # Process queue
        gen._process_completion_queue()

        # Session should be marked complete
        state = gen.session_graph_state["session_1"]
        assert state.is_complete is True
        assert "event_0" in state.completed_events
        assert "event_1" in state.completed_events
        assert state.event_completion_times["event_0"] == 100.0
        assert state.event_completion_times["event_1"] == 120.0

    def test_check_session_completed_processes_queue_first(self) -> None:
        """check_session_completed processes queue before checking status."""
        gen = object.__new__(OTelTraceReplayDataGenerator)

        # Set up session state

        mock_graph = MagicMock()
        mock_graph.events = {"event_0": MagicMock()}

        gen.session_graph_state = {
            "session_1": ReplaySessionState(
                session_id="session_1",
                graph=mock_graph,
                ready_events=set(),
                dispatched_events=set(),
                completed_events=set(),
                event_completion_times={},
                is_active=True,
                is_complete=False,  # Not yet complete
            )
        }

        # Create queue with completion notification
        call_count = [0]

        def get_nowait_side_effect() -> Any:
            call_count[0] += 1
            if call_count[0] == 1:
                return {
                    "session_id": "session_1",
                    "completion_time": 123.456,
                    "failed": False,
                    "event_completion_times": {"event_0": 100.0},
                }
            else:
                raise Exception("Queue empty")

        mock_queue = MagicMock()
        mock_queue.get_nowait = get_nowait_side_effect
        gen.session_completion_queue = mock_queue

        assert gen.session_graph_state["session_1"].is_complete is False
        # Check completion - should process queue first
        result = gen.check_session_completed("session_1")

        # Should return True because queue notification marked it complete
        assert result is True
        assert gen.session_graph_state["session_1"].is_complete is True

    def test_check_session_completed_returns_false_for_incomplete(self) -> None:
        """check_session_completed returns False for incomplete sessions."""
        gen = object.__new__(OTelTraceReplayDataGenerator)

        # Set up incomplete session state

        mock_graph = MagicMock()
        mock_graph.events = {"event_0": MagicMock(), "event_1": MagicMock()}

        gen.session_graph_state = {
            "session_1": ReplaySessionState(
                session_id="session_1",
                graph=mock_graph,
                ready_events=set(),
                dispatched_events=set(),
                completed_events={"event_0"},  # Only 1 of 2 events complete
                event_completion_times={"event_0": 100.0},
                is_active=True,
                is_complete=False,
            )
        }

        # Empty queue
        mock_queue = MagicMock()
        mock_queue.get_nowait = MagicMock(side_effect=Exception("Queue empty"))
        gen.session_completion_queue = mock_queue

        # Check completion
        result = gen.check_session_completed("session_1")

        # Should return False
        assert result is False

    def test_activate_session_marks_root_events_ready(self) -> None:
        """activate_session marks root events (no predecessors) as ready."""
        gen = object.__new__(OTelTraceReplayDataGenerator)

        # Set up session with graph

        # Create mock graph with root and non-root events
        root_event = MagicMock()
        root_event.predecessor_event_ids = []

        child_event = MagicMock()
        child_event.predecessor_event_ids = ["event_0"]

        mock_graph = MagicMock()
        mock_graph.events = {
            "event_0": root_event,
            "event_1": child_event,
        }

        gen.session_graph_state = {
            "session_1": ReplaySessionState(
                session_id="session_1",
                graph=mock_graph,
                ready_events=set(),
                dispatched_events=set(),
                completed_events=set(),
                event_completion_times={},
                is_active=False,
                is_complete=False,
            )
        }
        state = gen.session_graph_state["session_1"]
        assert state.is_active is False

        # Activate session
        gen.activate_session("session_1")

        # Root event should be ready
        state = gen.session_graph_state["session_1"]
        assert state.is_active is True
        assert "event_0" in state.ready_events
        assert "event_1" not in state.ready_events  # Child event not ready yet

    def test_multiple_sessions_complete_independently(self) -> None:
        """Multiple sessions can complete independently via queue."""
        gen = object.__new__(OTelTraceReplayDataGenerator)

        # Set up two sessions

        mock_graph_1 = MagicMock()
        mock_graph_1.events = {"event_0": MagicMock()}
        mock_graph_2 = MagicMock()
        mock_graph_2.events = {"event_0": MagicMock()}

        gen.session_graph_state = {
            "session_1": ReplaySessionState(
                session_id="session_1",
                graph=mock_graph_1,
                ready_events=set(),
                dispatched_events=set(),
                completed_events=set(),
                event_completion_times={},
                is_active=True,
                is_complete=False,
            ),
            "session_2": ReplaySessionState(
                session_id="session_2",
                graph=mock_graph_2,
                ready_events=set(),
                dispatched_events=set(),
                completed_events=set(),
                event_completion_times={},
                is_active=True,
                is_complete=False,
            ),
        }

        # Queue has completions for both sessions
        completions = [
            {
                "session_id": "session_1",
                "completion_time": 100.0,
                "failed": False,
                "event_completion_times": {"event_0": 100.0},
            },
            {
                "session_id": "session_2",
                "completion_time": 200.0,
                "failed": False,
                "event_completion_times": {"event_0": 200.0},
            },
        ]

        call_count = [0]

        def get_nowait_side_effect() -> Any:
            if call_count[0] < len(completions):
                result = completions[call_count[0]]
                call_count[0] += 1
                return result
            else:
                raise Exception("Queue empty")

        mock_queue = MagicMock()
        mock_queue.get_nowait = get_nowait_side_effect
        gen.session_completion_queue = mock_queue

        # Process queue
        gen._process_completion_queue()

        # Both sessions should be complete
        assert gen.session_graph_state["session_1"].is_complete is True
        assert gen.session_graph_state["session_2"].is_complete is True


# ---------------------------------------------------------------------------
# End-to-end: build graph from simple_chain.json + simulate output substitution
# ---------------------------------------------------------------------------


class TestEndToEndSimpleChain:
    """
    End-to-end test using simple_chain.json to verify:
    - Graph construction from trace
    - Output substitution with actual generated text
    - Session-to-worker affinity assumptions
    """

    @pytest.fixture
    def graph_and_calls(self) -> Any:
        if not SIMPLE_CHAIN_JSON.exists():
            pytest.skip(f"Test trace not found: {SIMPLE_CHAIN_JSON}")
        data = json.loads(SIMPLE_CHAIN_JSON.read_text())
        spans = data.get("spans", [])
        calls, _ = build_raw_calls(spans)
        graph = build_graph(calls)
        return graph, calls

    def test_graph_has_three_events(self, graph_and_calls: Any) -> None:
        """Verify simple_chain.json produces 3-event graph."""
        graph, _ = graph_and_calls
        assert len(graph.events) == 3

    def test_event_001_has_output_segment(self, graph_and_calls: Any) -> None:
        """Second event should have output segment from first event."""
        graph, _ = graph_and_calls
        event_ids = sorted(graph.events.keys())
        assert len(event_ids) >= 2, f"Expected at least 2 events, got {len(event_ids)}"
        event_001 = graph.events[event_ids[1]]
        seg_types = [s.type for s in event_001.call.input_segments]
        assert "output" in seg_types, f"Expected output segment in {event_ids[1]}, got: {seg_types}"

    def test_output_substitution_end_to_end(self, graph_and_calls: Any) -> None:
        """
        Simulate: event_000 completes with DIFFERENT output than recorded.
        Verify: event_001's messages use the new output after substitution.
        """
        graph, calls = graph_and_calls

        # Build events from graph
        session_id = "test_session"
        from dataclasses import replace as dc_replace

        events = []
        for event in graph.events.values():
            gc = event.call
            qualified_event_id = f"{session_id}:{event.event_id}"
            qualified_predecessor_ids = [f"{session_id}:{pid}" for pid in event.predecessor_event_ids]
            qualified_segments = [
                dc_replace(seg, source_event_id=f"{session_id}:{seg.source_event_id}")
                if seg.source_event_id is not None
                else seg
                for seg in gc.input_segments
            ]
            events.append(
                ReplaySessionEvent(
                    call_id=gc.call_id,
                    event_id=qualified_event_id,
                    session_index=0,
                    t_start_ms=event.t_start_ms,
                    t_end_ms=event.t_end_ms,
                    model=gc.model,
                    messages=gc.messages,
                    expected_output=gc.expected_output,
                    input_segments=qualified_segments,
                    expected_output_tokens=gc.expected_output_tokens,
                    temperature=gc.temperature,
                    max_tokens_recorded=gc.max_tokens_recorded,
                    predecessor_event_ids=qualified_predecessor_ids,
                    wait_ms=event.wait_ms,
                )
            )
        events.sort(key=lambda e: e.t_start_ms)

        # Set up registry, tracker, queue
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()
        queue: mp.Queue[Any] = mp.Queue()
        gen = object.__new__(OTelTraceReplayDataGenerator)
        gen.all_events = events
        gen._session_events = {0: events}
        gen.output_registry = registry
        gen.worker_tracker = tracker
        gen.session_completion_queue = queue
        gen.api_config = make_api_config()
        mock_otel_config = MagicMock()
        mock_otel_config.attribute_to_header_map = {}
        mock_otel_config.attribute_to_label_map = {}
        mock_otel_config.bad_tool_call_handling = BadToolCallHandling.NONE
        gen.otel_config = mock_otel_config
        gen.replay_config = mock_otel_config
        gen.session_graph_state = {session_id: MagicMock(graph=graph, random_string=None)}

        from inference_perf.apis import LazyLoadInferenceAPIData

        # Get event IDs
        event_ids = sorted(graph.events.keys())
        first_event_id = event_ids[0]
        second_event_id = event_ids[1]

        qualified_first_event_id = f"{session_id}:{first_event_id}"
        qualified_second_event_id = f"{session_id}:{second_event_id}"

        # Load first event
        event_000 = next(e for e in events if e.event_id == qualified_first_event_id)
        idx_000 = events.index(event_000)
        result_000 = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=idx_000))
        assert isinstance(result_000, SessionChatCompletionAPIData)

        # Simulate first event completing with DIFFERENT output
        actual_output_000 = "ACTUAL REPLAY OUTPUT: France's capital is Paris, city of lights!"
        # Provide the input messages that the first event had (from the trace)
        first_event_messages = result_000.original_messages
        registry.record(qualified_first_event_id, actual_output_000, first_event_messages)

        # Load second event
        event_001 = next(e for e in events if e.event_id == qualified_second_event_id)
        idx_001 = events.index(event_001)

        # Verify it has output segment
        output_segs = [s for s in event_001.input_segments if s.type == "output"]
        assert len(output_segs) >= 1, f"{second_event_id} should have at least one output segment"

        result_001 = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=idx_001))
        assert isinstance(result_001, SessionChatCompletionAPIData)

        # Wait for predecessors and substitute
        asyncio.run(result_001.wait_for_predecessors_and_substitute())

        # The assistant message should be the ACTUAL output
        assistant_messages = [m for m in result_001.messages if m.role == "assistant"]
        assert len(assistant_messages) >= 1
        assert assistant_messages[0].content == actual_output_000, (
            f"Expected substituted output, got: {assistant_messages[0].content!r}"
        )

    # Note: Timeout test removed because it takes 3600s to complete.
    # The timeout behavior is tested in TestEventOutputRegistry.test_require_async_timeout
    # which uses a short 0.1s timeout. The architectural correctness (asyncio.Event-based
    # waiting, session-to-worker affinity, output substitution) is thoroughly tested


# ---------------------------------------------------------------------------
# Error path tests for process_failure
# ---------------------------------------------------------------------------


class TestSessionChatCompletionAPIDataErrorPaths:
    """Test error handling paths in SessionChatCompletionAPIData."""

    @pytest.mark.asyncio
    async def test_process_failure_marks_session_failed(self) -> None:
        """process_failure marks session as failed and registers empty output."""
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()

        api_data = SessionChatCompletionAPIData(
            messages=[ChatMessage(role="user", content="Test")],
            max_tokens=50,
            event_id="session_1:event_0",
            registry=registry,
            worker_tracker=tracker,
            completion_queue=None,
            total_events_in_session=1,
            predecessor_event_ids=[],
        )

        # Simulate a failure
        exception = ValueError("Test error")
        config = make_mock_api_config_streaming(streaming=False)
        tokenizer = make_mock_tokenizer()
        result = await api_data.process_failure(None, config, tokenizer, exception)

        # Session should be marked as failed
        assert tracker.is_session_failed("session_1")

        # Registry should record failure for the event (not empty output)
        assert registry.is_event_failed("session_1:event_0")
        assert registry.get_output_by_event_id("session_1:event_0") is None

        # Should return empty inference info
        assert isinstance(result, SessionInferenceInfo)
        assert result.output_text == ""
        assert result.request_metrics.text.input_tokens == 0
        assert result.response_metrics is not None
        assert result.response_metrics.output_tokens == 0

    @pytest.mark.asyncio
    async def test_wait_for_predecessors_post_wait_failure_check(self) -> None:
        """Test failure propagation: when predecessor fails, successor skips via EventFailedError."""
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()

        # Create event with predecessor
        api_data = SessionChatCompletionAPIData(
            messages=[ChatMessage(role="user", content="Test")],
            max_tokens=50,
            event_id="session_1:event_1",
            registry=registry,
            worker_tracker=tracker,
            completion_queue=None,
            total_events_in_session=2,
            predecessor_event_ids=["session_1:event_0"],
        )

        # Simulate predecessor failure (no real output written, just failure signal)
        registry.record_failure("session_1:event_0")

        # Wait for predecessors - should detect failure via EventFailedError and skip
        await api_data.wait_for_predecessors_and_substitute()

        # Should have set skip_request flag
        assert api_data.skip_request is True

        # Should have propagated failure in registry (not written empty output)
        assert registry.is_event_failed("session_1:event_1")
        assert registry.get_output_by_event_id("session_1:event_1") is None

        # Event should NOT be marked as completed in tracker (skipped events don't count)
        assert not tracker.is_event_completed("session_1", "event_1")

    # by the other tests in this suite.


# ---------------------------------------------------------------------------
# New failure propagation integration tests
# ---------------------------------------------------------------------------


class TestFailurePropagation:
    """Integration tests for failure propagation through an event chain."""

    def _make_node(
        self,
        event_id: str,
        registry: EventOutputRegistry,
        tracker: WorkerSessionTracker,
        predecessor_event_ids: Optional[List[str]] = None,
        total_events: int = 3,
        completion_queue: Any = None,
    ) -> SessionChatCompletionAPIData:
        return SessionChatCompletionAPIData(
            messages=[ChatMessage(role="user", content="Test")],
            max_tokens=50,
            event_id=event_id,
            registry=registry,
            worker_tracker=tracker,
            completion_queue=completion_queue,
            total_events_in_session=total_events,
            predecessor_event_ids=predecessor_event_ids or [],
        )

    @pytest.mark.asyncio
    async def test_failure_propagates_through_chain(self) -> None:
        """event_0 fails → event_1 skips via EventFailedError → event_2 skips via EventFailedError.

        No empty strings are written to the registry. No record_event_completed calls.
        """
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()

        event_1 = self._make_node("session_1:event_1", registry, tracker, ["session_1:event_0"])
        event_2 = self._make_node("session_1:event_2", registry, tracker, ["session_1:event_1"])

        # Simulate event_0 failure
        tracker.mark_session_failed("session_1")
        registry.record_failure("session_1:event_0")

        # event_1 processes — should skip
        await event_1.wait_for_predecessors_and_substitute()
        assert event_1.skip_request is True
        assert registry.is_event_failed("session_1:event_1")
        assert registry.get_output_by_event_id("session_1:event_1") is None

        # event_2 processes — should also skip via event_1's failure
        await event_2.wait_for_predecessors_and_substitute()
        assert event_2.skip_request is True
        assert registry.is_event_failed("session_1:event_2")
        assert registry.get_output_by_event_id("session_1:event_2") is None

        # tracker should have no completed events (no record_event_completed calls)
        assert tracker.get_session_event_count("session_1") == 0

    @pytest.mark.asyncio
    async def test_cancelled_count_in_failure_notification(self) -> None:
        """process_failure on first event of 3-event session reports cancelled_events=2."""
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()

        notifications: List[Any] = []
        mock_queue = MagicMock()
        mock_queue.put_nowait = lambda data: notifications.append(data)

        event_0 = self._make_node("session_1:event_0", registry, tracker, total_events=3, completion_queue=mock_queue)

        exception = ValueError("HTTP 500")
        config = MagicMock()
        config.streaming = False
        tokenizer = MagicMock()
        tokenizer.count_tokens.return_value = 0

        await event_0.process_failure(None, config, tokenizer, exception)

        assert len(notifications) == 1
        data = notifications[0]
        assert data["failed"] is True
        assert data["session_id"] == "session_1"
        # 3 total - 0 completed before failure - 1 (the failing event itself) = 2 cancelled
        assert data["cancelled_events"] == 2


# ---------------------------------------------------------------------------
# Random Session ID Injection tests
# ---------------------------------------------------------------------------


class TestRandomSessionIDInjection:
    """Test random session ID injection for KV-cache invalidation."""

    def test_random_string_generation_is_unique_per_session(self) -> None:
        """Each session gets a unique random string, but events in same session share it."""
        import uuid

        # Generate two different random strings (simulating two different sessions)
        random_str1 = uuid.uuid4().hex[:16]
        random_str2 = uuid.uuid4().hex[:16]

        # Verify they are different and correct length
        assert random_str1 != random_str2
        assert len(random_str1) == 16
        assert len(random_str2) == 16

    def test_events_in_same_session_share_random_string(self) -> None:
        """Events in the same session should use the same random string."""
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()

        # Same session random string for both events
        session_random_str = "a1b2c3d4e5f6g7h8"

        api_data1 = SessionChatCompletionAPIData(
            messages=[ChatMessage(role="user", content="Hello")],
            max_tokens=50,
            event_id="session_1:event_0",
            registry=registry,
            worker_tracker=tracker,
            completion_queue=None,
            total_events_in_session=2,
            inject_random_session_id=True,
            session_random_string=session_random_str,
        )

        api_data2 = SessionChatCompletionAPIData(
            messages=[ChatMessage(role="user", content="World")],
            max_tokens=50,
            event_id="session_1:event_1",
            registry=registry,
            worker_tracker=tracker,
            completion_queue=None,
            total_events_in_session=2,
            inject_random_session_id=True,
            session_random_string=session_random_str,
        )

        # Both events in the same session should have the same random string
        assert api_data1.session_random_string == api_data2.session_random_string
        assert api_data1.session_random_string == session_random_str

    @pytest.mark.asyncio
    async def test_unique_segments_get_random_string_injected(self) -> None:
        """Unique segments have random session string injected when enabled."""
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()

        # Create messages with unique segment
        messages = [
            {"role": "user", "content": "What is the capital of France?"},
        ]

        segments = [
            InputSegment(type="unique", message_count=1, token_count=10, source_event_id=None),
        ]

        api_data = SessionChatCompletionAPIData(
            messages=[ChatMessage(role="user", content="What is the capital of France?")],
            max_tokens=50,
            event_id="session_1:event_0",
            registry=registry,
            worker_tracker=tracker,
            completion_queue=None,
            total_events_in_session=1,
            inject_random_session_id=True,
            session_random_string="test123456789abc",  # Provide session random string
            input_segments=segments,
            original_messages=messages,
        )

        # Trigger substitution
        await api_data.wait_for_predecessors_and_substitute()

        # Message should have random string injected
        assert len(api_data.messages) == 1
        assert api_data.messages[0].content is not None
        assert "[SESS:test123456789abc]" in api_data.messages[0].content
        assert "What is the capital of France?" in api_data.messages[0].content

    @pytest.mark.asyncio
    async def test_output_segments_do_not_get_random_string(self) -> None:
        """Output segments should NOT have random string injected."""
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()

        # Register predecessor output
        registry.record("session_1:event_0", "Paris is the capital.", [])

        # Create messages with output segment
        messages = [
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "Original output"},
        ]

        segments = [
            InputSegment(type="unique", message_count=1, token_count=10, source_event_id=None),
            InputSegment(type="output", message_count=1, token_count=5, source_event_id="session_1:event_0"),
        ]

        api_data = SessionChatCompletionAPIData(
            messages=[
                ChatMessage(role="user", content="What is the capital of France?"),
                ChatMessage(role="assistant", content="Original output"),
            ],
            max_tokens=50,
            event_id="session_1:event_1",
            registry=registry,
            worker_tracker=tracker,
            completion_queue=None,
            total_events_in_session=2,
            predecessor_event_ids=["session_1:event_0"],
            inject_random_session_id=True,
            session_random_string="test123456789abc",  # Provide session random string
            input_segments=segments,
            original_messages=messages,
        )

        # Trigger substitution
        await api_data.wait_for_predecessors_and_substitute()

        # First message (unique) should have random string
        assert api_data.messages[0].content is not None
        assert "[SESS:test123456789abc]" in api_data.messages[0].content
        # Second message (output) should be substituted but NOT have random string
        assert api_data.messages[1].content == "Paris is the capital."
        assert "[SESS:" not in api_data.messages[1].content

    @pytest.mark.asyncio
    async def test_injection_disabled_by_default(self) -> None:
        """Random string injection is disabled when inject_random_session_id=False."""
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()

        messages = [
            {"role": "user", "content": "What is the capital of France?"},
        ]

        segments = [
            InputSegment(type="unique", message_count=1, token_count=10, source_event_id=None),
        ]

        api_data = SessionChatCompletionAPIData(
            messages=[ChatMessage(role="user", content="What is the capital of France?")],
            max_tokens=50,
            event_id="session_1:event_0",
            registry=registry,
            worker_tracker=tracker,
            completion_queue=None,
            total_events_in_session=1,
            inject_random_session_id=False,  # Disabled
            input_segments=segments,
            original_messages=messages,
        )

        # Trigger substitution
        await api_data.wait_for_predecessors_and_substitute()

        # Message should NOT have random string injected
        assert api_data.messages[0].content == "What is the capital of France?"
        assert "[ID:" not in api_data.messages[0].content

    @pytest.mark.asyncio
    async def test_no_injection_for_original_session_when_flag_disabled(self) -> None:
        """Case 1: inject_random_session_id=False, original session (not duplicate) -> NO injection."""
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()

        messages = [
            {"role": "user", "content": "What is the capital of France?"},
        ]

        segments = [
            InputSegment(type="unique", message_count=1, token_count=10, source_event_id=None),
        ]

        api_data = SessionChatCompletionAPIData(
            messages=[ChatMessage(role="user", content="What is the capital of France?")],
            max_tokens=50,
            event_id="session_001:event_0",  # Original session (no _dup)
            registry=registry,
            worker_tracker=tracker,
            completion_queue=None,
            total_events_in_session=1,
            inject_random_session_id=False,  # Flag disabled
            session_random_string="test123456789abc",  # Random string available but not used
            input_segments=segments,
            original_messages=messages,
        )

        # Trigger substitution
        await api_data.wait_for_predecessors_and_substitute()

        # Message should NOT have random string injected (flag off, not duplicate)
        assert api_data.messages[0].content == "What is the capital of France?"
        assert "[SESS:" not in api_data.messages[0].content

    @pytest.mark.asyncio
    async def test_injection_for_duplicate_session_when_flag_disabled(self) -> None:
        """Case 2: inject_random_session_id=False, duplicate session -> YES injection (automatic)."""
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()

        messages = [
            {"role": "user", "content": "What is the capital of France?"},
        ]

        segments = [
            InputSegment(type="unique", message_count=1, token_count=10, source_event_id=None),
        ]

        api_data = SessionChatCompletionAPIData(
            messages=[ChatMessage(role="user", content="What is the capital of France?")],
            max_tokens=50,
            event_id="session_001_dup1:event_0",  # Duplicate session (has _dup1)
            registry=registry,
            worker_tracker=tracker,
            completion_queue=None,
            total_events_in_session=1,
            inject_random_session_id=False,  # Flag disabled
            session_random_string="test123456789abc",  # Random string available
            input_segments=segments,
            original_messages=messages,
        )

        # Trigger substitution
        await api_data.wait_for_predecessors_and_substitute()

        # Message SHOULD have random string injected (duplicate session, automatic)
        assert api_data.messages[0].content is not None
        assert "[SESS:test123456789abc]" in api_data.messages[0].content
        assert "What is the capital of France?" in api_data.messages[0].content

    @pytest.mark.asyncio
    async def test_injection_for_original_session_when_flag_enabled(self) -> None:
        """Case 3: inject_random_session_id=True, original session -> YES injection (flag enabled)."""
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()

        messages = [
            {"role": "user", "content": "What is the capital of France?"},
        ]

        segments = [
            InputSegment(type="unique", message_count=1, token_count=10, source_event_id=None),
        ]

        api_data = SessionChatCompletionAPIData(
            messages=[ChatMessage(role="user", content="What is the capital of France?")],
            max_tokens=50,
            event_id="session_001:event_0",  # Original session (no _dup)
            registry=registry,
            worker_tracker=tracker,
            completion_queue=None,
            total_events_in_session=1,
            inject_random_session_id=True,  # Flag enabled
            session_random_string="test123456789abc",  # Random string available
            input_segments=segments,
            original_messages=messages,
        )

        # Trigger substitution
        await api_data.wait_for_predecessors_and_substitute()

        # Message SHOULD have random string injected (flag enabled)
        assert api_data.messages[0].content is not None
        assert "[SESS:test123456789abc]" in api_data.messages[0].content
        assert "What is the capital of France?" in api_data.messages[0].content

    @pytest.mark.asyncio
    async def test_injection_for_duplicate_session_when_flag_enabled(self) -> None:
        """Case 4: inject_random_session_id=True, duplicate session -> YES injection (both conditions)."""
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()

        messages = [
            {"role": "user", "content": "What is the capital of France?"},
        ]

        segments = [
            InputSegment(type="unique", message_count=1, token_count=10, source_event_id=None),
        ]

        api_data = SessionChatCompletionAPIData(
            messages=[ChatMessage(role="user", content="What is the capital of France?")],
            max_tokens=50,
            event_id="session_001_dup1:event_0",  # Duplicate session (has _dup1)
            registry=registry,
            worker_tracker=tracker,
            completion_queue=None,
            total_events_in_session=1,
            inject_random_session_id=True,  # Flag enabled
            session_random_string="test123456789abc",  # Random string available
            input_segments=segments,
            original_messages=messages,
        )

        # Trigger substitution
        await api_data.wait_for_predecessors_and_substitute()

        # Message SHOULD have random string injected (both flag and duplicate)
        assert api_data.messages[0].content is not None
        assert "[SESS:test123456789abc]" in api_data.messages[0].content
        assert "What is the capital of France?" in api_data.messages[0].content

    def test_is_duplicate_session_detection(self) -> None:
        """Test the _is_duplicate_session() method with various session ID patterns."""

        # Test cases that SHOULD be detected as duplicates
        assert ReplayGraphSessionGeneratorBase.is_duplicate_session("session_001_dup1") is True
        assert ReplayGraphSessionGeneratorBase.is_duplicate_session("my_session_dup123") is True
        assert ReplayGraphSessionGeneratorBase.is_duplicate_session("trace_abc_dup5") is True
        assert ReplayGraphSessionGeneratorBase.is_duplicate_session("a_dup999") is True

        # Test cases that should NOT be detected as duplicates
        assert ReplayGraphSessionGeneratorBase.is_duplicate_session("session_001") is False
        assert ReplayGraphSessionGeneratorBase.is_duplicate_session("my_dup_session") is False  # _dup not at end
        assert ReplayGraphSessionGeneratorBase.is_duplicate_session("session_dup") is False  # No number after _dup
        assert (
            ReplayGraphSessionGeneratorBase.is_duplicate_session("duplicate_session") is False
        )  # Contains "dup" but wrong pattern
        assert ReplayGraphSessionGeneratorBase.is_duplicate_session("session_001_duplicate") is False  # Wrong suffix


# ---------------------------------------------------------------------------
# _build_replay_schedule: random session ID assignment for duplicates
# ---------------------------------------------------------------------------


def _make_simple_graph(event_count: int = 1) -> ReplayGraph:
    """Build a minimal ReplayGraph with `event_count` independent root events."""
    events: dict[str, GraphEvent] = {}
    for i in range(event_count):
        eid = f"event_{i}"
        events[eid] = GraphEvent(
            event_id=eid,
            call=GraphCall(
                call_id=f"call_{i}",
                model="test-model",
                messages=[{"role": "user", "content": f"msg {i}"}],
                expected_output="ok",
                input_segments=[InputSegment(type="unique", message_count=1, token_count=5)],
                total_input_tokens=5,
                expected_output_tokens=2,
                temperature=None,
                max_tokens_recorded=None,
            ),
            predecessor_event_ids=[],
            predecessor_dependency_types={},
            wait_ms=0,
            t_start_ms=i * 1000,
            t_end_ms=i * 1000 + 500,
        )
    return ReplayGraph(
        events=events,
        root_event_ids=[f"event_{i}" for i in range(event_count)],
        source_file="test",
    )


def _make_generator(
    replay_config: Optional[SessionReplayConfig] = None,
) -> ReplayGraphSessionGeneratorBase:
    """Create a ReplayGraphSessionGeneratorBase bypassing __init__."""
    gen = object.__new__(ReplayGraphSessionGeneratorBase)
    gen.replay_config = replay_config
    gen.output_registry = EventOutputRegistry()
    gen.worker_tracker = WorkerSessionTracker()
    gen.session_completion_queue = None
    gen.num_workers = 1
    gen.sessions = []
    gen._session_ids = []
    gen.session_graph_state = {}
    gen._session_events = {}
    gen.all_events = []
    gen._skipped_session_count = 0
    return gen


class TestBuildReplayScheduleRandomSessionID:
    """Verify _build_replay_schedule assigns random_string correctly for duplicate sessions."""

    def test_duplicate_sessions_get_random_string_without_flag(self) -> None:
        """Duplicate sessions (suffix _dup{N}) get a random_string even when inject_random_session_id=False."""
        graph = _make_simple_graph()
        gen = _make_generator(replay_config=SessionReplayConfig(inject_random_session_id=False))
        gen.sessions = [
            ReplaySession(session_id="sess_001_dup1", source_id="src", session_index=0, graph=graph),
            ReplaySession(session_id="sess_001_dup2", source_id="src", session_index=1, graph=graph),
        ]

        gen._build_replay_schedule()

        for sid in ("sess_001_dup1", "sess_001_dup2"):
            state = gen.session_graph_state[sid]
            assert state.random_string is not None, f"Duplicate session {sid} should have a random_string"
            assert len(state.random_string) == 16

    def test_original_session_no_random_string_without_flag(self) -> None:
        """Original (non-duplicate) sessions have no random_string when inject_random_session_id=False."""
        graph = _make_simple_graph()
        gen = _make_generator(replay_config=SessionReplayConfig(inject_random_session_id=False))
        gen.sessions = [
            ReplaySession(session_id="sess_001", source_id="src", session_index=0, graph=graph),
        ]

        gen._build_replay_schedule()

        state = gen.session_graph_state["sess_001"]
        assert state.random_string is None

    def test_original_session_gets_random_string_with_flag(self) -> None:
        """Original sessions get a random_string when inject_random_session_id=True."""
        graph = _make_simple_graph()
        gen = _make_generator(replay_config=SessionReplayConfig(inject_random_session_id=True))
        gen.sessions = [
            ReplaySession(session_id="sess_001", source_id="src", session_index=0, graph=graph),
        ]

        gen._build_replay_schedule()

        state = gen.session_graph_state["sess_001"]
        assert state.random_string is not None
        assert len(state.random_string) == 16

    def test_each_session_gets_unique_random_string(self) -> None:
        """Every session that receives a random_string gets a distinct value."""
        graph = _make_simple_graph()
        gen = _make_generator(replay_config=SessionReplayConfig(inject_random_session_id=True))
        gen.sessions = [
            ReplaySession(session_id=f"sess_{i}", source_id="src", session_index=i, graph=graph) for i in range(10)
        ]

        gen._build_replay_schedule()

        strings = [gen.session_graph_state[f"sess_{i}"].random_string for i in range(10)]
        assert all(s is not None for s in strings)
        assert len(set(strings)) == 10, "All random strings should be unique"

    def test_no_random_string_without_replay_config(self) -> None:
        """No random_string is assigned when replay_config is None, even for duplicates."""
        graph = _make_simple_graph()
        gen = _make_generator(replay_config=None)
        gen.sessions = [
            ReplaySession(session_id="sess_001_dup1", source_id="src", session_index=0, graph=graph),
        ]

        gen._build_replay_schedule()

        state = gen.session_graph_state["sess_001_dup1"]
        # replay_config is None, so the condition `(self.replay_config and ...) or is_duplicate`
        # still triggers for duplicates because is_duplicate is True regardless of replay_config
        assert state.random_string is not None

    def test_duplicate_via_initialize_sessions_gets_random_string(self) -> None:
        """Sessions duplicated by initialize_sessions (duplicate_sessions_target) get random_string."""
        graph = _make_simple_graph()
        gen = _make_generator(
            replay_config=SessionReplayConfig(
                inject_random_session_id=False,
                duplicate_sessions_target=3,
            )
        )

        original_sessions = [
            ReplaySession(session_id="sess_001", source_id="src", session_index=0, graph=graph),
        ]

        gen.initialize_sessions(original_sessions)

        # Should have 3 sessions: 1 original + 2 duplicates
        assert len(gen.sessions) == 3

        # Original session: no random_string (flag is off, not a duplicate)
        assert gen.session_graph_state["sess_001"].random_string is None

        # Duplicates: should have random_string
        dup_states = {sid: state for sid, state in gen.session_graph_state.items() if sid != "sess_001"}
        assert len(dup_states) == 2
        for sid, state in dup_states.items():
            assert state.random_string is not None, f"Duplicate {sid} should have random_string"
            assert len(state.random_string) == 16

    def test_events_are_created_for_all_sessions_including_duplicates(self) -> None:
        """_build_replay_schedule creates events for both original and duplicate sessions."""
        graph = _make_simple_graph(event_count=2)
        gen = _make_generator(
            replay_config=SessionReplayConfig(
                inject_random_session_id=False,
                duplicate_sessions_target=3,
            )
        )

        original_sessions = [
            ReplaySession(session_id="sess_001", source_id="src", session_index=0, graph=graph),
        ]

        gen.initialize_sessions(original_sessions)

        # 3 sessions x 2 events each = 6 total events
        assert len(gen.all_events) == 6


# ---------------------------------------------------------------------------
# bad_tool_call_handling tests
# ---------------------------------------------------------------------------

# Registry of known server-side tool-call parser bugs that produce malformed
# `function.arguments` strings. Add new entries here when a new bug pattern
# is discovered — the parameterized tests below will automatically cover them.
#
# Each entry: (id, malformed_args_string, human_readable_description)
MALFORMED_ARGS_CASES = [
    (
        "xml_parser_leak",
        "</parameter></function>",
        "vLLM tool parser can leak closing XML markers into the arguments string, thus making it an invalid json",
    ),
    (
        "truncated_mid_string",
        '{"command": "',
        "Truncation race in non-streaming concurrent path — arguments string ends mid-JSON",
    ),
]

VALID_ARGS = '{"location": "Paris"}'


def _make_tool_call(args: str, call_id: str = "call_0", name: str = "get_weather") -> dict[str, Any]:
    return {"id": call_id, "type": "function", "function": {"name": name, "arguments": args}}


def _make_assistant_message_with_tool_call(args: str, call_id: str = "call_0") -> dict[str, Any]:
    return {"role": "assistant", "content": None, "tool_calls": [_make_tool_call(args, call_id)]}


def _make_api_data_with_tool_call_output(
    registry: EventOutputRegistry,
    tracker: WorkerSessionTracker,
    recorded_args: str,
    handling: BadToolCallHandling = BadToolCallHandling.USE_RECORDED,
) -> SessionChatCompletionAPIData:
    """Two-event chain: event_0 produces a tool_call; event_1 reads it via an output segment."""
    recorded_assistant = _make_assistant_message_with_tool_call(recorded_args)
    original_messages = [
        {"role": "user", "content": "What is the weather?"},
        recorded_assistant,  # slot that will be substituted
    ]
    segments = [
        InputSegment(type="unique", message_count=1, token_count=5),
        InputSegment(type="output", message_count=1, token_count=10, source_event_id="session_0:event_0"),
    ]
    return SessionChatCompletionAPIData(
        messages=[
            ChatMessage(role="user", content="What is the weather?"),
            ChatMessage(role="assistant", content=None, tool_calls=[_make_tool_call(recorded_args)]),
        ],
        max_tokens=50,
        event_id="session_0:event_1",
        registry=registry,
        worker_tracker=tracker,
        completion_queue=None,
        total_events_in_session=2,
        predecessor_event_ids=["session_0:event_0"],
        input_segments=segments,
        original_messages=original_messages,
        override_tool_call_max_tokens=False,
        bad_tool_call_handling=handling,
    )


class TestBadToolCallHandling:
    """End-to-end tests for bad_tool_call_handling via wait_for_predecessors_and_substitute.

    Parameterized over MALFORMED_ARGS_CASES so every known parser-bug pattern is
    covered automatically. To add a new pattern, append an entry to that registry.
    """

    def _register_live_message(self, registry: EventOutputRegistry, event_id: str, args: str) -> None:
        """Seed the registry with a live assistant tool_call response."""
        live_message = _make_assistant_message_with_tool_call(args, call_id="live_call_0")
        registry.record(event_id, "", [], output_message=live_message)

    @pytest.mark.asyncio
    async def test_clean_live_tool_call_passes_through(self) -> None:
        """With use_recorded, a valid live tool_call is used as-is (no substitution)."""
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()
        self._register_live_message(registry, "session_0:event_0", VALID_ARGS)

        api_data = _make_api_data_with_tool_call_output(registry, tracker, VALID_ARGS)
        await api_data.wait_for_predecessors_and_substitute()

        # event_1 must not be skipped — valid live args should proceed normally
        assert not api_data.skip_request
        # event_1 must not be marked failed in the registry
        assert not registry.is_event_failed("session_0:event_1")
        # The live message (valid args) was used directly, not replaced
        assert api_data.messages[1].tool_calls is not None
        assert api_data.messages[1].tool_calls[0]["function"]["arguments"] == VALID_ARGS
        # No substitution fired, so telemetry must be empty
        assert tracker.get_session_recorded_substitution_event_ids("session_0") == []
        # No failure reason set
        assert api_data._substitution_failure_reason is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("case_id,malformed_args,desc", MALFORMED_ARGS_CASES)
    async def test_malformed_live_tool_call_substitutes_recorded(self, case_id: str, malformed_args: str, desc: str) -> None:
        """Malformed live tool_call arguments are replaced with the recorded assistant message."""
        print(f"\n[{case_id}] {desc}")
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()
        # Live response has malformed args; recorded slot (in original_messages) has valid args.
        self._register_live_message(registry, "session_0:event_0", malformed_args)

        api_data = _make_api_data_with_tool_call_output(registry, tracker, VALID_ARGS)
        await api_data.wait_for_predecessors_and_substitute()

        # event_1 must not be skipped — recorded fallback was clean so replay can continue
        assert not api_data.skip_request, f"[{case_id}] should not skip when recorded fallback is clean"
        # event_1 must not be marked failed in the registry
        assert not registry.is_event_failed("session_0:event_1")
        # The recorded message (valid args) was substituted in place of the malformed live one
        assert api_data.messages[1].tool_calls is not None
        assert api_data.messages[1].tool_calls[0]["function"]["arguments"] == VALID_ARGS
        # Telemetry must record event_0 as the predecessor whose live response was replaced
        subs = tracker.get_session_recorded_substitution_event_ids("session_0")
        assert "event_0" in subs, f"[{case_id}] substitution telemetry must name the predecessor"
        # No failure reason set — substitution succeeded cleanly
        assert api_data._substitution_failure_reason is None

    @pytest.mark.asyncio
    @pytest.mark.parametrize("case_id,malformed_args,desc", MALFORMED_ARGS_CASES)
    async def test_malformed_live_and_recorded_hard_fails(self, case_id: str, malformed_args: str, desc: str) -> None:
        """When both the live response and the recorded fallback are malformed, the event hard-fails exactly once."""
        print(f"\n[{case_id}] {desc}")
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()
        # Both live and recorded have the same malformed args — no clean fallback.
        self._register_live_message(registry, "session_0:event_0", malformed_args)

        api_data = _make_api_data_with_tool_call_output(registry, tracker, malformed_args)
        await api_data.wait_for_predecessors_and_substitute()

        # event_1 must be skipped — no clean message to send to the model
        assert api_data.skip_request, f"[{case_id}] must skip when both live and recorded are malformed"
        # The registry must mark event_1 as failed so downstream events in the DAG cascade-fail
        assert registry.is_event_failed("session_0:event_1")
        # No telemetry recorded — the substitution never completed successfully
        assert tracker.get_session_recorded_substitution_event_ids("session_0") == []
        # The failure reason must name the recorded fallback specifically, not the generic plain-text message,
        # so the log is actionable (tells the operator the trace itself was captured from a buggy parser)
        assert api_data._substitution_failure_reason is not None
        assert "recorded fallback" in api_data._substitution_failure_reason

    @pytest.mark.asyncio
    @pytest.mark.parametrize("case_id,malformed_args,desc", MALFORMED_ARGS_CASES)
    async def test_handling_none_does_not_substitute(self, case_id: str, malformed_args: str, desc: str) -> None:
        """With bad_tool_call_handling=none, malformed args pass through unchanged (upstream behavior)."""
        print(f"\n[{case_id}] {desc}")
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()
        self._register_live_message(registry, "session_0:event_0", malformed_args)

        api_data = _make_api_data_with_tool_call_output(registry, tracker, VALID_ARGS, handling=BadToolCallHandling.NONE)
        await api_data.wait_for_predecessors_and_substitute()

        # event_1 must not be skipped — handling=none never intervenes regardless of arg validity
        assert not api_data.skip_request, f"[{case_id}] handling=none must never skip"
        # event_1 must not be marked failed in the registry
        assert not registry.is_event_failed("session_0:event_1")
        # The live malformed args passed through byte-for-byte; no recorded fallback was consulted
        assert api_data.messages[1].tool_calls is not None
        assert api_data.messages[1].tool_calls[0]["function"]["arguments"] == malformed_args
        # No substitution fired, so telemetry must be empty
        assert tracker.get_session_recorded_substitution_event_ids("session_0") == []
        # No failure reason set
        assert api_data._substitution_failure_reason is None


class TestWorkerSessionEviction:
    """Verify a worker frees a session's built graph once all its events drain.

    In the lazy path each worker builds its own graphs and the parent's cleanup_session
    never runs in the worker, so without per-worker eviction a worker retains every graph
    it ever built for the whole stage. evict_worker_session (triggered when the last event
    of a session drains) must free the graph, event list, traversal state, registry entries,
    and worker-tracker bookkeeping. Every event drains exactly once via completion, skip, or
    request failure — eviction fires only after the last of them.
    """

    def _session_is_resident(self, gen: ReplayGraphSessionGeneratorBase, session_id: str, idx: int) -> bool:
        return (
            session_id in gen.session_graph_state
            or idx in gen._session_events
            or (idx < len(gen.sessions) and gen.sessions[idx] is not None)
        )

    def test_evicted_after_all_events_complete(self) -> None:
        """A 2-event session is freed only after both events complete, not after the first."""
        graph = _make_simple_graph(event_count=2)
        gen = _make_generator()
        gen.initialize_sessions([ReplaySession(session_id="s", source_id="src", session_index=0, graph=graph)])
        assert self._session_is_resident(gen, "s", 0)

        events = gen.get_session_events(0)
        assert len(events) == 2
        datas = [gen.load_lazy_data(e) for e in events]
        for d in datas:
            assert d.generator is gen  # back-reference wired

        info = SessionInferenceInfo(output_text="x", request_metrics=RequestMetrics(text=Text(input_tokens=1)))
        # First event completes — session must still be resident.
        datas[0].on_completion(info)
        assert self._session_is_resident(gen, "s", 0), "session evicted too early (before last event)"

        # Last event completes — now fully drained, must be evicted.
        datas[1].on_completion(info)
        assert not self._session_is_resident(gen, "s", 0), "session not evicted after all events completed"
        assert "s" not in gen.worker_tracker._drained_events

    def test_evicted_after_failure_and_skips_drain(self) -> None:
        """Root failure + successor skips still drain every event, so the session is freed."""
        graph = _make_simple_graph(event_count=3)
        gen = _make_generator()
        gen.initialize_sessions([ReplaySession(session_id="s", source_id="src", session_index=0, graph=graph)])

        events = gen.get_session_events(0)
        datas = [gen.load_lazy_data(e) for e in events]

        # Root event fails (request failure path).
        asyncio.run(datas[0].process_failure(None, make_mock_api_config_streaming(), make_mock_tokenizer(), Exception("boom")))
        assert self._session_is_resident(gen, "s", 0), "evicted before successors drained"

        # The two successors are dequeued and skip via the session-already-failed path.
        for d in datas[1:]:
            asyncio.run(d.wait_for_predecessors_and_substitute())
            assert d.skip_request is True

        # All three events have now drained — session must be evicted.
        assert not self._session_is_resident(gen, "s", 0), "session not evicted after failure + skips"


class TestUnbuildableSessionSlot:
    """Regression tests for the three bugs around un-buildable lazy session slots.

    Bug 1 (dispatch crash): dispatch_session calls get_session_info after is_session_buildable
    returns False; get_session_info → _get_session raises RuntimeError for a None slot.

    Bug 2 (eviction never fires): total_events_in_session is taken from len(state.graph.events)
    but _build_session_schedule skips empty-message events, so the drain counter never reaches
    the inflated total and evict_worker_session never fires.

    Bug 3 (repeated _build_session): _ensure_session_built short-circuits on
    `sessions[idx] is not None` only; un-buildable slots stay None so _build_session is
    re-invoked on every subsequent call.
    """

    def test_ensure_session_built_does_not_repeat_for_unbuildable_slot(self) -> None:
        """Bug 3: _build_session must be called at most once per slot even if it returns None."""
        gen = _make_generator()
        gen.initialize_sessions_lazy(["bad_session"])

        build_calls = 0

        def failing_build(idx: int) -> None:
            nonlocal build_calls
            build_calls += 1
            return None

        gen._build_session = failing_build  # type: ignore[assignment]

        # First call — attempts build, finds nothing.
        gen._ensure_session_built(0)
        assert build_calls == 1

        # Subsequent calls — slot already attempted, must NOT re-invoke _build_session.
        gen._ensure_session_built(0)
        gen._ensure_session_built(0)
        assert build_calls == 1, f"_build_session re-invoked {build_calls} times for an un-buildable slot"

    def test_is_session_buildable_returns_false_without_crashing(self) -> None:
        """Bug 3 + Bug 1 precondition: is_session_buildable must return False cleanly."""
        gen = _make_generator()
        gen.initialize_sessions_lazy(["bad_session"])
        gen._build_session = lambda idx: None  # type: ignore[assignment]

        assert not gen.is_session_buildable(0)
        # _session_ids must still be intact after a failed build (used by dispatch to
        # add the session to completed_session_ids without touching the None slot).
        assert gen._session_ids[0] == "bad_session"

    def test_eviction_fires_despite_empty_message_event(self) -> None:
        """Bug 2: evict_worker_session must fire even when one graph event has no messages.

        _build_session_schedule skips empty-message events (they are never dispatched),
        so total_events_in_session must be derived from the scheduled count — not from
        len(state.graph.events) — otherwise the drain counter never reaches the total
        and the session is never evicted.
        """
        # Build a graph where event_0 has no messages (will be skipped by schedule builder)
        # and event_1 has messages (will be dispatched and drained).
        events: dict[str, GraphEvent] = {
            "event_0": GraphEvent(
                event_id="event_0",
                call=GraphCall(
                    call_id="call_0",
                    model="test-model",
                    messages=[],  # empty — skipped by _build_session_schedule
                    expected_output="",
                    input_segments=[],
                    total_input_tokens=0,
                    expected_output_tokens=0,
                    temperature=None,
                    max_tokens_recorded=None,
                ),
                predecessor_event_ids=[],
                predecessor_dependency_types={},
                wait_ms=0,
                t_start_ms=0,
                t_end_ms=500,
            ),
            "event_1": GraphEvent(
                event_id="event_1",
                call=GraphCall(
                    call_id="call_1",
                    model="test-model",
                    messages=[{"role": "user", "content": "hello"}],
                    expected_output="ok",
                    input_segments=[InputSegment(type="unique", message_count=1, token_count=5)],
                    total_input_tokens=5,
                    expected_output_tokens=2,
                    temperature=None,
                    max_tokens_recorded=None,
                ),
                predecessor_event_ids=[],
                predecessor_dependency_types={},
                wait_ms=0,
                t_start_ms=1000,
                t_end_ms=1500,
            ),
        }
        graph = ReplayGraph(events=events, root_event_ids=["event_0", "event_1"], source_file="test")

        gen = _make_generator()
        gen.initialize_sessions([ReplaySession(session_id="s", source_id="src", session_index=0, graph=graph)])

        # Only event_1 is scheduled (event_0 has no messages).
        scheduled = gen.get_session_events(0)
        assert len(scheduled) == 1, f"expected 1 scheduled event, got {len(scheduled)}"

        data = gen.load_lazy_data(scheduled[0])
        assert data.generator is gen

        # total_events_in_session must equal the scheduled count (1), not len(graph.events) (2).
        assert data.total_events_in_session == 1, (
            f"total_events_in_session is {data.total_events_in_session}; "
            "should be 1 (scheduled) not 2 (raw graph) — otherwise eviction never fires"
        )

        # Draining the one scheduled event must trigger eviction.
        info = SessionInferenceInfo(output_text="x", request_metrics=RequestMetrics(text=Text(input_tokens=1)))
        data.on_completion(info)
        assert "s" not in gen.session_graph_state, "session not evicted after its only scheduled event completed"

    def test_is_session_buildable_returns_false_for_zero_event_session(self) -> None:
        """Bug 5: a session whose graph builds but schedules zero events must not be dispatched.

        If is_session_buildable returns True for such a session, dispatch adds it to
        active_session_indices but no worker events ever fire, so check_session_completed
        never returns True and the stage stalls until timeout.
        """
        # All events have empty messages — _build_session_schedule will skip every one.
        events: dict[str, GraphEvent] = {
            "event_0": GraphEvent(
                event_id="event_0",
                call=GraphCall(
                    call_id="call_0",
                    model="test-model",
                    messages=[],
                    expected_output="",
                    input_segments=[],
                    total_input_tokens=0,
                    expected_output_tokens=0,
                    temperature=None,
                    max_tokens_recorded=None,
                ),
                predecessor_event_ids=[],
                predecessor_dependency_types={},
                wait_ms=0,
                t_start_ms=0,
                t_end_ms=500,
            ),
        }
        graph = ReplayGraph(events=events, root_event_ids=["event_0"], source_file="test")
        gen = _make_generator()
        gen.initialize_sessions([ReplaySession(session_id="s", source_id="src", session_index=0, graph=graph)])

        # Graph built successfully (session is not None) but zero events scheduled.
        assert gen.sessions[0] is not None, "session should have been built"
        assert gen._session_events.get(0, []) == [], "expected zero scheduled events"

        # is_session_buildable must return False to prevent a stalled stage.
        assert not gen.is_session_buildable(0), (
            "is_session_buildable returned True for a zero-event session — "
            "dispatch would add it to active_session_indices with nothing to complete it"
        )


# ---------------------------------------------------------------------------
# Structured tool-call preservation through the real graph-build path.
#
# These tests exercise extract_messages -> build_graph (via build_raw_calls)
# and assert on GraphCall.messages — the dicts that become the wire payload.
# They guard against _replay_message_to_dict flattening recorded tool calls
# into the internal "<|tool_call|>...<|end|>" marker string, which would drop
# structured tool_calls and tool-result tool_call_id linkage.
# ---------------------------------------------------------------------------


def _graph_messages_from_input(input_messages: List[dict[str, Any]]) -> List[dict[str, Any]]:
    """Run input messages through the real span-extraction + graph builder and
    return the resulting GraphCall.messages (the wire dicts)."""
    span = {
        "trace_id": "t",
        "span_id": "s",
        "parent_span_id": None,
        "name": "chat gpt-4",
        "start_time": "2026-01-01T00:00:00.000000",
        "end_time": "2026-01-01T00:00:01.000000",
        "attributes": {
            "gen_ai.operation.name": "chat",
            "gen_ai.request.model": "gpt-4",
            "gen_ai.input.messages": json.dumps(input_messages),
            "gen_ai.output.text": "ok",
        },
        "resource_attributes": {"service.name": "t"},
        "status": {"code": 1, "message": ""},
    }
    raw_calls, _ = build_raw_calls([span])
    graph = build_graph(raw_calls, source_file="t")
    return next(iter(graph.events.values())).call.messages


class TestStructuredToolCallPreservation:
    """GraphCall.messages must carry structured tool_calls / tool_call_id."""

    def test_assistant_tool_calls_only_openai_format(self) -> None:
        msgs = _graph_messages_from_input(
            [
                {"role": "user", "content": "weather in Paris?"},
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                        }
                    ],
                },
            ]
        )
        assistant = msgs[1]
        assert "<|tool_call|>" not in json.dumps(assistant), "tool call must not be flattened to marker text"
        assert assistant["tool_calls"][0]["id"] == "c1"
        assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
        assert assistant["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'

    def test_assistant_content_plus_tool_calls(self) -> None:
        msgs = _graph_messages_from_input(
            [
                {
                    "role": "assistant",
                    "content": "Let me check.",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "get_weather", "arguments": '{"city": "Paris"}'},
                        }
                    ],
                },
            ]
        )
        assistant = msgs[0]
        assert "<|tool_call|>" not in json.dumps(assistant)
        assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
        # Preamble text is preserved alongside the structured call.
        assert "Let me check." in assistant.get("content", "")

    def test_assistant_tool_calls_direct_format(self) -> None:
        msgs = _graph_messages_from_input(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "c1", "name": "get_weather", "arguments": '{"city": "Paris"}'}],
                },
            ]
        )
        assistant = msgs[0]
        assert "<|tool_call|>" not in json.dumps(assistant)
        assert assistant["tool_calls"][0]["function"]["name"] == "get_weather"
        assert assistant["tool_calls"][0]["function"]["arguments"] == '{"city": "Paris"}'

    def test_tool_result_parts_list_keeps_tool_call_id(self) -> None:
        msgs = _graph_messages_from_input(
            [
                {
                    "role": "tool",
                    "content": [{"type": "tool_call_response", "id": "c1", "result": '{"temp_c": 18}'}],
                },
            ]
        )
        tool_msg = msgs[0]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "c1"
        assert "18" in tool_msg.get("content", "")

    def test_tool_result_string_content_keeps_tool_call_id(self) -> None:
        # A role:tool result with STRING content + a sibling tool_call_id must
        # retain tool_call_id on the wire (previously it became a plain
        # ReplayMessage in extract_messages and the id was dropped).
        msgs = _graph_messages_from_input(
            [
                {"role": "tool", "tool_call_id": "c1", "content": '{"temp_c": 18}'},
            ]
        )
        tool_msg = msgs[0]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "c1"
        assert "18" in tool_msg.get("content", "")

    def test_tool_call_object_arguments_serialized_to_json_string(self) -> None:
        # OpenAI/vLLM require function.arguments to be a JSON string. Recorded
        # traces often carry it as a parsed object; it must be serialised, not
        # passed through as an object (which triggers a 400 on the server).
        msgs = _graph_messages_from_input(
            [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "c1",
                            "type": "function",
                            "function": {"name": "make_todos", "arguments": {"todos": ["a", "b"]}},
                        }
                    ],
                },
            ]
        )
        args = msgs[0]["tool_calls"][0]["function"]["arguments"]
        assert isinstance(args, str), "arguments must be a JSON string, not an object"
        assert json.loads(args) == {"todos": ["a", "b"]}

    def test_tool_result_with_nested_block_list_result(self) -> None:
        # A tool_call_response whose `result` is itself a list of content blocks
        # (e.g. [{"type": "text", "text": "..."}]) must flatten to string content,
        # not crash the "".join and not leave a list on the wire.
        msgs = _graph_messages_from_input(
            [
                {
                    "role": "tool",
                    "content": [
                        {
                            "type": "tool_call_response",
                            "id": "c1",
                            "result": [{"type": "text", "text": '{"temp_c": 18}'}],
                        }
                    ],
                },
            ]
        )
        tool_msg = msgs[0]
        assert tool_msg["tool_call_id"] == "c1"
        assert isinstance(tool_msg.get("content"), str)
        assert "18" in tool_msg["content"]

    def test_tool_call_missing_id_skips_trace(self) -> None:
        # A tool_call with no id cannot be replayed faithfully (OpenAI/vLLM reject
        # tool_calls[].id=null with a 400, and the role:tool result is linked by
        # that id). build_graph's caller wraps build_graph in try/except and skips
        # the trace, so _normalize_tool_call must raise rather than emit id:null.
        with pytest.raises(ValueError, match="no id"):
            _graph_messages_from_input(
                [
                    {
                        "role": "assistant",
                        "tool_calls": [{"type": "function", "function": {"name": "get_weather", "arguments": "{}"}}],
                    },
                ]
            )

    def test_tool_call_non_dict_function_does_not_crash(self) -> None:
        # A malformed tool_call whose `function` is a bare string (not a nested
        # dict) must not raise AttributeError; it falls back to the direct/parts
        # shape (top-level name/arguments).
        msgs = _graph_messages_from_input(
            [
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "c1", "function": "get_weather", "name": "get_weather", "arguments": "{}"}],
                },
            ]
        )
        assert msgs[0]["tool_calls"][0]["function"]["name"] == "get_weather"

    def test_tool_result_on_user_role_parts_format_normalizes_to_tool(self) -> None:
        # Anthropic-native harnesses put a tool result on a
        # role:user message, rather than a dedicated
        # role:tool message. The OpenAI-compatible wire target only defines tool results as role:tool
        # messages, so the emitted role must be normalized.
        msgs = _graph_messages_from_input(
            [
                {
                    "role": "user",
                    "parts": [
                        {
                            "type": "tool_call_response",
                            "id": "call_cad5f02bc9c8424abdc072de",
                            "result": '[{"type": "text", "text": "Todos have been modified successfully."}]',
                        }
                    ],
                },
            ]
        )
        tool_msg = msgs[0]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "call_cad5f02bc9c8424abdc072de"
        assert "Todos have been modified successfully" in tool_msg["content"]

    def test_tool_result_on_user_role_content_list_format_normalizes_to_tool(self) -> None:
        # Same normalization as above, but for the raw-dict "content" list shape
        # (Shape 2) rather than the "parts" shape (Shape 1).
        msgs = _graph_messages_from_input(
            [
                {
                    "role": "user",
                    "content": [{"type": "tool_call_response", "id": "c1", "result": '{"temp_c": 18}'}],
                },
            ]
        )
        tool_msg = msgs[0]
        assert tool_msg["role"] == "tool"
        assert tool_msg["tool_call_id"] == "c1"
        assert "18" in tool_msg.get("content", "")

    def test_multiple_tool_results_keeps_first(self) -> None:
        # One OpenAI wire message answers exactly one tool_call_id. A message that
        # bundles several tool_call_response parts keeps the FIRST (id + result)
        # and drops the rest, rather than concatenating results under one id.
        msgs = _graph_messages_from_input(
            [
                {
                    "role": "tool",
                    "content": [
                        {"type": "tool_call_response", "id": "c1", "result": "first"},
                        {"type": "tool_call_response", "id": "c2", "result": "second"},
                    ],
                },
            ]
        )
        tool_msg = msgs[0]
        assert tool_msg["tool_call_id"] == "c1"
        assert tool_msg["content"] == "first"
        assert "second" not in tool_msg["content"]

    def test_tool_call_id_not_leaked_onto_user_message(self) -> None:
        # tool_call_id belongs on a role:tool message only. A role:user message
        # carrying a stray tool_call_id must not emit it on the wire (OpenAI/vLLM
        # reject tool_call_id on non-tool roles).
        msgs = _graph_messages_from_input(
            [
                {"role": "user", "tool_call_id": "c1", "content": "hello"},
            ]
        )
        assert "tool_call_id" not in msgs[0]


class TestLoadLazyDataPreservesToolLinkage:
    """load_lazy_data must build ChatMessages (the wire payload when no
    substitution runs, e.g. disable_output_substitution=true) that retain
    tool_call_id and structured tool_calls — not just original_messages."""

    def _make_gen_from_input(self, input_messages: List[dict[str, Any]]) -> Any:
        span = {
            "trace_id": "t",
            "span_id": "s",
            "parent_span_id": None,
            "name": "chat gpt-4",
            "start_time": "2026-01-01T00:00:00.000000",
            "end_time": "2026-01-01T00:00:01.000000",
            "attributes": {
                "gen_ai.operation.name": "chat",
                "gen_ai.request.model": "gpt-4",
                "gen_ai.input.messages": json.dumps(input_messages),
                "gen_ai.output.text": "ok",
            },
            "resource_attributes": {"service.name": "t"},
            "status": {"code": 1, "message": ""},
        }
        raw_calls, _ = build_raw_calls([span])
        graph = build_graph(raw_calls, source_file="t")
        session_id = "session_0"
        gc = next(iter(graph.events.values()))
        event = ReplaySessionEvent(
            call_id=gc.call.call_id,
            event_id=f"{session_id}:{gc.event_id}",
            session_index=0,
            t_start_ms=gc.t_start_ms,
            t_end_ms=gc.t_end_ms,
            model=gc.call.model,
            messages=gc.call.messages,
            expected_output=gc.call.expected_output,
            input_segments=gc.call.input_segments,
            expected_output_tokens=gc.call.expected_output_tokens,
            temperature=gc.call.temperature,
            max_tokens_recorded=gc.call.max_tokens_recorded,
            predecessor_event_ids=[],
            wait_ms=0,
        )
        gen = object.__new__(OTelTraceReplayDataGenerator)
        gen.all_events = [event]
        gen._session_events = {0: [event]}
        gen.output_registry = EventOutputRegistry()
        gen.worker_tracker = WorkerSessionTracker()
        gen.session_completion_queue = None
        gen.api_config = make_api_config()
        cfg = MagicMock()
        cfg.attribute_to_header_map = {}
        cfg.attribute_to_label_map = {}
        cfg.bad_tool_call_handling = BadToolCallHandling.NONE
        gen.otel_config = cfg
        gen.replay_config = cfg
        gen.session_graph_state = {session_id: MagicMock(graph=graph, random_string=None)}
        return gen

    def test_tool_result_chatmessage_keeps_tool_call_id(self) -> None:
        from inference_perf.apis import LazyLoadInferenceAPIData

        gen = self._make_gen_from_input(
            [
                {"role": "user", "content": "weather?"},
                {
                    "role": "assistant",
                    "tool_calls": [{"id": "c1", "type": "function", "function": {"name": "get_weather", "arguments": "{}"}}],
                },
                {"role": "tool", "tool_call_id": "c1", "content": '{"temp_c": 18}'},
            ]
        )
        api_data = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0))
        # The role:tool ChatMessage must carry tool_call_id (the wire payload
        # when disable_output_substitution=true skips the substitution rebuild).
        tool_cm = next(m for m in api_data.messages if m.role == "tool")
        assert tool_cm.tool_call_id == "c1"
        assert "tool_call_id" in tool_cm.to_dict()
        # The assistant ChatMessage must carry structured tool_calls.
        asst_cm = next(m for m in api_data.messages if m.role == "assistant")
        assert asst_cm.tool_calls is not None
        assert asst_cm.tool_calls[0]["function"]["name"] == "get_weather"


class TestDisableOutputSubstitutionValidation:
    """disable_output_substitution contradicts random session-ID injection, so
    the config validator must reject the combination up front rather than let
    the substitution pass run anyway (and silently ignore the flag)."""

    def test_disable_substitution_alone_is_valid(self) -> None:
        cfg = OTelTraceReplayConfig(trace_files=["/tmp/t.json"], disable_output_substitution=True)
        assert cfg.disable_output_substitution is True

    def test_disable_substitution_with_random_injection_raises(self) -> None:
        with pytest.raises(ValidationError, match="inject_random_session_id"):
            OTelTraceReplayConfig(
                trace_files=["/tmp/t.json"],
                disable_output_substitution=True,
                inject_random_session_id=True,
            )

    def test_disable_substitution_with_duplicate_sessions_raises(self) -> None:
        with pytest.raises(ValidationError, match="duplicate_sessions_target"):
            OTelTraceReplayConfig(
                trace_files=["/tmp/t.json"],
                disable_output_substitution=True,
                duplicate_sessions_target=10,
            )

    def test_random_injection_without_disable_substitution_is_valid(self) -> None:
        cfg = OTelTraceReplayConfig(
            trace_files=["/tmp/t.json"],
            inject_random_session_id=True,
            duplicate_sessions_target=10,
        )
        assert cfg.disable_output_substitution is False
