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
- OTelChatCompletionAPIData: output substitution and completion notification
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

# Ensure project root is on path
sys.path.insert(0, str(Path(__file__).parent.parent))

from inference_perf.datagen.otel_trace_replay_datagen import (
    EventFailedError,
    EventOutputRegistry,
    OTelChatCompletionAPIData,
    OTelInferenceInfo,
    OTelTraceReplayDataGenerator,
    OTelTraceReplayEvent,
    WorkerSessionTracker,
)
from inference_perf.datagen.otel_trace_to_replay_graph import (
    InputSegment,
    build_graph,
    build_raw_calls,
)
from inference_perf.apis.chat import ChatMessage
from inference_perf.config import APIConfig, APIType


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
            return await reg.require_async("event_001", timeout_sec=2.0)

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
# OTelChatCompletionAPIData tests
# ---------------------------------------------------------------------------


class TestOTelChatCompletionAPIData:
    """Test output capture, substitution, and completion notification."""

    def _make_api_data(
        self,
        event_id: str,
        registry: EventOutputRegistry,
        worker_tracker: WorkerSessionTracker,
        completion_queue: Optional[mp.Queue[Any]] = None,
        total_events: int = 1,
        predecessor_event_ids: Optional[List[str]] = None,
    ) -> OTelChatCompletionAPIData:
        return OTelChatCompletionAPIData(
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
        # Returns OTelInferenceInfo
        assert isinstance(info, OTelInferenceInfo)
        assert info.output_text == "Paris is the capital of France."

    @pytest.mark.asyncio
    async def test_on_completion_records_in_tracker(self) -> None:
        """on_completion records event completion in WorkerSessionTracker."""
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()
        api_data = self._make_api_data("session_1:event_0", registry, tracker, total_events=2)

        info = OTelInferenceInfo(
            output_text="test output",
            input_tokens=10,
            output_tokens=5,
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
        info_0 = OTelInferenceInfo(output_text="first", input_tokens=5, output_tokens=3)
        api_data_0.on_completion(info_0)

        # No notification yet (only 1 of 2 events complete)
        assert len(completion_notifications) == 0

        # Now complete event_1 (the last event)
        api_data_1 = self._make_api_data("session_1:event_1", registry, tracker, mock_queue, total_events=2)
        info_1 = OTelInferenceInfo(output_text="done", input_tokens=5, output_tokens=3)
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

        api_data = OTelChatCompletionAPIData(
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
        from inference_perf.datagen.otel_trace_replay_datagen import SessionGraphState

        mock_graph = MagicMock()
        mock_graph.events = {"event_0": MagicMock(), "event_1": MagicMock()}

        gen.session_graph_state = {
            "session_1": SessionGraphState(
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
        from inference_perf.datagen.otel_trace_replay_datagen import SessionGraphState

        mock_graph = MagicMock()
        mock_graph.events = {"event_0": MagicMock()}

        gen.session_graph_state = {
            "session_1": SessionGraphState(
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
        from inference_perf.datagen.otel_trace_replay_datagen import SessionGraphState

        mock_graph = MagicMock()
        mock_graph.events = {"event_0": MagicMock(), "event_1": MagicMock()}

        gen.session_graph_state = {
            "session_1": SessionGraphState(
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
        from inference_perf.datagen.otel_trace_replay_datagen import SessionGraphState

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
            "session_1": SessionGraphState(
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
        from inference_perf.datagen.otel_trace_replay_datagen import SessionGraphState

        mock_graph_1 = MagicMock()
        mock_graph_1.events = {"event_0": MagicMock()}
        mock_graph_2 = MagicMock()
        mock_graph_2.events = {"event_0": MagicMock()}

        gen.session_graph_state = {
            "session_1": SessionGraphState(
                session_id="session_1",
                graph=mock_graph_1,
                ready_events=set(),
                dispatched_events=set(),
                completed_events=set(),
                event_completion_times={},
                is_active=True,
                is_complete=False,
            ),
            "session_2": SessionGraphState(
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
        calls = build_raw_calls(spans)
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
                OTelTraceReplayEvent(
                    call_id=gc.call_id,
                    event_id=qualified_event_id,
                    file_index=0,
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
        gen.output_registry = registry
        gen.worker_tracker = tracker
        gen.session_completion_queue = queue
        gen.api_config = make_api_config()
        gen.session_graph_state = {session_id: MagicMock(graph=graph)}

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
        assert isinstance(result_000, OTelChatCompletionAPIData)

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
        assert isinstance(result_001, OTelChatCompletionAPIData)

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


class TestOTelChatCompletionAPIDataErrorPaths:
    """Test error handling paths in OTelChatCompletionAPIData."""

    @pytest.mark.asyncio
    async def test_process_failure_marks_session_failed(self) -> None:
        """process_failure marks session as failed and registers empty output."""
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()

        api_data = OTelChatCompletionAPIData(
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
        assert isinstance(result, OTelInferenceInfo)
        assert result.output_text == ""
        assert result.input_tokens == 0
        assert result.output_tokens == 0

    @pytest.mark.asyncio
    async def test_wait_for_predecessors_post_wait_failure_check(self) -> None:
        """Test failure propagation: when predecessor fails, successor skips via EventFailedError."""
        registry = EventOutputRegistry()
        tracker = WorkerSessionTracker()

        # Create event with predecessor
        api_data = OTelChatCompletionAPIData(
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
    ) -> OTelChatCompletionAPIData:
        return OTelChatCompletionAPIData(
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
