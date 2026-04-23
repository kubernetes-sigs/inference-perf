# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Shared graph-backed SessionGenerator runtime.

This module contains the session replay runtime that is agnostic to how a
ReplayGraph was produced. Concrete generators are responsible for producing
ReplaySession objects; this base class handles session lifecycle, worker
affinity, lazy request materialization, and session completion tracking.
"""

from __future__ import annotations

import asyncio
import json
import logging
import time
from dataclasses import dataclass, field, replace as dc_replace
from multiprocessing.managers import SyncManager
from typing import Any, Dict, List, Optional, Set

from aiohttp import ClientResponse

from inference_perf.apis import ChatCompletionAPIData, InferenceInfo, LazyLoadInferenceAPIData, SessionLifecycleMetric
from inference_perf.apis.chat import ChatMessage
from inference_perf.apis.streaming_parser import parse_sse_stream
from inference_perf.config import APIConfig, APIType, DataConfig
from inference_perf.datagen.base import LazyLoadDataMixin, SessionGenerator
from inference_perf.datagen.replay_graph_types import InputSegment, ReplayGraph
from inference_perf.utils.custom_tokenizer import CustomTokenizer

logger = logging.getLogger(__name__)


class EventFailedError(Exception):
    """Raised by EventOutputRegistry.require_async when the awaited event failed."""

    def __init__(self, event_id: str) -> None:
        super().__init__(f"Predecessor event {event_id!r} failed")
        self.event_id = event_id


class SessionInferenceInfo(InferenceInfo):
    """InferenceInfo subclass that also carries the raw output text."""

    output_text: Optional[str] = None
    output_tokens: int = 0
    output_token_times: list[float] = field(default_factory=list)


class WorkerSessionTracker:
    """Per-worker tracking of event completions and session failures."""

    def __init__(self) -> None:
        self._event_completions: Dict[str, Dict[str, float]] = {}
        self._failed_sessions: Set[str] = set()

    def record_event_completed(self, session_id: str, event_id: str, completion_time: float) -> None:
        if session_id not in self._event_completions:
            self._event_completions[session_id] = {}
        self._event_completions[session_id][event_id] = completion_time

    def is_event_completed(self, session_id: str, event_id: str) -> bool:
        return session_id in self._event_completions and event_id in self._event_completions[session_id]

    def get_event_completion_time(self, session_id: str, event_id: str) -> Optional[float]:
        return self._event_completions.get(session_id, {}).get(event_id)

    def mark_session_failed(self, session_id: str) -> None:
        self._failed_sessions.add(session_id)

    def is_session_failed(self, session_id: str) -> bool:
        return session_id in self._failed_sessions

    def get_session_event_count(self, session_id: str) -> int:
        return len(self._event_completions.get(session_id, {}))

    def get_session_completion_times(self, session_id: str) -> Dict[str, float]:
        return self._event_completions.get(session_id, {}).copy()


class EventOutputRegistry:
    """Per-worker registry mapping event_id → actual output text and input messages."""

    def __init__(self) -> None:
        self._event_output_text: Dict[str, str] = {}
        self._event_input_messages: Dict[str, Any] = {}
        self._event_signals: Dict[str, asyncio.Event] = {}
        self._failed_event_ids: Set[str] = set()

    def record(self, event_id: str, output_text: str, messages: List[Any]) -> None:
        if event_id in self._event_output_text:
            raise ValueError(
                f"Event {event_id} has already been recorded. "
                f"Each event should only complete once. This indicates a bug in the replay logic."
            )

        self._event_output_text[event_id] = output_text
        self._event_input_messages[event_id] = list(messages) if messages else []

        if event_id in self._event_signals:
            self._event_signals[event_id].set()
            logger.debug(f"Set asyncio.Event signal for event {event_id}")

    def get_output_by_event_id(self, event_id: str) -> Optional[str]:
        return self._event_output_text.get(event_id)

    def get_messages_by_event_id(self, event_id: str) -> Optional[List[Any]]:
        return self._event_input_messages.get(event_id)

    def get_event_ids(self) -> List[str]:
        return list(self._event_output_text.keys())

    def record_failure(self, event_id: str) -> None:
        self._failed_event_ids.add(event_id)
        if event_id not in self._event_signals:
            self._event_signals[event_id] = asyncio.Event()
        self._event_signals[event_id].set()
        logger.debug(f"Recorded failure for event {event_id}")

    def is_event_failed(self, event_id: str) -> bool:
        return event_id in self._failed_event_ids

    async def require_async(self, event_id: str, timeout_sec: float = 3600.0) -> str:
        if event_id in self._failed_event_ids:
            raise EventFailedError(event_id)

        output = self._event_output_text.get(event_id)
        if output is not None:
            return output

        if event_id not in self._event_signals:
            self._event_signals[event_id] = asyncio.Event()
        signal = self._event_signals[event_id]

        if event_id in self._failed_event_ids:
            raise EventFailedError(event_id)
        output = self._event_output_text.get(event_id)
        if output is not None:
            return output

        logger.debug(f"Event {event_id} waiting on asyncio signal (zero threads)")

        try:
            await asyncio.wait_for(signal.wait(), timeout=timeout_sec)
        except asyncio.TimeoutError as e:
            raise TimeoutError(
                f"EventOutputRegistry: output for '{event_id}' not available after "
                f"{timeout_sec:.1f}s. Check that the predecessor is not blocked or failed."
            ) from e

        if event_id in self._failed_event_ids:
            raise EventFailedError(event_id)

        output = self._event_output_text.get(event_id)
        assert output is not None, (
            f"asyncio signal fired for {event_id} but output missing from local cache — this is a bug in record()"
        )
        logger.debug(f"Event {event_id} woke from asyncio signal")
        return output


class SessionChatCompletionAPIData(ChatCompletionAPIData):
    """ChatCompletionAPIData subclass for graph-backed session replay."""

    model_config = {"arbitrary_types_allowed": True}

    event_id: str
    registry: EventOutputRegistry
    worker_tracker: WorkerSessionTracker
    completion_queue: Any
    total_events_in_session: int
    predecessor_event_ids: List[str] = field(default_factory=list)
    wait_ms: int = 0
    input_segments: List[InputSegment] = field(default_factory=list)
    original_messages: List[Dict[str, Any]] = field(default_factory=list)
    expected_output_content: Optional[str] = None
    skip_request: bool = False

    def _extract_session_id(self) -> str:
        return self.event_id.split(":")[0] if ":" in self.event_id else self.event_id

    async def wait_for_predecessors_and_substitute(self) -> None:
        session_id = self._extract_session_id()

        if self.worker_tracker.is_session_failed(session_id):
            logger.info(f"Event {self.event_id} skipping - session {session_id} has failed (pre-wait check)")
            self.skip_request = True
            self.registry.record_failure(self.event_id)
            return

        if self.predecessor_event_ids:
            logger.debug(f"Event {self.event_id} waiting for {len(self.predecessor_event_ids)} predecessor(s)")
            try:
                await asyncio.gather(
                    *[self.registry.require_async(event_id, timeout_sec=3600.0) for event_id in self.predecessor_event_ids]
                )
            except EventFailedError:
                logger.info(f"Event {self.event_id} skipping - predecessor failed")
                self.skip_request = True
                self.registry.record_failure(self.event_id)
                return
            logger.debug(f"Event {self.event_id} all predecessors done")

        if self.wait_ms > 0:
            wait_sec = self.wait_ms / 1000.0
            logger.debug(f"Event {self.event_id} waiting {wait_sec:.3f}s (wait_ms={self.wait_ms})")
            await asyncio.sleep(wait_sec)

        if any(seg.type == "output" or seg.type == "shared" for seg in self.input_segments):
            logger.debug(f"Event {self.event_id} substituting output/shared segments")
            substituted = self._build_messages_with_substitution()
            self.messages = [ChatMessage(role=m["role"], content=m["content"]) for m in substituted]
            logger.debug(f"Event {self.event_id} substitution complete, {len(self.messages)} messages")

    def _build_messages_with_substitution(self) -> List[Dict[str, Any]]:
        if not self.input_segments:
            return self.original_messages

        result = []
        cursor = 0

        for seg in self.input_segments:
            seg_msgs = self.original_messages[cursor : cursor + seg.message_count]

            if seg.type == "output":
                if seg.source_event_id:
                    actual_output = self.registry.get_output_by_event_id(seg.source_event_id)
                    logger.debug(
                        f"Registry get for event {self.event_id} output segment from {seg.source_event_id} generated: {actual_output}"
                    )

                    if actual_output:
                        for msg in seg_msgs:
                            substituted = dict(msg)
                            substituted["content"] = actual_output
                            result.append(substituted)
                        logger.debug(
                            f"Event {self.event_id}: substituted output segment with actual output from {seg.source_event_id}"
                        )
                    else:
                        logger.warning(
                            f"Event {self.event_id}: output segment from {seg.source_event_id} "
                            f"not available, using recorded content"
                        )
                        result.extend(seg_msgs)
                else:
                    logger.warning(f"Event {self.event_id}: output segment has no source_event_id, using recorded content")
                    result.extend(seg_msgs)
            elif seg.type == "shared":
                if seg.source_event_id is None:
                    logger.error(f"CRITICAL: Event {self.event_id} shared segment has no source_event_id")
                    result.extend(seg_msgs)
                    continue
                seg_msgs_from_parent = self.registry.get_messages_by_event_id(seg.source_event_id)
                if seg_msgs_from_parent is None:
                    logger.error(
                        f"CRITICAL: Event {self.event_id} shared segment from {seg.source_event_id} "
                        f"has no messages in registry (should not happen after require_async)"
                    )
                    result.extend(seg_msgs)
                else:
                    logger.debug(
                        f"Registry get for event {self.event_id} from {seg.source_event_id} "
                        f"shared segment num messages in parent event: {len(seg_msgs_from_parent)}"
                    )
                    if len(seg_msgs_from_parent) != len(seg_msgs):
                        logger.error(
                            f"Event {self.event_id} shared segment from {seg.source_event_id} "
                            f"had different number of messages in parent event: {len(seg_msgs_from_parent)}, "
                            f"num messages in seg: {len(seg_msgs)}"
                        )
                    for msg in seg_msgs_from_parent:
                        result.append(dict(msg))
            elif seg.type == "unique":
                for msg in seg_msgs:
                    result.append(msg)
            else:
                result.extend(seg_msgs)

            cursor += seg.message_count

        return result

    def on_completion(self, info: InferenceInfo) -> None:
        output_text = info.output_text if isinstance(info, SessionInferenceInfo) else ""
        output_text = output_text or ""
        self.registry.record(self.event_id, output_text, self.messages)
        logger.debug(
            f"calling registry record for event {self.event_id} num input messages {len(self.messages)} and output: {output_text}"
        )
        completion_time = time.perf_counter()
        session_id = self._extract_session_id()
        event_id = self.event_id.split(":", 1)[1] if ":" in self.event_id else self.event_id
        self.worker_tracker.record_event_completed(session_id, event_id, completion_time)
        logger.debug(f"Recorded event completion in worker tracker for {self.event_id}")

        completed_count = self.worker_tracker.get_session_event_count(session_id)

        if completed_count == self.total_events_in_session:
            logger.debug(f"Session {session_id} completed all {self.total_events_in_session} events in worker")

            completion_data = {
                "session_id": session_id,
                "completion_time": completion_time,
                "failed": self.worker_tracker.is_session_failed(session_id),
                "event_completion_times": self.worker_tracker.get_session_completion_times(session_id),
            }

            if self.completion_queue is not None:
                try:
                    self.completion_queue.put_nowait(completion_data)
                    logger.debug(f"Pushed session {session_id} completion to queue")
                except Exception as e:
                    logger.error(f"Failed to push session {session_id} completion to queue: {e}")

    async def process_response(
        self,
        response: ClientResponse,
        config: APIConfig,
        tokenizer: CustomTokenizer,
        lora_adapter: Optional[str] = None,
    ) -> SessionInferenceInfo:
        """Process the LLM response, capture output text, and register it."""
        logger.debug(f"process_response called for event {self.event_id}")
        output_text: str = ""

        if config.streaming:
            # Use shared streaming parser with chat-specific content extraction
            output_text, output_token_times, raw_content, _, _ = await parse_sse_stream(
                response, extract_content=lambda data: data.get("choices", [{}])[0].get("delta", {}).get("content")
            )

            prompt_text = "".join([msg.content for msg in self.messages if msg.content])
            prompt_len = tokenizer.count_tokens(prompt_text)
            output_len = tokenizer.count_tokens(output_text)
            info = SessionInferenceInfo(
                input_tokens=prompt_len,
                output_tokens=output_len,
                output_token_times=output_token_times,
                lora_adapter=lora_adapter,
                output_text=output_text or None,
                extra_info={"raw_response": raw_content},
            )
        else:
            data = await response.json()
            prompt_len = tokenizer.count_tokens("".join([m.content for m in self.messages]))
            choices = data.get("choices", [])
            if choices:
                output_text = "".join([choice.get("message", {}).get("content", "") for choice in choices])
            output_len = tokenizer.count_tokens(output_text)
            info = SessionInferenceInfo(
                input_tokens=prompt_len,
                output_tokens=output_len,
                lora_adapter=lora_adapter,
                output_text=output_text or None,
            )

        # Register output and notify successors.
        self.on_completion(info)

        if output_text:
            logger.debug(f"Registered output for event {self.event_id}: {len(output_text)} chars : {output_text}")
        else:
            logger.debug(f"Registered empty output for event {self.event_id}")

        return info

    async def process_failure(
        self,
        response: Optional[ClientResponse],
        config: APIConfig,
        tokenizer: CustomTokenizer,
        exception: Exception,
        lora_adapter: Optional[str] = None,
    ) -> InferenceInfo:
        logger.error(f"Request failed for event {self.event_id}: {type(exception).__name__}: {str(exception)}")

        session_id = self._extract_session_id()
        was_already_failed = self.worker_tracker.is_session_failed(session_id)
        self.worker_tracker.mark_session_failed(session_id)
        self.registry.record_failure(self.event_id)

        if not was_already_failed and self.completion_queue is not None:
            completion_time = time.perf_counter()
            completed_so_far = self.worker_tracker.get_session_event_count(session_id)
            cancelled = self.total_events_in_session - completed_so_far - 1
            completion_data = {
                "session_id": session_id,
                "completion_time": completion_time,
                "failed": True,
                "cancelled_events": cancelled,
                "event_completion_times": self.worker_tracker.get_session_completion_times(session_id),
            }

            try:
                logger.debug(f"Pushing immediate failure notification for session {session_id}")
                self.completion_queue.put_nowait(completion_data)
                logger.info(f"Session {session_id} failure notification sent to main process (cancelled_events={cancelled})")
            except Exception as e:
                logger.error(f"Failed to push session {session_id} failure notification to queue: {e}")

        return SessionInferenceInfo(
            input_tokens=0,
            output_tokens=0,
            lora_adapter=lora_adapter,
            output_text="",
        )


@dataclass
class ReplaySessionState:
    """Tracks graph traversal state for one session."""

    session_id: str
    graph: ReplayGraph
    ready_events: Set[str]
    dispatched_events: Set[str]
    completed_events: Set[str]
    event_completion_times: Dict[str, float]
    is_active: bool = False
    is_complete: bool = False
    failed: bool = False
    cancelled_events: int = 0


@dataclass
class ReplaySessionEvent:
    """Represents a single replayable event derived from a graph event."""

    call_id: str
    event_id: str
    session_index: int
    t_start_ms: int
    t_end_ms: int
    model: str
    messages: List[Dict[str, Any]]
    expected_output: str
    input_segments: List[InputSegment]
    expected_output_tokens: int
    temperature: Optional[float]
    max_tokens_recorded: Optional[int]
    predecessor_event_ids: List[str] = field(default_factory=list)
    wait_ms: int = 0


@dataclass
class ReplaySession:
    """Represents one replayable session backed by a ReplayGraph."""

    session_id: str
    source_id: str
    session_index: int
    graph: ReplayGraph
    start_offset_ms: int = 0


class ReplayGraphSessionGeneratorBase(SessionGenerator, LazyLoadDataMixin):
    """Shared runtime for ReplayGraph-backed session replay generators."""

    def __init__(
        self,
        api_config: APIConfig,
        config: DataConfig,
        tokenizer: Optional[CustomTokenizer],
        mp_manager: Optional[SyncManager] = None,
        base_seed: Optional[int] = None,
        num_workers: int = 1,
    ) -> None:
        super().__init__(api_config, config, tokenizer)
        self.config = config
        self.mp_manager = mp_manager
        self.num_workers = max(1, num_workers)
        self.base_seed = base_seed if base_seed is not None else 42

        self.output_registry = EventOutputRegistry()
        self.worker_tracker = WorkerSessionTracker()
        if mp_manager is not None:
            self.session_completion_queue: Any = mp_manager.Queue()
        else:
            self.session_completion_queue = None

        self.sessions: List[ReplaySession] = []
        self.session_graph_state: Dict[str, ReplaySessionState] = {}
        self.all_events: List[ReplaySessionEvent] = []

    def initialize_sessions(self, sessions: List[ReplaySession]) -> None:
        """Finalize generator state from prepared sessions."""
        self.sessions = sessions
        if not self.sessions:
            raise ValueError("No valid replay sessions found")
        self._build_replay_schedule()
        logger.debug("Loaded %d sessions with %d total events", len(self.sessions), len(self.all_events))

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Chat]

    def is_preferred_worker_requested(self) -> bool:
        return True

    def _build_replay_schedule(self) -> None:
        self.all_events = []

        for session in self.sessions:
            state = ReplaySessionState(
                session_id=session.session_id,
                graph=session.graph,
                ready_events=set(),
                dispatched_events=set(),
                completed_events=set(),
                event_completion_times={},
                is_active=False,
                is_complete=False,
            )
            self.session_graph_state[session.session_id] = state

            for event in session.graph.events.values():
                gc = event.call

                if not gc.messages:
                    logger.warning("Call %s in event %s has no messages, skipping", gc.call_id, event.event_id)
                    continue

                qualified_event_id = f"{session.session_id}:{event.event_id}"
                qualified_predecessor_ids = [f"{session.session_id}:{pid}" for pid in event.predecessor_event_ids]
                qualified_segments = [
                    dc_replace(seg, source_event_id=f"{session.session_id}:{seg.source_event_id}")
                    if seg.source_event_id is not None
                    else seg
                    for seg in gc.input_segments
                ]

                self.all_events.append(
                    ReplaySessionEvent(
                        call_id=gc.call_id,
                        event_id=qualified_event_id,
                        session_index=session.session_index,
                        t_start_ms=event.t_start_ms,
                        t_end_ms=event.t_end_ms,
                        model=gc.model,
                        messages=gc.messages,  # type: ignore[arg-type]
                        expected_output=gc.expected_output,
                        input_segments=qualified_segments,
                        expected_output_tokens=gc.expected_output_tokens,
                        temperature=gc.temperature,
                        max_tokens_recorded=gc.max_tokens_recorded,
                        predecessor_event_ids=qualified_predecessor_ids,
                        wait_ms=event.wait_ms,
                    )
                )

        logger.info(
            "Built replay schedule: %d events across %d sessions (graph-based traversal)",
            len(self.all_events),
            len(self.sessions),
        )

    def get_session_count(self) -> int:
        return len(self.sessions)

    def get_session_event_indices(self, session_index: int) -> List[int]:
        if session_index < 0 or session_index >= len(self.sessions):
            raise IndexError(f"Session index {session_index} out of range (total: {len(self.sessions)})")

        session = self.sessions[session_index]
        return [i for i, event in enumerate(self.all_events) if event.session_index == session.session_index]

    def get_session_info(self, session_index: int) -> Dict[str, Any]:
        if session_index < 0 or session_index >= len(self.sessions):
            raise IndexError(f"Session index {session_index} out of range (total: {len(self.sessions)})")

        session = self.sessions[session_index]
        event_indices = self.get_session_event_indices(session_index)

        return {
            "session_id": session.session_id,
            "file_path": session.source_id,
            "source_id": session.source_id,
            "session_index": session.session_index,
            "num_events": len(event_indices),
            "num_graph_events": len(session.graph.events),
            "start_offset_ms": session.start_offset_ms,
        }

    def get_session_events(self, session_index: int) -> List[LazyLoadInferenceAPIData]:
        session = self.sessions[session_index]
        event_indices = self.get_session_event_indices(session_index)
        session_worker_id = abs(hash(session.session_id)) % self.num_workers
        return [LazyLoadInferenceAPIData(data_index=idx, preferred_worker_id=session_worker_id) for idx in event_indices]

    def build_session_metric(
        self,
        session_id: str,
        stage_id: int,
        start_time: float,
        end_time: float,
    ) -> SessionLifecycleMetric:
        state = self.session_graph_state.get(session_id)
        if state is None:
            raise ValueError(f"Unknown session: {session_id}")

        source_id = ""
        for session in self.sessions:
            if session.session_id == session_id:
                source_id = session.source_id
                break

        num_events = len(state.graph.events)
        num_events_completed = len(state.completed_events)
        num_events_cancelled = state.cancelled_events if state.failed else 0

        return SessionLifecycleMetric(
            session_id=session_id,
            stage_id=stage_id,
            file_path=source_id,
            start_time=start_time,
            end_time=end_time,
            duration_sec=end_time - start_time,
            num_events=num_events,
            num_events_completed=num_events_completed,
            num_events_cancelled=num_events_cancelled,
        )

    def activate_session(self, session_id: str) -> None:
        state = self.session_graph_state.get(session_id)
        if state is None:
            logger.warning("Attempted to activate unknown session: %s", session_id)
            return

        state.is_active = True
        root_events = {event_id for event_id, event in state.graph.events.items() if not event.predecessor_event_ids}
        state.ready_events.update(root_events)
        logger.debug("Activated session %s with %d root events", session_id, len(root_events))

    def _process_completion_queue(self) -> None:
        if self.session_completion_queue is None:
            return

        try:
            while True:
                completion_data = self.session_completion_queue.get_nowait()
                completed_session_id = completion_data["session_id"]

                completed_state = self.session_graph_state.get(completed_session_id)
                if completed_state is not None:
                    event_times = completion_data.get("event_completion_times", {})
                    for event_id, completion_time in event_times.items():
                        if event_id not in completed_state.completed_events:
                            completed_state.completed_events.add(event_id)
                            completed_state.event_completion_times[event_id] = completion_time

                    completed_state.is_complete = True
                    completed_state.failed = completion_data.get("failed", False)
                    completed_state.cancelled_events = completion_data.get("cancelled_events", 0)
                    logger.debug(
                        "Session %s marked complete from queue notification (failed=%s)",
                        completed_session_id,
                        completed_state.failed,
                    )
        except Exception:
            pass

    def get_session_state(self, session_id: str) -> Optional[ReplaySessionState]:
        return self.session_graph_state.get(session_id)

    def check_session_completed(self, session_id: str) -> bool:
        self._process_completion_queue()

        state = self.session_graph_state.get(session_id)
        if state is None:
            logger.warning("Attempted to check unknown session: %s", session_id)
            return False

        if state.is_complete:
            return True

        shared_failed = getattr(self, "shared_failed_sessions", None)
        if shared_failed is not None:
            is_failed = session_id in shared_failed
            if is_failed:
                state.is_complete = True
                logger.info("Session %s marked as complete due to failure", session_id)
                return True

        return False

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> SessionChatCompletionAPIData:
        n = data.data_index
        if n >= len(self.all_events):
            raise IndexError(f"Event index {n} out of range (total: {len(self.all_events)})")

        event = self.all_events[n]

        chat_messages = []
        original_messages: List[Dict[str, Any]] = []
        for msg in event.messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
            else:
                role = getattr(msg, "role", "user")
                content = getattr(msg, "text", "")

            if isinstance(content, list):
                content_parts = []
                for block in content:
                    if isinstance(block, dict):
                        if block.get("type") == "text":
                            content_parts.append(block.get("text", ""))
                        else:
                            content_parts.append(json.dumps(block))
                    else:
                        content_parts.append(str(block))
                content = " ".join(content_parts)

            content_str = str(content)
            chat_messages.append(ChatMessage(role=role, content=content_str))
            original_messages.append({"role": role, "content": content_str})

        max_tokens = event.expected_output_tokens
        session_id = event.event_id.split(":")[0] if ":" in event.event_id else event.event_id
        state = self.session_graph_state.get(session_id)
        total_events = len(state.graph.events) if state else 0

        return SessionChatCompletionAPIData(
            messages=chat_messages,
            max_tokens=max_tokens,
            event_id=event.event_id,
            registry=self.output_registry,
            worker_tracker=getattr(self, "worker_tracker", WorkerSessionTracker()),
            completion_queue=getattr(self, "session_completion_queue", None),
            total_events_in_session=total_events,
            predecessor_event_ids=event.predecessor_event_ids,
            wait_ms=event.wait_ms,
            input_segments=event.input_segments,
            original_messages=original_messages,
            expected_output_content=event.expected_output,
            otel_context=data.otel_context,
            session_id=data.session_id,
            preferred_worker_id=data.preferred_worker_id,
        )

    def cleanup_session(self, session_id: str) -> None:
        state = self.session_graph_state.get(session_id)
        if state is None:
            logger.warning("Attempted to cleanup unknown session: %s", session_id)
            return

        event_count = len(state.graph.events)
        for event_id in state.graph.events.keys():
            qualified_event_id = f"{session_id}:{event_id}"
            self.output_registry._event_output_text.pop(qualified_event_id, None)
            self.output_registry._event_input_messages.pop(qualified_event_id, None)
            self.output_registry._event_signals.pop(qualified_event_id, None)
            self.output_registry._failed_event_ids.discard(qualified_event_id)

        del self.session_graph_state[session_id]
        logger.debug("Cleaned up session %s: removed %d events from memory", session_id, event_count)
