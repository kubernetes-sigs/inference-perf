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
import re
import time
import uuid
from dataclasses import dataclass, field, replace as dc_replace
from multiprocessing.managers import SyncManager
from typing import Any, Dict, List, Optional, Set, Tuple

from aiohttp import ClientResponse

from inference_perf.apis import (
    ChatCompletionAPIData,
    ErrorResponseInfo,
    InferenceInfo,
    LazyLoadInferenceAPIData,
    SessionLifecycleMetric,
    StreamedResponseMetrics,
    UnaryResponseMetrics,
)
from inference_perf.apis.chat import ChatMessage
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.apis.streaming_parser import parse_sse_stream
from inference_perf.config import APIConfig, APIType, DataConfig, SessionReplayConfig
from inference_perf.config.datagen.replay import BadToolCallHandling
from inference_perf.datagen.base import LazyLoadDataMixin, SessionGenerator
from inference_perf.datagen.replay_graph_types import InputSegment, ReplayGraph
from inference_perf.utils.custom_tokenizer import CustomTokenizer

logger = logging.getLogger(__name__)


class SessionReplayLazyLoadData(LazyLoadInferenceAPIData):
    """LazyLoadInferenceAPIData extended with per-session addressing for OTel trace replay.

    Instead of a global data_index into all_events, the worker identifies the
    request by (session_index, local_event_index) so it can build only the
    requested session's graph on demand without materializing all sessions upfront.
    """

    data_index: int = -1
    session_index: int
    local_event_index: int


# --- bad_tool_call_handling ------------------------------------------------
# Server-side tool-call parsers can emit malformed JSON in
# tool_calls[i].function.arguments — for example vLLM's `qwen3_xml` parser
# leaks closing XML markers (`</parameter></function>`) into the JSON string
# value at decode time. vLLM still returns 200 on the response, but on the
# *next* turn the chat template's `json.loads(arguments)` raises and vLLM
# returns HTTP 400. Replaying the bad bytes verbatim therefore halts the
# session.
#
# This mitigation lives ENTIRELY in the substitution path
# (`_build_messages_with_substitution`). The response path is byte-identical
# to upstream main: it stores the raw tool_calls in the registry exactly as
# the model emitted them. At substitution time, when a downstream event
# pulls a predecessor's stored message, we run `_detect_bad_tool_calls` on
# its `tool_calls` and, if `bad_tool_call_handling=use_recorded`, substitute
# the recorded assistant message at this slot.
#
# Gated per-event by the `bad_tool_call_handling` field on the OTel replay
# config. When the value is `none` (the default), the inline detection is
# short-circuited and behavior is identical to upstream main.


def _detect_bad_tool_calls(
    tool_calls: Optional[List[Dict[str, Any]]],
) -> List[Tuple[int, str, str]]:
    """Return [(index, function_name, json_error)] for tool_calls whose
    `arguments` field is a string that fails json.loads(). Empty list = ok."""
    bad: List[Tuple[int, str, str]] = []
    if not tool_calls:
        return bad
    for i, tc in enumerate(tool_calls):
        fn = tc.get("function") or {}
        args = fn.get("arguments", "")
        if not isinstance(args, str):
            continue
        try:
            json.loads(args)
        except json.JSONDecodeError as e:
            bad.append((i, fn.get("name", "?"), str(e)))
    return bad


# --- end bad_tool_call_handling --------------------------------------------


class EventFailedError(Exception):
    """Raised by EventOutputRegistry.require_async when the awaited event failed."""

    def __init__(self, event_id: str) -> None:
        super().__init__(f"Predecessor event {event_id!r} failed")
        self.event_id = event_id


class SessionInferenceInfo(InferenceInfo):
    """InferenceInfo subclass that also carries the raw output text."""

    output_text: Optional[str] = None
    output_message: Optional[Dict[str, Any]] = None

    def __init__(
        self,
        *,
        request_metrics: Optional[RequestMetrics] = None,
        input_tokens: Optional[int] = None,
        output_tokens: Optional[int] = None,
        **kwargs: Any,
    ) -> None:
        if request_metrics is None:
            request_metrics = RequestMetrics(text=Text(input_tokens=input_tokens or 0))
            if "response_metrics" not in kwargs:
                kwargs["response_metrics"] = UnaryResponseMetrics(output_tokens=output_tokens or 0)
        super().__init__(request_metrics=request_metrics, **kwargs)


class WorkerSessionTracker:
    """Per-worker tracking of event completions and session failures."""

    def __init__(self) -> None:
        self._event_completions: Dict[str, Dict[str, float]] = {}
        self._failed_sessions: Set[str] = set()
        # Per-session set of predecessor event_ids whose live tool_call
        # response was detected malformed at substitution time and replaced
        # with the recorded assistant message. Empty when
        # bad_tool_call_handling is `none`. Stored as a set so a
        # predecessor with multiple downstream consumers (DAG fan-out) is
        # counted once, not once per consumer.
        self._recorded_substitution_event_ids: Dict[str, Set[str]] = {}
        # Per-session set of event_ids that have reached a terminal state on this worker —
        # whether they completed, were skipped, or failed. Used to detect when a session is
        # fully drained so the worker can evict its (otherwise never-freed) built graph.
        # A set (not a counter) makes drain accounting idempotent: an event that passes
        # through more than one terminal path is counted at most once.
        self._drained_events: Dict[str, Set[str]] = {}

    def record_event_drained(self, session_id: str, event_id: str) -> int:
        """Mark one event as drained (terminal) and return the session's drained count.

        Called from every terminal path (completion, skip, request failure). Idempotent
        per event_id. The caller compares the returned count against the session's total
        event count to decide whether the session is fully drained and can be evicted.
        """
        drained = self._drained_events.setdefault(session_id, set())
        drained.add(event_id)
        return len(drained)

    def forget_session(self, session_id: str) -> None:
        """Drop all per-worker tracking for a session (called after eviction)."""
        self._event_completions.pop(session_id, None)
        self._failed_sessions.discard(session_id)
        self._drained_events.pop(session_id, None)
        self._recorded_substitution_event_ids.pop(session_id, None)

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

    def record_recorded_substitution(self, session_id: str, event_id: str) -> None:
        """Tag a predecessor event_id whose live tool_call response was
        replaced with the recorded message. Idempotent."""
        self._recorded_substitution_event_ids.setdefault(session_id, set()).add(event_id)

    def get_session_recorded_substitution_event_ids(self, session_id: str) -> List[str]:
        # Sorted for deterministic test/log output.
        return sorted(self._recorded_substitution_event_ids.get(session_id, set()))


class EventOutputRegistry:
    """Per-worker registry mapping event_id → actual output text and input messages."""

    def __init__(self) -> None:
        self._event_output_text: Dict[str, str] = {}
        self._event_output_message: Dict[str, Dict[str, Any]] = {}
        self._event_input_messages: Dict[str, Any] = {}
        self._event_signals: Dict[str, asyncio.Event] = {}
        self._failed_event_ids: Set[str] = set()

    def record(
        self,
        event_id: str,
        output_text: str,
        messages: List[Any],
        output_message: Optional[Dict[str, Any]] = None,
    ) -> None:
        if event_id in self._event_output_text:
            raise ValueError(
                f"Event {event_id} has already been recorded. "
                f"Each event should only complete once. This indicates a bug in the replay logic."
            )

        self._event_output_text[event_id] = output_text
        self._event_input_messages[event_id] = list(messages) if messages else []
        if output_message is not None:
            self._event_output_message[event_id] = output_message

        if event_id in self._event_signals:
            self._event_signals[event_id].set()
            logger.debug(f"Set asyncio.Event signal for event {event_id}")

    def get_output_by_event_id(self, event_id: str) -> Optional[str]:
        return self._event_output_text.get(event_id)

    def get_message_by_event_id(self, event_id: str) -> Optional[Dict[str, Any]]:
        return self._event_output_message.get(event_id)

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
    # Back-reference to the worker's datagen so the last event of a session can evict the
    # session's lazily-built graph from this worker (the worker never calls cleanup_session
    # otherwise, so built graphs would accumulate for the whole stage). Set in load_lazy_data.
    generator: Optional["ReplayGraphSessionGeneratorBase"] = None
    predecessor_event_ids: List[str] = field(default_factory=list)
    wait_ms: int = 0
    input_segments: List[InputSegment] = field(default_factory=list)
    original_messages: List[Dict[str, Any]] = field(default_factory=list)
    expected_output_content: Optional[str] = None
    skip_request: bool = False
    expected_output_is_tool_call: bool = False
    expected_output_tool_names: Optional[List[str]] = None
    # KV-cache invalidation configuration
    inject_random_session_id: bool = False
    session_random_string: Optional[str] = None
    override_tool_call_max_tokens: bool = False
    # Mitigation for tool-call responses with malformed JSON in `arguments`.
    # `none` (default) is byte-identical to upstream main; `use_recorded`
    # substitutes the recorded assistant message at the affected slot.
    bad_tool_call_handling: BadToolCallHandling = BadToolCallHandling.NONE
    # When True, output/shared segments are NOT substituted with live predecessor
    # output; the recorded assistant messages are sent as-is. Predecessor wait
    # timing is still enforced.
    disable_output_substitution: bool = False
    # Set by _build_messages_with_substitution when it calls record_failure
    # early (e.g. recorded fallback also malformed). Lets the caller pass the
    # right reason string to _fail_and_notify instead of a generic fallback.
    _substitution_failure_reason: Optional[str] = None

    async def to_request_body(
        self, effective_model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool
    ) -> Dict[str, Any]:
        payload = await super().to_request_body(effective_model_name, max_tokens, ignore_eos, streaming)

        if self.expected_output_is_tool_call and self.tool_definitions:
            if self.override_tool_call_max_tokens:
                payload["ignore_eos"] = False
                # The recorded output_tokens might come from a different model/tokenizer.
                # The replay model may need significantly more tokens to express the
                # same tool call (different tokenizer, different tool-call preamble).
                # Use a generous cap and let ignore_eos=False stop generation naturally.
                payload["max_tokens"] = max(payload.get("max_tokens", 0) * 4, 4096)

            if "tool_choice" in payload:
                logger.warning(
                    f"Event {self.event_id}: payload already has tool_choice={payload['tool_choice']!r}; "
                    f"overwriting with replay-enforced value."
                )
            names = self.expected_output_tool_names or []
            # Build available set from self.tool_definitions (the raw list). The name
            # field is set explicitly in to_request_body and is NOT passed through
            # _clean_parameters, so it survives schema cleaning unchanged.
            available = {t["name"] for t in self.tool_definitions if "name" in t}
            if len(names) == 1 and names[0] in available:
                # Force the exact function the original trace recorded. This maximises
                # faithfulness to the trace and ensures the successor's role:tool messages
                # (which reference this function by name/index) remain coherent.
                payload["tool_choice"] = {"type": "function", "function": {"name": names[0]}}
            else:
                # Fall back to "required" when:
                # - the recorded tool name is not in this call's tool_definitions (the
                #   trace's tool lists don't always match its outputs — vLLM rejects
                #   a tool_choice that names a function not present in tools), or
                # - there were multiple tool calls (vLLM only accepts one name at a time).
                payload["tool_choice"] = "required"

        return payload

    def _extract_session_id(self) -> str:
        return self.event_id.split(":")[0] if ":" in self.event_id else self.event_id

    def _fail_and_notify(self, session_id: str, reason: str) -> None:
        """Mark this event and session as failed, notify the completion queue.

        Called from wait_for_predecessors_and_substitute when we decide to skip
        the request before it reaches the model server (predecessor failed,
        session already failed, or substitution produced an invalid message).
        Mirrors the queue notification logic in process_failure so the main-
        process completion loop is not left waiting indefinitely.
        """
        self.skip_request = True
        was_already_failed = self.worker_tracker.is_session_failed(session_id)
        self.worker_tracker.mark_session_failed(session_id)
        self.registry.record_failure(self.event_id)
        logger.info(f"Event {self.event_id} skipping — {reason}")

        if not was_already_failed and self.completion_queue is not None:
            completion_time = time.perf_counter()
            completed_so_far = self.worker_tracker.get_session_event_count(session_id)
            cancelled = self.total_events_in_session - completed_so_far - 1
            completion_data = {
                "session_id": session_id,
                "completion_time": completion_time,
                "failed": True,
                "failure_reason": reason,
                "cancelled_events": cancelled,
                "event_completion_times": self.worker_tracker.get_session_completion_times(session_id),
            }
            try:
                self.completion_queue.put_nowait(completion_data)
                logger.debug(f"Pushed skip-failure notification for session {session_id} (cancelled_events={cancelled})")
            except Exception as e:
                logger.error(f"Failed to push skip-failure notification for session {session_id}: {e}")

        # This event is now terminal (skipped). Count it toward the worker drain so the
        # session is evicted once its last event drains.
        self._mark_drained_and_maybe_evict(session_id)

    def _mark_drained_and_maybe_evict(self, session_id: str) -> None:
        """Record this event as drained on the worker; evict the session once all drain.

        Every event ends in exactly one terminal state on the worker — completed, skipped,
        or request-failed — and each terminal path calls this exactly once. When the number
        of drained events reaches total_events_in_session, the session is fully done on this
        worker and its lazily-built graph can be freed.

        This is what keeps a worker's resident memory bounded to roughly the concurrent
        working set. Without it, each worker retains every session's graph it ever built for
        the lifetime of the stage (the parent calls cleanup_session, but workers never do),
        so memory grows with the number of sessions processed — catastrophically for large
        corpora or high duplicate_sessions_target, especially when sessions fail fast and the
        session pool churns quickly.

        Eviction is safe here precisely because the session is fully drained: no further
        events for it remain in flight or in the worker's queue, so nothing will try to read
        the freed graph. (A session that fails mid-way still drains every event — the
        successors flow through the skip path and are counted here too.)
        """
        if self.generator is None:
            return
        drained = self.worker_tracker.record_event_drained(session_id, self.event_id)
        if drained >= self.total_events_in_session:
            self.generator.evict_worker_session(session_id)

    async def wait_for_predecessors_and_substitute(self) -> None:
        session_id = self._extract_session_id()

        if self.worker_tracker.is_session_failed(session_id):
            self._fail_and_notify(session_id, "session already failed (pre-wait check)")
            return

        if self.predecessor_event_ids:
            logger.debug(f"Event {self.event_id} waiting for {len(self.predecessor_event_ids)} predecessor(s)")
            try:
                await asyncio.gather(
                    *[self.registry.require_async(event_id, timeout_sec=3600.0) for event_id in self.predecessor_event_ids]
                )
            except EventFailedError:
                self._fail_and_notify(session_id, "predecessor failed")
                return
            except (TimeoutError, asyncio.TimeoutError) as e:
                self._fail_and_notify(session_id, f"predecessor wait failed: {type(e).__name__}")
                return
            logger.debug(f"Event {self.event_id} all predecessors done")

        if self.wait_ms > 0:
            wait_sec = self.wait_ms / 1000.0
            logger.debug(f"Event {self.event_id} waiting {wait_sec:.3f}s (wait_ms={self.wait_ms})")
            await asyncio.sleep(wait_sec)

        # Substitute output segments with actual predecessor outputs, or inject random session ID into unique segments
        needs_substitution = (not self.disable_output_substitution) and any(
            seg.type == "output" or seg.type == "shared" for seg in self.input_segments
        )
        # Inject random string if flag is enabled OR session is a duplicate
        is_duplicate = ReplayGraphSessionGeneratorBase.is_duplicate_session(session_id)
        needs_random_injection = (self.inject_random_session_id or is_duplicate) and any(
            seg.type == "unique" for seg in self.input_segments
        )
        if needs_substitution or needs_random_injection:
            if needs_substitution:
                logger.debug(f"Event {self.event_id} substituting output/shared segments")
            if needs_random_injection:
                reason = "flag enabled" if self.inject_random_session_id else "duplicate session"
                logger.debug(f"Event {self.event_id} injecting random session ID ({reason})")

            substituted = self._build_messages_with_substitution()
            # _build_messages_with_substitution calls record_failure and returns
            # early when substitution is not possible (e.g. tool call expected but
            # live model returned plain text). Detect that and skip the request.
            if self.registry.is_event_failed(self.event_id):
                reason = self._substitution_failure_reason or "Unknown"
                self._fail_and_notify(session_id, reason)
                return
            self.messages = [
                ChatMessage(
                    role=m["role"],
                    content=m.get("content"),
                    reasoning_content=m.get("reasoning") or m.get("reasoning_content"),
                    tool_calls=m.get("tool_calls"),
                    tool_call_id=m.get("tool_call_id"),
                )
                for m in substituted
            ]
            logger.debug(f"Event {self.event_id} substitution/injection complete, {len(self.messages)} messages")

    def _build_messages_with_substitution(self) -> List[Dict[str, Any]]:
        # NOTE: when input_segments is empty, the original_messages list is returned
        # by reference (not copied). Callers must not mutate the returned list.
        if not self.input_segments:
            return self.original_messages

        result: List[Dict[str, Any]] = []
        cursor = 0

        # Track where live tool-call assistant messages were inserted so we can
        # rewrite the tool_call_id values in the role:tool messages that follow.
        # Each entry is (index_of_assistant_in_result, live_tool_calls_list).
        #
        # We use a post-pass rather than rewriting inline because the role:tool
        # messages live in a later segment ("unique") that hasn't been added to
        # `result` yet when we process the "output" segment.
        #
        # index_of_assistant_in_result == len(result) at the time of the append,
        # which equals the index the message will occupy after the append.
        pending_id_rewrites: List[Tuple[int, List[Dict[str, Any]]]] = []

        for seg in self.input_segments:
            seg_msgs = self.original_messages[cursor : cursor + seg.message_count]

            if seg.type == "output":
                # An output segment must cover exactly one message — the assistant
                # turn that will be replaced by the predecessor's live output.
                if seg.message_count != 1:
                    logger.error(
                        f"Event {self.event_id}: output segment has message_count={seg.message_count} "
                        f"(expected 1). Using recorded messages to avoid index corruption."
                    )
                    result.extend(seg_msgs)
                    cursor += seg.message_count
                    continue

                if seg.source_event_id:
                    actual_message = self.registry.get_message_by_event_id(seg.source_event_id)
                    if actual_message:
                        live_tool_calls = actual_message.get("tool_calls")
                        # Inline detection: if the live model produced tool_calls
                        # AND bad_tool_call_handling is enabled, check each
                        # `arguments` for valid JSON. Any failure means the next
                        # request would 400 on the chat template's json.loads.
                        bad_live = (
                            _detect_bad_tool_calls(live_tool_calls)
                            if (self.bad_tool_call_handling == BadToolCallHandling.USE_RECORDED and live_tool_calls)
                            else []
                        )
                        if bad_live:
                            # Substitute the recorded assistant message at this slot.
                            # The recorded message is `seg_msgs[0]` (output segments
                            # cover exactly one message — the assistant turn). Its
                            # `tool_call_id` flows naturally into the role:tool
                            # successors that follow in `original_messages`, so no
                            # entry is appended to pending_id_rewrites for this
                            # position. The wire body for the demoted slot is
                            # structurally identical to a healthy replay.
                            recorded_message = seg_msgs[0]
                            recorded_tool_calls = recorded_message.get("tool_calls") or []
                            # Defensive: if the recorded trace ALSO has malformed
                            # tool_calls at this slot, we have no clean fallback.
                            # Hard-fail the event; EventFailedError will cascade
                            # to downstream events that await this one. Parallel
                            # DAG branches continue.
                            bad_recorded = _detect_bad_tool_calls(recorded_tool_calls)
                            if bad_recorded:
                                logger.error(
                                    f"Event {self.event_id}: bad_tool_call_handling=use_recorded "
                                    f"detected malformed live tool_calls from {seg.source_event_id}, "
                                    f"but the recorded fallback is also malformed "
                                    f"(errors={[e for (_, _, e) in bad_recorded]}). Failing event."
                                )
                                self._substitution_failure_reason = (
                                    f"recorded fallback for {seg.source_event_id} is also malformed"
                                )
                                self.registry.record_failure(self.event_id)
                                return result  # partial; caller checks is_event_failed
                            # Tag the predecessor event_id for telemetry. Set
                            # semantics dedupe across DAG fan-out.
                            session_id = self._extract_session_id()
                            pred_event_id = (
                                seg.source_event_id.split(":", 1)[1] if ":" in seg.source_event_id else seg.source_event_id
                            )
                            self.worker_tracker.record_recorded_substitution(session_id, pred_event_id)
                            result.append(recorded_message)
                            logger.warning(
                                f"Event {self.event_id}: substituted RECORDED message for "
                                f"{seg.source_event_id} (live had {len(bad_live)} malformed "
                                f"tool_call(s); recorded structurally clean)"
                            )
                        else:
                            if live_tool_calls:
                                # Record the position of this assistant message so the post-pass
                                # can rewrite tool_call_id in the role:tool messages that follow.
                                pending_id_rewrites.append((len(result), live_tool_calls))
                            result.append(actual_message)
                            logger.debug(
                                f"Event {self.event_id}: substituted output segment with structured message from {seg.source_event_id}"
                            )
                    else:
                        actual_output = self.registry.get_output_by_event_id(seg.source_event_id)
                        logger.debug(
                            f"Registry get for event {self.event_id} output segment from {seg.source_event_id} generated: {actual_output}"
                        )
                        logger.warning(
                            f"Event {self.event_id}: unable to get the actual output message from {seg.source_event_id}. "
                            f"Using output text instead. "
                        )
                        if actual_output:
                            if self.expected_output_is_tool_call:
                                # The live model returned plain text where a tool call was
                                # expected (tool_choice was either absent or ignored). The
                                # successor's role:tool messages will have dangling
                                # tool_call_id references and the model server will likely
                                # reject the next request. Treat this event as failed so
                                # downstream events skip rather than send broken requests.
                                logger.warning(
                                    f"Event {self.event_id}: original output was a tool call but live model "
                                    f"returned plain text. Marking event as failed to prevent downstream "
                                    f"requests with dangling tool_call_id references."
                                )
                                self._substitution_failure_reason = (
                                    "substitution failed (tool call expected but plain text returned)"
                                )
                                self.registry.record_failure(self.event_id)
                                return result  # partial result; caller should check skip_request
                            for msg in seg_msgs:
                                substituted = dict(msg)
                                substituted["content"] = actual_output
                                result.append(substituted)
                            logger.debug(
                                f"Event {self.event_id}: substituted output segment with text output from {seg.source_event_id}"
                            )
                        else:
                            logger.debug(
                                f"Event {self.event_id}: output segment from {seg.source_event_id} "
                                f"not available, using recorded content"
                            )
                            result.extend(seg_msgs)
                else:
                    logger.debug(f"Event {self.event_id}: output segment has no source_event_id, using recorded content")
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
                    # Only take the first seg.message_count messages from the parent.
                    # The shared segment represents a prefix of the parent's messages,
                    # not necessarily all of them. This handles cases where the parent
                    # has more messages than the shared prefix length.
                    seg_msgs_from_parent = seg_msgs_from_parent[: seg.message_count]

                    logger.debug(
                        f"Registry get for event {self.event_id} from {seg.source_event_id} "
                        f"shared segment: using {len(seg_msgs_from_parent)} messages (prefix of parent's messages)"
                    )

                    # Validate that we have the expected number of messages after slicing
                    if len(seg_msgs_from_parent) != seg.message_count:
                        logger.warning(
                            f"Event {self.event_id} shared segment from {seg.source_event_id} "
                            f"expected {seg.message_count} messages but parent only has {len(seg_msgs_from_parent)}. "
                            f"Using recorded messages as fallback."
                        )
                        result.extend(seg_msgs)
                    else:
                        for msg in seg_msgs_from_parent:
                            if isinstance(msg, ChatMessage):
                                result.append({k: v for k, v in msg.model_dump().items() if v is not None})
                            else:
                                result.append(dict(msg))
            elif seg.type == "unique":
                # Unique message - inject random session string if:
                # 1. inject_random_session_id flag is enabled, OR
                # 2. Session is a duplicate (matches pattern: {id}_dup{number})
                for msg in seg_msgs:
                    session_id = self._extract_session_id()
                    is_duplicate = ReplayGraphSessionGeneratorBase.is_duplicate_session(session_id)
                    should_inject = (self.inject_random_session_id or is_duplicate) and self.session_random_string

                    if should_inject:
                        # Use the session random string passed from SessionGraphState
                        # Inject random string into message content
                        msg_copy = dict(msg)
                        original_content = msg_copy.get("content", "")

                        # Prepend random session identifier to content
                        msg_copy["content"] = f"[SESS:{self.session_random_string}] {original_content}"
                        result.append(msg_copy)
                        reason = "flag enabled" if self.inject_random_session_id else "duplicate session"
                        logger.debug(f"Event {self.event_id}: injected random session string ({reason})")
                    else:
                        result.append(msg)
            else:
                result.extend(seg_msgs)

            cursor += seg.message_count

        # Post-pass: rewrite tool_call_id values in role:tool messages so they match
        # the live tool call IDs instead of the recorded (now stale) ones.
        #
        # Why by index rather than by name: the live model may call the same function
        # twice, making name-based matching ambiguous. Index is unambiguous — the i-th
        # role:tool message corresponds to the i-th tool call in the preceding assistant
        # message (guaranteed by the OpenAI spec).
        #
        # We scan forward from the assistant message and rewrite only role:tool messages
        # that carry a tool_call_id, skipping any intervening non-tool messages
        # (which are valid in some trace formats).
        for assistant_idx, live_tool_calls in pending_id_rewrites:
            tool_result_idx = 0
            for result_idx in range(assistant_idx + 1, len(result)):
                if tool_result_idx >= len(live_tool_calls):
                    break
                msg = result[result_idx]
                if msg.get("role") == "tool":
                    live_id = live_tool_calls[tool_result_idx].get("id")
                    if live_id:
                        msg = dict(msg)  # copy before mutating — the dict may be shared
                        msg["tool_call_id"] = live_id
                        result[result_idx] = msg
                        logger.debug(
                            f"Event {self.event_id}: rewrote tool_call_id at position {result_idx} "
                            f"to live ID {live_id!r} (index {tool_result_idx})"
                        )
                    tool_result_idx += 1

        return result

    def on_completion(self, info: InferenceInfo) -> None:
        output_text = info.output_text if isinstance(info, SessionInferenceInfo) else ""
        output_text = output_text or ""
        output_message = info.output_message if isinstance(info, SessionInferenceInfo) else None
        self.registry.record(self.event_id, output_text, self.messages, output_message=output_message)
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
            # Telemetry: only emit recorded-substitution keys when at least
            # one substitution fired in this session, so upstream-default runs
            # (handling=none, or handling set but no malformed tool_calls
            # observed) produce an identical wire format.
            recorded_subst_ids = self.worker_tracker.get_session_recorded_substitution_event_ids(session_id)
            if recorded_subst_ids:
                completion_data["recorded_substitution_event_ids"] = recorded_subst_ids
                completion_data["n_recorded_substitutions"] = len(recorded_subst_ids)

            if self.completion_queue is not None:
                try:
                    self.completion_queue.put_nowait(completion_data)
                    logger.debug(f"Pushed session {session_id} completion to queue")
                except Exception as e:
                    logger.error(f"Failed to push session {session_id} completion to queue: {e}")

        # This event is now terminal (completed). Count it toward the worker drain and evict
        # the session once its last event drains. Done last: eviction clears worker_tracker
        # state for the session, so the completion-queue notification above must be built first.
        self._mark_drained_and_maybe_evict(session_id)

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

        def _get_text(content: Any) -> str:
            if isinstance(content, str):
                return content
            if isinstance(content, list):
                return "".join(
                    [
                        item.get("text", "")
                        for item in content
                        if isinstance(item, dict) and item.get("type") in ("text", "input_text")
                    ]
                )
            return ""

        if config.streaming:
            # Accumulate tool_call chunks and reasoning_content alongside text content.
            # delta.tool_calls is a list of partial objects; each chunk carries an
            # index that identifies which tool call it belongs to.
            tool_call_chunks: Dict[int, Dict[str, Any]] = {}
            reasoning_content_chunks: list[str] = []

            def _extract_streaming_content(data: Dict[str, Any]) -> Optional[str]:
                delta = data.get("choices", [{}])[0].get("delta", {})
                for chunk in delta.get("tool_calls") or []:
                    idx = chunk.get("index", 0)
                    if idx not in tool_call_chunks:
                        tool_call_chunks[idx] = {
                            "id": chunk.get("id", ""),
                            "type": chunk.get("type", "function"),
                            "function": {"name": "", "arguments": ""},
                        }
                    fn = chunk.get("function") or {}
                    if fn.get("name"):
                        tool_call_chunks[idx]["function"]["name"] += fn["name"]
                    if fn.get("arguments"):
                        tool_call_chunks[idx]["function"]["arguments"] += fn["arguments"]
                    if chunk.get("id"):
                        tool_call_chunks[idx]["id"] = chunk["id"]

                # Accumulate reasoning chunks (prefer "reasoning", fall back to "reasoning_content")
                reasoning_chunk = delta.get("reasoning") or delta.get("reasoning_content")
                if reasoning_chunk is not None:
                    reasoning_content_chunks.append(str(reasoning_chunk))

                content = delta.get("content")
                return str(content) if content is not None else None

            text_content, chunk_times, raw_content, response_chunks, server_usage = await parse_sse_stream(
                response, extract_content=_extract_streaming_content
            )

            # Combine reasoning_content with text_content in output_text (used for token count)
            reasoning_text = "".join(reasoning_content_chunks) if reasoning_content_chunks else ""
            if reasoning_text:
                output_text = reasoning_text + text_content
            else:
                output_text = text_content

            streaming_output_message: Optional[Dict[str, Any]] = None
            if tool_call_chunks:
                live_tool_calls = [tool_call_chunks[i] for i in sorted(tool_call_chunks)]
                streaming_output_message = {"role": "assistant", "tool_calls": live_tool_calls}
                if text_content:
                    streaming_output_message["content"] = text_content
            else:
                streaming_output_message = {"role": "assistant", "content": text_content}

            if reasoning_text:
                streaming_output_message["reasoning_content"] = reasoning_text

            prompt_text = "".join([_get_text(msg.content) for msg in self.messages if msg.content])
            prompt_len = tokenizer.count_tokens(prompt_text)
            server_completion_tokens = server_usage.get("completion_tokens") if server_usage else None
            if server_completion_tokens is not None:
                output_len = int(server_completion_tokens)
            else:
                tc_text = ""
                if tool_call_chunks:
                    tc_text = json.dumps([tool_call_chunks[i] for i in sorted(tool_call_chunks)], ensure_ascii=False)
                output_len = tokenizer.count_tokens(output_text + tc_text)
            info = SessionInferenceInfo(
                request_metrics=RequestMetrics(text=Text(input_tokens=prompt_len)),
                response_metrics=StreamedResponseMetrics(
                    response_chunks=response_chunks,
                    chunk_times=chunk_times,
                    output_tokens=output_len,
                    output_token_times=chunk_times,
                    server_usage=server_usage,
                ),
                lora_adapter=lora_adapter,
                output_text=output_text or None,
                output_message=streaming_output_message,
                extra_info={"raw_response": raw_content},
            )
        else:
            data = await response.json()
            prompt_len = tokenizer.count_tokens("".join([_get_text(m.content) for m in self.messages]))
            choices = data.get("choices", [])
            output_message: Optional[Dict[str, Any]] = None
            tool_calls = None
            if choices:
                msg_dict = choices[0].get("message", {})
                text_content = msg_dict.get("content", "") or ""
                tool_calls = msg_dict.get("tool_calls")
                reasoning_content = msg_dict.get("reasoning") or msg_dict.get("reasoning_content")

                # Combine reasoning_content with output text for the content field
                if reasoning_content:
                    output_text = reasoning_content + text_content
                else:
                    output_text = text_content

                if tool_calls:
                    output_message = {"role": "assistant", "tool_calls": tool_calls}
                    if text_content:
                        output_message["content"] = text_content
                else:
                    output_message = {"role": "assistant", "content": text_content}

                if reasoning_content:
                    output_message["reasoning_content"] = reasoning_content
            usage = data.get("usage") or {}
            server_completion_tokens = usage.get("completion_tokens")
            if server_completion_tokens is not None:
                output_len = int(server_completion_tokens)
            else:
                tc_text = ""
                if tool_calls:
                    tc_text = json.dumps(tool_calls, ensure_ascii=False)
                output_len = tokenizer.count_tokens(output_text + tc_text)
            info = SessionInferenceInfo(
                request_metrics=RequestMetrics(text=Text(input_tokens=prompt_len)),
                response_metrics=UnaryResponseMetrics(output_tokens=output_len),
                lora_adapter=lora_adapter,
                output_text=output_text or None,
                output_message=output_message,
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

        # This event is now terminal (request failed). Count it toward the worker drain. Note
        # that when an event fails, its successors are still queued; they will be dequeued and
        # flow through the skip path, each counted there. Eviction therefore only fires when
        # the last of them drains — after which nothing re-reads the session — so the graph is
        # never rebuilt by a late successor.
        self._mark_drained_and_maybe_evict(session_id)

        return SessionInferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=0)),
            response_metrics=UnaryResponseMetrics(output_tokens=0),
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
    failure_reason: Optional[str] = None
    cancelled_events: int = 0
    # Populated from completion_data when the worker finishes the
    # session. None means the worker did not push the keys (i.e.
    # bad_tool_call_handling=none); empty list means handling was
    # enabled but no substitutions fired.
    n_recorded_substitutions: Optional[int] = None
    recorded_substitution_event_ids: Optional[List[str]] = None
    random_string: Optional[str] = None  # Random string for KV-cache invalidation (shared by all events in session)


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
    tool_definitions: Optional[List[Dict[str, Any]]] = None


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
        replay_config: Optional[SessionReplayConfig] = None,
    ) -> None:
        super().__init__(api_config, config, tokenizer)
        self.config = config
        self.replay_config = replay_config
        self.mp_manager = mp_manager
        self.num_workers = max(1, num_workers)
        self.base_seed = base_seed if base_seed is not None else 42

        self.output_registry = EventOutputRegistry()
        self.worker_tracker = WorkerSessionTracker()
        if mp_manager is not None:
            self.session_completion_queue: Any = mp_manager.Queue()
        else:
            self.session_completion_queue = None

        self.sessions: List[Optional[ReplaySession]] = []
        self._session_ids: List[str] = []
        self._session_id_to_index: Dict[str, int] = {}
        self.session_graph_state: Dict[str, ReplaySessionState] = {}
        # Per-session event lists keyed by session_index. Populated on demand by
        # _ensure_session_built (lazy path) or all at once by initialize_sessions (eager path).
        self._session_events: Dict[int, List[ReplaySessionEvent]] = {}
        # Flat list kept only for the eager initialize_sessions() path / back-compat.
        self.all_events: List[ReplaySessionEvent] = []
        self._skipped_session_count: int = 0

    def initialize_sessions(self, sessions: List[ReplaySession]) -> None:
        """Finalize generator state from fully-built sessions (eager path)."""
        # Duplicate sessions if needed to meet total session requirements
        if self.replay_config and self.replay_config.duplicate_sessions_target is not None:
            sessions = self._duplicate_sessions_if_needed(sessions, self.replay_config.duplicate_sessions_target)

        if not sessions:
            raise ValueError("No valid replay sessions found")

        # Assign session_index to match position in list after shuffling/duplication
        for i, session in enumerate(sessions):
            session.session_index = i

        self.sessions = list(sessions)
        self._build_replay_schedule()
        logger.info(
            "Built replay schedule: %d events across %d sessions (eager)",
            len(self.all_events),
            len(self.sessions),
        )

    def initialize_sessions_lazy(self, session_ids: List[str]) -> None:
        """Set up placeholder slots for on-demand graph building (lazy path).

        Allocates None session slots and records their IDs so get_session_count()
        works immediately. Each session's graph is built later by _ensure_session_built,
        triggered the first time the session is dispatched (parent) or replayed (worker).
        """
        if not session_ids:
            raise ValueError("No valid trace records found after filtering")
        self.sessions = [None] * len(session_ids)
        self._session_ids = list(session_ids)
        self._session_id_to_index = {sid: i for i, sid in enumerate(session_ids)}
        self._session_events = {}
        self.all_events = []
        logger.info("Lazy init: %d session slots allocated", len(session_ids))

    def _build_session(self, session_index: int) -> Optional[ReplaySession]:
        """Build the ReplaySession for one slot. Implemented by lazy subclasses."""
        raise NotImplementedError("Lazy generators must implement _build_session()")

    def _ensure_session_built(self, session_index: int) -> None:
        """Build and register session_index's graph if not already done. Idempotent."""
        if session_index < 0 or session_index >= len(self.sessions):
            raise IndexError(f"Session index {session_index} out of range (total: {len(self.sessions)})")
        if self.sessions[session_index] is not None or session_index in self._session_events:
            return
        session = self._build_session(session_index)
        if session is None:
            # No graph (e.g. malformed spans, or all calls errored with include_errors=False).
            # Register an empty event list as the "already attempted" sentinel so this slot is
            # not retried on subsequent calls. The dispatcher skips it (see is_session_buildable).
            self._session_events[session_index] = []
            self._skipped_session_count += 1
            return
        session.session_index = session_index
        self.sessions[session_index] = session
        events = self._build_session_schedule(session)
        self._session_events[session_index] = events
        logger.debug("Built session %s: %d events", session.session_id, len(events))

    def is_session_buildable(self, session_index: int) -> bool:
        """Return True if session_index has (or can build) a graph, False if it produced none.

        Builds the session on demand (idempotent). Lets the dispatcher skip un-buildable
        slots without raising. Skipped sessions are not reported anywhere (see dispatch_session).
        """
        self._ensure_session_built(session_index)
        session = self.sessions[session_index]
        if session is None:
            logger.warning(f"Skipping session {session_index} ({self._session_ids[session_index]!r}): no graph could be built")
            return False
        events = self._session_events.get(session_index, [])
        if not events:
            logger.warning(
                f"Skipping session {session_index} ({self._session_ids[session_index]!r}): graph built but no schedulable events"
            )
            self._skipped_session_count += 1
            return False
        logger.info("Dispatching session %s: %d events", session.session_id, len(events))
        return True

    @staticmethod
    def _duplicate_sessions_if_needed(sessions: List[ReplaySession], target_sessions: int) -> List[ReplaySession]:
        """Duplicate sessions to ensure we have enough for high-concurrency testing.

        This is useful when the trace corpus is smaller than needed for stress testing.
        Sessions are duplicated with unique IDs to avoid conflicts.

        Args:
            sessions: List of sessions to potentially duplicate
            target_sessions: Target number of sessions to reach by duplication

        Returns:
            List of sessions (original + duplicates if needed)
        """
        current_count = len(sessions)

        if current_count >= target_sessions:
            logger.info(f"Session corpus sufficient: {current_count} sessions available (target: {target_sessions})")
            return sessions

        # Calculate how many duplicates we need
        duplicates_needed = target_sessions - current_count
        logger.warning(
            f"Session corpus small: {current_count} sessions available. "
            f"Duplicating to reach {target_sessions} sessions for stress testing."
        )

        # Duplicate sessions in round-robin fashion
        original_sessions = list(sessions)
        duplicate_count = 0
        session_idx = 0

        while len(sessions) < target_sessions:
            # Get next session to duplicate (round-robin)
            source_session = original_sessions[session_idx % len(original_sessions)]
            session_idx += 1
            duplicate_count += 1

            # Create duplicate with unique ID
            # Note: session_index will be reassigned in initialize_sessions() to match list position
            duplicate_session = ReplaySession(
                session_id=f"{source_session.session_id}_dup{duplicate_count}",
                source_id=source_session.source_id,
                session_index=-1,  # Placeholder, will be reassigned after duplication
                graph=source_session.graph,
                start_offset_ms=source_session.start_offset_ms,
            )

            sessions.append(duplicate_session)

        logger.info(f"Duplicated {duplicates_needed} sessions. Total sessions now: {len(sessions)}")
        return sessions

    @staticmethod
    def is_duplicate_session(session_id: str) -> bool:
        """Check if a session is a duplicate based on its ID.

        Duplicates are created with the pattern: {original_id}_dup{number}
        This method uses regex to robustly detect this pattern.

        Args:
            session_id: The session ID to check

        Returns:
            True if the session is a duplicate, False otherwise
        """
        # Match pattern: anything followed by _dup and one or more digits at the end
        return bool(re.search(r"_dup\d+$", session_id))

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Chat]

    def is_preferred_worker_requested(self) -> bool:
        return True

    def _build_replay_schedule(self) -> None:
        """Build the full schedule from self.sessions (eager path / tests)."""
        self.all_events = []
        self._session_events = {}
        self._session_ids = [s.session_id for s in self.sessions if s is not None]
        self._session_id_to_index = {sid: i for i, sid in enumerate(self._session_ids)}
        for session in self.sessions:
            if session is None:
                continue
            events = self._build_session_schedule(session)
            self._session_events[session.session_index] = events
            self.all_events.extend(events)

    def _build_session_schedule(self, session: ReplaySession) -> List[ReplaySessionEvent]:
        """Register one session's graph state and return its qualified events.

        Builds the per-session ReplaySessionState (incl. the KV-cache-invalidation
        random_string, which is generated for duplicate sessions or when
        inject_random_session_id is set) and stores it in session_graph_state.
        """
        random_string = None
        is_duplicate = ReplayGraphSessionGeneratorBase.is_duplicate_session(session.session_id)
        if (self.replay_config and self.replay_config.inject_random_session_id) or is_duplicate:
            random_string = uuid.uuid4().hex[:16]
            logger.debug(f"Generated random string for session {session.session_id}: {random_string}")

        state = ReplaySessionState(
            session_id=session.session_id,
            graph=session.graph,
            ready_events=set(),
            dispatched_events=set(),
            completed_events=set(),
            event_completion_times={},
            is_active=False,
            is_complete=False,
            random_string=random_string,
        )
        self.session_graph_state[session.session_id] = state

        events: List[ReplaySessionEvent] = []
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

            events.append(
                ReplaySessionEvent(
                    call_id=gc.call_id,
                    event_id=qualified_event_id,
                    session_index=session.session_index,
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
                    wait_ms=min(event.wait_ms, self.replay_config.max_wait_ms) if self.replay_config else event.wait_ms,
                    tool_definitions=gc.tool_definitions,
                )
            )
        return events

    def get_session_count(self) -> int:
        return len(self._session_ids)

    def _get_session(self, session_index: int) -> ReplaySession:
        """Return the built session at session_index, building it on demand if needed."""
        if session_index < 0 or session_index >= len(self._session_ids):
            raise IndexError(f"Session index {session_index} out of range (total: {len(self._session_ids)})")
        self._ensure_session_built(session_index)
        session = self.sessions[session_index]
        if session is None:
            raise RuntimeError(f"Session {session_index} ({self._session_ids[session_index]!r}) produced no graph")
        return session

    def get_session_event_indices(self, session_index: int) -> List[int]:
        self._ensure_session_built(session_index)
        return list(range(len(self._session_events.get(session_index, []))))

    def get_session_info(self, session_index: int) -> Dict[str, Any]:
        session = self._get_session(session_index)
        num_events = len(self._session_events.get(session_index, []))
        return {
            "session_id": session.session_id,
            "file_path": session.source_id,
            "source_id": session.source_id,
            "session_index": session.session_index,
            "num_events": num_events,
            "num_graph_events": len(session.graph.events),
            "start_offset_ms": session.start_offset_ms,
        }

    def get_session_events(self, session_index: int) -> List[LazyLoadInferenceAPIData]:
        session = self._get_session(session_index)
        events = self._session_events.get(session_index, [])
        session_worker_id = abs(hash(session.session_id)) % self.num_workers
        return [
            SessionReplayLazyLoadData(
                session_index=session_index,
                local_event_index=local,
                preferred_worker_id=session_worker_id,
            )
            for local in range(len(events))
        ]

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
            if session is not None and session.session_id == session_id:
                source_id = session.source_id
                break

        session_index = self._session_id_to_index.get(session_id)
        num_events = (
            len(self._session_events[session_index])
            if session_index is not None and session_index in self._session_events
            else len(state.graph.events)
        )
        num_events_completed = len(state.completed_events)
        num_events_cancelled = state.cancelled_events if state.failed else 0

        error = None
        if state.failure_reason:
            error = ErrorResponseInfo(error_type="SessionReplayError", error_msg=state.failure_reason)

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
            error=error,
            n_recorded_substitutions=state.n_recorded_substitutions,
            recorded_substitution_event_ids=state.recorded_substitution_event_ids,
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
                    completed_state.failure_reason = completion_data.get("failure_reason")
                    completed_state.cancelled_events = completion_data.get("cancelled_events", 0)
                    # Bad tool-call handling telemetry. The two keys are
                    # gated worker-side behind `len(...) > 0`, so their
                    # absence here is meaningful (no substitution path
                    # exercised) and we propagate that absence to the
                    # session metric as None.
                    if "n_recorded_substitutions" in completion_data:
                        completed_state.n_recorded_substitutions = completion_data["n_recorded_substitutions"]
                        completed_state.recorded_substitution_event_ids = completion_data.get(
                            "recorded_substitution_event_ids", []
                        )
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

    def _resolve_event(self, data: LazyLoadInferenceAPIData) -> ReplaySessionEvent:
        """Resolve a queued lazy item to its ReplaySessionEvent.

        Per-session addressing (lazy path): build session_index on demand, index
        local_event_index. Legacy global addressing (eager path): index all_events.
        """
        if isinstance(data, SessionReplayLazyLoadData):
            self._ensure_session_built(data.session_index)
            events = self._session_events.get(data.session_index, [])
            if data.local_event_index >= len(events):
                raise IndexError(
                    f"Local event index {data.local_event_index} out of range for session {data.session_index} (total: {len(events)})"
                )
            return events[data.local_event_index]

        n = data.data_index
        if n < 0 or n >= len(self.all_events):
            raise IndexError(f"Event index {n} out of range (total: {len(self.all_events)})")
        return self.all_events[n]

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> SessionChatCompletionAPIData:
        event = self._resolve_event(data)

        chat_messages = []
        original_messages: List[Dict[str, Any]] = []
        for msg in event.messages:
            if isinstance(msg, dict):
                role = msg.get("role", "user")
                content = msg.get("content", "")
                tool_calls = msg.get("tool_calls")
                tool_call_id = msg.get("tool_call_id")
                reasoning_content = msg.get("reasoning") or msg.get("reasoning_content")
            else:
                role = getattr(msg, "role", "user")
                content = getattr(msg, "text", "")
                tool_calls = getattr(msg, "tool_calls", None)
                tool_call_id = getattr(msg, "tool_call_id", None)
                reasoning_content = getattr(msg, "reasoning", None) or getattr(msg, "reasoning_content", None)

            if tool_calls is not None:
                # Preserve any assistant preamble content alongside the tool calls
                # so chat_messages (which becomes the wire payload when no
                # substitution/injection runs) is not lossy.
                tc_content = content if isinstance(content, str) and content else None
                chat_messages.append(
                    ChatMessage(role=role, content=tc_content, tool_calls=tool_calls, reasoning_content=reasoning_content)
                )
                tc_msg: Dict[str, Any] = {"role": role, "tool_calls": tool_calls}
                if tc_content is not None:
                    tc_msg["content"] = tc_content
                if reasoning_content:
                    tc_msg["reasoning_content"] = reasoning_content
                original_messages.append(tc_msg)
                continue

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
            # Carry tool_call_id onto the ChatMessage too (not just original_messages),
            # so role:tool linkage survives on the wire when substitution is disabled.
            # tool_call_id belongs on a role:tool message only — don't leak it onto
            # user/assistant messages even if the recorded message carried one.
            wire_tool_call_id = tool_call_id if role == "tool" else None
            chat_messages.append(
                ChatMessage(
                    role=role, content=content_str, tool_call_id=wire_tool_call_id, reasoning_content=reasoning_content
                )
            )
            orig_msg: Dict[str, Any] = {"role": role, "content": content_str}
            if wire_tool_call_id is not None:
                orig_msg["tool_call_id"] = wire_tool_call_id
            if reasoning_content:
                orig_msg["reasoning_content"] = reasoning_content
            original_messages.append(orig_msg)

        max_tokens = event.expected_output_tokens
        session_id = event.event_id.split(":")[0] if ":" in event.event_id else event.event_id
        raw_event_id = event.event_id.split(":", 1)[1] if ":" in event.event_id else event.event_id
        state = self.session_graph_state.get(session_id)
        session_index = event.session_index
        total_events = len(self._session_events.get(session_index, [])) if state else 0

        gc = state.graph.events[raw_event_id].call if state and raw_event_id in state.graph.events else None

        return SessionChatCompletionAPIData(
            messages=chat_messages,
            max_tokens=max_tokens,
            tool_definitions=event.tool_definitions,
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
            expected_output_is_tool_call=gc.expected_output_is_tool_call if gc else False,
            expected_output_tool_names=gc.expected_output_tool_names if gc else None,
            otel_context=data.otel_context,
            session_id=data.session_id,
            preferred_worker_id=data.preferred_worker_id,
            # Pass KV-cache invalidation configuration and session random string
            inject_random_session_id=self.replay_config.inject_random_session_id if self.replay_config else False,
            session_random_string=state.random_string if state else None,
            override_tool_call_max_tokens=self.replay_config.override_tool_call_max_tokens if self.replay_config else False,
            # Mitigation knob: read once per event from replay_config. Default
            # NONE keeps the wire format byte-identical to upstream main.
            bad_tool_call_handling=getattr(self.replay_config, "bad_tool_call_handling", BadToolCallHandling.NONE)
            if self.replay_config
            else BadToolCallHandling.NONE,
            disable_output_substitution=getattr(self.replay_config, "disable_output_substitution", False)
            if self.replay_config
            else False,
            # Back-reference so the event can evict this session from the worker once drained.
            generator=self,
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
            self.output_registry._event_output_message.pop(qualified_event_id, None)
            self.output_registry._event_input_messages.pop(qualified_event_id, None)
            self.output_registry._event_signals.pop(qualified_event_id, None)
            self.output_registry._failed_event_ids.discard(qualified_event_id)

        del self.session_graph_state[session_id]

        # Free the lazily-built graph and event list so a completed session's memory is
        # released (it was only retained between dispatch and completion). Without this the
        # parent accumulates every dispatched session's graph for the whole run, which blows
        # up RAM for large corpora / high duplicate_sessions_target. Re-access (if it ever
        # happened) would rebuild on demand via _ensure_session_built.
        idx = self._session_id_to_index.get(session_id)
        if idx is not None:
            if idx < len(self.sessions):
                self.sessions[idx] = None
            self._session_events.pop(idx, None)

        logger.debug("Cleaned up session %s: removed %d events from memory", session_id, event_count)

    def evict_worker_session(self, session_id: str) -> None:
        """Free a fully-drained session's memory inside a worker process.

        The parent process frees sessions via cleanup_session in its dispatch loop, but
        workers never call cleanup_session — so in the lazy path each worker would retain
        every graph it ever built for the whole stage. This is called by the data object
        (_mark_drained_and_maybe_evict) when a session's last event drains on this worker,
        making per-worker memory track the concurrent working set instead of the full corpus.

        Reuses cleanup_session to drop the built graph, event list, traversal state, and
        output-registry entries, then clears the per-worker WorkerSessionTracker bookkeeping
        for the session (which cleanup_session does not touch).
        """
        self.cleanup_session(session_id)
        tracker = getattr(self, "worker_tracker", None)
        if tracker is not None:
            tracker.forget_session(session_id)
        logger.debug("Worker evicted fully-drained session %s", session_id)
