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
    output_message: Optional[Dict[str, Any]] = None


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

    async def to_payload(
        self, effective_model_name: str, max_tokens: int, ignore_eos: bool, streaming: bool
    ) -> Dict[str, Any]:
        payload = await super().to_payload(effective_model_name, max_tokens, ignore_eos, streaming)

        if self.expected_output_is_tool_call and self.tool_definitions:
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
            # field is set explicitly in to_payload and is NOT passed through
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
                "cancelled_events": cancelled,
                "event_completion_times": self.worker_tracker.get_session_completion_times(session_id),
            }
            try:
                self.completion_queue.put_nowait(completion_data)
                logger.debug(f"Pushed skip-failure notification for session {session_id} (cancelled_events={cancelled})")
            except Exception as e:
                logger.error(f"Failed to push skip-failure notification for session {session_id}: {e}")

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
            logger.debug(f"Event {self.event_id} all predecessors done")

        if self.wait_ms > 0:
            wait_sec = self.wait_ms / 1000.0
            logger.debug(f"Event {self.event_id} waiting {wait_sec:.3f}s (wait_ms={self.wait_ms})")
            await asyncio.sleep(wait_sec)

        # Substitute output segments with actual predecessor outputs, or inject random session ID into unique segments
        needs_substitution = any(seg.type == "output" or seg.type == "shared" for seg in self.input_segments)
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
                self._fail_and_notify(session_id, "substitution failed (tool call expected but plain text returned)")
                return
            self.messages = [
                ChatMessage(
                    role=m["role"],
                    content=m.get("content"),
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
                    # Only take the first seg.message_count messages from the parent.
                    # The shared segment represents a prefix of the parent's messages,
                    # not necessarily all of them. This handles cases where the parent
                    # has more messages than the shared prefix length.
                    seg_msgs_from_parent = seg_msgs_from_parent[:seg.message_count]
                    
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
            # Accumulate tool_call chunks alongside text content.
            # delta.tool_calls is a list of partial objects; each chunk carries an
            # index that identifies which tool call it belongs to.
            tool_call_chunks: Dict[int, Dict[str, Any]] = {}

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
                content = delta.get("content")
                return str(content) if content is not None else None

            output_text, chunk_times, raw_content, response_chunks, server_usage = await parse_sse_stream(
                response, extract_content=_extract_streaming_content
            )

            streaming_output_message: Optional[Dict[str, Any]] = None
            if tool_call_chunks:
                live_tool_calls = [tool_call_chunks[i] for i in sorted(tool_call_chunks)]
                streaming_output_message = {"role": "assistant", "tool_calls": live_tool_calls}
                if output_text:
                    streaming_output_message["content"] = output_text
            else:
                streaming_output_message = {"role": "assistant", "content": output_text}

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
                output_text = msg_dict.get("content", "") or ""
                tool_calls = msg_dict.get("tool_calls")
                if tool_calls:
                    output_message = {"role": "assistant", "tool_calls": tool_calls}
                    if output_text:
                        output_message["content"] = output_text
                else:
                    output_message = {"role": "assistant", "content": output_text}
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
    cancelled_events: int = 0
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

        self.sessions: List[ReplaySession] = []
        self.session_graph_state: Dict[str, ReplaySessionState] = {}
        self.all_events: List[ReplaySessionEvent] = []

    def initialize_sessions(self, sessions: List[ReplaySession]) -> None:
        """Finalize generator state from prepared sessions."""
        # Duplicate sessions if needed to meet total session requirements
        if self.replay_config and self.replay_config.duplicate_sessions_target is not None:
            sessions = self._duplicate_sessions_if_needed(sessions, self.replay_config.duplicate_sessions_target)

        self.sessions = sessions
        if not self.sessions:
            raise ValueError("No valid replay sessions found")

        # Assign session_index to match position in list after shuffling/duplication
        for i, session in enumerate(self.sessions):
            session.session_index = i

        self._build_replay_schedule()
        logger.debug("Loaded %d sessions with %d total events", len(self.sessions), len(self.all_events))

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
        self.all_events = []

        for session in self.sessions:
            # Always generate random string for each session
            # Used for KV-cache invalidation when:
            # 1. inject_random_session_id flag is enabled, OR
            # 2. Session is a duplicate (contains "_dup" in session_id)
            random_string = None
            is_duplicate = ReplayGraphSessionGeneratorBase.is_duplicate_session(session.session_id)
            if (self.replay_config and self.replay_config.inject_random_session_id) or is_duplicate:
                random_string = uuid.uuid4().hex[:16]
                logger.debug(f"Generated random string for session {session.session_id}: {random_string}")

            # Initialize session graph state
            state = ReplaySessionState(
                session_id=session.session_id,
                graph=session.graph,
                ready_events=set(),  # Will be populated when session is activated
                dispatched_events=set(),
                completed_events=set(),
                event_completion_times={},
                is_active=False,
                is_complete=False,
                random_string=random_string,
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
                tool_calls = msg.get("tool_calls")
                tool_call_id = msg.get("tool_call_id")
            else:
                role = getattr(msg, "role", "user")
                content = getattr(msg, "text", "")
                tool_calls = getattr(msg, "tool_calls", None)
                tool_call_id = getattr(msg, "tool_call_id", None)

            if tool_calls is not None:
                chat_messages.append(ChatMessage(role=role, tool_calls=tool_calls))
                original_messages.append({"role": role, "tool_calls": tool_calls})
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
            chat_messages.append(ChatMessage(role=role, content=content_str))
            orig_msg: Dict[str, Any] = {"role": role, "content": content_str}
            if tool_call_id is not None:
                orig_msg["tool_call_id"] = tool_call_id
            original_messages.append(orig_msg)

        max_tokens = event.expected_output_tokens
        session_id = event.event_id.split(":")[0] if ":" in event.event_id else event.event_id
        raw_event_id = event.event_id.split(":", 1)[1] if ":" in event.event_id else event.event_id
        state = self.session_graph_state.get(session_id)
        total_events = len(state.graph.events) if state else 0

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
        logger.debug("Cleaned up session %s: removed %d events from memory", session_id, event_count)
