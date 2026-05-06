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
Convert OTel trace JSON to a replay graph JSON.

This script extracts LLM call events from OpenTelemetry traces and converts them
into a graph suitable for replay testing. Each event in the graph represents a single
LLM call. Edges encode predecessor relationships and wait times (tool/agent processing
time between events).

Graph structure
---------------
Each event contains:
  - event_id: unique identifier
  - call: a single LLM call
    The call contains:
      - call_id: original span_id
      - model: model name
      - messages: original message list (for replay)
      - input_segments: ordered list of segments describing the prompt at message granularity
          Each segment: {type, message_count, token_count, source_event_id (if output/shared)}
            type = "shared"   — leading messages identical to a predecessor call's messages
                                (KV cache hit opportunity)
            type = "output"   — an assistant message whose content is a predecessor call's output
                                (injected result from a predecessor)
            type = "unique"   — messages unique to this call
      - expected_output_tokens: how many tokens to generate
      - total_input_tokens: total prompt token count
      - temperature, max_tokens_recorded: original decoding params (informational)
  - predecessor_event_ids: list of event_ids that must complete before this event starts
  - wait_ms: delay (ms) after the last predecessor finishes before this event starts

Token count estimation
----------------------
Uses gen_ai.usage.prompt_tokens / completion_tokens from the span if present.
Falls back to len(text) // 4 (rough chars-per-token estimate) per message.
"""

import argparse
import json
import logging
from enum import Enum
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from inference_perf.datagen.export_replay_graph_to_dot import export_to_dot
from inference_perf.datagen.otel_trace_utils import (
    reconstruct_llm_output,
    reconstruct_llm_input,
    reconstruct_each_part_in_message_info,
)
from inference_perf.datagen.replay_graph_types import (
    ComplexReplayMessage,
    GraphCall,
    GraphEvent,
    InputSegment,
    ReplayGraph,
    ReplayMessage,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def parse_iso(ts: str) -> float:
    """Parse ISO-8601 timestamp to seconds since epoch."""
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def estimate_tokens(text: str) -> int:
    """Estimate token count from character length (rough: 4 chars per token)."""
    return max(1, len(text) // 4)


def message_content_text(msg: ReplayMessage) -> str:
    """Extract the text content of a message (handles string or list content)."""
    content = msg.text
    if isinstance(content, list):
        parts = []
        for blk in content:
            if isinstance(blk, dict):
                parts.append(json.dumps(blk, ensure_ascii=False, sort_keys=True))
            else:
                parts.append(str(blk))
        return " ".join(parts)
    return str(content)


def message_tokens(msg: ReplayMessage) -> int:
    """Estimate token count for a single message."""
    return estimate_tokens(message_content_text(msg))


def _replay_message_to_dict(x: ReplayMessage) -> Dict[str, Any]:
    """Convert a ReplayMessage to an OpenAI-compatible dict for the graph.

    For ComplexReplayMessage with tool_call_response parts, extracts tool_call_id
    so it survives into the replay runtime for ID rewriting.
    """
    if isinstance(x, ComplexReplayMessage):
        info = x.message_info
        parts = info.get("parts")
        if parts and x.role == "tool":
            tool_result_parts = [p for p in parts if p.get("type") == "tool_call_response"]
            if tool_result_parts:
                result_part = tool_result_parts[0]
                return {"role": "tool", "content": result_part.get("result", ""), "tool_call_id": result_part.get("id", "")}

    return {"role": x.role, "content": x.text}


def messages_equal(a: ReplayMessage, b: ReplayMessage) -> bool:
    """Return True if two messages have the same role and content."""
    return a.role == b.role and message_content_text(a) == message_content_text(b)


def output_matches_message(output_text: str, msg: ReplayMessage, allow_partial_match: bool = False) -> bool:
    """Return True if msg is an assistant message whose content matches output_text."""
    if msg.role != "assistant":
        return False
    msg_text = message_content_text(msg)
    if msg_text == output_text:
        return True
    if not allow_partial_match:
        return False
    else:  # try partial match
        if output_text in msg_text:  # output_text is entirely contained in msg_text
            return True
    return False


# ---------------------------------------------------------------------------
# Span extraction helpers
# ---------------------------------------------------------------------------


def _convert_content_and_tool_calls_to_parts(message: Dict[str, Any]) -> Dict[str, Any]:
    """
    Convert a message with both 'content' and 'tool_calls' fields into parts format.

    Transforms:
        {"role": "assistant", "content": "text", "tool_calls": [...]}
    Into:
        {"role": "assistant", "parts": [{"type": "text", "content": "text"}, {"type": "tool_call", ...}]}
    """
    parts = []

    # Add content as a text part if present
    content = message.get("content")
    if content:
        if isinstance(content, str):
            parts.append({"type": "text", "content": content})
        elif isinstance(content, list):
            # Content is already a list of parts
            parts.extend(content)

    # Add tool_calls as tool_call parts
    tool_calls = message.get("tool_calls", [])
    for tc in tool_calls:
        if isinstance(tc, dict):
            # OpenAI format: {"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}
            if "function" in tc:
                parts.append(
                    {
                        "type": "tool_call",
                        "id": tc.get("id"),  # type: ignore[dict-item]
                        "name": tc["function"].get("name"),
                        "arguments": tc["function"].get("arguments"),
                    }
                )
            # Direct format: {"name": "...", "arguments": "..."}
            elif "name" in tc:
                parts.append(
                    {
                        "type": "tool_call",
                        "id": tc.get("id"),  # type: ignore[dict-item]
                        "name": tc.get("name"),  # type: ignore[dict-item]
                        "arguments": tc.get("arguments"),  # type: ignore[dict-item]
                    }
                )

    # Create new message with parts
    result = {"role": message.get("role", "assistant"), "parts": parts}
    return result


def extract_messages(span: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Extract messages from span attributes. Returns empty list if not found."""
    attrs = span.get("attributes") or {}
    raw = attrs.get("gen_ai.input.messages")
    res = []
    if raw is None:
        return []
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception as err:
            raise ValueError(f"Failed to parse messages JSON: {raw}") from err

    if isinstance(raw, list):
        for x in raw:
            # sometimes the content field contains a dictionary with several properties
            role = x["role"]
            if "content" in x:
                content = x["content"]
                # Check if message also has tool_calls - convert to parts format
                if "tool_calls" in x and x["tool_calls"] is not None:
                    # Transform message with content + tool_calls into parts format
                    message_with_parts = _convert_content_and_tool_calls_to_parts(x)
                    res.append(
                        ComplexReplayMessage(
                            role=role,
                            message_info=message_with_parts,
                            raw_reconstructed_text=reconstruct_llm_input(message_with_parts),
                        )
                    )
                elif isinstance(content, str):
                    res.append(ReplayMessage(role=role, text=content))  # type: ignore[arg-type]
                else:
                    res.append(
                        ComplexReplayMessage(role=role, message_info=x, raw_reconstructed_text=reconstruct_llm_input(x))
                    )
            else:
                """ This is the case here:
                {
                    "role": "assistant",
                    "tool_calls": [{
                        "id": "call_123",
                        "type": "function",
                        "function": {"name": "get_weather", "arguments": "{\"city\": \"NYC\"}"}
                        }]
                }

                """
                res.append(ComplexReplayMessage(role=role, message_info=x, raw_reconstructed_text=reconstruct_llm_input(x)))
        return res  # type: ignore[return-value]
    else:
        return []
    return []


def extract_output_message(span: Dict[str, Any]) -> Optional[ReplayMessage]:
    """Extract output message from span attributes. Returns a ReplayMessage or ComplexReplayMessage object."""
    attrs = span.get("attributes") or {}
    for k in ("gen_ai.output.text", "gen_ai.completion", "gen_ai.output"):
        if k in attrs and isinstance(attrs[k], str):
            return ReplayMessage(role="assistant", text=attrs[k])
    out = attrs.get("gen_ai.output.messages")
    if isinstance(out, str):
        try:
            msgs = json.loads(out)
            if len(msgs) > 1:
                raise ValueError(f"Unexpected output messages fromat: expected a single message, got {len(msgs)} messages")
            return ComplexReplayMessage(
                role="assistant",
                message_info=reconstruct_each_part_in_message_info(msgs[0]),
                raw_reconstructed_text=reconstruct_llm_output(msgs[0]),
            )
        except Exception as err:
            raise ValueError(f"Failed parsing {out}") from err
    if isinstance(out, list) and out:
        return ReplayMessage(role="assistant", text=message_content_text(out[-1]))
    return None


def is_llm_span(span: Dict[str, Any], include_errors: bool = False) -> bool:
    """Check if span represents an LLM call."""
    name = span.get("name", "") or ""
    attrs = span.get("attributes") or {}
    is_llm = name.startswith("chat ") or "gen_ai.input.messages" in attrs
    if not is_llm:
        return False
    if not include_errors:
        status = span.get("status", {})
        if status.get("code", 0) == 2:
            return False
    return True


# ---------------------------------------------------------------------------
# Raw call (one per LLM span)
# ---------------------------------------------------------------------------


@dataclass
class RawCall:
    """A single LLM call extracted from a span."""

    call_id: str  # span_id
    trace_id: str
    t_start_ms: int  # ms relative to earliest span in file
    t_end_ms: int
    model: str
    messages: List[ReplayMessage]  # original message list (required)
    out_message: Optional[ReplayMessage]
    prompt_tokens: Optional[int]  # from gen_ai.usage.prompt_tokens
    completion_tokens: Optional[int]  # from gen_ai.usage.completion_tokens
    temperature: Optional[float]
    max_tokens_recorded: Optional[int]
    tool_definitions: Optional[List[Dict[str, Any]]] = None


def filter_duplicate_spans(spans: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Filter out duplicate spans based on start_time, end_time, and attributes.
    (this is added to support exgentic traces)

    Two spans are considered duplicates if they have identical:
    - start_time
    - end_time
    - attributes (all key-value pairs)

    When duplicates are found, only the first occurrence is kept.

    Args:
        spans: List of span dictionaries

    Returns:
        List of unique spans (duplicates removed)
    """
    seen_signatures: Set[str] = set()
    unique_spans: List[Dict[str, Any]] = []
    sorted_spans = sorted(spans, key=lambda s: s["span_id"])  # to make filtering consistant between runs
    for span in sorted_spans:
        # Create a signature for the span based on start_time, end_time, and attributes
        start_time = span.get("start_time", "")
        end_time = span.get("end_time", "")
        attributes = span.get("attributes", {})

        # Convert attributes dict to a sorted JSON string for consistent comparison
        attrs_str = json.dumps(attributes, sort_keys=True, ensure_ascii=False)

        # Create a unique signature
        signature = f"{start_time}|{end_time}|{attrs_str}"

        if signature not in seen_signatures:
            seen_signatures.add(signature)
            unique_spans.append(span)

    return unique_spans


def build_raw_calls(spans: List[Dict[str, Any]], include_errors: bool = False) -> List[RawCall]:
    """Extract and sort raw LLM calls from spans.

    First filters out duplicate spans (identical start_time, end_time, and attributes),
    then extracts LLM calls from the remaining unique spans.
    """
    # Filter out duplicate spans first
    unique_spans = filter_duplicate_spans(spans)

    llm_spans = [s for s in unique_spans if is_llm_span(s, include_errors=include_errors)]
    if not llm_spans:
        return []

    t0 = min(parse_iso(s["start_time"]) for s in llm_spans)
    llm_spans.sort(key=lambda s: (parse_iso(s["start_time"]), s.get("span_id", "")))

    calls: List[RawCall] = []
    for s in llm_spans:
        attrs = s.get("attributes") or {}
        messages = extract_messages(s)
        out_message = extract_output_message(s)
        t_start = int(round((parse_iso(s["start_time"]) - t0) * 1000))
        t_end = int(round((parse_iso(s["end_time"]) - t0) * 1000)) if s.get("end_time") else t_start

        prompt_tokens = attrs.get("gen_ai.usage.prompt_tokens")
        if prompt_tokens is None:
            prompt_tokens = attrs.get("gen_ai.usage.input_tokens")
        completion_tokens = attrs.get("gen_ai.usage.completion_tokens")
        if completion_tokens is None:
            completion_tokens = attrs.get("gen_ai.usage.output_tokens")
        if prompt_tokens is not None:
            prompt_tokens = int(prompt_tokens)

        if completion_tokens is not None:
            completion_tokens = int(completion_tokens)
        tool_definitions_raw = attrs.get("gen_ai.tool.definitions")
        tool_definitions: Optional[List[Dict[str, Any]]] = None
        if isinstance(tool_definitions_raw, str):
            try:
                tool_definitions = json.loads(tool_definitions_raw)
            except Exception:
                logger.warning(f"Span {s.get('span_id')}: failed to parse gen_ai.tool.definitions as JSON, ignoring")
        elif isinstance(tool_definitions_raw, list):
            tool_definitions = tool_definitions_raw
        calls.append(
            RawCall(
                call_id=s.get("span_id") or "",
                trace_id=s.get("trace_id") or "",
                t_start_ms=t_start,
                t_end_ms=t_end,
                model=str(attrs.get("gen_ai.request.model") or ""),
                messages=messages,  # type: ignore[arg-type]
                out_message=out_message,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                temperature=attrs.get("gen_ai.request.temperature"),
                max_tokens_recorded=attrs.get("gen_ai.request.max_tokens"),
                tool_definitions=tool_definitions,
            )
        )
    return calls


# ---------------------------------------------------------------------------
# Causal dependency detection (message-level)


class DEPENDENCY_TYPE(Enum):
    """Types of dependencies between different llm call nodes."""

    CAUSAL_FULL_MATCH = "full_match"  # full text of output a matches the full text of b
    CAUSAL_TOOL_CALL_IDS_MATCHED = "tool_call_ids_matched"  # all tool call IDs from output a appear in b's messages
    CAUSAL_SPLIT_PARTS_MATCHED = (
        "split_parts_matched"  # the output of a is split into parts, each part appears in b's list of messages.
    )
    CAUSAL_CONTENT_AND_SPLIT_TOOLS_MATCH = "content_and_split_tools_match"  # the output of a contains [content, tool_call_1, tool_call_2, etc], b's list of messages contains [content+tool_call_1, content+tool_call_2, etc).
    CAUSAL_DROP_CONTENT_SPLIT_PARTS = "drop_content_split_parts"  # the output of a contains [content, tool_call_1, tool_call_2, etc], b's list of message contains [tool_call_1, tool_call_2]
    TEMPORAL = "temporal"  # outputs are not matches, this is a temporal dependency only


# ---------------------------------------------------------------------------
def _try_match_tool_call_ids(a_parts: List[Dict[str, Any]], b_messages: List[Dict[str, Any]]) -> bool:
    """
    Check if all tool call IDs from a_parts appear in b_messages.
    Args:
        a_parts: List of part dictionaries from output message A
        b_messages: List of messages from call B

    Returns:
        True if all tool calls in a_parts have IDs and all those IDs appear in b_messages
    """
    # Extract tool call parts that have IDs
    tool_call_parts = [p for p in a_parts if p["type"] == "tool_call"]
    if not tool_call_parts:
        return False

    # Get all tool call IDs from a_parts
    tool_call_ids = [p.get("id") for p in tool_call_parts]

    # Check if all tool calls have IDs
    if not all(tc_id is not None for tc_id in tool_call_ids):
        return False

    # Extract all tool call IDs from b_messages
    b_tool_call_ids: set[str] = set()
    for msg in b_messages:
        # we message_info may contain parts to tool_calls.
        if isinstance(msg, ComplexReplayMessage):
            if "tool_calls" in msg.message_info:
                for tool_call in msg.message_info["tool_calls"]:
                    if tool_call.get("id"):
                        b_tool_call_ids.add(tool_call["id"])
            elif "parts" in msg.message_info:
                for part in msg.message_info["parts"]:
                    if part["type"] == "tool_call" and part.get("id"):
                        b_tool_call_ids.add(part["id"])

    # Check if all tool call IDs from a appear in b (order doesn't matter)
    return all(tc_id in b_tool_call_ids for tc_id in tool_call_ids)


def _extract_tool_call_ids(msg: Any) -> set[str]:
    """Return the set of tool-call IDs referenced in a single message.

    Handles ComplexReplayMessage (parts or tool_calls format) and plain dicts
    (OpenAI tool_calls format). Used by decompose_input to pre-compute per-message
    ID sets once, avoiding repeated extraction in the inner pred loop.
    """
    ids: set[str] = set()
    if isinstance(msg, ComplexReplayMessage):
        if "tool_calls" in msg.message_info:
            for tc in msg.message_info["tool_calls"]:
                if tc.get("id"):
                    ids.add(tc["id"])
        elif "parts" in msg.message_info:
            for part in msg.message_info["parts"]:
                if part.get("type") == "tool_call" and part.get("id"):
                    ids.add(part["id"])
    elif isinstance(msg, dict):
        for tc in msg.get("tool_calls") or []:
            if tc.get("id"):
                ids.add(tc["id"])
    return ids


def get_causal_dep(
    a: RawCall, b: RawCall, output_matches_for_substitutions: Optional[Dict[tuple[str, str], List[int]]] = None
) -> Optional[DEPENDENCY_TYPE]:
    """Return the type of causal dependency if call B causally depends on call A, None otherwise.

    A call B depends on A if any assistant message in B's message list has content
    that matches A's output text (full content match, not a snippet).
    This means A's output was injected into B's prompt as a prior assistant turn.
    We start with trying to detect FULL_MATCH dependency. Then, if not detected, we proceed to TOOL_CALL_IDS_MATCHED. If this dependency is not detected either we proceed to the other options.

    Args:
        output_matches_for_substitutions: Optional cache storing message indices with EXACT output matches.
                                         Key: (pred_call_id, curr_call_id), Value: list of matching indices

    Returns:
        FULL_MATCH: Full text of output A matches the full text in B
        TOOL_CALL_IDS_MATCHED: All tool call IDs from output A's tool calls appear in B's messages
                               (order doesn't matter, just presence of IDs)
        SPLIT_PARTS_MATCHED: Output of A is split into parts, each part appears separately in B's messages
        CONTENT_AND_SPLIT_TOOLS_MATCH: Output of A contains [content, tool_call_1, tool_call_2, etc],
                                        B's messages contain [content+tool_call_1, content+tool_call_2, etc]
        DROP_CONTENT_SPLIT_PARTS: Output of A contains [content, tool_call_1, tool_call_2, etc],
                                   B's messages contain [tool_call_1, tool_call_2] (content dropped)
        None: No causal dependency detected
    """
    if not a.out_message or not b.messages:
        return None
    a_out = a.out_message.text or ""
    for msg_idx, msg in enumerate(b.messages):
        if output_matches_message(a_out, msg, allow_partial_match=True):
            # Cache only exact matches (for reuse in decompose_input)
            if output_matches_for_substitutions is not None and output_matches_message(a_out, msg, allow_partial_match=False):
                cache_key = (a.call_id, b.call_id)
                if cache_key not in output_matches_for_substitutions:
                    output_matches_for_substitutions[cache_key] = []
                output_matches_for_substitutions[cache_key].append(msg_idx)
            return DEPENDENCY_TYPE.CAUSAL_FULL_MATCH
    # Try matching by tool call IDs before falling back to multi-part text comparison.
    # This handles the common case where the output is stored in OTel "parts"
    # format but appears in the successor's input in OpenAI "tool_calls" format —
    # the two representations look different as text but share the same IDs.
    # Only attempt when there are actual tool-call parts with IDs to match against.
    if isinstance(a.out_message, ComplexReplayMessage) and "parts" in a.out_message.message_info:
        tool_parts_with_ids = [p for p in a.out_message.message_info["parts"] if p.get("type") == "tool_call" and p.get("id")]
        if tool_parts_with_ids and _try_match_tool_call_ids(tool_parts_with_ids, b.messages):  # type: ignore[arg-type]
            return DEPENDENCY_TYPE.CAUSAL_TOOL_CALL_IDS_MATCHED
    # try matching parts (only relevant when there are multiple parts that expand
    # into separate messages in the successor — single tool calls are covered above)
    if (
        isinstance(a.out_message, ComplexReplayMessage)
        and "parts" in a.out_message.message_info
        and len(a.out_message.message_info["parts"]) > 1
    ):
        # this means this output message contains several parts, and will be interpreted as more than one message in the calls history
        parts = a.out_message.message_info["parts"]
        parts_text = a.out_message.message_info["parts_text"]

        # Determine structure: check if first part is content (text) or tool_call
        first_part_is_content = parts[0]["type"] != "tool_call"

        # Case 1: Only tool calls [tool_call_1, tool_call_2, ..., tool_call_n]
        # Case 2: Content + tool calls [content, tool_call_1, tool_call_2, ..., tool_call_n]

        if not first_part_is_content:
            # Case 1: Only tool calls - each tool call appears separately in input messages
            if _try_match_parts(parts, parts_text, b.messages, combine_content_with_tools=False):
                return DEPENDENCY_TYPE.CAUSAL_SPLIT_PARTS_MATCHED
        else:
            # Case 2: Content + tool calls - try two matching strategies:
            # Strategy A: Each part appears separately (content has offset 1, tools have offset 2)
            if _try_match_parts(parts, parts_text, b.messages, combine_content_with_tools=False):
                return DEPENDENCY_TYPE.CAUSAL_SPLIT_PARTS_MATCHED
            # Strategy B: Content is combined with each tool call
            # [content + tool_call_1, content + tool_call_2, ..., content + tool_call_n]
            # Combined messages are treated as tool calls (offset 2)
            if _try_match_parts(parts, parts_text, b.messages, combine_content_with_tools=True):
                return DEPENDENCY_TYPE.CAUSAL_CONTENT_AND_SPLIT_TOOLS_MATCH
            # Strategy C: Content is dropped, only tool calls remain
            # Check if tool calls alone (without content) match
            tool_call_parts = [p for p in parts if p["type"] == "tool_call"]
            tool_call_texts = [parts_text[i] for i, p in enumerate(parts) if p["type"] == "tool_call"]
            if tool_call_parts and _try_match_parts(
                tool_call_parts, tool_call_texts, b.messages, combine_content_with_tools=False
            ):
                return DEPENDENCY_TYPE.CAUSAL_DROP_CONTENT_SPLIT_PARTS

        return None
    return None


def _try_match_parts(
    parts: List[Dict[str, Any]], parts_text: List[str], b_messages: List[Any], combine_content_with_tools: bool
) -> bool:
    """
    Unified function to match parts in b_messages.

    Args:
        parts: List of part dictionaries with 'type' field
        parts_text: List of text representations for each part
        b_messages: List of messages to search in
        combine_content_with_tools: If True and first part is content, expect content combined with each tool call

    Returns:
        True if a valid match is found
    """
    if not parts or not parts_text or not b_messages:
        return False

    # Determine what we're matching
    first_part_is_content = parts[0]["type"] != "tool_call"

    # For combine mode, we need content + tool calls
    if combine_content_with_tools:
        if not first_part_is_content or len(parts) < 2:
            return False
        if not all(part["type"] == "tool_call" for part in parts[1:]):
            return False

        # When combining: skip content in parts list, but check for it in each message
        content_text = parts_text[0]
        parts_to_match = parts[1:]
        parts_text_to_match = parts_text[1:]
    else:
        # Standard mode: match all parts separately
        content_text = None
        parts_to_match = parts
        parts_text_to_match = parts_text

    # Find candidates for the first part to match
    first_part_text = parts_text_to_match[0]
    first_part_match_candidates = []

    for i in range(len(b_messages)):
        msg = b_messages[i]

        # Prepare text to match for first part
        if combine_content_with_tools:
            # Concatenate content with first tool call text
            text_to_match = (content_text or "") + "\n" + (first_part_text or "")
        else:
            # Just the first part text
            text_to_match = first_part_text

        # Check if the text matches
        if output_matches_message(text_to_match, msg, allow_partial_match=True):
            first_part_match_candidates.append(i)

    # Try each candidate position
    for candidate_i in first_part_match_candidates:
        candidate_ok = True

        # Calculate offset after first matched part
        if combine_content_with_tools:
            # Combined content+tool is treated as tool call (offset 2)
            offset = 2
        else:
            # Separate parts: 1 for content, 2 for tool_call
            offset = 1 if parts_to_match[0]["type"] != "tool_call" else 2

        # Check remaining parts
        for part, part_text in zip(parts_to_match[1:], parts_text_to_match[1:], strict=False):
            next_index_to_check = candidate_i + offset
            if next_index_to_check >= len(b_messages):
                candidate_ok = False
                break

            msg = b_messages[next_index_to_check]

            # Prepare text to match
            if combine_content_with_tools:
                # Concatenate content with tool call text
                text_to_match = (content_text or "") + "\n" + (part_text or "")
            else:
                # Just the part text
                text_to_match = part_text

            # Check if the text matches the message
            if not output_matches_message(text_to_match, msg, allow_partial_match=True):
                candidate_ok = False
                break

            # Update offset for next part
            if combine_content_with_tools:
                # Combined content+tool is treated as tool call (offset 2)
                offset += 2
            else:
                # Separate parts: 1 for content, 2 for tool_call
                offset += 1 if part["type"] != "tool_call" else 2

        if candidate_ok:
            return True

    return False


# ---------------------------------------------------------------------------
# Input segment decomposition (message-level)
# ---------------------------------------------------------------------------


def decompose_input(
    call: RawCall,
    predecessors: List[RawCall],
    predecessor_event_ids: List[str],
    output_matches_for_substitutions: Optional[Dict[tuple[str, str], List[int]]] = None,
) -> List[InputSegment]:
    """Decompose a call's message list into segments relative to its predecessors.

    Algorithm:
    1. Find the predecessor whose message list shares the longest common prefix
       with this call's messages (message-by-message equality).
    2. After the shared prefix, scan remaining messages for any assistant message
       whose content matches a predecessor's output text.
    3. Whatever remains is "unique".

    Token counts are derived from recorded prompt_tokens (proportional to message counts)
    or estimated per-message at 4 chars/token.
    """
    messages = call.messages
    total_msgs = len(messages)

    if total_msgs == 0:
        return [InputSegment(type="unique", message_count=0, token_count=0)]

    # Total token count for this call's input
    total_tokens = call.prompt_tokens if call.prompt_tokens is not None else sum(message_tokens(m) for m in messages)

    def msgs_to_tokens(msg_list: List[Dict[str, Any]]) -> int:
        """Convert a list of messages to token count proportionally."""
        if total_msgs == 0 or total_tokens == 0:
            return 0
        msg_chars = sum(len(message_content_text(m)) for m in msg_list)  # type: ignore[arg-type,misc]
        total_chars = sum(len(message_content_text(m)) for m in messages)
        if total_chars == 0:
            return 0
        return max(0, round(total_tokens * msg_chars / total_chars))

    if not predecessors:
        return [InputSegment(type="unique", message_count=total_msgs, token_count=total_tokens)]

    # Step 1: Find the predecessor with the longest common message prefix
    best_pred_idx: int = -1
    best_prefix_count: int = 0
    for idx, pred in enumerate(predecessors):
        prefix_len = 0
        for _i, (ma, mb) in enumerate(zip(messages, pred.messages, strict=False)):
            if messages_equal(ma, mb):
                prefix_len += 1
            else:
                break
        if prefix_len > best_prefix_count:
            best_prefix_count = prefix_len
            best_pred_idx = idx

    segments: List[InputSegment] = []
    cursor = 0  # current message index

    # Segment 1: shared prefix (if at least 1 message is shared)
    if best_prefix_count >= 1 and best_pred_idx >= 0:
        shared_msgs = messages[:best_prefix_count]
        segments.append(
            InputSegment(
                type="shared",
                message_count=best_prefix_count,
                token_count=msgs_to_tokens(shared_msgs),  # type: ignore[arg-type]
                source_event_id=predecessor_event_ids[best_pred_idx],
            )
        )
        cursor = best_prefix_count

    # Step 2: After the shared prefix, scan for injected outputs from predecessors.
    # Each predecessor's output may appear as an assistant message in the remaining messages.
    # Pre-compute the tool-call ID set for each message once so the inner pred loop
    # doesn't repeat the extraction for every (pred, msg) pair.
    msg_tool_id_sets: List[set[str]] = [_extract_tool_call_ids(m) for m in messages]

    while cursor < total_msgs:
        # Find the earliest remaining message that matches any predecessor's output
        best_out_pred_idx: int = -1
        best_out_msg_idx: int = total_msgs  # position in messages[]

        for pred_idx, pred in enumerate(predecessors):
            if not pred.out_message:
                continue

            # Check cache first for exact matches
            cache_key = (pred.call_id, call.call_id)
            if output_matches_for_substitutions and cache_key in output_matches_for_substitutions:
                # Use cached exact match indices
                matched_indices = output_matches_for_substitutions[cache_key]
                for msg_idx in matched_indices:
                    if cursor <= msg_idx < total_msgs:
                        if msg_idx < best_out_msg_idx:
                            best_out_msg_idx = msg_idx
                            best_out_pred_idx = pred_idx
                        break
            else:
                # Fallback: search for exact matches or tool-call ID matches
                pred_out = pred.out_message.text or ""
                # Build tool-call ID set for this predecessor's output (for fallback ID matching).
                pred_tool_ids: set[str] = set()
                if isinstance(pred.out_message, ComplexReplayMessage) and "parts" in pred.out_message.message_info:
                    pred_tool_ids = {
                        p["id"] for p in pred.out_message.message_info["parts"] if p.get("type") == "tool_call" and p.get("id")
                    }
                for msg_idx in range(cursor, total_msgs):
                    matched = output_matches_message(pred_out, messages[msg_idx])
                    # Fallback: match by tool-call IDs when text representations differ
                    if not matched and pred_tool_ids:
                        matched = bool(pred_tool_ids <= msg_tool_id_sets[msg_idx])
                    if matched:
                        if msg_idx < best_out_msg_idx:
                            best_out_msg_idx = msg_idx
                            best_out_pred_idx = pred_idx
                        break  # found earliest occurrence for this pred

        if best_out_pred_idx == -1:
            # No more injected outputs — rest is unique
            remaining_msgs = messages[cursor:]
            if remaining_msgs:
                segments.append(
                    InputSegment(
                        type="unique",
                        message_count=len(remaining_msgs),
                        token_count=msgs_to_tokens(remaining_msgs),  # type: ignore[arg-type]
                    )
                )
            break

        # Gap before the output message — unique messages
        if best_out_msg_idx > cursor:
            gap_msgs = messages[cursor:best_out_msg_idx]
            segments.append(
                InputSegment(
                    type="unique",
                    message_count=len(gap_msgs),
                    token_count=msgs_to_tokens(gap_msgs),  # type: ignore[arg-type]
                )
            )
            cursor = best_out_msg_idx

        # The injected output message
        out_msg = messages[cursor]
        _ct = predecessors[best_out_pred_idx].completion_tokens
        out_tokens: int = _ct if _ct is not None else msgs_to_tokens([out_msg])  # type: ignore[list-item]
        segments.append(
            InputSegment(
                type="output",
                message_count=1,
                token_count=out_tokens,
                source_event_id=predecessor_event_ids[best_out_pred_idx],
            )
        )
        cursor += 1

    # Ensure we have at least one segment
    if not segments:
        segments.append(InputSegment(type="unique", message_count=total_msgs, token_count=total_tokens))

    return segments


# ---------------------------------------------------------------------------
# Graph data structures
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph(
    calls: List[RawCall],
    source_file: str = "",
) -> ReplayGraph:
    """Build a ReplayGraph from a list of raw calls.

    Each RawCall becomes exactly one GraphEvent. Predecessor relationships are
    inferred from causal dependencies (output→input message matching) with a
    fallback to the immediately preceding call for timing-only chains.

    Steps:
    1. For each call, find its direct predecessor calls (causal dep or timing fallback).
    2. Apply transitive reduction: remove predecessors that are already ancestors
       of another predecessor (keep only direct edges).
    3. Decompose each call's messages into segments relative to all ancestor calls.
    4. Build GraphEvent objects with predecessor_event_ids and wait_ms.
    """
    if not calls:
        return ReplayGraph(events={}, root_event_ids=[], source_file=source_file)

    n = len(calls)
    # Assign event IDs 1:1 with calls, incorporating span_id for traceability
    event_ids = [f"event_{i:03d}_{calls[i].call_id}" for i in range(n)]

    # ---------------------------------------------------------------------------
    # Step 1: Find direct predecessors for each call
    # ---------------------------------------------------------------------------
    # predecessor_indices[i] = dict mapping predecessor index to dependency type
    predecessor_indices: List[Dict[int, DEPENDENCY_TYPE]] = [{} for _ in range(n)]

    def get_causal_ancestors(node_idx: int) -> Set[int]:
        """Get all causal ancestors of a node (excluding temporal edges)."""
        ancestors: Set[int] = set()
        stack = [
            pred_idx for pred_idx, dep_type in predecessor_indices[node_idx].items() if dep_type != DEPENDENCY_TYPE.TEMPORAL
        ]
        while stack:
            ancestor = stack.pop()
            if ancestor not in ancestors:
                ancestors.add(ancestor)
                stack.extend(
                    [
                        pred_idx
                        for pred_idx, dep_type in predecessor_indices[ancestor].items()
                        if dep_type != DEPENDENCY_TYPE.TEMPORAL
                    ]
                )
        return ancestors

    def is_valid_predecessor(predecessor_candidate: Any, curr_call: Any) -> bool:
        # checks if candidate can be a predecessor to curr_call. Make sure times are not overlapping
        # since the events are sorted, we can assume the candidate doesn't start after curr_call
        if curr_call.t_start_ms < predecessor_candidate.t_end_ms:
            # curr starts before the candidate ends.
            return False
        return True

    # Cache for exact output matches: (pred_call_id, curr_call_id) -> list of matching message indices. Is used when input segments are created.
    output_matches_for_substitutions: Dict[tuple[str, str], List[int]] = {}

    for i in range(1, n):
        # Collect all calls that causally feed call i
        curr_causal_preds: Dict[int, DEPENDENCY_TYPE] = {}
        # Track indices that are transitively connected (ancestors of found predecessors)
        transitive_preds: Set[int] = set()

        for j in range(i - 1, -1, -1):
            # If j is already known to be a transitive predecessor, skip expensive check
            if j in transitive_preds:
                continue

            dep_type = get_causal_dep(calls[j], calls[i], output_matches_for_substitutions)
            if dep_type is not None:
                curr_causal_preds[j] = dep_type
                # Add all of j's causal ancestors as transitive predecessors
                # so we skip checking them in future iterations
                transitive_preds.update(get_causal_ancestors(j))

        # All predecessors in curr_causal_preds are direct (not transitive)
        # because we skipped transitive ones in the loop above
        predecessor_indices[i].update(curr_causal_preds)

        """
        Add a temporal fallback predecessor (the closest non-overlapping event) if one exists.
        This ensures we don't have long wait times when causal predecessors are distant.
        Causal detection (output matching) doesn't catch all dependencies, so we use
        temporal proximity as a conservative fallback to maintain realistic timing.
        """
        predecessor_index = None  # Will remain None if no valid predecessor found
        # Look for the closest possible predecessor. It's not necessarily the immediate predecessor, as they can be executed in parallel
        for j in range(i - 1, -1, -1):
            if is_valid_predecessor(calls[j], calls[i]):
                predecessor_index = j
                break
        # Only add temporal predecessor if one was found and it's not already a predecessor
        if predecessor_index is not None and predecessor_index not in predecessor_indices[i]:
            predecessor_indices[i][predecessor_index] = DEPENDENCY_TYPE.TEMPORAL

    # ---------------------------------------------------------------------------
    # Step 2: Compute all ancestors per call (for segment decomposition)
    # ---------------------------------------------------------------------------
    def all_ancestor_indices(call_idx: int) -> List[int]:
        """Return all ancestor call indices (transitive closure of predecessors)."""
        visited: Set[int] = set()
        stack = list(predecessor_indices[call_idx].keys())
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(predecessor_indices[node].keys())
        return list(visited)

    # ---------------------------------------------------------------------------
    # Step 3: Build GraphEvents
    # ---------------------------------------------------------------------------
    events: Dict[str, GraphEvent] = {}
    for i, (eid, rc) in enumerate(zip(event_ids, calls, strict=False)):
        # All ancestor calls (for segment decomposition — includes transitive predecessors)
        ancestor_idxs = all_ancestor_indices(i)
        ancestor_calls = [calls[j] for j in ancestor_idxs]
        ancestor_event_ids = [event_ids[j] for j in ancestor_idxs]

        # Decompose input into message-level segments
        segments = decompose_input(rc, ancestor_calls, ancestor_event_ids, output_matches_for_substitutions)

        # Validate that segment message counts sum to total messages
        total_segment_messages = sum(seg.message_count for seg in segments)
        actual_message_count = len(rc.messages)
        if total_segment_messages != actual_message_count:
            logger.warning(
                f"Segment validation failed for call {rc.call_id}: "
                f"segment messages ({total_segment_messages}) != actual messages ({actual_message_count})"
            )

        total_input_tokens = rc.prompt_tokens if rc.prompt_tokens is not None else sum(message_tokens(m) for m in rc.messages)
        expected_output_tokens = (
            rc.completion_tokens
            if rc.completion_tokens is not None
            else estimate_tokens(rc.out_message.text or "" if rc.out_message else "")
        )

        # Detect whether the recorded output was a tool call and collect names.
        # ComplexReplayMessage.message_info uses OTel "parts" format after
        # reconstruct_each_part_in_message_info, or OpenAI "tool_calls" format
        # when passed through _convert_content_and_tool_calls_to_parts.
        expected_output_tool_names: List[str] = []
        if isinstance(rc.out_message, ComplexReplayMessage):
            info = rc.out_message.message_info
            # OTel parts format: {"parts": [{"type": "tool_call", "name": ...}, ...]}
            for part in info.get("parts", []):
                if part.get("type") == "tool_call" and part.get("name"):
                    expected_output_tool_names.append(part["name"])
            # OpenAI tool_calls format: {"tool_calls": [{"function": {"name": ...}}, ...]}
            if not expected_output_tool_names:
                for tc in info.get("tool_calls", []):
                    name = tc.get("function", {}).get("name") or tc.get("name")
                    if name:
                        expected_output_tool_names.append(name)

        # If the recorded output was a tool call but no tool definitions were captured,
        # we cannot inject tool_choice at replay time. Treat it as a plain-text output
        # so substitution does not fail-fast when the live model returns text.
        has_tool_definitions = bool(rc.tool_definitions)
        effective_is_tool_call = bool(expected_output_tool_names) and has_tool_definitions
        if bool(expected_output_tool_names) and not has_tool_definitions:
            logger.warning(
                f"Span {rc.call_id}: recorded output is a tool call "
                f"({expected_output_tool_names}) but no tool definitions were captured "
                f"in the span (gen_ai.tool.definitions missing). "
                f"tool_choice cannot be forced at replay time, so the live model will "
                f"return plain text. expected_output_is_tool_call is being set to False "
                f"so that substitution treats the live response as plain text rather than "
                f"failing the session chain with a dangling tool_call_id error."
            )

        graph_call = GraphCall(
            call_id=rc.call_id,
            model=rc.model,
            messages=[
                _replay_message_to_dict(x) for x in rc.messages
            ],  # convert to a list of dictionaries representing a message with role and content only.
            expected_output=(rc.out_message.text or "" if rc.out_message else ""),
            input_segments=segments,
            total_input_tokens=total_input_tokens,
            expected_output_tokens=expected_output_tokens,
            temperature=rc.temperature,
            max_tokens_recorded=rc.max_tokens_recorded,
            tool_definitions=rc.tool_definitions,
            expected_output_is_tool_call=effective_is_tool_call,
            expected_output_tool_names=expected_output_tool_names or None,
        )

        # Compute wait_ms: gap between when the last predecessor ends and this call starts
        pred_idxs = predecessor_indices[i]
        if pred_idxs:
            last_pred_end_ms = max(calls[j].t_end_ms for j in pred_idxs.keys())
            wait_ms = max(0, rc.t_start_ms - last_pred_end_ms)
        else:
            wait_ms = 0

        # Build predecessor dependency types mapping
        predecessor_dependency_types = {event_ids[j]: dep_type.value for j, dep_type in pred_idxs.items()}

        events[eid] = GraphEvent(
            event_id=eid,
            call=graph_call,
            predecessor_event_ids=list(predecessor_dependency_types.keys()),
            predecessor_dependency_types=predecessor_dependency_types,
            wait_ms=wait_ms,
            t_start_ms=rc.t_start_ms,
            t_end_ms=rc.t_end_ms,
        )

    root_event_ids = [event_ids[i] for i in range(n) if not predecessor_indices[i]]

    return ReplayGraph(events=events, root_event_ids=root_event_ids, source_file=source_file)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def segment_to_dict(seg: InputSegment) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "type": seg.type,
        "message_count": seg.message_count,
        "token_count": seg.token_count,
    }
    if seg.source_event_id is not None:
        d["source_event_id"] = seg.source_event_id
    return d


def graph_call_to_dict(gc: GraphCall) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "call_id": gc.call_id,
        "model": gc.model,
        "total_input_tokens": gc.total_input_tokens,
        "expected_output_tokens": gc.expected_output_tokens,
        "temperature": gc.temperature,
        "max_tokens_recorded": gc.max_tokens_recorded,
        "input_segments": [segment_to_dict(s) for s in gc.input_segments],
        "messages": gc.messages,
        "expected_output": gc.expected_output,
    }
    if gc.tool_definitions is not None:
        d["tool_definitions"] = gc.tool_definitions
    if gc.expected_output_is_tool_call:
        d["expected_output_is_tool_call"] = gc.expected_output_is_tool_call
    if gc.expected_output_tool_names is not None:
        d["expected_output_tool_names"] = gc.expected_output_tool_names
    return d


def graph_event_to_dict(event: GraphEvent) -> Dict[str, Any]:
    return {
        "event_id": event.event_id,
        "t_start_ms": event.t_start_ms,
        "t_end_ms": event.t_end_ms,
        "predecessor_event_ids": event.predecessor_event_ids,
        "predecessor_dependency_types": event.predecessor_dependency_types,
        "wait_ms": event.wait_ms,
        "call": graph_call_to_dict(event.call),
    }


def graph_to_dict(graph: ReplayGraph) -> Dict[str, Any]:
    return {
        "source_file": graph.source_file,
        "root_event_ids": graph.root_event_ids,
        "event_count": len(graph.events),
        "events": {eid: graph_event_to_dict(event) for eid, event in graph.events.items()},
    }


# ---------------------------------------------------------------------------
# Human-readable pretty-print
# ---------------------------------------------------------------------------


def _fmt_ms(ms: int) -> str:
    """Format milliseconds as a human-readable duration."""
    if ms < 1000:
        return f"{ms}ms"
    return f"{ms / 1000:.1f}s"


def _shorten_string(s: str, max_length: int = 100) -> str:
    if len(s) < max_length:
        return s
    side_length = (max_length - 3) // 2  # 3 for '...'
    return f"{s[:side_length]} ... ... {s[-side_length:]}"


def _segment_label(seg: InputSegment, messages: List[Dict[str, str]]) -> str:
    """One-line label for an input segment."""
    type_labels = {"shared": "SHARED", "output": "OUTPUT", "unique": "UNIQUE"}
    label = type_labels.get(seg.type, seg.type.upper())
    src = f" <- {seg.source_event_id}" if seg.source_event_id else ""
    msg_str = "\n\t\t\t".join(f"{x['role']} : {_shorten_string(x['content'])}" for x in messages)
    return f"{label}({seg.message_count}msg/{seg.token_count}t{src})\n\t\t\t{msg_str}"


def _topo_order(graph: ReplayGraph) -> List[str]:
    """Return event IDs in topological order (BFS from roots)."""
    # Build successor map from predecessor_event_ids
    successors: Dict[str, List[str]] = {eid: [] for eid in graph.events}
    for eid, event in graph.events.items():
        for pred_id in event.predecessor_event_ids:
            if pred_id in successors:
                successors[pred_id].append(eid)

    visited: Set[str] = set()
    queue = list(graph.root_event_ids)
    order: List[str] = []
    while queue:
        eid = queue.pop(0)
        if eid in visited:
            continue
        visited.add(eid)
        order.append(eid)
        for succ_id in successors.get(eid, []):
            queue.append(succ_id)
    return order


def map_input_seq_to_messages(gc: Any) -> list[Any]:
    """
    returns a list of tuples, each tuple contains the sequence, and the corresponding messages
    """
    curr_msg_index = 0
    res = []
    for seq in gc.input_segments:
        res.append((seq, gc.messages[curr_msg_index : curr_msg_index + seq.message_count]))
        curr_msg_index += seq.message_count
    return res


def print_graph(graph: ReplayGraph) -> None:
    """Pretty-print the replay graph to stdout with box-drawing characters."""
    order = _topo_order(graph)
    source_name = graph.source_file.split("/")[-1] if graph.source_file else ""

    title = (
        f"REPLAY GRAPH   {len(graph.events)} events   source: {source_name}"
        if source_name
        else f"REPLAY GRAPH   {len(graph.events)} events"
    )
    print()
    print(f"  {title}")
    print("  " + "-" * len(title))
    print()
    print("  Legend:  SHARED = KV-cache prefix reuse (identical leading messages)")
    print("           OUTPUT = predecessor output injected as assistant message")
    print("           UNIQUE = messages unique to this call")
    print()

    for eid in order:
        event = graph.events[eid]
        is_root = eid in graph.root_event_ids
        duration_ms = event.t_end_ms - event.t_start_ms
        gc = event.call

        tags = []
        if is_root:
            tags.append("ROOT")
        tag_str = "   " + " | ".join(tags) if tags else ""

        print(
            f"  ╔══ EVENT {eid}"
            f"   t={_fmt_ms(event.t_start_ms)} -> {_fmt_ms(event.t_end_ms)}"
            f"  (duration {_fmt_ms(duration_ms)})"
            f"{tag_str}"
        )
        print("  ║")

        if event.predecessor_event_ids:
            preds_str = ", ".join(event.predecessor_event_ids)
            print(f"  ║   waits for: [{preds_str}]  then +{_fmt_ms(event.wait_ms)}")
        else:
            print("  ║   (no predecessors — starts immediately)")
        print("  ║")

        temp_str = f"  temperature={gc.temperature}" if gc.temperature is not None else ""
        tools_str = f"  tools={len(gc.tool_definitions)}" if gc.tool_definitions else ""
        print(f"  ║   CALL {gc.call_id}   model={gc.model}{temp_str}{tools_str}")
        print(f"  ║     Input  ({gc.total_input_tokens} tokens, {len(gc.messages)} messages):")
        for seg, messages in map_input_seq_to_messages(gc):
            offset = "       "
            segment_label = _segment_label(seg, messages).replace("\n", f"\n{offset}")
            print(f"  ║{offset}* {segment_label}")
        out_note = f"   (max_tokens_recorded={gc.max_tokens_recorded})" if gc.max_tokens_recorded else ""
        print(f"  ║     Output: {gc.expected_output_tokens} tokens expected{out_note}")

        print("  ╚" + "=" * 58)
        print()


def summarize_graph(graph: ReplayGraph) -> str:
    """Return a compact one-line-per-event summary string (for logging/testing)."""
    lines = []
    for eid in _topo_order(graph):
        event = graph.events[eid]
        gc = event.call
        preds = (
            f"after [{', '.join(event.predecessor_event_ids)}] +{event.wait_ms}ms" if event.predecessor_event_ids else "ROOT"
        )
        seg_str = " ".join(_segment_label(s, m) for s, m in map_input_seq_to_messages(gc))
        tools_str = f"  tools={len(gc.tool_definitions)}" if gc.tool_definitions else ""
        lines.append(f"[{eid}] {preds}  t={event.t_start_ms}-{event.t_end_ms}ms{tools_str}")
        lines.append(f"    {gc.call_id}: [{seg_str}] -> O({gc.expected_output_tokens}t)")
    return "\n".join(lines)


def visualize_graph(graph: Any, output_file: Any) -> None:
    """
    Export graph to DOT format and optionally render to PNG.

    Args:
        graph: ReplayGraph object
        test_name: Name of the test (used for filename)
        output_dir: Directory to save output files
    """

    # Convert graph to JSON format expected by export_to_dot
    graph_dict = graph_to_dict(graph)

    # Export to DOT
    export_to_dot(graph_dict, str(output_file))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    """Main entry point."""
    ap = argparse.ArgumentParser(
        description="Convert OTel trace JSON to a replay graph JSON",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--input", required=True, help="OTel-like JSON trace file")
    ap.add_argument("--output", required=True, help="Output replay graph JSON file")
    ap.add_argument("--include_errors", action="store_true", help="Include spans with error status")
    ap.add_argument("--summary", action="store_true", help="Print human-readable graph summary")
    ap.add_argument(
        "--vis_output",
        default=None,
        help="If provided, is the path to the graph structure to be displayed in https://viz-js.com/",
    )
    args = ap.parse_args()

    data = json.loads(Path(args.input).read_text(encoding="utf-8"))
    spans = data.get("spans") or []
    if not spans:
        raise SystemExit("No spans found in trace JSON")

    calls = build_raw_calls(spans, include_errors=args.include_errors)
    if not calls:
        raise SystemExit("No LLM spans found in trace file")

    graph = build_graph(calls, source_file=args.input)

    out_path = Path(args.output)
    out_path.write_text(
        json.dumps(graph_to_dict(graph), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    print(f"Wrote replay graph ({len(graph.events)} events, {len(calls)} calls) to {args.output}")

    if args.summary:
        print_graph(graph)
    if args.vis_output:
        visualize_graph(graph, args.vis_output)


if __name__ == "__main__":
    main()
