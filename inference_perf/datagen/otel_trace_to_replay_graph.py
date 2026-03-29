#!/usr/bin/env python3
"""
Convert OTel trace JSON to a replay graph JSON.

This script extracts LLM call events from OpenTelemetry traces and converts them
into a graph suitable for replay testing. Each node in the graph represents a single
LLM call. Edges encode predecessor relationships and wait times (tool/agent processing
time between nodes).

Graph structure
---------------
Each node contains:
  - node_id: unique identifier
  - call: a single LLM call
    The call contains:
      - call_id: original span_id
      - model: model name
      - messages: original message list (for replay)
      - input_segments: ordered list of segments describing the prompt at message granularity
          Each segment: {type, message_count, token_count, source_node_id (if output/shared)}
            type = "shared"   — leading messages identical to a predecessor call's messages
                                (KV cache hit opportunity)
            type = "output"   — an assistant message whose content is a predecessor call's output
                                (injected result from a predecessor)
            type = "unique"   — messages unique to this call
      - expected_output_tokens: how many tokens to generate
      - total_input_tokens: total prompt token count
      - temperature, max_tokens_recorded: original decoding params (informational)
  - predecessor_node_ids: list of node_ids that must complete before this node starts
  - wait_ms: delay (ms) after the last predecessor finishes before this node starts

Token count estimation
----------------------
Uses gen_ai.usage.prompt_tokens / completion_tokens from the span if present.
Falls back to len(text) // 4 (rough chars-per-token estimate) per message.
"""

import argparse
import json
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

from inference_perf.datagen.export_replay_graph_to_dot import export_to_dot
from inference_perf.datagen.otel_trace_utils import reconstruct_llm_output, reconstruct_llm_input,  \
    reconstruct_each_part_in_message_info

logger = logging.getLogger(__name__)


@dataclass
class OtelMessage:
    role: str
    text: str


class ComplexOtelMessage(OtelMessage):  # usually, this message type can be user in the list of input messages.
    def __init__(self, role: str, message_info: dict[str, Any], raw_reconstructed_text: str):
        super().__init__(role=role, text=raw_reconstructed_text)
        self.message_info = message_info


# ---------------------------------------------------------------------------
# Text helpers
# ---------------------------------------------------------------------------


def parse_iso(ts: str) -> float:
    """Parse ISO-8601 timestamp to seconds since epoch."""
    dt = datetime.fromisoformat(ts.replace("Z", "+00:00"))
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.timestamp()


def norm_text(s: str) -> str:
    """Normalize text by collapsing whitespace."""
    return re.sub(r"\s+", " ", s or "").strip()


def estimate_tokens(text: str) -> int:
    """Estimate token count from character length (rough: 4 chars per token)."""
    return max(1, len(text) // 4)


def message_content_text(msg: Dict[str, Any]) -> str:
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


def message_tokens(msg: Dict[str, Any]) -> int:
    """Estimate token count for a single message."""
    return estimate_tokens(message_content_text(msg))


def messages_equal(a: Dict[str, Any], b: Dict[str, Any]) -> bool:
    """Return True if two messages have the same role and content."""
    return a.role == b.role and norm_text(message_content_text(a)) == norm_text(message_content_text(b))


def output_matches_message(output_text: str, msg: Dict[str, Any]) -> bool:
    """Return True if msg is an assistant message whose content matches output_text."""
    if msg.role != "assistant":
        return False
    msg_text = norm_text(message_content_text(msg))
    out_text = norm_text(output_text)
    return msg_text == out_text


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
    content = message.get('content')
    if content:
        if isinstance(content, str):
            parts.append({"type": "text", "content": content})
        elif isinstance(content, list):
            # Content is already a list of parts
            parts.extend(content)

    # Add tool_calls as tool_call parts
    tool_calls = message.get('tool_calls', [])
    for tc in tool_calls:
        if isinstance(tc, dict):
            # OpenAI format: {"id": "...", "type": "function", "function": {"name": "...", "arguments": "..."}}
            if 'function' in tc:
                parts.append({
                    "type": "tool_call",
                    "id": tc.get('id'),
                    "name": tc['function'].get('name'),
                    "arguments": tc['function'].get('arguments')
                })
            # Direct format: {"name": "...", "arguments": "..."}
            elif 'name' in tc:
                parts.append({
                    "type": "tool_call",
                    "id": tc.get('id'),
                    "name": tc.get('name'),
                    "arguments": tc.get('arguments')
                })

    # Create new message with parts
    result = {"role": message.get('role', 'assistant'), "parts": parts}
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
                if "tool_calls" in x:
                    # Transform message with content + tool_calls into parts format
                    message_with_parts = _convert_content_and_tool_calls_to_parts(x)
                    res.append(ComplexOtelMessage(role=role, message_info=message_with_parts, raw_reconstructed_text=reconstruct_llm_input(message_with_parts)))
                elif isinstance(content, str):
                    res.append(OtelMessage(role=role, text=content))
                else:
                    res.append(ComplexOtelMessage(role=role, message_info=x, raw_reconstructed_text=reconstruct_llm_input(x)))
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
                res.append(ComplexOtelMessage(role=role, message_info=x, raw_reconstructed_text=reconstruct_llm_input(x)))
        return res
    else:
        return []
    return []


def extract_output_message(span: Dict[str, Any]) -> Optional[OtelMessage]:
    """Extract output message from span attributes. Returns an OtelMessage or ComplexOtelMessage object."""
    attrs = span.get("attributes") or {}
    for k in ("gen_ai.output.text", "gen_ai.completion", "gen_ai.output"):
        if k in attrs and isinstance(attrs[k], str):
            return OtelMessage(role="assistant", text=attrs[k])
    out = attrs.get("gen_ai.output.messages")
    if isinstance(out, str):
        try:
            msgs = json.loads(out)
            if len(msgs) > 1:
                raise ValueError(f"Unexpected output messages fromat: expected a single message, got {len(msgs)} messages")
            return ComplexOtelMessage(role="assistant", message_info=reconstruct_each_part_in_message_info(msgs[0]), raw_reconstructed_text=reconstruct_llm_output(msgs[0]))
        except Exception as err:
            raise ValueError(f"Failed parsing {out}") from err
    if isinstance(out, list) and out:
        return OtelMessage(role="assistant", text=message_content_text(out[-1]))
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
    messages: List[OtelMessage]  # original message list (required)
    out_message: Optional[OtelMessage]
    prompt_tokens: Optional[int]  # from gen_ai.usage.prompt_tokens
    completion_tokens: Optional[int]  # from gen_ai.usage.completion_tokens
    temperature: Optional[float]
    max_tokens_recorded: Optional[int]


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
    sorted_spans = sorted(spans, key=lambda s: s['span_id']) #to make filtering consistant between runs
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
        calls.append(
            RawCall(
                call_id=s.get("span_id") or "",
                trace_id=s.get("trace_id") or "",
                t_start_ms=t_start,
                t_end_ms=t_end,
                model=str(attrs.get("gen_ai.request.model") or ""),
                messages=messages,
                out_message=out_message,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                temperature=attrs.get("gen_ai.request.temperature"),
                max_tokens_recorded=attrs.get("gen_ai.request.max_tokens"),
            )
        )
    return calls


# ---------------------------------------------------------------------------
# Causal dependency detection (message-level)
# ---------------------------------------------------------------------------

def is_causal_dep(a: RawCall, b: RawCall) -> bool:
    """Return True if call B causally depends on call A.

    A call B depends on A if any assistant message in B's message list has content
    that matches A's output text (full content match, not a snippet).
    This means A's output was injected into B's prompt as a prior assistant turn.
    """
    if not a.out_message or not b.messages:
        return False
    #match entire output message:
    a_out = norm_text(a.out_message.text)
    for msg in b.messages:
        if output_matches_message(a_out, msg):
            return True
    #try matching parts
    if isinstance(a.out_message, ComplexOtelMessage) and "parts" in a.out_message.message_info and len(a.out_message.message_info["parts"]) > 1:
        #this means this output message contains several parts, and will be interpreted as more than one message in the calls history
        parts = a.out_message.message_info["parts"]
        parts_text = a.out_message.message_info["parts_text"]
        
        # Determine structure: check if first part is content (text) or tool_call
        first_part_is_content = parts[0]["type"] != "tool_call"
        
        # Case 1: Only tool calls [tool_call_1, tool_call_2, ..., tool_call_n]
        # Case 2: Content + tool calls [content, tool_call_1, tool_call_2, ..., tool_call_n]
        
        if not first_part_is_content:
            # Case 1: Only tool calls - each tool call appears separately in input messages
            return _try_match_parts(parts, parts_text, b.messages, combine_content_with_tools=False)
        else:
            # Case 2: Content + tool calls - try two matching strategies:
            # Strategy A: Each part appears separately (content has offset 1, tools have offset 2)
            if _try_match_parts(parts, parts_text, b.messages, combine_content_with_tools=False):
                return True
            # Strategy B: Content is combined with each tool call
            # [content + tool_call_1, content + tool_call_2, ..., content + tool_call_n]
            # Combined messages are treated as tool calls (offset 2)
            if _try_match_parts(parts, parts_text, b.messages, combine_content_with_tools=True):
                return True
        
        return False
    return False


def _try_match_parts(parts: list, parts_text: list, b_messages: list, combine_content_with_tools: bool) -> bool:
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
            text_to_match = content_text + '\n' +first_part_text
        else:
            # Just the first part text
            text_to_match = first_part_text
        
        # Check if the text matches
        if output_matches_message(text_to_match, msg):
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
        for part, part_text in zip(parts_to_match[1:], parts_text_to_match[1:]):
            next_index_to_check = candidate_i + offset
            if next_index_to_check >= len(b_messages):
                candidate_ok = False
                break
            
            msg = b_messages[next_index_to_check]
            
            # Prepare text to match
            if combine_content_with_tools:
                # Concatenate content with tool call text
                text_to_match = content_text + '\n' + part_text
            else:
                # Just the part text
                text_to_match = part_text
            
            # Check if the text matches the message
            if not output_matches_message(text_to_match, msg):
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


@dataclass
class InputSegment:
    """One segment of an LLM call's input prompt, at message granularity.

    type:
        "shared"  — leading messages identical to a predecessor call's messages
                    (KV cache hit opportunity — these tokens are already cached)
        "output"  — a single assistant message whose content is a predecessor's output
                    (injected result; at replay time, substitute with actual generated text)
        "unique"  — messages unique to this call (no predecessor shares them)

    message_count: how many messages this segment covers
    token_count: estimated or recorded token count for this segment
    source_node_id: which predecessor node this segment comes from (shared/output only)
    """

    type: str  # "shared" | "output" | "unique"
    message_count: int
    token_count: int
    source_node_id: Optional[str] = None


def decompose_input(
    call: RawCall,
    predecessors: List[RawCall],
    predecessor_node_ids: List[str],
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
        msg_chars = sum(len(message_content_text(m)) for m in msg_list)
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
                token_count=msgs_to_tokens(shared_msgs),
                source_node_id=predecessor_node_ids[best_pred_idx],
            )
        )
        cursor = best_prefix_count

    # Step 2: After the shared prefix, scan for injected outputs from predecessors
    # Each predecessor's output may appear as an assistant message in the remaining messages
    while cursor < total_msgs:
        # Find the earliest remaining message that matches any predecessor's output
        best_out_pred_idx: int = -1
        best_out_msg_idx: int = total_msgs  # position in messages[]

        for pred_idx, pred in enumerate(predecessors):
            if not pred.out_message:
                continue
            pred_out = norm_text(pred.out_message.text)
            # Search in remaining messages
            for msg_idx in range(cursor, total_msgs):
                if output_matches_message(pred_out, messages[msg_idx]):
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
                        token_count=msgs_to_tokens(remaining_msgs),
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
                    token_count=msgs_to_tokens(gap_msgs),
                )
            )
            cursor = best_out_msg_idx

        # The injected output message
        out_msg = messages[cursor]
        _ct = predecessors[best_out_pred_idx].completion_tokens
        out_tokens: int = _ct if _ct is not None else msgs_to_tokens([out_msg])
        segments.append(
            InputSegment(
                type="output",
                message_count=1,
                token_count=out_tokens,
                source_node_id=predecessor_node_ids[best_out_pred_idx],
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


@dataclass
class GraphCall:
    """An LLM call within a graph node, ready for replay."""

    call_id: str
    model: str
    messages: List[OtelMessage]  # original messages (for replay)
    expected_output: str  # original output
    input_segments: List[InputSegment]
    total_input_tokens: int
    expected_output_tokens: int  # set max_tokens to this; disable EOS for downstream prefix
    temperature: Optional[float]
    max_tokens_recorded: Optional[int]


@dataclass
class GraphNode:
    """A node in the replay graph. Contains exactly one LLM call.

    The replayer starts this node when ALL predecessor nodes have completed,
    then waits `wait_ms` before dispatching the call.
    """

    node_id: str
    call: GraphCall
    predecessor_node_ids: List[str]  # all nodes that must complete before this one starts
    causal_predecessor_ids: List[str]  # subset of predecessor_node_ids that are causal dependencies - this is stored only for visualization purpose
    wait_ms: int  # delay after last predecessor finishes (ms)
    # Timing info (informational, from original trace)
    t_start_ms: int
    t_end_ms: int


@dataclass
class ReplayGraph:
    """The complete replay graph for one combined trace file."""

    nodes: Dict[str, GraphNode]
    root_node_ids: List[str]  # nodes with no predecessors (start immediately)
    source_file: str


# ---------------------------------------------------------------------------
# Graph builder
# ---------------------------------------------------------------------------


def build_graph(
    calls: List[RawCall],
    source_file: str = "",
) -> ReplayGraph:
    """Build a ReplayGraph from a list of raw calls.

    Each RawCall becomes exactly one GraphNode. Predecessor relationships are
    inferred from causal dependencies (output→input message matching) with a
    fallback to the immediately preceding call for timing-only chains.

    Steps:
    1. For each call, find its direct predecessor calls (causal dep or timing fallback).
    2. Apply transitive reduction: remove predecessors that are already ancestors
       of another predecessor (keep only direct edges).
    3. Decompose each call's messages into segments relative to all ancestor calls.
    4. Build GraphNode objects with predecessor_node_ids and wait_ms.
    """
    if not calls:
        return ReplayGraph(nodes={}, root_node_ids=[], source_file=source_file)

    n = len(calls)
    # Assign node IDs 1:1 with calls, incorporating span_id for traceability
    node_ids = [f"node_{i:03d}_{calls[i].call_id}" for i in range(n)]

    # ---------------------------------------------------------------------------
    # Step 1: Find direct predecessors for each call
    # ---------------------------------------------------------------------------
    # predecessor_indices[i] = list of call indices that are DIRECT predecessors of call i
    predecessor_indices: List[List[int]] = [[] for _ in range(n)]
    # is_causal_edge[i] = True if predecessor_indices[i] was derived from causal deps
    #                      False if it was a timing-fallback assignment
    causal_preds: List[List[int]] = [[] for _ in range(n)] # the list of all causal predecessors per node

    def is_causal_ancestor(ancestor: int, descendant: int) -> bool:
        """Return True if ancestor is a (transitive) predecessor of descendant
        following only causal edges (not timing-fallback edges)."""
        visited: Set[int] = set()
        stack = list(causal_preds[descendant])
        while stack:
            node = stack.pop()
            if node == ancestor:
                return True
            if node not in visited:
                visited.add(node)
                stack.extend(causal_preds[node])
        return False

    def is_valid_predecessor(predecessor_candidate, curr_call):
        # checks if candidate can be a predecessor to curr_call. Make sure times are not overlapping
        # since the nodes are sorted, we can assume the candidate doesn't start after curr_call
        if curr_call.t_start_ms < predecessor_candidate.t_end_ms:
            # curr starts before the candidate ends.
            return False
        return True

    for i in range(1, n):
        # Collect all calls that causally feed call i
        curr_causal_preds: List[int] = []
        for j in range(i - 1, -1, -1):
            if is_causal_dep(calls[j], calls[i]):
                curr_causal_preds.append(j)

        if curr_causal_preds:
            # Transitive reduction: remove j if it's already a causal ancestor of another
            # causal pred k. Only traverse causal edges — timing-fallback edges do not
            # create transitive relationships that should suppress direct causal deps.
            direct_preds = [j for j in curr_causal_preds if not any(is_causal_ancestor(j, k) for k in curr_causal_preds if k != j)]
            predecessor_indices[i] = direct_preds
            causal_preds[i].extend(predecessor_indices[i])

        """
        Add a temporal fallback predecessor (the closest non-overlapping node) if one exists.
        This ensures we don't have long wait times when causal predecessors are distant.
        Causal detection (output matching) doesn't catch all dependencies, so we use
        temporal proximity as a conservative fallback to maintain realistic timing.
        """
        predecessor_index = None  # Will remain None if no valid predecessor found
        # Look for the closest possible predecessor. It's not necessarily the immediate predecessor, as they can be executed in parallel
        for j in range(i-1, -1, -1):
            if is_valid_predecessor(calls[j], calls[i]):
                predecessor_index = j
                break
        # Only add temporal predecessor if one was found and it's not already a causal predecessor
        if predecessor_index is not None and predecessor_index not in predecessor_indices[i]:
            predecessor_indices[i].append(predecessor_index)

    # ---------------------------------------------------------------------------
    # Step 2: Compute all ancestors per call (for segment decomposition)
    # ---------------------------------------------------------------------------
    def all_ancestor_indices(call_idx: int) -> List[int]:
        """Return all ancestor call indices (transitive closure of predecessors)."""
        visited: Set[int] = set()
        stack = list(predecessor_indices[call_idx])
        while stack:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                stack.extend(predecessor_indices[node])
        return list(visited)

    # ---------------------------------------------------------------------------
    # Step 3: Build GraphNodes
    # ---------------------------------------------------------------------------
    nodes: Dict[str, GraphNode] = {}
    for i, (nid, rc) in enumerate(zip(node_ids, calls, strict=False)):
        # All ancestor calls (for segment decomposition — includes transitive predecessors)
        ancestor_idxs = all_ancestor_indices(i)
        ancestor_calls = [calls[j] for j in ancestor_idxs]
        ancestor_node_ids = [node_ids[j] for j in ancestor_idxs]

        # Decompose input into message-level segments
        segments = decompose_input(rc, ancestor_calls, ancestor_node_ids)

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

        graph_call = GraphCall(
            call_id=rc.call_id,
            model=rc.model,
            messages=[
                {"role": x.role, "content": x.text} for x in rc.messages
            ],  # convert to a list of dictionaries representing a message with role and content only.
            expected_output=(rc.out_message.text or "" if rc.out_message else ""),
            input_segments=segments,
            total_input_tokens=total_input_tokens,
            expected_output_tokens=expected_output_tokens,
            temperature=rc.temperature,
            max_tokens_recorded=rc.max_tokens_recorded,
        )

        # Compute wait_ms: gap between when the last predecessor ends and this call starts
        pred_idxs = predecessor_indices[i]
        if pred_idxs:
            last_pred_end_ms = max(calls[j].t_end_ms for j in pred_idxs)
            wait_ms = max(0, rc.t_start_ms - last_pred_end_ms)
        else:
            wait_ms = 0

        nodes[nid] = GraphNode(
            node_id=nid,
            call=graph_call,
            predecessor_node_ids=[node_ids[j] for j in pred_idxs],
            causal_predecessor_ids=[node_ids[j] for j in causal_preds[i]],
            wait_ms=wait_ms,
            t_start_ms=rc.t_start_ms,
            t_end_ms=rc.t_end_ms,
        )

    root_node_ids = [node_ids[i] for i in range(n) if not predecessor_indices[i]]

    return ReplayGraph(nodes=nodes, root_node_ids=root_node_ids, source_file=source_file)


# ---------------------------------------------------------------------------
# Serialization
# ---------------------------------------------------------------------------


def segment_to_dict(seg: InputSegment) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "type": seg.type,
        "message_count": seg.message_count,
        "token_count": seg.token_count,
    }
    if seg.source_node_id is not None:
        d["source_node_id"] = seg.source_node_id
    return d


def graph_call_to_dict(gc: GraphCall) -> Dict[str, Any]:
    return {
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


def graph_node_to_dict(node: GraphNode) -> Dict[str, Any]:
    return {
        "node_id": node.node_id,
        "t_start_ms": node.t_start_ms,
        "t_end_ms": node.t_end_ms,
        "predecessor_node_ids": node.predecessor_node_ids,
        "causal_predecessor_ids": node.causal_predecessor_ids,
        "wait_ms": node.wait_ms,
        "call": graph_call_to_dict(node.call),
    }


def graph_to_dict(graph: ReplayGraph) -> Dict[str, Any]:
    return {
        "source_file": graph.source_file,
        "root_node_ids": graph.root_node_ids,
        "node_count": len(graph.nodes),
        "nodes": {nid: graph_node_to_dict(node) for nid, node in graph.nodes.items()},
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
    src = f" <- {seg.source_node_id}" if seg.source_node_id else ""
    messages = "\n\t\t\t".join(f"{x['role']} : {_shorten_string(x['content'])}" for x in messages)
    return f"{label}({seg.message_count}msg/{seg.token_count}t{src})\n\t\t\t{messages}"


def _topo_order(graph: ReplayGraph) -> List[str]:
    """Return node IDs in topological order (BFS from roots)."""
    # Build successor map from predecessor_node_ids
    successors: Dict[str, List[str]] = {nid: [] for nid in graph.nodes}
    for nid, node in graph.nodes.items():
        for pred_id in node.predecessor_node_ids:
            if pred_id in successors:
                successors[pred_id].append(nid)

    visited: Set[str] = set()
    queue = list(graph.root_node_ids)
    order: List[str] = []
    while queue:
        nid = queue.pop(0)
        if nid in visited:
            continue
        visited.add(nid)
        order.append(nid)
        for succ_id in successors.get(nid, []):
            queue.append(succ_id)
    return order


def map_input_seq_to_messages(gc):
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
        f"REPLAY GRAPH   {len(graph.nodes)} nodes   source: {source_name}"
        if source_name
        else f"REPLAY GRAPH   {len(graph.nodes)} nodes"
    )
    print()
    print(f"  {title}")
    print("  " + "-" * len(title))
    print()
    print("  Legend:  SHARED = KV-cache prefix reuse (identical leading messages)")
    print("           OUTPUT = predecessor output injected as assistant message")
    print("           UNIQUE = messages unique to this call")
    print()

    for nid in order:
        node = graph.nodes[nid]
        is_root = nid in graph.root_node_ids
        duration_ms = node.t_end_ms - node.t_start_ms
        gc = node.call

        tags = []
        if is_root:
            tags.append("ROOT")
        tag_str = "   " + " | ".join(tags) if tags else ""

        print(
            f"  ╔══ NODE {nid}"
            f"   t={_fmt_ms(node.t_start_ms)} -> {_fmt_ms(node.t_end_ms)}"
            f"  (duration {_fmt_ms(duration_ms)})"
            f"{tag_str}"
        )
        print("  ║")

        if node.predecessor_node_ids:
            preds_str = ", ".join(node.predecessor_node_ids)
            print(f"  ║   waits for: [{preds_str}]  then +{_fmt_ms(node.wait_ms)}")
        else:
            print("  ║   (no predecessors — starts immediately)")
        print("  ║")

        temp_str = f"  temperature={gc.temperature}" if gc.temperature is not None else ""
        print(f"  ║   CALL {gc.call_id}   model={gc.model}{temp_str}")
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
    """Return a compact one-line-per-node summary string (for logging/testing)."""
    lines = []
    for nid in _topo_order(graph):
        node = graph.nodes[nid]
        gc = node.call
        preds = f"after [{', '.join(node.predecessor_node_ids)}] +{node.wait_ms}ms" if node.predecessor_node_ids else "ROOT"
        seg_str = " ".join(_segment_label(s, m) for s, m in map_input_seq_to_messages(gc))
        lines.append(f"[{nid}] {preds}  t={node.t_start_ms}-{node.t_end_ms}ms")
        lines.append(f"    {gc.call_id}: [{seg_str}] -> O({gc.expected_output_tokens}t)")
    return "\n".join(lines)


def visualize_graph(graph, output_file) -> None:
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
    ap.add_argument("--vis_output", default=None, help="If provided, is the path to the graph structure to be displayed in https://viz-js.com/")
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
    print(f"Wrote replay graph ({len(graph.nodes)} nodes, {len(calls)} calls) to {args.output}")

    if args.summary:
        print_graph(graph)
    if args.vis_output:
        visualize_graph(graph, args.vis_output)


if __name__ == "__main__":
    main()

# Made with Bob
