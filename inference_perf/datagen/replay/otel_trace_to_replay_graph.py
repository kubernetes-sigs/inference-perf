#!/usr/bin/env python3
# Copyright 2025 The Kubernetes Authors.
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

"""
Convert OTel trace JSON to a replay graph JSON.

This module is the OTel-specific half of trace replay: it reads spans, keeps the
LLM calls, and turns each one into a `RawCall`. Everything after that (predecessor
inference, prompt segmentation, user-facing tagging, the graph structure itself)
lives in `replay_graph_builder` and is shared with every other trace format.

Token counts come from gen_ai.usage.prompt_tokens / completion_tokens when the span
has them; the builder estimates them from text otherwise.
"""

import argparse
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Set, Tuple

from inference_perf.datagen.replay.export_replay_graph_to_dot import export_to_dot
from inference_perf.datagen.replay.otel_trace_utils import (
    reconstruct_llm_output,
    reconstruct_llm_input,
    reconstruct_each_part_in_message_info,
)
from inference_perf.datagen.replay.replay_graph_types import (
    ComplexReplayMessage,
    GraphCall,
    GraphEvent,
    InputSegment,
    ReplayGraph,
    ReplayMessage,
)
from inference_perf.datagen.replay.replay_graph_builder import (  # noqa: F401
    # Used here.
    RawCall,
    _convert_content_and_tool_calls_to_parts,
    build_graph,
    tag_user_facing_events,
    # Re-exported: these used to be defined in this module.
    DEPENDENCY_TYPE,
    PredecessorFinder,
    decompose_input,
    estimate_tokens,
    find_predecessors_by_text_matching,
    get_causal_dep,
    message_content_text,
    message_tokens,
    messages_equal,
    output_matches_message,
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


def extract_messages(span: Dict[str, Any]) -> Tuple[List["ReplayMessage"], int]:
    """Extract messages from span attributes.

    Returns a tuple of (messages, developer_role_normalized_count).
    """
    attrs = span.get("attributes") or {}
    raw = attrs.get("gen_ai.input.messages")
    res = []
    normalized_count = 0
    if raw is None:
        return [], 0
    if isinstance(raw, str):
        try:
            raw = json.loads(raw)
        except Exception as err:
            raise ValueError(f"Failed to parse messages JSON: {raw}") from err

    if isinstance(raw, list):
        for x in raw:
            # sometimes the content field contains a dictionary with several properties
            role = x["role"]
            if role == "developer":
                role = "system"
                x["role"] = "system"
                normalized_count += 1
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
                    # A string-content message that also carries tool linkage
                    # (a role:tool result with tool_call_id, or an assistant
                    # message with tool_calls) must retain that structure, so
                    # keep it as a ComplexReplayMessage rather than a plain one.
                    if x.get("tool_call_id") is not None or x.get("tool_calls") is not None:
                        res.append(
                            ComplexReplayMessage(role=role, message_info=x, raw_reconstructed_text=reconstruct_llm_input(x))
                        )
                    else:
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
        return res, normalized_count  # type: ignore[return-value]
    else:
        return [], 0
    return [], 0


def extract_output_message(span: Dict[str, Any]) -> Optional[ReplayMessage]:
    """Extract output message from span attributes. Returns a ReplayMessage or ComplexReplayMessage object."""
    attrs = span.get("attributes") or {}
    for k in ("gen_ai.output.text", "gen_ai.completion", "gen_ai.output"):
        if k in attrs and isinstance(attrs[k], str):
            return ReplayMessage(role="assistant", text=attrs[k])
    out = attrs.get("gen_ai.output.messages")
    if isinstance(out, str):
        try:
            out = json.loads(out)
        except Exception as err:
            raise ValueError(f"Failed parsing {out}") from err
    if isinstance(out, list):
        if isinstance(out[0], dict):
            if len(out) > 1:
                raise ValueError(f"Unexpected output messages fromat: expected a single message, got {len(out)} messages")
            return ComplexReplayMessage(
                role="assistant",
                message_info=reconstruct_each_part_in_message_info(out[0]),
                raw_reconstructed_text=reconstruct_llm_output(out[0]),
            )
    return None


def is_llm_span(span: Dict[str, Any], include_errors: bool = False) -> bool:
    """Check if span represents an LLM call."""
    name = span.get("name", "") or ""
    attrs = span.get("attributes") or {}
    # Classify on gen_ai.operation.name (name fallback), not key-presence: a fixed-schema
    # source can carry an empty gen_ai.input.messages on every span.
    op = attrs.get("gen_ai.operation.name")
    is_llm = (op == "chat" or name.startswith("chat ")) and bool(attrs.get("gen_ai.input.messages"))
    if not is_llm:
        return False
    if not include_errors:
        status = span.get("status", {})
        if status.get("code", 0) == 2:
            return False
    return True


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


def build_raw_calls(spans: List[Dict[str, Any]], include_errors: bool = False) -> Tuple[List[RawCall], int]:
    """Extract and sort raw LLM calls from spans.

    First filters out duplicate spans (identical start_time, end_time, and attributes),
    then extracts LLM calls from the remaining unique spans.

    Returns a tuple of (calls, developer_role_normalized_count).
    """
    # Filter out duplicate spans first
    unique_spans = filter_duplicate_spans(spans)

    llm_spans = [s for s in unique_spans if is_llm_span(s, include_errors=include_errors)]
    if not llm_spans:
        return [], 0

    t0 = min(parse_iso(s["start_time"]) for s in llm_spans)
    llm_spans.sort(key=lambda s: (parse_iso(s["start_time"]), s.get("span_id", "")))

    calls: List[RawCall] = []
    total_normalized = 0
    for s in llm_spans:
        attrs = s.get("attributes") or {}
        messages, normalized_count = extract_messages(s)
        total_normalized += normalized_count
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

        # Filter out known large standard GenAI attributes
        excluded_keys = {
            "gen_ai.input.messages",
            "gen_ai.output.messages",
            "gen_ai.tool.definitions",
            "gen_ai.output.text",
            "gen_ai.completion",
            "gen_ai.output",
        }
        extra_attrs = {k: v for k, v in attrs.items() if k not in excluded_keys}

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
                tool_definitions=tool_definitions,
                extra_attributes=extra_attrs,
            )
        )
    return calls, total_normalized


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
    if gc.attributes is not None:
        d["attributes"] = gc.attributes
    return d


def graph_event_to_dict(event: GraphEvent) -> Dict[str, Any]:
    d: Dict[str, Any] = {
        "event_id": event.event_id,
        "t_start_ms": event.t_start_ms,
        "t_end_ms": event.t_end_ms,
        "predecessor_event_ids": event.predecessor_event_ids,
        "predecessor_dependency_types": event.predecessor_dependency_types,
        "wait_ms": event.wait_ms,
        "call": graph_call_to_dict(event.call),
    }
    if event.is_user_facing:
        d["is_user_facing"] = True
    if event.is_structured_output_call:
        d["is_structured_output_call"] = True
    if event.is_tool_internal:
        d["is_tool_internal"] = True
    return d


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


def _message_text(msg: Dict[str, Any]) -> str:
    """Readable text for a chat message. Tool-call assistant turns carry no 'content'
    key (they carry 'tool_calls'), so fall back to a compact tool-call rendering."""
    content = msg.get("content")
    if content:
        return str(content)
    tool_calls = msg.get("tool_calls")
    if tool_calls:
        names = [tc.get("function", {}).get("name", "?") for tc in tool_calls]
        return f"[tool_calls: {', '.join(names)}]"
    return ""


def _segment_label(seg: InputSegment, messages: List[Dict[str, str]]) -> str:
    """One-line label for an input segment."""
    type_labels = {"shared": "SHARED", "output": "OUTPUT", "unique": "UNIQUE"}
    label = type_labels.get(seg.type, seg.type.upper())
    src = f" <- {seg.source_event_id}" if seg.source_event_id else ""
    msg_str = "\n\t\t\t".join(f"{x['role']} : {_shorten_string(_message_text(x))}" for x in messages)
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

    calls, _ = build_raw_calls(spans, include_errors=args.include_errors)
    if not calls:
        raise SystemExit("No LLM spans found in trace file")

    graph = build_graph(calls, source_file=args.input)
    # Re-tag with full span list for structural tool-internal detection
    tag_user_facing_events(graph, all_spans=spans)

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
