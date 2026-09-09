#!/usr/bin/env python3
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

"""
Dump a synthetic agentic workload from a config to a JSON file.

Two output formats are supported via ``--format``:

  replay (default)
    The native inference-perf replay graph JSON.  This is the synthetic
    counterpart to ``otel_trace_to_replay_graph``: instead of extracting LLM
    calls from an OTel trace it builds one synthetic per-session replay graph
    procedurally (config -> theme -> tokenizer -> build_graph_for_session) and
    serialises it to the same format understood by the replay datagen.

  sharegpt
    ToolACE-ShareGPT JSONL (one record per graph event), compatible with
    ``Beryex/ToolACE-sharegpt``.  Schema per record:
      system       – system-prompt string (extracted from the leading system message)
      tools        – JSON-encoded list of tool definitions
      conversations – list of turns with roles human / gpt / function_call / observation
      metadata – graph metadata block (event_id, predecessors, token budgets,
                       input_segments) preserved so the file can be used for replay
                       or fine-tuning auditing; ignored by standard ShareGPT readers.

Synthetic graphs are per-session and deterministic in ``(config, session_index)``;
use ``--session-index`` to select which session graph to build.
"""

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List

from inference_perf.config.config import read_config
from inference_perf.config.datagen.config import DataGenType
from inference_perf.datagen.replay.otel_trace_to_replay_graph import (
    graph_event_to_dict,
    graph_to_dict,
    print_graph,
    visualize_graph,
)
from inference_perf.datagen.replay.replay_graph_types import ReplayGraph
from inference_perf.datagen.synthetic_agentic.synthetic_agentic_datagen import build_graph_for_session
from inference_perf.datagen.synthetic_agentic.synthetic_themes import GENERIC_THEME, load_theme
from inference_perf.utils.custom_tokenizer import CustomTokenizer


def _tool_names_to_fc_value(names: List[str]) -> str:
    """Encode tool name(s) as a ``function_call`` turn value (always a JSON list)."""
    return json.dumps([{"name": n, "arguments": "{}"} for n in names])


def _event_to_toolace(event_id: str, event: Dict[str, Any]) -> Dict[str, Any]:
    """Convert one graph-event dict to a ToolACE-ShareGPT record.

    Each ``role:assistant`` message with tool calls becomes one ``function_call``
    turn whose value is always a JSON list ``[{"name":…,"arguments":"…"},…]``,
    keeping arguments as a JSON string.  The ``role:tool`` result messages that
    immediately follow are gathered into a single ``observation`` turn as a
    JSON-encoded list ``[{"name":…,"results":…},…]``.
    Plain ``role:assistant`` messages become ``gpt`` turns.
    ``call.expected_output``, when non-empty, is appended as a final ``gpt`` turn.

    When ``expected_output_is_tool_call`` is set, a ``function_call`` turn is
    emitted from ``expected_output_tool_names`` (always a list) instead of a
    ``gpt`` turn.
    """
    call = event["call"]
    messages: List[Dict[str, Any]] = call["messages"]
    expected_output: str = call.get("expected_output", "") or ""
    tool_defs: List[Dict[str, Any]] = call.get("tool_definitions") or []

    system = ""
    conversations: List[Dict[str, str]] = []
    i = 0

    # Pull the leading system message into the top-level field.
    if messages and messages[0].get("role") == "system":
        system = messages[0].get("content", "")
        i = 1

    while i < len(messages):
        msg = messages[i]
        role = msg.get("role", "")

        if role == "user":
            conversations.append({"from": "human", "value": msg.get("content", "")})
            i += 1

        elif role == "assistant":
            tool_calls: List[Dict[str, Any]] = msg.get("tool_calls") or []
            if tool_calls:
                # Always a list, even for a single call.
                fc_value = json.dumps(
                    [
                        {
                            "name": tc.get("function", {}).get("name", ""),
                            "arguments": tc.get("function", {}).get("arguments", "{}"),
                        }
                        for tc in tool_calls
                    ]
                )
                conversations.append({"from": "function_call", "value": fc_value})
                # Collect the immediately-following role:tool messages into one observation.
                # Collect tool results keyed by tool_call_id, then emit in
                # tool_calls order so function_call[i] aligns with observation[i]
                # even when results arrive out of order (e.g. OTel graphs).
                result_by_id: Dict[str, str] = {}
                j = i + 1
                while j < len(messages) and messages[j].get("role") == "tool":
                    tool_msg = messages[j]
                    result_by_id[tool_msg.get("tool_call_id", "")] = tool_msg.get("content", "")
                    j += 1
                if result_by_id:
                    results = [
                        {"name": tc.get("function", {}).get("name", ""), "results": result_by_id.get(tc.get("id", ""), "")}
                        for tc in tool_calls
                    ]
                    conversations.append({"from": "observation", "value": json.dumps(results)})
                i = j
            else:
                conversations.append({"from": "gpt", "value": msg.get("content", "")})
                i += 1

        else:
            # Skip stray tool messages not consumed above (should not occur).
            i += 1

    if expected_output:
        if call.get("expected_output_is_tool_call"):
            tool_names: List[str] = call.get("expected_output_tool_names") or []
            conversations.append({"from": "function_call", "value": json.dumps([{"name": n, "arguments": "{}"} for n in tool_names])})
        else:
            conversations.append({"from": "gpt", "value": expected_output})
    elif call.get("expected_output_is_tool_call"):
        # expected_output is blank but the graph records it will be a tool call.
        tool_names = call.get("expected_output_tool_names") or []
        conversations.append({"from": "function_call", "value": json.dumps([{"name": n, "arguments": "{}"} for n in tool_names])})

    # Preserve graph metadata in a dedicated namespace so standard ShareGPT
    # readers ignore it while inference-perf tooling can recover replay context.
    inference_perf_meta: Dict[str, Any] = {
        "event_id": event_id,
        "predecessor_event_ids": event.get("predecessor_event_ids", []),
        "predecessor_dependency_types": event.get("predecessor_dependency_types", {}),
        "expected_output_tokens": call.get("expected_output_tokens"),
        "input_segments": call.get("input_segments", []),
        "temperature": call.get("temperature"),
        "model": call.get("model", ""),
    }
    if call.get("expected_output_is_tool_call"):
        inference_perf_meta["expected_output_is_tool_call"] = True
    if call.get("expected_output_tool_names") is not None:
        inference_perf_meta["expected_output_tool_names"] = call["expected_output_tool_names"]

    return {
        "system": system,
        "tools": json.dumps(tool_defs),
        "conversations": conversations,
        "metadata": inference_perf_meta,
    }


def graph_to_sharegpt(graph: ReplayGraph) -> List[Dict[str, Any]]:
    """Convert every event in *graph* to a ToolACE-ShareGPT record (one per event)."""
    return [_event_to_toolace(eid, graph_event_to_dict(event)) for eid, event in graph.events.items()]


def main() -> None:
    """Main entry point."""
    ap = argparse.ArgumentParser(
        description="Dump a synthetic agentic workload from a config to a JSON file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    ap.add_argument("--config", required=True, help="Synthetic agentic config YAML file")
    ap.add_argument("--session-index", type=int, default=0, help="Which session graph to build")
    ap.add_argument(
        "--theme",
        default=None,
        help="Which theme to render (default: first key of cfg.theme_mix)",
    )
    ap.add_argument("--output", required=True, help="Output file path")
    ap.add_argument(
        "--format",
        choices=["replay", "sharegpt"],
        default="replay",
        help=(
            "Output format: 'replay' (default) writes the native inference-perf replay graph JSON; "
            "'sharegpt' writes ToolACE-ShareGPT JSONL (one record per graph event, "
            "compatible with Beryex/ToolACE-sharegpt)"
        ),
    )
    ap.add_argument("--summary", action="store_true", help="Print human-readable graph summary")
    ap.add_argument(
        "--vis_output",
        default=None,
        help="If provided, is the path to the graph structure to be displayed in https://viz-js.com/",
    )
    args = ap.parse_args()

    config = read_config(args.config)
    if config.data.type != DataGenType.SyntheticAgentic or config.data.synthetic_agentic is None:
        raise SystemExit("Config must set data.type: synthetic_agentic with a data.synthetic_agentic block")
    cfg = config.data.synthetic_agentic

    theme_name = args.theme if args.theme is not None else next(iter(cfg.theme_mix))
    theme = GENERIC_THEME if theme_name == "generic" else load_theme(theme_name)

    if not (config.tokenizer and config.tokenizer.pretrained_model_name_or_path):
        raise SystemExit(
            "Synthetic graph build needs a tokenizer to size turns. Add a top-level "
            'tokenizer: {pretrained_model_name_or_path: "<model>"} block to your config.'
        )
    tokenizer = CustomTokenizer(config.tokenizer)

    graph = build_graph_for_session(cfg, theme, tokenizer, args.session_index)

    out_path = Path(args.output)
    if args.format == "sharegpt":
        records = graph_to_sharegpt(graph)
        out_path.write_text(
            "\n".join(json.dumps(r, ensure_ascii=False) for r in records),
            encoding="utf-8",
        )
        print(
            f"Wrote {len(records)} ShareGPT records ({len(graph.events)} events) for session "
            f"{args.session_index}, theme {theme_name} to {args.output}"
        )
    else:
        out_path.write_text(
            json.dumps(graph_to_dict(graph), indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        print(
            f"Wrote synthetic replay graph ({len(graph.events)} events) for session "
            f"{args.session_index}, theme {theme_name} to {args.output}"
        )

    if args.summary:
        print_graph(graph)
    if args.vis_output:
        visualize_graph(graph, args.vis_output)


if __name__ == "__main__":
    main()
