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


# !/usr/bin/env python3
"""
Export ReplayGraph to Graphviz DOT format for visualization.

Usage:
    python export_replay_graph_to_dot.py --input replay_graph.json --output graph.dot

Then visualize at: https://viz-js.com/
"""

from pathlib import Path
from typing import Any, Dict


def escape_label(text: str) -> str:
    """Escape special characters for DOT labels."""
    return text.replace('"', '\\"').replace("\n", "\\n")


def export_to_dot(graph_data: Dict[str, Any], output_file: str) -> None:
    """Convert ReplayGraph JSON to Graphviz DOT format."""

    events = graph_data.get("events", {})
    root_event_ids = set(graph_data.get("root_event_ids", []))
    source_file = graph_data.get("source_file", "")

    lines = []
    lines.append("digraph ReplayGraph {")
    lines.append("    rankdir=TB;")
    lines.append('    node [shape=box, style="rounded,filled", fontname="Arial"];')
    lines.append('    edge [fontname="Arial", fontsize=10];')
    lines.append("")

    # Add title as a label
    title = f"Replay Graph\\n{len(events)} events"
    if source_file:
        title += f"\\nSource: {source_file.split('/')[-1]}"
    lines.append('    labelloc="t";')
    lines.append(f'    label="{escape_label(title)}";')
    lines.append("    fontsize=16;")
    lines.append("")

    # Add events
    for event_id, event_data in events.items():
        call = event_data.get("call", {})
        t_start = event_data.get("t_start_ms", 0)
        t_end = event_data.get("t_end_ms", 0)
        duration = t_end - t_start
        wait_ms = event_data.get("wait_ms", 0)

        # Build label
        input_tokens = call.get("total_input_tokens", 0)
        output_tokens = call.get("expected_output_tokens", 0)
        call_id = call.get("call_id", "")

        # Use call_id as the primary identifier, with event_id as secondary
        label_parts = [
            f"{call_id}",
            f"({event_id})",
            f"Start: {t_start:.0f}ms",
            f"i.token: {input_tokens} | o.token: {output_tokens}",
            f"Duration: {duration:.1f}ms",
        ]

        # Add wait time if non-zero
        if wait_ms > 0:
            label_parts.insert(3, f"Wait: {wait_ms:.0f}ms")

        # Add segment info - always show all segment types
        segments = call.get("input_segments", [])
        shared_total = 0
        output_total = 0
        unique_total = 0
        shared_msgs = 0
        unique_msgs = 0

        for seg in segments:
            seg_type = seg.get("type", "")
            msg_count = seg.get("message_count", 0)
            token_count = seg.get("token_count", 0)
            if seg_type == "shared":
                shared_total += token_count
                shared_msgs += msg_count
            elif seg_type == "output":
                output_total += token_count
            elif seg_type == "unique":
                unique_total += token_count
                unique_msgs += msg_count

        # Always show all three segment types
        seg_summary = [
            f"shared:{shared_msgs}m/{shared_total}t",
            f"output:{output_total}t",
            f"unq:{unique_msgs}m/{unique_total}t",
        ]
        label_parts.append(" | ".join(seg_summary))

        label = "\\n".join(label_parts)

        # Color based on event type
        if event_id in root_event_ids:
            fillcolor = "lightgreen"
        else:
            fillcolor = "lightblue"

        lines.append(f'    "{event_id}" [label="{escape_label(label)}", fillcolor={fillcolor}];')

    lines.append("")

    # Add legend
    lines.append("    // Legend")
    lines.append("    subgraph cluster_legend {")
    lines.append('        label="Edge Types";')
    lines.append("        style=filled;")
    lines.append("        color=lightgray;")
    lines.append("        fontsize=12;")
    lines.append('        legend_full [label="Full Match", shape=box, style=filled, fillcolor=white];')
    lines.append('        legend_tool_ids [label="Tool Call IDs", shape=box, style=filled, fillcolor=white];')
    lines.append('        legend_split [label="Split Parts", shape=box, style=filled, fillcolor=white];')
    lines.append('        legend_combined [label="Content+Tools", shape=box, style=filled, fillcolor=white];')
    lines.append('        legend_drop [label="Drop Content", shape=box, style=filled, fillcolor=white];')
    lines.append('        legend_temporal [label="Temporal", shape=box, style=filled, fillcolor=white];')
    lines.append('        legend_full -> legend_tool_ids [style=bold, color=blue, label=""];')
    lines.append('        legend_tool_ids -> legend_split [style=bold, color=cyan, label=""];')
    lines.append('        legend_split -> legend_combined [style=bold, color=green, label=""];')
    lines.append('        legend_combined -> legend_drop [style=bold, color=purple, label=""];')
    lines.append('        legend_drop -> legend_temporal [style=bold, color=orange, label=""];')
    lines.append('        legend_temporal -> legend_full [style=bold, color=black, label=""];')
    lines.append("    }")
    lines.append("")

    # Add edges
    for event_id, event_data in events.items():
        predecessor_ids = event_data.get("predecessor_event_ids", [])
        predecessor_dependency_types = event_data.get("predecessor_dependency_types", {})
        wait_ms = event_data.get("wait_ms", 0)

        for pred_id in predecessor_ids:
            dep_type = predecessor_dependency_types.get(pred_id, "temporal")

            # Style edges differently based on dependency type
            if dep_type == "full_match":
                edge_style = "style=bold, color=blue"
                edge_label_prefix = "FM"
            elif dep_type == "tool_call_ids_matched":
                edge_style = "style=bold, color=cyan"
                edge_label_prefix = "TID"
            elif dep_type == "split_parts_matched":
                edge_style = "style=bold, color=green"
                edge_label_prefix = "SP"
            elif dep_type == "content_and_split_tools_match":
                edge_style = "style=bold, color=purple"
                edge_label_prefix = "CT"
            elif dep_type == "drop_content_split_parts":
                edge_style = "style=bold, color=orange"
                edge_label_prefix = "DC"
            else:  # temporal
                edge_style = "style=bold, color=black"
                edge_label_prefix = "T"

            # Build edge label
            edge_label = edge_label_prefix
            if wait_ms > 0:
                edge_label += f" +{wait_ms:.0f}ms"

            lines.append(f'    "{pred_id}" -> "{event_id}" [{edge_style}, label="{edge_label}"];')

    lines.append("}")

    # Write to file
    output_path = Path(output_file)
    output_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"\n📊 Graph visualization saved to: {output_file}")
    print("   View online at: https://viz-js.com/")
