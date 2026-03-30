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

import argparse
import json
from pathlib import Path
from typing import Any, Dict
import subprocess


def escape_label(text: str) -> str:
    """Escape special characters for DOT labels."""
    return text.replace('"', '\\"').replace('\n', '\\n')


def export_to_dot(graph_data: Dict[str, Any], output_file: str) -> None:
    """Convert ReplayGraph JSON to Graphviz DOT format."""

    nodes = graph_data.get("nodes", {})
    root_node_ids = set(graph_data.get("root_node_ids", []))
    source_file = graph_data.get("source_file", "")

    lines = []
    lines.append("digraph ReplayGraph {")
    lines.append('    rankdir=TB;')
    lines.append('    node [shape=box, style="rounded,filled", fontname="Arial"];')
    lines.append('    edge [fontname="Arial", fontsize=10];')
    lines.append('')

    # Add title as a label
    title = f"Replay Graph\\n{len(nodes)} nodes"
    if source_file:
        title += f"\\nSource: {source_file.split('/')[-1]}"
    lines.append(f'    labelloc="t";')
    lines.append(f'    label="{escape_label(title)}";')
    lines.append(f'    fontsize=16;')
    lines.append('')

    # Add nodes
    for node_id, node_data in nodes.items():
        call = node_data.get("call", {})
        t_start = node_data.get("t_start_ms", 0)
        t_end = node_data.get("t_end_ms", 0)
        duration = t_end - t_start
        wait_ms = node_data.get("wait_ms", 0)

        # Build label
        input_tokens = call.get("total_input_tokens", 0)
        output_tokens = call.get("expected_output_tokens", 0)
        call_id = call.get("call_id", "")

        # Use call_id as the primary identifier, with node_id as secondary
        label_parts = [
            f"{call_id}",
            f"({node_id})",
            f"Start: {t_start:.0f}ms",
            f"i.token: {input_tokens} | o.token: {output_tokens}",
            f"Duration: {duration:.1f}ms"
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
            f"unq:{unique_msgs}m/{unique_total}t"
        ]
        label_parts.append(" | ".join(seg_summary))

        label = "\\n".join(label_parts)

        # Color based on node type
        if node_id in root_node_ids:
            fillcolor = "lightgreen"
        else:
            fillcolor = "lightblue"

        lines.append(f'    {node_id} [label="{escape_label(label)}", fillcolor={fillcolor}];')

    lines.append('')

    # Add legend
    lines.append('    // Legend')
    lines.append('    subgraph cluster_legend {')
    lines.append('        label="Edge Types";')
    lines.append('        style=filled;')
    lines.append('        color=lightgray;')
    lines.append('        fontsize=12;')
    lines.append('        legend_causal [label="Causal\\n(output→input)", shape=box, style=filled, fillcolor=white];')
    lines.append(
        '        legend_temporal [label="Temporal\\n(timing fallback)", shape=box, style=filled, fillcolor=white];')
    lines.append('        legend_causal -> legend_temporal [style=bold, color=blue, label=""];')
    lines.append('        legend_temporal -> legend_causal [style=bold, color=black, label=""];')
    lines.append('    }')
    lines.append('')

    # Add edges
    for node_id, node_data in nodes.items():
        predecessor_ids = node_data.get("predecessor_node_ids", [])
        causal_predecessor_ids = set(node_data.get("causal_predecessor_ids", []))
        wait_ms = node_data.get("wait_ms", 0)

        for pred_id in predecessor_ids:
            is_causal = pred_id in causal_predecessor_ids

            # Style edges differently based on type
            if is_causal:
                # Causal edges: solid line, bold, blue
                edge_style = 'style=bold, color=blue'
                edge_type = 'causal'
            else:
                # Temporal edges: dashed line, gray
                edge_style = 'style=bold, color=black'
                edge_type = 'temporal'

            # Build edge label
            edge_label = ""
            if wait_ms > 0:
                edge_label += f" +{wait_ms:.0f}ms"

            lines.append(f'    {pred_id} -> {node_id} [{edge_style}, label="{edge_label}"];')

    lines.append('}')

    # Write to file
    output_path = Path(output_file)
    output_path.write_text('\n'.join(lines), encoding='utf-8')

    print(f"\n📊 Graph visualization saved to: {output_file}")
    print(f"   View online at: https://viz-js.com/")

