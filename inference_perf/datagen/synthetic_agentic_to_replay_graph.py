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
Dump a synthetic agentic replay graph JSON from a config.

This script is the synthetic counterpart to ``otel_trace_to_replay_graph``: instead
of extracting LLM calls from an OTel trace, it builds one synthetic per-session
replay graph procedurally from a ``synthetic_agentic`` config (config -> theme ->
tokenizer -> build_graph_for_session) and dumps it the same way (JSON, optional DOT
visualization, optional human-readable summary).

Synthetic graphs are per-session and deterministic in ``(config, session_index)``;
use ``--session-index`` to select which session graph to build.
"""

import argparse
import json
from pathlib import Path

from inference_perf.config.config import read_config
from inference_perf.config.datagen.config import DataGenType
from inference_perf.datagen.otel_trace_to_replay_graph import graph_to_dict, print_graph, visualize_graph
from inference_perf.datagen.synthetic_agentic import build_graph_for_session
from inference_perf.datagen.synthetic_themes import GENERIC_THEME, load_theme
from inference_perf.utils.custom_tokenizer import CustomTokenizer


def main() -> None:
    """Main entry point."""
    ap = argparse.ArgumentParser(
        description="Dump a synthetic agentic replay graph JSON from a config",
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
    ap.add_argument("--output", required=True, help="Output replay graph JSON file")
    ap.add_argument("--summary", action="store_true", help="Print human-readable graph summary")
    ap.add_argument(
        "--vis_output",
        default=None,
        help="If provided, is the path to the graph structure to be displayed in https://viz-js.com/",
    )
    args = ap.parse_args()

    config = read_config(args.config)
    if config.data.type != DataGenType.SyntheticAgentic or config.data.synthetic_agentic is None:
        raise SystemExit(
            "Config must set data.type: synthetic_agentic with a data.synthetic_agentic block"
        )
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
