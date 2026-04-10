# Copyright 2026 The Kubernetes Authors.
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

import re
from typing import List, Dict, Any, Optional
from rich.console import Console
from rich.table import Table
from rich import print as rprint
from inference_perf.utils.report_file import ReportFile


def extract_stage_id(report_name: str) -> Optional[int]:
    """Extract stage ID from report name (e.g. 'stage_0_lifecycle_metrics')."""
    match = re.match(r"stage_(\d+)_lifecycle_metrics", report_name)
    if match:
        return int(match.group(1))
    return None


def print_summary_table(reports: List[ReportFile]) -> None:
    """Print a summary table of all stages to stdout using rich."""
    stage_reports: Dict[int, Dict[str, Any]] = {}

    for report in reports:
        stage_id = extract_stage_id(report.name)
        if stage_id is not None:
            stage_reports[stage_id] = report.contents

    if not stage_reports:
        rprint("[yellow]No per-stage lifecycle metrics found to display summary table.[/yellow]")
        return

    # Sort stages by ID
    sorted_stages = sorted(stage_reports.keys())

    has_goodput = any("goodput_metrics" in r.get("successes", {}) for r in stage_reports.values())

    console = Console()

    # Table 1: Stage & Throughput Summary
    summary_table = Table(
        title="[bold magenta]Throughput and Goodput Summary[/bold magenta]", show_header=True, header_style="bold cyan"
    )
    summary_table.add_column("Stage", justify="right")
    summary_table.add_column("Req Rate", justify="right")
    summary_table.add_column("Achieved Rate", justify="right")
    summary_table.add_column("Error Rate", justify="right")
    summary_table.add_column("Req/s", justify="right")
    summary_table.add_column("In Tokens/s", justify="right")
    summary_table.add_column("Out Tokens/s", justify="right")
    summary_table.add_column("Tot Tokens/s", justify="right")
    if has_goodput:
        summary_table.add_column("Goodput %", justify="right")
        summary_table.add_column("Req Goodput", justify="right")
        summary_table.add_column("Token Goodput", justify="right")

    # Table 2: Request & Token Latency (ms)
    latency_table = Table(
        title="[bold magenta]Request & Token Latency (ms)[/bold magenta]", show_header=True, header_style="bold cyan"
    )
    latency_table.add_column("Stage", justify="right")
    latency_table.add_column("Req Lat Mean", justify="right")
    latency_table.add_column("Req Lat Med", justify="right")
    latency_table.add_column("Req Lat P90", justify="right")
    latency_table.add_column("TTFT Mean", justify="right")
    latency_table.add_column("TTFT Med", justify="right")
    latency_table.add_column("TTFT P90", justify="right")
    latency_table.add_column("ITL Mean", justify="right")
    latency_table.add_column("ITL Med", justify="right")
    latency_table.add_column("ITL P90", justify="right")

    # Table 3: Token Generation Speed (ms)
    speed_table = Table(
        title="[bold magenta]Token Generation Speed (ms)[/bold magenta]", show_header=True, header_style="bold cyan"
    )
    speed_table.add_column("Stage", justify="right")
    speed_table.add_column("TPOT Mean", justify="right")
    speed_table.add_column("TPOT Med", justify="right")
    speed_table.add_column("TPOT P90", justify="right")
    speed_table.add_column("Norm TPOT Mean", justify="right")
    speed_table.add_column("Norm TPOT Med", justify="right")
    speed_table.add_column("Norm TPOT P90", justify="right")

    # Table 4: Token Lengths
    token_table = Table(
        title="[bold magenta]Token Length Aggregates[/bold magenta]", show_header=True, header_style="bold cyan"
    )
    token_table.add_column("Stage", justify="right")
    token_table.add_column("Prompt Mean", justify="right")
    token_table.add_column("Prompt Med", justify="right")
    token_table.add_column("Prompt P90", justify="right")
    token_table.add_column("Output Mean", justify="right")
    token_table.add_column("Output Med", justify="right")
    token_table.add_column("Output P90", justify="right")

    for stage_id in sorted_stages:
        contents = stage_reports[stage_id]
        load_summary = contents.get("load_summary", {})
        successes = contents.get("successes", {})
        failures = contents.get("failures", {})

        req_rate = load_summary.get("requested_rate", 0.0)
        ach_rate = load_summary.get("achieved_rate", 0.0)

        # Error Rate calculation
        success_count = successes.get("count", 0)
        failed_count = failures.get("count", 0)
        total_count = success_count + failed_count
        error_rate = failed_count / total_count if total_count > 0 else 0.0
        error_rate_pct = error_rate * 100.0

        error_color = "red" if error_rate > 0.05 else ("yellow" if error_rate > 0 else "green")
        error_str = f"[{error_color}]{error_rate_pct:0.1f}%[/]"

        # Throughput extraction
        throughput = successes.get("throughput", {})
        req_per_sec = throughput.get("requests_per_sec", 0.0)
        in_tokens = throughput.get("input_tokens_per_sec", 0.0)
        out_tokens = throughput.get("output_tokens_per_sec", 0.0)
        tot_tokens = throughput.get("total_tokens_per_sec", 0.0)

        # Latency extraction (convert to ms)
        latency = successes.get("latency", {})

        # Request Latency (E2E)
        req_lat = latency.get("request_latency")
        req_lat_mean = req_lat_med = req_lat_p90 = "-"
        if req_lat:
            mean = req_lat.get("mean", 0.0) * 1000.0
            median = req_lat.get("median", 0.0) * 1000.0
            p90 = req_lat.get("p90", 0.0) * 1000.0
            req_lat_mean = f"{mean:0.1f}"
            req_lat_med = f"{median:0.1f}"
            req_lat_p90 = f"{p90:0.1f}"

        # TTFT
        ttft = latency.get("time_to_first_token")
        ttft_mean = ttft_med = ttft_p90 = "-"
        if ttft:
            mean = ttft.get("mean", 0.0) * 1000.0
            median = ttft.get("median", 0.0) * 1000.0
            p90 = ttft.get("p90", 0.0) * 1000.0
            ttft_mean = f"{mean:0.1f}"
            ttft_med = f"{median:0.1f}"
            ttft_p90 = f"{p90:0.1f}"

        # ITL
        itl = latency.get("inter_token_latency")
        itl_mean = itl_med = itl_p90 = "-"
        if itl:
            mean = itl.get("mean", 0.0) * 1000.0
            median = itl.get("median", 0.0) * 1000.0
            p90 = itl.get("p90", 0.0) * 1000.0
            itl_mean = f"{mean:0.1f}"
            itl_med = f"{median:0.1f}"
            itl_p90 = f"{p90:0.1f}"

        # TPOT
        tpot = latency.get("time_per_output_token")
        tpot_mean = tpot_med = tpot_p90 = "-"
        if tpot:
            mean = tpot.get("mean", 0.0) * 1000.0
            median = tpot.get("median", 0.0) * 1000.0
            p90 = tpot.get("p90", 0.0) * 1000.0
            tpot_mean = f"{mean:0.1f}"
            tpot_med = f"{median:0.1f}"
            tpot_p90 = f"{p90:0.1f}"

        # Normalized TPOT
        norm_tpot = latency.get("normalized_time_per_output_token")
        norm_tpot_mean = norm_tpot_med = norm_tpot_p90 = "-"
        if norm_tpot:
            mean = norm_tpot.get("mean", 0.0) * 1000.0
            median = norm_tpot.get("median", 0.0) * 1000.0
            p90 = norm_tpot.get("p90", 0.0) * 1000.0
            norm_tpot_mean = f"{mean:0.1f}"
            norm_tpot_med = f"{median:0.1f}"
            norm_tpot_p90 = f"{p90:0.1f}"

        # Token Length extraction
        prompt_len = successes.get("prompt_len")
        prompt_mean = prompt_med = prompt_p90 = "-"
        if prompt_len:
            prompt_mean = f"{prompt_len.get('mean', 0.0):0.1f}"
            prompt_med = f"{prompt_len.get('median', 0.0):0.1f}"
            prompt_p90 = f"{prompt_len.get('p90', 0.0):0.1f}"

        output_len = successes.get("output_len")
        output_mean = output_med = output_p90 = "-"
        if output_len:
            output_mean = f"{output_len.get('mean', 0.0):0.1f}"
            output_med = f"{output_len.get('median', 0.0):0.1f}"
            output_p90 = f"{output_len.get('p90', 0.0):0.1f}"

        # Populate Table 1
        summary_row = [
            str(stage_id),
            f"{req_rate:0.1f}",
            f"{ach_rate:0.1f}",
            error_str,
            f"{req_per_sec:0.1f}",
            f"{in_tokens:0.1f}",
            f"{out_tokens:0.1f}",
            f"{tot_tokens:0.1f}",
        ]
        if has_goodput:
            goodput_str = "-"
            req_goodput_str = "-"
            tok_goodput_str = "-"
            goodput_metrics = successes.get("goodput_metrics")
            if goodput_metrics:
                goodput_pct = goodput_metrics.get("goodput_pct", 0.0)
                req_goodput = goodput_metrics.get("request_goodput", 0.0)
                tok_goodput = goodput_metrics.get("token_goodput", 0.0)
                
                goodput_str = f"{goodput_pct:0.1f}%"
                req_goodput_str = f"{req_goodput:0.1f}"
                tok_goodput_str = f"{tok_goodput:0.1f}"
            summary_row.extend([goodput_str, req_goodput_str, tok_goodput_str])
        summary_table.add_row(*summary_row)

        # Populate Table 2
        latency_table.add_row(
            str(stage_id),
            req_lat_mean,
            req_lat_med,
            req_lat_p90,
            ttft_mean,
            ttft_med,
            ttft_p90,
            itl_mean,
            itl_med,
            itl_p90,
        )

        # Populate Table 3
        speed_table.add_row(
            str(stage_id),
            tpot_mean,
            tpot_med,
            tpot_p90,
            norm_tpot_mean,
            norm_tpot_med,
            norm_tpot_p90,
        )

        # Populate Table 4
        token_table.add_row(
            str(stage_id),
            prompt_mean,
            prompt_med,
            prompt_p90,
            output_mean,
            output_med,
            output_p90,
        )

    console.print(summary_table)
    console.print(latency_table)
    console.print(speed_table)
    console.print(token_table)
