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


def extract_session_stage_id(report_name: str) -> Optional[int]:
    """Extract stage ID from session report name (e.g. 'stage_0_session_lifecycle_metrics')."""
    match = re.match(r"stage_(\d+)_session_lifecycle_metrics", report_name)
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
                goodput_percentage = goodput_metrics.get("goodput_percentage", 0.0)
                req_goodput = goodput_metrics.get("request_goodput", 0.0)
                tok_goodput = goodput_metrics.get("token_goodput", 0.0)

                goodput_str = f"{goodput_percentage:0.1f}%"
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

    # Print session-level metrics if available
    print_session_summary_tables(reports)


def print_session_summary_tables(reports: List[ReportFile]) -> None:
    """Print session-level summary tables for session-based data generators."""
    session_reports: Dict[int, Dict[str, Any]] = {}

    # Extract session lifecycle metrics reports
    for report in reports:
        stage_id = extract_session_stage_id(report.name)
        if stage_id is not None:
            session_reports[stage_id] = report.contents

    if not session_reports:
        # No session metrics found, skip
        return

    # Sort stages by ID
    sorted_stages = sorted(session_reports.keys())

    console = Console()

    # Table 1: Session Summary
    session_summary_table = Table(
        title="[bold magenta]Session Summary[/bold magenta]", show_header=True, header_style="bold cyan"
    )
    session_summary_table.add_column("Stage", justify="right")
    session_summary_table.add_column("Sessions/s", justify="right")
    session_summary_table.add_column("Total Sessions", justify="right")
    session_summary_table.add_column("Succeeded", justify="right")
    session_summary_table.add_column("Failed", justify="right")
    session_summary_table.add_column("Error %", justify="right")
    session_summary_table.add_column("Total Events", justify="right")
    session_summary_table.add_column("Events Completed", justify="right")
    session_summary_table.add_column("Events Cancelled", justify="right")

    # Table 2: Session Duration & Events
    session_duration_table = Table(
        title="[bold magenta]Session Duration & Events[/bold magenta]", show_header=True, header_style="bold cyan"
    )
    session_duration_table.add_column("Stage", justify="right")
    session_duration_table.add_column("Duration Mean (s)", justify="right")
    session_duration_table.add_column("Duration Med (s)", justify="right")
    session_duration_table.add_column("Duration P90 (s)", justify="right")
    session_duration_table.add_column("Events Mean", justify="right")
    session_duration_table.add_column("Events Med", justify="right")
    session_duration_table.add_column("Events P90", justify="right")

    # Table 3: Session Token Totals
    session_tokens_table = Table(
        title="[bold magenta]Session Token Totals (per session)[/bold magenta]", show_header=True, header_style="bold cyan"
    )
    session_tokens_table.add_column("Stage", justify="right")
    session_tokens_table.add_column("In Tok/Sess Mean", justify="right")
    session_tokens_table.add_column("In Tok/Sess Med", justify="right")
    session_tokens_table.add_column("In Tok/Sess P90", justify="right")
    session_tokens_table.add_column("Out Tok/Sess Mean", justify="right")
    session_tokens_table.add_column("Out Tok/Sess Med", justify="right")
    session_tokens_table.add_column("Out Tok/Sess P90", justify="right")

    for stage_id in sorted_stages:
        contents = session_reports[stage_id]

        # Extract session summary metrics
        num_sessions = contents.get("num_sessions", 0)
        num_sessions_succeeded = contents.get("num_sessions_succeeded", 0)
        num_sessions_failed = contents.get("num_sessions_failed", 0)
        total_events = contents.get("total_events", 0)
        total_events_completed = contents.get("total_events_completed", 0)
        total_events_cancelled = contents.get("total_events_cancelled", 0)
        sessions_per_second = contents.get("sessions_per_second", 0.0)

        # Color code succeeded/failed sessions
        succeeded_str = f"[green]{num_sessions_succeeded}[/]"
        failed_color = "red" if num_sessions_failed > 0 else "green"
        failed_str = f"[{failed_color}]{num_sessions_failed}[/]"

        # Session error rate
        session_error_rate = num_sessions_failed / num_sessions if num_sessions > 0 else 0.0
        session_error_pct = session_error_rate * 100.0
        error_color = "red" if session_error_rate > 0.05 else ("yellow" if session_error_rate > 0 else "green")
        error_str = f"[{error_color}]{session_error_pct:.1f}%[/]"

        # Populate Table 1
        session_summary_table.add_row(
            str(stage_id),
            f"{sessions_per_second:.2f}",
            str(num_sessions),
            succeeded_str,
            failed_str,
            error_str,
            str(total_events),
            str(total_events_completed),
            str(total_events_cancelled),
        )

        # Extract session duration metrics
        session_duration = contents.get("session_duration_sec", {})
        duration_mean = session_duration.get("mean", 0.0)
        duration_median = session_duration.get("median", 0.0)
        duration_p90 = session_duration.get("p90", 0.0)

        # Extract events per session metrics
        num_events = contents.get("num_events", {})
        events_mean = num_events.get("mean", 0.0)
        events_median = num_events.get("median", 0.0)
        events_p90 = num_events.get("p90", 0.0)

        # Populate Table 2
        session_duration_table.add_row(
            str(stage_id),
            f"{duration_mean:.2f}",
            f"{duration_median:.2f}",
            f"{duration_p90:.2f}",
            f"{events_mean:.1f}",
            f"{events_median:.1f}",
            f"{events_p90:.1f}",
        )

        # Extract token metrics
        total_input_tokens = contents.get("total_input_tokens", {})
        input_mean = total_input_tokens.get("mean", 0.0)
        input_median = total_input_tokens.get("median", 0.0)
        input_p90 = total_input_tokens.get("p90", 0.0)

        total_output_tokens = contents.get("total_output_tokens", {})
        output_mean = total_output_tokens.get("mean", 0.0)
        output_median = total_output_tokens.get("median", 0.0)
        output_p90 = total_output_tokens.get("p90", 0.0)

        # Populate Table 3
        session_tokens_table.add_row(
            str(stage_id),
            f"{input_mean:.1f}",
            f"{input_median:.1f}",
            f"{input_p90:.1f}",
            f"{output_mean:.1f}",
            f"{output_median:.1f}",
            f"{output_p90:.1f}",
        )

    # Print all session tables
    console.print(session_summary_table)
    console.print(session_duration_table)
    console.print(session_tokens_table)
