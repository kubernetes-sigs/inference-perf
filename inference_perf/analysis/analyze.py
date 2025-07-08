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

import json
import logging
import operator
from pathlib import Path
from typing import Any, Dict, List, Tuple

logger = logging.getLogger(__name__)


def _extract_latency_metric(latency_data: Dict[str, Any], metric_name: str, convert_to_ms: bool = False) -> float | None:
    """Helper to extract a metric's mean value from latency data."""
    metric_data = latency_data.get(metric_name)
    if isinstance(metric_data, dict):
        mean_val = metric_data.get("mean")
        if isinstance(mean_val, (int, float)):
            return mean_val * 1000 if convert_to_ms else mean_val
    return None


def _extract_throughput_metric(throughput_data: Dict[str, Any], metric_name: str) -> float | None:
    """Helper to extract a throughput metric's value."""
    metric_value = throughput_data.get(metric_name)
    if isinstance(metric_value, (int, float)):
        return float(metric_value)
    return None


def _generate_plot(charts_to_generate: List[Dict[str, Any]], suptitle: str, output_path: Path) -> None:
    """Generates and saves a plot with multiple subplots."""
    import matplotlib.pyplot as plt

    if not charts_to_generate:
        logger.warning(f"No data available to generate chart: {output_path.name}")
        return

    num_charts = len(charts_to_generate)
    fig, axes = plt.subplots(1, num_charts, figsize=(7 * num_charts, 6), squeeze=False)
    fig.suptitle(suptitle, fontsize=16)

    for i, chart_info in enumerate(charts_to_generate):
        ax = axes[0, i]
        data = chart_info["data"]
        qps_values = [x[0] for x in data]
        y_values = [x[1] for x in data]

        ax.plot(qps_values, y_values, marker="o", linestyle="-")
        ax.set_title(chart_info["title"])
        ax.set_xlabel(chart_info.get("xlabel", "QPS (requested rate)"))
        ax.set_ylabel(chart_info["ylabel"])
        ax.grid(True)

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(output_path)
    logger.info(f"Chart saved to {output_path}")
    plt.close(fig)


def analyze_reports(report_dir: str) -> None:
    """
    Analyzes performance reports to generate charts.

    Args:
        report_dir: The directory containing the report files.
    """
    try:
        # Check for matplotlib and provide a helpful error message if it's not installed.
        import matplotlib  # noqa: F401
    except ImportError:
        logger.error(
            "matplotlib is not installed. Please install it to use the --analyze feature.\n"
            "You can install it via 'pip install .[analysis]'"
        )
        return

    logger.info(f"Analyzing reports in {report_dir}")

    # Find stage lifecycle metrics files
    report_path = Path(report_dir)
    stage_files = list(report_path.glob("stage_*_lifecycle_metrics.json"))

    if not stage_files:
        logger.error(f"No stage lifecycle metrics files found in {report_dir}")
        return

    # Latency data
    qps_vs_ttft: List[Tuple[float, float]] = []
    qps_vs_ntpot: List[Tuple[float, float]] = []
    qps_vs_itl: List[Tuple[float, float]] = []
    # Throughput data
    qps_vs_itps: List[Tuple[float, float]] = []
    qps_vs_otps: List[Tuple[float, float]] = []
    qps_vs_ttps: List[Tuple[float, float]] = []
    # Throughput vs Latency data
    ttft_vs_otps: List[Tuple[float, float]] = []
    ntpot_vs_otps: List[Tuple[float, float]] = []
    itl_vs_otps: List[Tuple[float, float]] = []

    for stage_file in stage_files:
        try:
            with open(stage_file, "r") as f:
                report_data = json.load(f)

            # Get QPS from report file
            qps = report_data.get("load_summary", {}).get("requested_rate")
            if qps is None:
                logger.warning(f"Could not find requested_rate in {stage_file.name}. Skipping.")
                continue

            success_data = report_data.get("successes", {})
            if not success_data:
                logger.warning(f"No success data in {stage_file.name}. Skipping.")
                continue

            # Extract latency metrics if they exist
            ttft, ntpot, itl = None, None, None
            latency_data = success_data.get("latency", {})
            if latency_data:
                ttft = _extract_latency_metric(latency_data, "time_to_first_token", convert_to_ms=True)
                if ttft is not None:
                    qps_vs_ttft.append((qps, ttft))

                ntpot = _extract_latency_metric(latency_data, "normalized_time_per_output_token", convert_to_ms=True)
                if ntpot is not None:
                    qps_vs_ntpot.append((qps, ntpot))

                itl = _extract_latency_metric(latency_data, "inter_token_latency", convert_to_ms=True)
                if itl is not None:
                    qps_vs_itl.append((qps, itl))

            # Extract throughput metrics if they exist
            otps = None
            throughput_data = success_data.get("throughput", {})
            if throughput_data:
                itps = _extract_throughput_metric(throughput_data, "input_tokens_per_sec")
                if itps is not None:
                    qps_vs_itps.append((qps, itps))

                otps = _extract_throughput_metric(throughput_data, "output_tokens_per_sec")
                if otps is not None:
                    qps_vs_otps.append((qps, otps))

                ttps = _extract_throughput_metric(throughput_data, "total_tokens_per_sec")
                if ttps is not None:
                    qps_vs_ttps.append((qps, ttps))

            # Populate latency vs throughput data
            if otps is not None:
                if ttft is not None:
                    ttft_vs_otps.append((ttft, otps))
                if ntpot is not None:
                    ntpot_vs_otps.append((ntpot, otps))
                if itl is not None:
                    itl_vs_otps.append((itl, otps))

        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {stage_file.name}")
            continue
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing {stage_file.name}: {e}")
            continue

    # --- Generate Latency Plot ---
    latency_charts_to_generate = []
    if qps_vs_ttft:
        latency_charts_to_generate.append(
            {
                "title": "Time to First Token vs. QPS",
                "ylabel": "Mean TTFT (ms)",
                "data": sorted(qps_vs_ttft, key=operator.itemgetter(0)),
            }
        )
    if qps_vs_ntpot:
        latency_charts_to_generate.append(
            {
                "title": "Norm. Time per Output Token vs. QPS",
                "ylabel": "Mean Norm. Time (ms/token)",
                "data": sorted(qps_vs_ntpot, key=operator.itemgetter(0)),
            }
        )
    if qps_vs_itl:
        latency_charts_to_generate.append(
            {
                "title": "Inter-Token Latency vs. QPS",
                "ylabel": "Mean ITL (ms)",
                "data": sorted(qps_vs_itl, key=operator.itemgetter(0)),
            }
        )

    _generate_plot(
        latency_charts_to_generate,
        "Latency vs Request Rate",
        report_path / "latency_vs_qps.png",
    )

    # --- Generate Throughput Plot ---
    throughput_charts_to_generate = []
    if qps_vs_itps:
        throughput_charts_to_generate.append(
            {
                "title": "Input Tokens/sec vs. QPS",
                "ylabel": "Tokens/sec",
                "data": sorted(qps_vs_itps, key=operator.itemgetter(0)),
            }
        )
    if qps_vs_otps:
        throughput_charts_to_generate.append(
            {
                "title": "Output Tokens/sec vs. QPS",
                "ylabel": "Tokens/sec",
                "data": sorted(qps_vs_otps, key=operator.itemgetter(0)),
            }
        )
    if qps_vs_ttps:
        throughput_charts_to_generate.append(
            {
                "title": "Total Tokens/sec vs. QPS",
                "ylabel": "Tokens/sec",
                "data": sorted(qps_vs_ttps, key=operator.itemgetter(0)),
            }
        )

    _generate_plot(
        throughput_charts_to_generate,
        "Throughput vs Request Rate",
        report_path / "throughput_vs_qps.png",
    )

    # --- Generate Throughput vs Latency Curve Plot ---
    throughput_latency_charts_to_generate = []
    if ntpot_vs_otps:
        throughput_latency_charts_to_generate.append(
            {
                "title": "Throughput vs. Norm. Time per Output Token",
                "xlabel": "Mean Norm. Time (ms/token)",
                "ylabel": "Output Tokens/sec",
                "data": sorted(ntpot_vs_otps, key=operator.itemgetter(0)),
            }
        )
    if ttft_vs_otps:
        throughput_latency_charts_to_generate.append(
            {
                "title": "Throughput vs. Time to First Token",
                "xlabel": "Mean TTFT (ms)",
                "ylabel": "Output Tokens/sec",
                "data": sorted(ttft_vs_otps, key=operator.itemgetter(0)),
            }
        )
    if itl_vs_otps:
        throughput_latency_charts_to_generate.append(
            {
                "title": "Throughput vs. Inter-Token Latency",
                "xlabel": "Mean ITL (ms)",
                "ylabel": "Output Tokens/sec",
                "data": sorted(itl_vs_otps, key=operator.itemgetter(0)),
            }
        )

    _generate_plot(
        throughput_latency_charts_to_generate,
        "Latency vs Throughput",
        report_path / "throughput_vs_latency.png",
    )
