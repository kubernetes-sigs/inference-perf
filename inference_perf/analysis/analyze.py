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
from pathlib import Path
from typing import Any, Dict, List, Tuple

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def _extract_metric(latency_data: Dict[str, Any], metric_name: str, convert_to_ms: bool = False) -> float | None:
    """Helper to extract a metric's mean value from latency data."""
    metric_data = latency_data.get(metric_name)
    if metric_data and "mean" in metric_data and metric_data["mean"] is not None:
        value = metric_data["mean"]
        return value * 1000 if convert_to_ms else value
    return None


def analyze_reports(report_dir: str):
    """
    Analyzes performance reports to generate charts.

    Args:
        report_dir: The directory containing the report files.
    """
    logger.info(f"Analyzing reports in {report_dir}")

    # Find stage lifecycle metrics files
    report_path = Path(report_dir)
    stage_files = list(report_path.glob("stage_*_lifecycle_metrics.json"))

    if not stage_files:
        logger.error(f"No stage lifecycle metrics files found in {report_dir}")
        return

    qps_vs_ttft: List[Tuple[float, float]] = []
    qps_vs_ntpot: List[Tuple[float, float]] = []
    qps_vs_itl: List[Tuple[float, float]] = []

    for stage_file in stage_files:
        try:
            with open(stage_file, "r") as f:
                report_data = json.load(f)

            # Get QPS from report file
            qps = report_data.get("load_summary", {}).get("requested_rate")
            if qps is None:
                logger.warning(f"Could not find requested_rate in {stage_file.name}. Skipping.")
                continue

            latency_data = report_data.get("successes", {}).get("latency", {})
            if not latency_data:
                logger.warning(f"No latency data in successes for {stage_file.name}. Skipping.")
                continue

            # Extract metrics if they exist
            ttft = _extract_metric(latency_data, "time_to_first_token", convert_to_ms=True)
            if ttft is not None:
                qps_vs_ttft.append((qps, ttft))

            ntpot = _extract_metric(latency_data, "normalized_time_per_output_token", convert_to_ms=True)
            if ntpot is not None:
                qps_vs_ntpot.append((qps, ntpot))

            itl = _extract_metric(latency_data, "inter_token_latency", convert_to_ms=True)
            if itl is not None:
                qps_vs_itl.append((qps, itl))

        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {stage_file.name}")
            continue
        except Exception as e:
            logger.error(f"An unexpected error occurred while processing {stage_file.name}: {e}")
            continue

    charts_to_generate = []
    if qps_vs_ttft:
        charts_to_generate.append(
            {
                "title": "Time to First Token vs. QPS",
                "ylabel": "Mean TTFT (ms)",
                "data": sorted(qps_vs_ttft, key=lambda x: x[0]),
            }
        )
    if qps_vs_ntpot:
        charts_to_generate.append(
            {
                "title": "Norm. Time per Output Token vs. QPS",
                "ylabel": "Mean Norm. Time (ms/output token)",
                "data": sorted(qps_vs_ntpot, key=lambda x: x[0]),
            }
        )
    if qps_vs_itl:
        charts_to_generate.append(
            {
                "title": "Inter-Token Latency vs. QPS",
                "ylabel": "Mean ITL (ms)",
                "data": sorted(qps_vs_itl, key=lambda x: x[0]),
            }
        )

    if not charts_to_generate:
        logger.error("No data points collected to generate any charts.")
        return

    # Generate plots
    num_charts = len(charts_to_generate)
    fig, axes = plt.subplots(1, num_charts, figsize=(7 * num_charts, 6), squeeze=False)
    fig.suptitle("Latency vs Request Rate", fontsize=16)

    for i, chart_info in enumerate(charts_to_generate):
        ax = axes[0, i]
        data = chart_info["data"]
        qps_values = [x[0] for x in data]
        y_values = [x[1] for x in data]

        ax.plot(qps_values, y_values, marker="o", linestyle="-")
        ax.set_title(chart_info["title"])
        ax.set_xlabel("QPS (requested rate)")
        ax.set_ylabel(chart_info["ylabel"])
        ax.grid(True)

    fig.tight_layout(rect=[0, 0.03, 1, 0.95])

    chart_path = report_path / "latency_vs_qps.png"
    plt.savefig(chart_path)
    logger.info(f"Chart saved to {chart_path}")
