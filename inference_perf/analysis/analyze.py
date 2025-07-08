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

import argparse
import json
import logging
from pathlib import Path

import matplotlib.pyplot as plt

logger = logging.getLogger(__name__)


def analyze_reports(report_dir: str):
    """
    Analyzes performance reports to generate charts.

    Args:
        report_dir: The directory containing the report files.
    """
    logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger.info(f"Analyzing reports in {report_dir}")

    # Find stage lifecycle metrics files
    report_path = Path(report_dir)
    stage_files = list(report_path.glob("stage_*_lifecycle_metrics.json"))

    if not stage_files:
        logger.error(f"No stage lifecycle metrics files found in {report_dir}")
        return

    qps_vs_ttft = []

    for stage_file in stage_files:
        try:
            with open(stage_file, "r") as f:
                report_data = json.load(f)

            # Get QPS from report file
            qps = report_data.get("load_summary", {}).get("requested_rate")
            if qps is None:
                logger.warning(f"Could not find requested_rate in {stage_file.name}. Skipping.")
                continue

            # Get TTFT from report
            ttft_data = report_data.get("successes", {}).get("latency", {}).get("time_to_first_token")
            if ttft_data and "mean" in ttft_data and ttft_data["mean"] is not None:
                ttft_mean_s = ttft_data["mean"]
                ttft_mean_ms = ttft_mean_s * 1000
                qps_vs_ttft.append((qps, ttft_mean_ms))
            else:
                logger.warning(f"Could not find mean Time to First Token in {stage_file.name}. Skipping.")
        except json.JSONDecodeError:
            logger.error(f"Error decoding JSON from {stage_file.name}")
            continue

    if not qps_vs_ttft:
        logger.error("No data points collected to generate a chart.")
        return

    # Sort by QPS for plotting
    qps_vs_ttft.sort(key=lambda x: x[0])
    qps_values = [x[0] for x in qps_vs_ttft]
    ttft_values = [x[1] for x in qps_vs_ttft]

    # Generate plot
    plt.figure(figsize=(10, 6))
    plt.plot(qps_values, ttft_values, marker="o", linestyle="-")
    plt.title("Time to First Token vs. QPS")
    plt.xlabel("QPS (requested rate)")
    plt.ylabel("Mean Time to First Token (ms)")
    plt.grid(True)

    chart_path = report_path / "ttft_vs_qps.png"
    plt.savefig(chart_path)
    logger.info(f"Chart saved to {chart_path}")


def main():
    """Command-line entry point."""
    parser = argparse.ArgumentParser(description="Analyze inference-perf reports.")
    parser.add_argument("-r", "--report-dir", required=True, help="Directory containing the reports to analyze.")
    args = parser.parse_args()
    analyze_reports(args.report_dir)


if __name__ == "__main__":
    main()
