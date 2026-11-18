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

import json
import logging
import operator
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

# Values accepted for the `rate` argument of analyze_reports. Each one names a
# `<value>_rate` key in the load_summary of a stage report.
RATE_CHOICES = ("achieved", "requested")


class AnalysisOptionError(ValueError):
    """An analyze_reports option (labels, rate or latency_stat) that is not valid.

    It is a ValueError, so callers that catch ValueError also catch it.
    """


_SUMMARY_STATS = ("mean", "median", "min", "max")
_PERCENTILE_STAT = re.compile(r"p(\d+(?:\.\d+)?)")
_PLOTTED_LATENCY_METRICS = ("time_to_first_token", "normalized_time_per_output_token", "inter_token_latency")


def _latency_stat_key(stat: str) -> str:
    """Returns the key that holds a latency statistic in a stage report.

    Reports write the 50th percentile as "median" and other percentiles as
    "p<value>" (see reportgen.base.summarize). So "p50" reads "median" and
    "p90.0" reads "p90". Names are not case-sensitive.
    """
    name = stat.lower()
    if name in _SUMMARY_STATS:
        return name
    match = _PERCENTILE_STAT.fullmatch(name)
    if match:
        percentile = float(match.group(1))
        if percentile <= 100:
            return "median" if percentile == 50 else f"p{percentile:g}"
    raise AnalysisOptionError(
        f"Latency statistic '{stat}' is not valid. Use mean, median, min, max or a percentile from p0 to p100, such as p90."
    )


def _latency_stat_label(stat_key: str) -> str:
    """Returns the axis label prefix for a latency statistic key: "mean" gives "Mean", "p90" gives "P90"."""
    return stat_key.capitalize() if stat_key in _SUMMARY_STATS else stat_key.upper()


def _check_labels(report_dirs: List[str], labels: List[str]) -> None:
    """Raises AnalysisOptionError if the labels cannot name each report directory once."""
    if len(labels) != len(report_dirs):
        raise AnalysisOptionError(
            f"The number of labels ({len(labels)}) is not equal to the number of report directories ({len(report_dirs)}). "
            "Give one label for each report directory."
        )
    if len(set(labels)) != len(labels):
        raise AnalysisOptionError("Two or more labels are the same. Give each report directory a different label.")
    if len({Path(report_dir) for report_dir in report_dirs}) != len(report_dirs):
        raise AnalysisOptionError("A report directory occurs more than one time. Give each report directory only one time.")


def _extract_latency_metric(
    latency_data: Dict[str, Any], metric_name: str, convert_to_ms: bool = False, stat: str = "mean"
) -> Optional[float]:
    """Helper to extract one statistic of a metric from latency data. The default statistic is the mean."""
    metric_data = latency_data.get(metric_name)
    if isinstance(metric_data, dict):
        value = metric_data.get(stat)
        if isinstance(value, (int, float)):
            return value * 1000 if convert_to_ms else value
    return None


def _metrics_missing_stat(latency_data: Dict[str, object], stat: str) -> List[str]:
    """Returns the plotted latency metrics that are in the report but have no value for `stat`."""
    missing = []
    for name in _PLOTTED_LATENCY_METRICS:
        values = latency_data.get(name)
        if isinstance(values, dict) and stat not in values:
            missing.append(name)
    return missing


def _extract_throughput_metric(throughput_data: Dict[str, Any], metric_name: str) -> Optional[float]:
    """Helper to extract a throughput metric's value."""
    metric_value = throughput_data.get(metric_name)
    if isinstance(metric_value, (int, float)):
        return float(metric_value)
    return None


def _generate_multi_plot(
    chartset_to_generate: List[List[Dict[str, Any]]],
    num_charts: int,
    names: List[str],
    suptitle: str,
    output_path: Path,
) -> None:
    """Generates and saves a plot with multiple subplots."""
    import matplotlib.pyplot as plt

    if not num_charts:
        logger.debug("No chart data available")
        return

    if not chartset_to_generate:
        logger.warning(f"No data available to generate chart: {output_path.name}")
        return

    fig, axes = plt.subplots(1, num_charts, figsize=(7 * num_charts, 6), squeeze=False)
    fig.suptitle(suptitle, fontsize=22)

    for charts_to_generate in chartset_to_generate:
        if not charts_to_generate:
            logger.warning(f"No data available to generate chart: {output_path.name}")
            return
        for i, chart_info in enumerate(charts_to_generate):
            ax = axes[0, i]
            data = chart_info["data"]
            qps_values = [x[0] for x in data]
            y_values = [x[1] for x in data]

            ax.plot(qps_values, y_values, marker="o", linestyle="-")
            ax.set_title(chart_info["title"], fontsize=20)
            ax.set_xlabel(chart_info["xlabel"], fontsize=16)
            ax.set_ylabel(chart_info["ylabel"], fontsize=16)
            ax.tick_params(axis="both", labelsize=14)
            if not names:
                ax.legend([f"Report {idx + 1}" for idx in range(len(chartset_to_generate))])
            else:
                ax.legend(names)
            ax.grid(True)

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(output_path)
    logger.info(f"Chart saved to {output_path}")
    plt.close(fig)


def _generate_plot(charts_to_generate: List[Dict[str, Any]], suptitle: str, output_path: Path) -> None:
    """Generates and saves a plot with multiple subplots."""
    import matplotlib.pyplot as plt

    if not charts_to_generate:
        logger.warning(f"No data available to generate chart: {output_path.name}")
        return

    num_charts = len(charts_to_generate)
    fig, axes = plt.subplots(1, num_charts, figsize=(7 * num_charts, 6), squeeze=False)
    fig.suptitle(suptitle, fontsize=22)

    for i, chart_info in enumerate(charts_to_generate):
        ax = axes[0, i]
        data = chart_info["data"]
        qps_values = [x[0] for x in data]
        y_values = [x[1] for x in data]

        ax.plot(qps_values, y_values, marker="o", linestyle="-")
        ax.set_title(chart_info["title"], fontsize=20)
        ax.set_xlabel(chart_info["xlabel"], fontsize=16)
        ax.set_ylabel(chart_info["ylabel"], fontsize=16)
        ax.tick_params(axis="both", labelsize=14)
        ax.grid(True)

    fig.tight_layout(rect=(0, 0.03, 1, 0.95))
    plt.savefig(output_path)
    logger.info(f"Chart saved to {output_path}")
    plt.close(fig)


def analyze_reports(
    report_dirs: List[str],
    analysis_dir: Optional[str] = None,
    labels: Optional[List[str]] = None,
    rate: str = "achieved",
    latency_stat: str = "mean",
) -> None:
    """
    Analyzes performance reports to generate charts.

    Args:
        report_dirs: The directories that contain the stage_*_lifecycle_metrics.json files.
        analysis_dir: The directory for the unified charts that compare all reports. If None, no unified charts.
        labels: Display names for report_dirs, in the same order, for the unified chart legends.
            If None, the names are "Report 1", "Report 2" and so on.
        rate: The rate on the QPS axis. "achieved" reads load_summary.achieved_rate.
            "requested" reads load_summary.requested_rate, the rate set for the stage.
        latency_stat: The latency statistic to plot: "mean", "median", "min", "max" or a percentile
            such as "p90". A percentile must be in report.request_lifecycle.percentiles of the run.

    Raises:
        AnalysisOptionError: If labels, rate or latency_stat is not valid. It is a ValueError.
    """
    if rate not in RATE_CHOICES:
        raise AnalysisOptionError(f"Rate '{rate}' is not valid. Use one of: {', '.join(RATE_CHOICES)}.")
    stat_key = _latency_stat_key(latency_stat)
    if labels is not None:
        _check_labels(report_dirs, labels)

    rate_key = f"{rate}_rate"
    qps_xlabel = f"QPS ({rate} rate)"
    stat_label = _latency_stat_label(stat_key)

    try:
        # Check for matplotlib and provide a helpful error message if it's not installed.
        import matplotlib  # noqa: F401
    except ImportError:
        logger.error(
            "matplotlib is not installed. Please install it to use the --analyze feature.\n"
            "You can install it via 'pip install .[analysis]'"
        )
        return

    chartset = {}

    for report_dir in report_dirs:
        logger.info(f"Analyzing reports in {report_dir}")

        # Find stage lifecycle metrics files
        report_path = Path(report_dir)
        stage_files = list(report_path.glob("stage_*_lifecycle_metrics.json"))

        if not stage_files:
            logger.error(f"No stage lifecycle metrics files found in {report_dir}")
            return

        # Latency data (concurrency)
        concurrency_vs_ttft: List[Tuple[float, float]] = []
        concurrency_vs_ntpot: List[Tuple[float, float]] = []
        concurrency_vs_itl: List[Tuple[float, float]] = []
        # Throughput data (concurrency)
        concurrency_vs_itps: List[Tuple[float, float]] = []
        concurrency_vs_otps: List[Tuple[float, float]] = []
        concurrency_vs_ttps: List[Tuple[float, float]] = []
        # Latency data (QPS)
        qps_vs_ttft: List[Tuple[float, float]] = []
        qps_vs_ntpot: List[Tuple[float, float]] = []
        qps_vs_itl: List[Tuple[float, float]] = []
        # Throughput data (QPS)
        qps_vs_itps: List[Tuple[float, float]] = []
        qps_vs_otps: List[Tuple[float, float]] = []
        qps_vs_ttps: List[Tuple[float, float]] = []
        qps_vs_goodput_percentage: List[Tuple[float, float]] = []
        qps_vs_request_goodput_rate: List[Tuple[float, float]] = []
        # Throughput vs Latency data
        ttft_vs_otps: List[Tuple[float, float]] = []
        ntpot_vs_otps: List[Tuple[float, float]] = []
        itl_vs_otps: List[Tuple[float, float]] = []

        for stage_file in stage_files:
            try:
                with open(stage_file, "r") as f:
                    report_data = json.load(f)

                # Get concurrency
                concurrency = report_data.get("load_summary", {}).get("concurrency", None)

                # Get QPS from report file
                qps = report_data.get("load_summary", {}).get(rate_key)
                if qps is None:
                    logger.warning(f"Could not find {rate_key} in {stage_file.name}. Skipping.")
                    continue

                success_data = report_data.get("successes", {})
                if not success_data:
                    logger.warning(f"No success data in {stage_file.name}. Skipping.")
                    continue

                # Extract latency metrics if they exist
                ttft, ntpot, itl = None, None, None
                latency_data = success_data.get("latency", {})
                if latency_data:
                    missing = _metrics_missing_stat(latency_data, stat_key)
                    if missing:
                        logger.warning(
                            f"No '{stat_key}' value for {', '.join(missing)} in {stage_file.name}. Skipping these points. "
                            "To record a percentile, add it to report.request_lifecycle.percentiles in the run config."
                        )

                    ttft = _extract_latency_metric(latency_data, "time_to_first_token", convert_to_ms=True, stat=stat_key)
                    if ttft is not None:
                        if concurrency:
                            concurrency_vs_ttft.append((concurrency, ttft))
                        else:
                            qps_vs_ttft.append((qps, ttft))

                    ntpot = _extract_latency_metric(
                        latency_data,
                        "normalized_time_per_output_token",
                        convert_to_ms=True,
                        stat=stat_key,
                    )
                    if ntpot is not None:
                        if concurrency:
                            concurrency_vs_ntpot.append((concurrency, ntpot))
                        else:
                            qps_vs_ntpot.append((qps, ntpot))

                    itl = _extract_latency_metric(latency_data, "inter_token_latency", convert_to_ms=True, stat=stat_key)
                    if itl is not None:
                        if concurrency:
                            concurrency_vs_itl.append((concurrency, itl))
                        else:
                            qps_vs_itl.append((qps, itl))

                # Extract throughput metrics if they exist
                otps = None
                throughput_data = success_data.get("throughput", {})
                if throughput_data:
                    itps = _extract_throughput_metric(throughput_data, "input_tokens_per_sec")
                    if itps is not None:
                        if concurrency:
                            concurrency_vs_itps.append((concurrency, itps))
                        else:
                            qps_vs_itps.append((qps, itps))

                    otps = _extract_throughput_metric(throughput_data, "output_tokens_per_sec")
                    if otps is not None:
                        if concurrency:
                            concurrency_vs_otps.append((concurrency, otps))
                        else:
                            qps_vs_otps.append((qps, otps))

                    ttps = _extract_throughput_metric(throughput_data, "total_tokens_per_sec")
                    if ttps is not None:
                        if concurrency:
                            concurrency_vs_ttps.append((concurrency, ttps))
                        else:
                            qps_vs_ttps.append((qps, ttps))

                # Extract goodput metrics if they exist
                goodput_metrics = success_data.get("goodput_metrics", {})
                if goodput_metrics:
                    goodput_percentage = goodput_metrics.get("goodput_percentage")
                    if goodput_percentage is not None:
                        if not concurrency:
                            qps_vs_goodput_percentage.append((qps, goodput_percentage))

                    req_goodput_rate = goodput_metrics.get("request_goodput")
                    if req_goodput_rate is None:
                        req_goodput_rate = goodput_metrics.get("request_goodput_rate")
                    if req_goodput_rate is not None:
                        if not concurrency:
                            qps_vs_request_goodput_rate.append((qps, req_goodput_rate))

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

        # --- Generate Concurrency Latency Plot ---
        concurrency_latency_charts_to_generate = []
        if concurrency_vs_ttft:
            concurrency_latency_charts_to_generate.append(
                {
                    "title": "Time to First Token vs. Concurrency",
                    "xlabel": "Concurrency",
                    "ylabel": f"{stat_label} TTFT (ms)",
                    "data": sorted(concurrency_vs_ttft, key=operator.itemgetter(0)),
                }
            )
        if concurrency_vs_ntpot:
            concurrency_latency_charts_to_generate.append(
                {
                    "title": "Norm. Time per Output Token vs. Concurrency",
                    "xlabel": "Concurrency",
                    "ylabel": f"{stat_label} Norm. Time (ms/token)",
                    "data": sorted(concurrency_vs_ntpot, key=operator.itemgetter(0)),
                }
            )
        if concurrency_vs_itl:
            concurrency_latency_charts_to_generate.append(
                {
                    "title": "Inter-Token Latency vs. Concurrency",
                    "xlabel": "Concurrency",
                    "ylabel": f"{stat_label} ITL (ms)",
                    "data": sorted(concurrency_vs_itl, key=operator.itemgetter(0)),
                }
            )

        _generate_plot(
            concurrency_latency_charts_to_generate,
            "Latency vs Concurrency",
            report_path / "latency_vs_concurrency.png",
        )

        # --- Generate Concurrency Throughput Plot ---
        concurrency_throughput_charts_to_generate = []
        if concurrency_vs_itps:
            concurrency_throughput_charts_to_generate.append(
                {
                    "title": "Input Tokens/sec vs. Concurrency",
                    "xlabel": "Concurrency",
                    "ylabel": "Tokens/sec",
                    "data": sorted(concurrency_vs_itps, key=operator.itemgetter(0)),
                }
            )
        if concurrency_vs_otps:
            concurrency_throughput_charts_to_generate.append(
                {
                    "title": "Output Tokens/sec vs. Concurrency",
                    "xlabel": "Concurrency",
                    "ylabel": "Tokens/sec",
                    "data": sorted(concurrency_vs_otps, key=operator.itemgetter(0)),
                }
            )
        if concurrency_vs_ttps:
            concurrency_throughput_charts_to_generate.append(
                {
                    "title": "Total Tokens/sec vs. Concurrency",
                    "xlabel": "Concurrency",
                    "ylabel": "Tokens/sec",
                    "data": sorted(concurrency_vs_ttps, key=operator.itemgetter(0)),
                }
            )

        _generate_plot(
            concurrency_throughput_charts_to_generate,
            "Throughput vs Concurrency",
            report_path / "throughput_vs_concurrency.png",
        )

        # --- Generate QPS Latency Plot ---
        latency_charts_to_generate = []
        if qps_vs_ttft:
            latency_charts_to_generate.append(
                {
                    "title": "Time to First Token vs. QPS",
                    "xlabel": qps_xlabel,
                    "ylabel": f"{stat_label} TTFT (ms)",
                    "data": sorted(qps_vs_ttft, key=operator.itemgetter(0)),
                }
            )
        if qps_vs_ntpot:
            latency_charts_to_generate.append(
                {
                    "title": "Norm. Time per Output Token vs. QPS",
                    "xlabel": qps_xlabel,
                    "ylabel": f"{stat_label} Norm. Time (ms/token)",
                    "data": sorted(qps_vs_ntpot, key=operator.itemgetter(0)),
                }
            )
        if qps_vs_itl:
            latency_charts_to_generate.append(
                {
                    "title": "Inter-Token Latency vs. QPS",
                    "xlabel": qps_xlabel,
                    "ylabel": f"{stat_label} ITL (ms)",
                    "data": sorted(qps_vs_itl, key=operator.itemgetter(0)),
                }
            )

        _generate_plot(
            latency_charts_to_generate,
            "Latency vs Request Rate",
            report_path / "latency_vs_qps.png",
        )

        # --- Generate QPS Throughput Plot ---
        throughput_charts_to_generate = []
        if qps_vs_itps:
            throughput_charts_to_generate.append(
                {
                    "title": "Input Tokens/sec vs. QPS",
                    "xlabel": qps_xlabel,
                    "ylabel": "Tokens/sec",
                    "data": sorted(qps_vs_itps, key=operator.itemgetter(0)),
                }
            )
        if qps_vs_otps:
            throughput_charts_to_generate.append(
                {
                    "title": "Output Tokens/sec vs. QPS",
                    "xlabel": qps_xlabel,
                    "ylabel": "Tokens/sec",
                    "data": sorted(qps_vs_otps, key=operator.itemgetter(0)),
                }
            )
        if qps_vs_ttps:
            throughput_charts_to_generate.append(
                {
                    "title": "Total Tokens/sec vs. QPS",
                    "xlabel": qps_xlabel,
                    "ylabel": "Tokens/sec",
                    "data": sorted(qps_vs_ttps, key=operator.itemgetter(0)),
                }
            )

        _generate_plot(
            throughput_charts_to_generate,
            "Throughput vs Request Rate",
            report_path / "throughput_vs_qps.png",
        )

        # --- Generate QPS Goodput Plot ---
        goodput_charts_to_generate = []
        if qps_vs_goodput_percentage:
            goodput_charts_to_generate.append(
                {
                    "title": "Goodput % vs. QPS",
                    "xlabel": qps_xlabel,
                    "ylabel": "Goodput (%)",
                    "data": sorted(qps_vs_goodput_percentage, key=operator.itemgetter(0)),
                }
            )
        if qps_vs_request_goodput_rate:
            goodput_charts_to_generate.append(
                {
                    "title": "Request Goodput Rate vs. QPS",
                    "xlabel": qps_xlabel,
                    "ylabel": "Goodput Rate (req/s)",
                    "data": sorted(qps_vs_request_goodput_rate, key=operator.itemgetter(0)),
                }
            )

        if goodput_charts_to_generate:
            _generate_plot(
                goodput_charts_to_generate,
                "Goodput vs Request Rate",
                report_path / "goodput_vs_qps.png",
            )

        # --- Generate Throughput vs Latency Curve Plot ---
        throughput_latency_charts_to_generate = []
        if ntpot_vs_otps:
            throughput_latency_charts_to_generate.append(
                {
                    "title": "Throughput vs. Norm. Time per Output Token",
                    "xlabel": f"{stat_label} Norm. Time (ms/token)",
                    "ylabel": "Output Tokens/sec",
                    "data": sorted(ntpot_vs_otps, key=operator.itemgetter(0)),
                }
            )
        if ttft_vs_otps:
            throughput_latency_charts_to_generate.append(
                {
                    "title": "Throughput vs. Time to First Token",
                    "xlabel": f"{stat_label} TTFT (ms)",
                    "ylabel": "Output Tokens/sec",
                    "data": sorted(ttft_vs_otps, key=operator.itemgetter(0)),
                }
            )
        if itl_vs_otps:
            throughput_latency_charts_to_generate.append(
                {
                    "title": "Throughput vs. Inter-Token Latency",
                    "xlabel": f"{stat_label} ITL (ms)",
                    "ylabel": "Output Tokens/sec",
                    "data": sorted(itl_vs_otps, key=operator.itemgetter(0)),
                }
            )

        _generate_plot(
            throughput_latency_charts_to_generate,
            "Latency vs Throughput",
            report_path / "throughput_vs_latency.png",
        )

        chartset[report_path] = [
            concurrency_latency_charts_to_generate,
            concurrency_throughput_charts_to_generate,
            latency_charts_to_generate,
            throughput_charts_to_generate,
            throughput_latency_charts_to_generate,
            goodput_charts_to_generate,
        ]

    if analysis_dir:
        logger.info(f"Unified analysis reporting in {analysis_dir}")
        analysis_path = Path(analysis_dir)

        # Creating a legend file in the analysis path with list of report paths analyzed
        if labels is not None:
            label_by_path = dict(zip((Path(report_dir) for report_dir in report_dirs), labels, strict=True))
            report_names = [label_by_path[path] for path in chartset]
        else:
            report_names = [f"Report {idx + 1}" for idx in range(len(chartset.keys()))]
        report_legend_dict = {name: str(path) for name, path in zip(report_names, chartset, strict=True)}
        with open(analysis_path / "analyzed_reports_legend.json", "w") as f:
            json.dump(report_legend_dict, f, indent=2)

        # --- Generate Concurrency Latency Plot ---
        setof_concurrency_latency_charts_to_generate = [charts_to_generate[0] for charts_to_generate in chartset.values()]
        if len(setof_concurrency_latency_charts_to_generate) > 0:
            _generate_multi_plot(
                setof_concurrency_latency_charts_to_generate,
                len(max(setof_concurrency_latency_charts_to_generate, key=len)),
                report_names,
                "Latency vs Concurrency",
                analysis_path / "latency_vs_concurrency.png",
            )

        # --- Generate Concurrency Throughput Plot ---
        setof_concurrency_throughput_charts_to_generate = [charts_to_generate[1] for charts_to_generate in chartset.values()]
        if len(setof_concurrency_throughput_charts_to_generate) > 0:
            _generate_multi_plot(
                setof_concurrency_throughput_charts_to_generate,
                len(max(setof_concurrency_throughput_charts_to_generate, key=len)),
                report_names,
                "Throughput vs Concurrency",
                analysis_path / "throughput_vs_concurrency.png",
            )

        # --- Generate QPS Latency Plot ---
        setof_latency_charts_to_generate = [charts_to_generate[2] for charts_to_generate in chartset.values()]
        if len(setof_latency_charts_to_generate) > 0:
            _generate_multi_plot(
                setof_latency_charts_to_generate,
                len(max(setof_latency_charts_to_generate, key=len)),
                report_names,
                "Latency vs Request Rate",
                analysis_path / "latency_vs_qps.png",
            )

        # --- Generate QPS Throughput Plot ---
        setof_throughput_charts_to_generate = [charts_to_generate[3] for charts_to_generate in chartset.values()]
        if len(setof_throughput_charts_to_generate) > 0:
            _generate_multi_plot(
                setof_throughput_charts_to_generate,
                len(max(setof_throughput_charts_to_generate, key=len)),
                report_names,
                "Throughput vs Request Rate",
                analysis_path / "throughput_vs_qps.png",
            )

        # --- Generate Throughput vs Latency Curve Plot ---
        setof_throughput_latency_charts_to_generate = [charts_to_generate[4] for charts_to_generate in chartset.values()]
        if len(setof_throughput_latency_charts_to_generate) > 0:
            _generate_multi_plot(
                setof_throughput_latency_charts_to_generate,
                len(max(setof_throughput_latency_charts_to_generate, key=len)),
                report_names,
                "Latency vs Throughput",
                analysis_path / "throughput_vs_latency.png",
            )

        # --- Generate QPS Goodput Plot (Unified) ---
        setof_goodput_charts_to_generate = [
            charts_to_generate[5] for charts_to_generate in chartset.values() if len(charts_to_generate) > 5
        ]
        if len(setof_goodput_charts_to_generate) > 0:
            _generate_multi_plot(
                setof_goodput_charts_to_generate,
                len(max(setof_goodput_charts_to_generate, key=len)),
                report_names,
                "Goodput vs Request Rate",
                analysis_path / "goodput_vs_qps.png",
            )
