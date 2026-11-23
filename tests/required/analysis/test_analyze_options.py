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
"""Report labels, rate axis and latency statistic options of analyze_reports (issue #741)."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import TYPE_CHECKING, Dict, List, Optional, Tuple

import numpy as np
import pytest

from inference_perf.analysis.analyze import (
    AnalysisOptionError,
    _extract_latency_metric,
    _latency_stat_key,
    analyze_reports,
)

if TYPE_CHECKING:
    from matplotlib.figure import Figure

# (x label, y label, x values, y values) of the first line in one subplot.
Subplot = Tuple[str, str, List[float], List[float]]


# Every statistic of one latency metric, in seconds: mean=base, min=base/2, max=4*base,
# median=2*base, p90=3*base. There is no p97, so p97 is a missing percentile.
def _latency_stats(base: float) -> Dict[str, float]:
    return {"mean": base, "min": base / 2, "max": 4 * base, "median": 2 * base, "p90": 3 * base}


# One stage report. TTFT uses base=ttft, NTPOT base=ttft/10 and ITL base=ttft/100, so every
# metric and every statistic plots a different value. Output throughput is 100 * achieved,
# goodput is 90% and request goodput is achieved / 2. With concurrency set, the stage goes
# to the concurrency charts in place of the QPS charts.
def _stage_report(requested: float, achieved: float, ttft: float, concurrency: Optional[int] = None) -> Dict[str, object]:
    load_summary: Dict[str, object] = {"count": 10, "requested_rate": requested, "achieved_rate": achieved}
    if concurrency is not None:
        load_summary["concurrency"] = concurrency
    return {
        "load_summary": load_summary,
        "successes": {
            "count": 10,
            "latency": {
                "time_to_first_token": _latency_stats(ttft),
                "normalized_time_per_output_token": _latency_stats(ttft / 10),
                "inter_token_latency": _latency_stats(ttft / 100),
            },
            "throughput": {"output_tokens_per_sec": 100.0 * achieved},
            "goodput_metrics": {"goodput_percentage": 90.0, "request_goodput": achieved / 2},
        },
    }


# Writes tmp_path/<name> with two stages: stage 0 requested 1.0 / achieved 0.75 / TTFT 0.25 s,
# stage 1 requested 2.0 / achieved 1.5 / TTFT 0.5 s. concurrency=True gives the stages
# concurrency 4 and 8.
def _write_report(tmp_path: Path, name: str, concurrency: bool = False) -> Path:
    report_dir = tmp_path / name
    report_dir.mkdir()
    stages = [(1.0, 0.75, 0.25, 4), (2.0, 1.5, 0.5, 8)]
    for stage, (requested, achieved, ttft, users) in enumerate(stages):
        report = _stage_report(requested, achieved, ttft, users if concurrency else None)
        (report_dir / f"stage_{stage}_lifecycle_metrics.json").write_text(json.dumps(report))
    return report_dir


# Replaces plt.savefig so each chart's Figure is kept under its output path, e.g.
# figures[str(report_dir / "latency_vs_qps.png")], and nothing is written to disk.
# Skips the test when matplotlib (the "analysis" extra) is not installed.
@pytest.fixture
def figures(monkeypatch: pytest.MonkeyPatch) -> Dict[str, Figure]:
    plt = pytest.importorskip("matplotlib.pyplot")
    saved: Dict[str, Figure] = {}

    def fake_savefig(fname: Path, *args: object, **kwargs: object) -> None:
        saved[str(fname)] = plt.gcf()

    monkeypatch.setattr(plt, "savefig", fake_savefig)
    return saved


# Every subplot of a saved chart, left to right. For latency_vs_qps with the defaults, the first
# one is ("QPS (achieved rate)", "Mean TTFT (ms)", [0.75, 1.5], [250.0, 500.0]).
def _subplots(figure: Figure) -> List[Subplot]:
    result: List[Subplot] = []
    for ax in figure.axes:
        line = ax.get_lines()[0]
        xs = np.asarray(line.get_xdata()).tolist()
        ys = np.asarray(line.get_ydata()).tolist()
        result.append((ax.get_xlabel(), ax.get_ylabel(), xs, ys))
    return result


# Checks the subplots of one chart against (x label, y label, x values, y values) rows.
# Labels must be equal, and values equal to within pytest.approx.
def _assert_subplots(figure: Figure, expected: List[Subplot]) -> None:
    actual = _subplots(figure)
    assert [(xlabel, ylabel) for xlabel, ylabel, _, _ in actual] == [(xlabel, ylabel) for xlabel, ylabel, _, _ in expected]
    for (_, _, xs, ys), (_, _, expected_xs, expected_ys) in zip(actual, expected, strict=True):
        assert xs == pytest.approx(expected_xs)
        assert ys == pytest.approx(expected_ys)


# Default call: x is achieved_rate (0.75, 1.5), the axis says "QPS (achieved rate)", and the
# latency charts plot the mean. Before #741 this chart plotted achieved_rate under the label
# "QPS (requested rate)".
def test_default_plots_achieved_rate_and_names_it(tmp_path: Path, figures: Dict[str, Figure]) -> None:
    report = _write_report(tmp_path, "run")

    analyze_reports([str(report)])

    _assert_subplots(
        figures[str(report / "latency_vs_qps.png")],
        [
            ("QPS (achieved rate)", "Mean TTFT (ms)", [0.75, 1.5], [250.0, 500.0]),
            ("QPS (achieved rate)", "Mean Norm. Time (ms/token)", [0.75, 1.5], [25.0, 50.0]),
            ("QPS (achieved rate)", "Mean ITL (ms)", [0.75, 1.5], [2.5, 5.0]),
        ],
    )
    for xlabel, _, xs, _ in _subplots(figures[str(report / "throughput_vs_qps.png")]):
        assert (xlabel, xs) == ("QPS (achieved rate)", [0.75, 1.5])


# rate="requested": x is requested_rate (1.0, 2.0) and every QPS chart says "QPS (requested rate)",
# the goodput chart too. The latency values do not change: mean TTFT 250 ms and 500 ms.
def test_requested_rate_plots_configured_rate_and_names_it(tmp_path: Path, figures: Dict[str, Figure]) -> None:
    report = _write_report(tmp_path, "run")

    analyze_reports([str(report)], rate="requested")

    xlabel, _, xs, ys = _subplots(figures[str(report / "latency_vs_qps.png")])[0]
    assert (xlabel, xs) == ("QPS (requested rate)", [1.0, 2.0])
    assert ys == pytest.approx([250.0, 500.0])
    for chart in ("latency_vs_qps.png", "throughput_vs_qps.png", "goodput_vs_qps.png"):
        for xlabel, _, xs, _ in _subplots(figures[str(report / chart)]):
            assert (chart, xlabel, xs) == (chart, "QPS (requested rate)", [1.0, 2.0])


# latency_stat="p90" plots the p90 values (3 * base, in ms) of TTFT, NTPOT and ITL, with a "P90"
# label on the latency axes and on the x axes of the throughput-vs-latency chart.
def test_p90_latency_stat_plots_p90_and_names_it(tmp_path: Path, figures: Dict[str, Figure]) -> None:
    report = _write_report(tmp_path, "run")

    analyze_reports([str(report)], latency_stat="p90")

    _assert_subplots(
        figures[str(report / "latency_vs_qps.png")],
        [
            ("QPS (achieved rate)", "P90 TTFT (ms)", [0.75, 1.5], [750.0, 1500.0]),
            ("QPS (achieved rate)", "P90 Norm. Time (ms/token)", [0.75, 1.5], [75.0, 150.0]),
            ("QPS (achieved rate)", "P90 ITL (ms)", [0.75, 1.5], [7.5, 15.0]),
        ],
    )
    _assert_subplots(
        figures[str(report / "throughput_vs_latency.png")],
        [
            ("P90 Norm. Time (ms/token)", "Output Tokens/sec", [75.0, 150.0], [75.0, 150.0]),
            ("P90 TTFT (ms)", "Output Tokens/sec", [750.0, 1500.0], [75.0, 150.0]),
            ("P90 ITL (ms)", "Output Tokens/sec", [7.5, 15.0], [75.0, 150.0]),
        ],
    )


# Stages with concurrency 4 and 8 and latency_stat="p90": the concurrency latency chart
# plots the p90 values against concurrency, with "P90" in each y label.
def test_p90_latency_stat_names_the_concurrency_chart(tmp_path: Path, figures: Dict[str, Figure]) -> None:
    report = _write_report(tmp_path, "run", concurrency=True)

    analyze_reports([str(report)], latency_stat="p90")

    _assert_subplots(
        figures[str(report / "latency_vs_concurrency.png")],
        [
            ("Concurrency", "P90 TTFT (ms)", [4.0, 8.0], [750.0, 1500.0]),
            ("Concurrency", "P90 Norm. Time (ms/token)", [4.0, 8.0], [75.0, 150.0]),
            ("Concurrency", "P90 ITL (ms)", [4.0, 8.0], [7.5, 15.0]),
        ],
    )


# Reports store p50 under "median", so latency_stat="p50" plots the median TTFT
# (500 ms and 1000 ms) and labels the axis "Median TTFT (ms)".
def test_p50_latency_stat_reads_median(tmp_path: Path, figures: Dict[str, Figure]) -> None:
    report = _write_report(tmp_path, "run")

    analyze_reports([str(report)], latency_stat="p50")

    _, ylabel, _, ys = _subplots(figures[str(report / "latency_vs_qps.png")])[0]
    assert ylabel == "Median TTFT (ms)"
    assert ys == pytest.approx([500.0, 1000.0])


# {"time_to_first_token": {"mean": 0.1, "p99": 0.4}}: stat="p99" gives 0.4 (400 ms with
# convert_to_ms), no stat gives the mean 0.1, and a stat the report lacks gives None.
def test_extract_latency_metric_reads_the_given_stat() -> None:
    latency = {"time_to_first_token": {"mean": 0.1, "p99": 0.4}}

    assert _extract_latency_metric(latency, "time_to_first_token", stat="p99") == 0.4
    assert _extract_latency_metric(latency, "time_to_first_token", convert_to_ms=True, stat="p99") == pytest.approx(400.0)
    assert _extract_latency_metric(latency, "time_to_first_token") == 0.1
    assert _extract_latency_metric(latency, "time_to_first_token", stat="p90") is None


# Names map to report keys the way reportgen writes them: "p50" -> "median",
# "p90.0" -> "p90", "P99" -> "p99", and "p0.1" stays "p0.1".
@pytest.mark.parametrize(
    ("stat", "key"),
    [
        ("mean", "mean"),
        ("median", "median"),
        ("min", "min"),
        ("max", "max"),
        ("p50", "median"),
        ("p90", "p90"),
        ("p90.0", "p90"),
        ("P99", "p99"),
        ("p99.9", "p99.9"),
        ("p0.1", "p0.1"),
        ("p100", "p100"),
    ],
)
def test_latency_stat_key_matches_report_keys(stat: str, key: str) -> None:
    assert _latency_stat_key(stat) == key


# Names that are not a summary statistic or a percentile from p0 to p100 raise AnalysisOptionError.
@pytest.mark.parametrize("stat", ["avg", "90", "p", "p101", "p-1", "p90x", ""])
def test_unknown_latency_stat_raises(stat: str) -> None:
    with pytest.raises(AnalysisOptionError, match="is not valid"):
        _latency_stat_key(stat)


# Two reports with labels ["vLLM", "SGLang"]: the unified chart legend and
# analyzed_reports_legend.json use those names in place of "Report 1" and "Report 2".
def test_labels_name_the_unified_legend(tmp_path: Path, figures: Dict[str, Figure]) -> None:
    first, second = _write_report(tmp_path, "a"), _write_report(tmp_path, "b")
    out = tmp_path / "out"
    out.mkdir()

    analyze_reports([str(first), str(second)], str(out), labels=["vLLM", "SGLang"])

    legend = figures[str(out / "latency_vs_qps.png")].axes[0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["vLLM", "SGLang"]
    assert json.loads((out / "analyzed_reports_legend.json").read_text()) == {"vLLM": str(first), "SGLang": str(second)}


# No options: the unified legend is "Report 1", "Report 2", the latency axis says
# "Mean TTFT (ms)" and x is achieved_rate, as before the options existed.
def test_default_labels_and_stat_are_unchanged(tmp_path: Path, figures: Dict[str, Figure]) -> None:
    first, second = _write_report(tmp_path, "a"), _write_report(tmp_path, "b")
    out = tmp_path / "out"
    out.mkdir()

    analyze_reports([str(first), str(second)], str(out))

    unified = figures[str(out / "latency_vs_qps.png")]
    legend = unified.axes[0].get_legend()
    assert legend is not None
    assert [text.get_text() for text in legend.get_texts()] == ["Report 1", "Report 2"]
    assert json.loads((out / "analyzed_reports_legend.json").read_text()) == {"Report 1": str(first), "Report 2": str(second)}
    xlabel, ylabel, xs, ys = _subplots(unified)[0]
    assert (xlabel, ylabel) == ("QPS (achieved rate)", "Mean TTFT (ms)")
    assert xs == [0.75, 1.5]
    assert ys == pytest.approx([250.0, 500.0])


# Bad options raise AnalysisOptionError, a ValueError, before any report is read: 1 label for
# 2 reports, a repeated label, a labelled directory given twice, and rate="configured".
@pytest.mark.parametrize(
    ("dirs", "labels", "rate", "message"),
    [
        (["a", "b"], ["only"], "achieved", r"number of labels \(1\) is not equal to the number of report directories \(2\)"),
        (["a", "b"], ["same", "same"], "achieved", "labels are the same"),
        (["a", "a"], ["x", "y"], "achieved", "report directory occurs more than one time"),
        (["a"], None, "configured", "Rate 'configured' is not valid"),
    ],
)
def test_bad_options_raise(dirs: List[str], labels: Optional[List[str]], rate: str, message: str) -> None:
    with pytest.raises(AnalysisOptionError, match=message) as error:
        analyze_reports(dirs, labels=labels, rate=rate)
    assert isinstance(error.value, ValueError)


# latency_stat="p97" with reports that have mean/min/max/median/p90 but no p97: each stage logs a
# warning that names p97, the metrics and the file, and no latency_vs_qps chart is saved.
def test_missing_percentile_warns_and_skips_points(
    tmp_path: Path, figures: Dict[str, Figure], caplog: pytest.LogCaptureFixture
) -> None:
    report = _write_report(tmp_path, "run")

    with caplog.at_level(logging.WARNING, logger="inference_perf.analysis.analyze"):
        analyze_reports([str(report)], latency_stat="p97")

    metrics = "time_to_first_token, normalized_time_per_output_token, inter_token_latency"
    assert f"No 'p97' value for {metrics} in stage_0_lifecycle_metrics.json" in caplog.text
    assert f"No 'p97' value for {metrics} in stage_1_lifecycle_metrics.json" in caplog.text
    assert str(report / "latency_vs_qps.png") not in figures
    assert str(report / "throughput_vs_qps.png") in figures


# Stage 0 has no requested_rate. With rate="requested" it is skipped with a warning,
# and only stage 1 (requested 2.0, TTFT 500 ms) is plotted.
def test_stage_without_requested_rate_is_skipped(
    tmp_path: Path, figures: Dict[str, Figure], caplog: pytest.LogCaptureFixture
) -> None:
    report = _write_report(tmp_path, "run")
    stage_0 = report / "stage_0_lifecycle_metrics.json"
    contents = json.loads(stage_0.read_text())
    del contents["load_summary"]["requested_rate"]
    stage_0.write_text(json.dumps(contents))

    with caplog.at_level(logging.WARNING, logger="inference_perf.analysis.analyze"):
        analyze_reports([str(report)], rate="requested")

    assert "Could not find requested_rate in stage_0_lifecycle_metrics.json. Skipping." in caplog.text
    _, _, xs, ys = _subplots(figures[str(report / "latency_vs_qps.png")])[0]
    assert xs == [2.0]
    assert ys == pytest.approx([500.0])
