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
"""Tests for the cross-report (global) consistency validator."""

from __future__ import annotations

from typing import List

from inference_perf.reportgen.cross_report.validation import CrossReportValidator
from inference_perf.reportgen.validation import ValidationReport, run_validators
from inference_perf.utils import ReportFile

from .helpers import make_report_set, tampered

PER_REQUEST_FILE = "per_request_lifecycle_metrics.json"
STAGE_0_FILE = "stage_0_lifecycle_metrics.json"


def _validate(reports: List[ReportFile]) -> ValidationReport:
    return run_validators([CrossReportValidator()], reports)


def test_consistent_report_set_is_clean() -> None:
    result = _validate(make_report_set())

    assert not result.all_errors(), [f.message for f in result.all_errors()]
    assert not result.all_warnings(), [f.message for f in result.all_warnings()]


def test_stage_counts_not_summing_to_run_summary_is_a_global_error() -> None:
    reports, contents = tampered(make_report_set(), STAGE_0_FILE)
    contents["load_summary"]["count"] += 1
    contents["successes"]["count"] += 1

    result = _validate(reports)
    checks = {f.check for f in result.global_findings.errors}
    assert "cross_report.stage_totals" in checks


def test_per_request_record_count_mismatch_is_a_global_error() -> None:
    reports, records = tampered(make_report_set(), PER_REQUEST_FILE)
    records.pop()

    result = _validate(reports)
    checks = {f.check for f in result.global_findings.errors}
    assert "cross_report.per_request_count" in checks


def test_gap_in_stage_numbering_is_a_warning() -> None:
    reports = [
        ReportFile(name="stage_2_lifecycle_metrics", contents=r.get_contents())
        if r.get_filename() == "stage_1_lifecycle_metrics.json"
        else r
        for r in make_report_set(with_br_partials=False)
    ]

    result = _validate(reports)
    assert any(f.check == "cross_report.stage_indices" for f in result.global_findings.warnings)


def test_more_stage_reports_than_configured_stages_is_a_global_error() -> None:
    reports = make_report_set()
    reports = [
        ReportFile(name="config", file_type="yaml", contents={"load": {"stages": [{"rate": 1}]}})
        if r.get_filename() == "config.yaml"
        else r
        for r in reports
    ]

    result = _validate(reports)
    assert any(f.check == "cross_report.configured_stages" for f in result.global_findings.errors)


def test_fewer_stage_reports_than_configured_stages_is_a_warning() -> None:
    reports = [r for r in make_report_set(with_br_partials=False) if r.get_filename() != "stage_1_lifecycle_metrics.json"]

    result = _validate(reports)
    assert any(f.check == "cross_report.configured_stages" for f in result.global_findings.warnings)


def test_decreasing_stage_rate_is_a_warning() -> None:
    reports, contents = tampered(make_report_set(), STAGE_0_FILE)
    contents["load_summary"]["requested_rate"] = 2.0  # stage 1 keeps the helpers' rate of 1.0

    result = _validate(reports)
    assert any(f.check == "cross_report.stage_rates" for f in result.global_findings.warnings)


def test_equal_stage_rates_are_clean() -> None:
    # Monotonically increasing means non-decreasing: a sweep may repeat a
    # rate, and the helpers give every stage the same rate.
    result = _validate(make_report_set())
    assert not any(f.check == "cross_report.stage_rates" for f in result.global_findings.warnings)


def test_non_numeric_stage_rate_skips_the_rate_ordering_check() -> None:
    # Rates are floats today; a future rate type has no defined ordering, so
    # the check must skip instead of comparing across types.
    reports, contents = tampered(make_report_set(), STAGE_0_FILE)
    contents["load_summary"]["requested_rate"] = {"distribution": "poisson", "mean": 2.0}

    result = _validate(reports)
    assert not any(f.check == "cross_report.stage_rates" for f in result.global_findings.warnings)
