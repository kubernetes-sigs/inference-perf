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
"""Tests for the run-summary lifecycle validator."""

from __future__ import annotations

from typing import List

from inference_perf.reportgen.summary.validation import SummaryLifecycleValidator
from inference_perf.reportgen.validation import Finding, ValidationReport, run_validators
from inference_perf.utils import ReportFile

from .helpers import make_report_set, replace_contents, tampered

SUMMARY_FILE = "summary_lifecycle_metrics.json"


def _validate(reports: List[ReportFile]) -> ValidationReport:
    return run_validators([SummaryLifecycleValidator()], reports)


def _errors_for_check(result: ValidationReport, check: str) -> List[Finding]:
    return [f for f in result.all_errors() if f.check == check]


def test_real_generator_output_is_clean() -> None:
    result = _validate(make_report_set())

    assert not result.all_errors(), [f.message for f in result.all_errors()]
    assert not result.all_warnings(), [f.message for f in result.all_warnings()]
    assert result.reports[SUMMARY_FILE].is_clean()


def test_absent_summary_is_skipped_silently() -> None:
    reports = [r for r in make_report_set() if r.get_filename() != SUMMARY_FILE]
    result = _validate(reports)

    assert SUMMARY_FILE not in result.reports
    assert not result.all_errors() and not result.all_warnings()


def test_structurally_broken_summary_halts_with_a_single_error() -> None:
    reports = replace_contents(make_report_set(), SUMMARY_FILE, ["not", "a", "summary"])
    result = _validate(reports)

    errors = result.reports[SUMMARY_FILE].errors
    assert len(errors) == 1
    assert errors[0].check == "summary.structure"


def test_unreconciled_counts_is_an_error() -> None:
    reports, contents = tampered(make_report_set(), SUMMARY_FILE)
    contents["successes"]["count"] += 1

    result = _validate(reports)
    assert _errors_for_check(result, "summary.counts")


def test_mean_outside_min_max_is_an_error() -> None:
    reports, contents = tampered(make_report_set(), SUMMARY_FILE)
    latency = contents["successes"]["latency"]["request_latency"]
    latency["mean"] = latency["max"] * 10

    result = _validate(reports)
    assert any("mean" in f.message for f in _errors_for_check(result, "summary.distributions"))


def test_percentile_inversion_is_an_error() -> None:
    reports, contents = tampered(make_report_set(), SUMMARY_FILE)
    latency = contents["successes"]["latency"]["request_latency"]
    latency["median"], latency["p90"] = latency["p90"], latency["median"]

    result = _validate(reports)
    assert any("ordering" in f.message for f in _errors_for_check(result, "summary.distributions"))


def test_negative_latency_is_an_error_but_negative_schedule_delay_is_a_warning() -> None:
    reports, contents = tampered(make_report_set(), SUMMARY_FILE)
    contents["successes"]["latency"]["request_latency"]["min"] = -0.5

    result = _validate(reports)
    assert any("negative" in f.message for f in _errors_for_check(result, "summary.distributions"))

    reports, contents = tampered(make_report_set(), SUMMARY_FILE)
    delay = contents["load_summary"]["schedule_delay"]
    delay["min"] = -0.5
    delay["median"] = min(delay["median"], 0.0)

    result = _validate(reports)
    warnings = [f for f in result.all_warnings() if f.check == "summary.distributions"]
    assert warnings and not _errors_for_check(result, "summary.distributions")


def test_token_total_disagreeing_with_mean_is_an_error() -> None:
    reports, contents = tampered(make_report_set(), SUMMARY_FILE)
    contents["successes"]["output_tokens"]["total"] += 100

    result = _validate(reports)
    assert _errors_for_check(result, "summary.tokens")


def test_cached_exceeding_total_prompt_tokens_is_an_error() -> None:
    reports, contents = tampered(make_report_set(), SUMMARY_FILE)
    prompt = contents["successes"]["prompt_tokens"]
    prompt["cached"] = prompt["total"] + 1
    prompt["uncached"] = 0.0

    result = _validate(reports)
    assert any("cached" in f.message for f in _errors_for_check(result, "summary.tokens"))


def test_throughput_not_reconciling_with_count_is_an_error() -> None:
    reports, contents = tampered(make_report_set(), SUMMARY_FILE)
    contents["successes"]["throughput"]["requests_per_sec"] *= 2

    result = _validate(reports)
    assert _errors_for_check(result, "summary.throughput")


def test_goodput_exceeding_total_is_an_error() -> None:
    reports, contents = tampered(make_report_set(), SUMMARY_FILE)
    contents["successes"]["goodput_metrics"] = {
        "good_requests": 10,
        "total_requests": contents["successes"]["count"],
        "goodput_percentage": 120.0,
    }

    result = _validate(reports)
    errors = _errors_for_check(result, "summary.goodput")
    assert any("good_requests" in f.message for f in errors)
    assert any("percentage" in f.message for f in errors)


def test_token_count_mismatches_is_a_warning() -> None:
    reports, contents = tampered(make_report_set(), SUMMARY_FILE)
    contents["successes"]["token_count_mismatches"] = 3

    result = _validate(reports)
    warnings = [f for f in result.all_warnings() if f.check == "summary.token_mismatches"]
    assert warnings and "3" in warnings[0].message
