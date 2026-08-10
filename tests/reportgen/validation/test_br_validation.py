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
"""Tests for the BR0.2 partial-report validator."""

from __future__ import annotations

from typing import List

from inference_perf.reportgen.br.v0_2.validation import BrPartialValidator
from inference_perf.reportgen.validation import Finding, ValidationReport, run_validators
from inference_perf.utils import ReportFile

from .helpers import make_report_set, tampered

PARTIAL_0_FILE = "inference-perf.partial.stage_0.yaml"


def _validate(reports: List[ReportFile]) -> ValidationReport:
    return run_validators([BrPartialValidator()], reports)


def _errors_for_check(result: ValidationReport, check: str) -> List[Finding]:
    return [f for f in result.all_errors() if f.check == check]


def test_real_partials_are_clean_and_covered() -> None:
    result = _validate(make_report_set())

    assert not result.all_errors(), [f.message for f in result.all_errors()]
    assert not result.all_warnings(), [f.message for f in result.all_warnings()]
    assert result.reports[PARTIAL_0_FILE].is_clean()
    assert result.reports["inference-perf.partial.stage_1.yaml"].is_clean()


def test_no_partials_is_skipped_silently() -> None:
    result = _validate(make_report_set(with_br_partials=False))

    assert not result.reports
    assert not result.all_errors() and not result.all_warnings()


def test_schema_invalid_partial_is_an_error() -> None:
    reports, contents = tampered(make_report_set(), PARTIAL_0_FILE)
    contents["run"] = "not-a-run-block"

    result = _validate(reports)
    assert _errors_for_check(result, "br_partial.schema")


def test_wrong_version_is_an_error() -> None:
    reports, contents = tampered(make_report_set(), PARTIAL_0_FILE)
    contents["version"] = "0.0.9"

    result = _validate(reports)
    assert _errors_for_check(result, "br_partial.version")


def test_null_value_violates_the_merge_contract() -> None:
    reports, contents = tampered(make_report_set(), PARTIAL_0_FILE)
    contents["results"]["request_performance"]["aggregate"]["requests"]["failures"] = None

    result = _validate(reports)
    errors = _errors_for_check(result, "br_partial.nulls")
    assert errors and "failures" in errors[0].message


def test_disagreement_with_stage_lifecycle_is_an_error() -> None:
    reports, contents = tampered(make_report_set(), PARTIAL_0_FILE)
    aggregate = contents["results"]["request_performance"]["aggregate"]
    aggregate["latency"]["request_latency"]["mean"] *= 1.5

    result = _validate(reports)
    errors = _errors_for_check(result, "br_partial.lifecycle_agreement")
    assert errors and "request_latency" in errors[0].message
    assert errors[0].report == PARTIAL_0_FILE


def test_count_disagreement_with_stage_lifecycle_is_an_error() -> None:
    reports, contents = tampered(make_report_set(), PARTIAL_0_FILE)
    contents["results"]["request_performance"]["aggregate"]["requests"]["total"] += 1

    result = _validate(reports)
    assert _errors_for_check(result, "br_partial.lifecycle_agreement")


def test_run_time_disagreement_is_an_error() -> None:
    reports, contents = tampered(make_report_set(), PARTIAL_0_FILE)
    contents["run"]["time"]["duration"] = "PT99.000S"

    result = _validate(reports)
    assert _errors_for_check(result, "br_partial.run_time")


def test_agreement_is_skipped_without_the_stage_lifecycle_report() -> None:
    reports, contents = tampered(make_report_set(), PARTIAL_0_FILE)
    contents["results"]["request_performance"]["aggregate"]["requests"]["total"] += 1
    reports = [r for r in reports if r.get_filename() != "stage_0_lifecycle_metrics.json"]

    result = _validate(reports)
    assert not _errors_for_check(result, "br_partial.lifecycle_agreement")
