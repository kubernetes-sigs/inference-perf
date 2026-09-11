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
"""Tests for the validation framework: sequential checks, halting, crash
containment, and the shape of the emitted validation.json."""

from __future__ import annotations

from typing import List, Sequence

from inference_perf.reportgen.validation import (
    Check,
    Finding,
    ReportSet,
    ReportSetValidator,
    Severity,
    StopValidation,
    run_validators,
    validate_reports,
)
from inference_perf.utils import ReportFile


class _ScriptedValidator(ReportSetValidator):
    """Runs a scripted list of checks and records which ones executed."""

    name = "scripted"

    def __init__(self, checks: Sequence[Check], covered: Sequence[str] = ()) -> None:
        self._checks = list(checks)
        self._covered = list(covered)
        self.executed: List[str] = []

    def covers(self, reports: ReportSet) -> List[str]:
        return list(self._covered)

    def checks(self) -> Sequence[Check]:
        return [self._wrap(i, check) for i, check in enumerate(self._checks)]

    def _wrap(self, i: int, check: Check) -> Check:
        def wrapped(reports: ReportSet) -> List[Finding]:
            self.executed.append(f"check_{i}")
            return check(reports)

        return wrapped


def _warning(report: str | None = None) -> Finding:
    return Finding(check="scripted.warn", severity=Severity.WARNING, message="w", report=report)


def _error(report: str | None = None) -> Finding:
    return Finding(check="scripted.err", severity=Severity.ERROR, message="e", report=report)


def test_findings_route_to_global_and_per_report_groups() -> None:
    validator = _ScriptedValidator(
        [
            lambda reports: [_warning(None)],
            lambda reports: [_error("a.json")],
        ]
    )
    result = run_validators([validator], [])

    assert [f.check for f in result.global_findings.warnings] == ["scripted.warn"]
    assert not result.global_findings.errors
    assert [f.check for f in result.reports["a.json"].errors] == ["scripted.err"]
    assert not result.reports["a.json"].warnings


def test_checks_run_sequentially_and_stop_validation_halts() -> None:
    def halt(reports: ReportSet) -> List[Finding]:
        raise StopValidation([_error("a.json")])

    validator = _ScriptedValidator([lambda reports: [], halt, lambda reports: [_error("b.json")]])
    result = run_validators([validator], [])

    assert validator.executed == ["check_0", "check_1"]
    assert "b.json" not in result.reports
    assert [f.check for f in result.reports["a.json"].errors] == ["scripted.err"]


def test_silent_stop_validation_records_nothing() -> None:
    def halt(reports: ReportSet) -> List[Finding]:
        raise StopValidation()

    validator = _ScriptedValidator([halt, lambda reports: [_error()]])
    result = run_validators([validator], [])

    assert validator.executed == ["check_0"]
    assert not result.all_errors() and not result.all_warnings()


def test_crashing_check_is_contained_and_siblings_still_run() -> None:
    def crash(reports: ReportSet) -> List[Finding]:
        raise ValueError("kaboom")

    validator = _ScriptedValidator([crash, lambda reports: [_warning("a.json")]])
    result = run_validators([validator], [])

    assert validator.executed == ["check_0", "check_1"]
    internal = result.global_findings.errors
    assert len(internal) == 1
    assert internal[0].check == "scripted.internal"
    assert "kaboom" in internal[0].message
    assert [f.check for f in result.reports["a.json"].warnings] == ["scripted.warn"]


def test_covered_files_appear_even_when_clean() -> None:
    validator = _ScriptedValidator([lambda reports: []], covered=["clean.json"])
    result = run_validators([validator], [])

    assert result.reports["clean.json"].is_clean()


def test_validate_reports_emits_validation_json_with_global_alias() -> None:
    report_file = validate_reports([ReportFile(name="config", file_type="yaml", contents={})])

    assert report_file.get_filename() == "validation.json"
    contents = report_file.get_contents()
    assert set(contents.keys()) == {"global", "reports"}
    assert contents["global"] == {"warnings": [], "errors": []}
