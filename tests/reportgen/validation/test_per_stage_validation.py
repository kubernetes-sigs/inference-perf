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
"""Tests for the per-stage lifecycle validator."""

from __future__ import annotations

from typing import List

from inference_perf.reportgen.per_stage.validation import PerStageLifecycleValidator
from inference_perf.reportgen.validation import ValidationReport, run_validators
from inference_perf.utils import ReportFile

from .helpers import make_report_set, replace_contents, tampered

STAGE_0_FILE = "stage_0_lifecycle_metrics.json"
STAGE_1_FILE = "stage_1_lifecycle_metrics.json"


def _validate(reports: List[ReportFile]) -> ValidationReport:
    return run_validators([PerStageLifecycleValidator()], reports)


def test_real_generator_output_is_clean_and_covers_every_stage() -> None:
    result = _validate(make_report_set())

    assert not result.all_errors(), [f.message for f in result.all_errors()]
    assert result.reports[STAGE_0_FILE].is_clean()
    assert result.reports[STAGE_1_FILE].is_clean()


def test_findings_are_attributed_to_the_broken_stage_file() -> None:
    reports, contents = tampered(make_report_set(), STAGE_1_FILE)
    contents["successes"]["count"] += 1

    result = _validate(reports)
    assert any(f.check == "per_stage.counts" for f in result.reports[STAGE_1_FILE].errors)
    assert result.reports[STAGE_0_FILE].is_clean()


def test_structurally_broken_stage_file_does_not_hide_other_stages() -> None:
    reports = replace_contents(make_report_set(), STAGE_0_FILE, "garbage")
    reports, contents = tampered(reports, STAGE_1_FILE)
    contents["successes"]["count"] += 1

    result = _validate(reports)
    assert any(f.check == "per_stage.structure" for f in result.reports[STAGE_0_FILE].errors)
    assert any(f.check == "per_stage.counts" for f in result.reports[STAGE_1_FILE].errors)


def test_negative_requested_rate_is_an_error() -> None:
    reports, contents = tampered(make_report_set(), STAGE_0_FILE)
    contents["load_summary"]["requested_rate"] = -1.0

    result = _validate(reports)
    assert any(f.check == "per_stage.load" for f in result.reports[STAGE_0_FILE].errors)


def test_no_stage_files_is_clean() -> None:
    reports = [r for r in make_report_set() if "stage_" not in r.get_filename()]
    result = _validate(reports)

    assert not result.all_errors() and not result.all_warnings()
    assert not result.reports
