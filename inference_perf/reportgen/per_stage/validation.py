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
"""Validation of the per-stage lifecycle reports.

A ``stage_<n>_lifecycle_metrics.json`` file is the same ``ResponsesSummary``
shape as the run summary, so the content checks are shared with
``summary/validation.py``. On top of those, the stage files carry the load
block (requested rate, send duration) that the run summary does not.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from inference_perf.reportgen.summary.validation import structure_findings, summary_content_findings
from inference_perf.reportgen.validation import (
    Check,
    Finding,
    ReportSet,
    ReportSetValidator,
    Severity,
    StopValidation,
    is_number,
)


def _stage_filename(stage_id: int) -> str:
    return f"stage_{stage_id}_lifecycle_metrics.json"


class PerStageLifecycleValidator(ReportSetValidator):
    name = "per_stage"

    def covers(self, reports: ReportSet) -> List[str]:
        return [_stage_filename(stage_id) for stage_id in sorted(reports.stage_lifecycle_files())]

    def checks(self) -> Sequence[Check]:
        return [self._check_structure, self._check_contents, self._check_load]

    def _well_formed(self, reports: ReportSet) -> Dict[int, Any]:
        """Stage files that passed the structure check; the rest are skipped."""
        return {
            stage_id: contents
            for stage_id, contents in reports.stage_lifecycle_files().items()
            if not structure_findings(_stage_filename(stage_id), contents, "unused")
        }

    def _check_structure(self, reports: ReportSet) -> List[Finding]:
        stage_files = reports.stage_lifecycle_files()
        if not stage_files:
            raise StopValidation()
        findings: List[Finding] = []
        for stage_id, contents in sorted(stage_files.items()):
            findings += structure_findings(_stage_filename(stage_id), contents, f"{self.name}.structure")
        return findings

    def _check_contents(self, reports: ReportSet) -> List[Finding]:
        findings: List[Finding] = []
        for stage_id, contents in sorted(self._well_formed(reports).items()):
            findings += summary_content_findings(_stage_filename(stage_id), contents, self.name)
        return findings

    def _check_load(self, reports: ReportSet) -> List[Finding]:
        """The stage-only load block: rates and send duration are non-negative."""
        findings: List[Finding] = []
        for stage_id, contents in sorted(self._well_formed(reports).items()):
            filename = _stage_filename(stage_id)
            load = contents.get("load_summary")
            if not isinstance(load, dict):
                continue
            for key in ("requested_rate", "achieved_rate", "send_duration", "concurrency"):
                value = load.get(key)
                if is_number(value) and value < 0:
                    findings.append(
                        Finding(
                            check=f"{self.name}.load",
                            severity=Severity.ERROR,
                            message=f"load_summary.{key} is negative: {value}",
                            report=filename,
                        )
                    )
        return findings
