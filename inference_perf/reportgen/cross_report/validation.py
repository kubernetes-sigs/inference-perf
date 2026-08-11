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
"""Cross-report consistency: the same run described by different files must
tell the same story.

These are the highest-value checks in the validation layer: every report file
is derived from the same request metrics, so any disagreement between files
(stage counts not summing to the run summary, per-request records missing
from aggregates) means requests were dropped or double-counted somewhere in
assembly — the #564/#602 regression family. Also home to ``stage_rates``, a
plausibility check on the load sweep itself: stages are expected to ramp the
request rate upward, so a decreasing rate warns. Findings that span files
carry no single filename and land under the ``global`` key of
``validation.json``.
"""

from __future__ import annotations

from typing import List, Optional, Sequence

from inference_perf.reportgen.summary.validation import structure_findings
from inference_perf.reportgen.validation import (
    CONFIG_FILENAME,
    PER_REQUEST_FILENAME,
    SUMMARY_LIFECYCLE_FILENAME,
    Check,
    Finding,
    ReportSet,
    ReportSetValidator,
    Severity,
    get_path,
    is_number,
)


def _global_error(check: str, message: str) -> Finding:
    return Finding(check=check, severity=Severity.ERROR, message=message)


def _global_warning(check: str, message: str) -> Finding:
    return Finding(check=check, severity=Severity.WARNING, message=message)


class CrossReportValidator(ReportSetValidator):
    name = "cross_report"

    def checks(self) -> Sequence[Check]:
        return [
            self._check_stage_totals,
            self._check_per_request_count,
            self._check_stage_indices,
            self._check_configured_stages,
            self._check_stage_rates,
        ]

    @staticmethod
    def _summary_count(reports: ReportSet, *path: str) -> Optional[float]:
        summary = reports.contents(SUMMARY_LIFECYCLE_FILENAME)
        if not isinstance(summary, dict):
            return None
        value = get_path(summary, *path)
        return float(value) if is_number(value) else None

    def _check_stage_totals(self, reports: ReportSet) -> List[Finding]:
        """Per-stage counts must sum to the run summary's counts."""
        stage_files = {
            stage_id: contents
            for stage_id, contents in reports.stage_lifecycle_files().items()
            if not structure_findings("unused", contents, "unused")
        }
        if not stage_files:
            return []

        findings: List[Finding] = []
        for path in (("load_summary", "count"), ("successes", "count"), ("failures", "count")):
            summary_value = self._summary_count(reports, *path)
            if summary_value is None:
                continue
            stage_values = [get_path(contents, *path) for contents in stage_files.values()]
            if not all(is_number(v) for v in stage_values):
                continue
            stage_sum = sum(float(v) for v in stage_values)
            if stage_sum != summary_value:
                dotted = ".".join(path)
                findings.append(
                    _global_error(
                        f"{self.name}.stage_totals",
                        f"per-stage {dotted} values sum to {stage_sum:g} but the run summary reports "
                        f"{summary_value:g} — requests dropped or double-counted across stages "
                        f"(stages: {sorted(stage_files)})",
                    )
                )
        return findings

    def _check_per_request_count(self, reports: ReportSet) -> List[Finding]:
        """Per-request records must reconcile with the run summary count."""
        records = reports.contents(PER_REQUEST_FILENAME)
        summary_count = self._summary_count(reports, "load_summary", "count")
        if not isinstance(records, list) or summary_count is None:
            return []
        if len(records) != summary_count:
            return [
                _global_error(
                    f"{self.name}.per_request_count",
                    f"{PER_REQUEST_FILENAME} has {len(records)} record(s) but the run summary reports "
                    f"{summary_count:g} request(s)",
                )
            ]
        return []

    def _check_stage_indices(self, reports: ReportSet) -> List[Finding]:
        """A gap in stage numbering means a stage produced no report at all."""
        stage_ids = sorted(reports.stage_lifecycle_files())
        if not stage_ids:
            return []
        missing = sorted(set(range(max(stage_ids) + 1)) - set(stage_ids))
        if missing:
            return [
                _global_warning(
                    f"{self.name}.stage_indices",
                    f"no per-stage lifecycle report for stage(s) {missing} — those stages recorded no requests",
                )
            ]
        return []

    def _check_configured_stages(self, reports: ReportSet) -> List[Finding]:
        """Emitted stage reports must correspond to stages the config declares."""
        config = reports.contents(CONFIG_FILENAME)
        stages = get_path(config, "load", "stages") if isinstance(config, dict) else None
        if not isinstance(stages, list):
            return []
        stage_ids = sorted(reports.stage_lifecycle_files())
        if not stage_ids:
            return []

        findings: List[Finding] = []
        extra = [stage_id for stage_id in stage_ids if stage_id >= len(stages)]
        if extra:
            findings.append(
                _global_error(
                    f"{self.name}.configured_stages",
                    f"per-stage report(s) for stage(s) {extra} but the config declares only {len(stages)} stage(s)",
                )
            )
        absent = sorted(set(range(len(stages))) - set(stage_ids))
        if absent:
            findings.append(
                _global_warning(
                    f"{self.name}.configured_stages",
                    f"config declares {len(stages)} stage(s) but stage(s) {absent} emitted no lifecycle report",
                )
            )
        return findings

    def _check_stage_rates(self, reports: ReportSet) -> List[Finding]:
        """Requested rates must be monotonically increasing across stages.

        Non-decreasing, precisely: a sweep may repeat a rate, but ramping down
        is a suspicious configuration, not an internal inconsistency — hence a
        warning. Rates are only ordered while they are all numeric; if any
        stage carries a future non-numeric rate type, the check skips rather
        than guess an ordering.
        """
        ordered: List[tuple[int, float]] = []
        for stage_id, contents in sorted(reports.stage_lifecycle_files().items()):
            rate = get_path(contents, "load_summary", "requested_rate")
            if not is_number(rate):
                return []
            ordered.append((stage_id, float(rate)))

        findings: List[Finding] = []
        for (prev_id, prev_rate), (stage_id, rate) in zip(ordered, ordered[1:], strict=False):
            if rate < prev_rate:
                findings.append(
                    _global_warning(
                        f"{self.name}.stage_rates",
                        f"stage requested rates are not monotonically increasing: stage {stage_id} "
                        f"requested_rate {rate:g} < stage {prev_id} requested_rate {prev_rate:g}",
                    )
                )
        return findings
