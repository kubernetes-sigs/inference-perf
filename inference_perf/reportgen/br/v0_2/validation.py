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
"""Validation of the emitted BR0.2 partial reports.

Three contracts are enforced per ``inference-perf.partial.stage_<n>.yaml``:

1. **Schema**: the partial must validate as a ``BenchmarkReportV021`` document
   on its own (required fields populated, optional sections absent).
2. **Mergeability**: no ``None`` values anywhere — a downstream composer's
   deep-merge must never see an inference-perf ``null`` clobber real data.
3. **Agreement**: the BR aggregate and the native per-stage lifecycle report
   are two projections of the same request metrics through two code paths
   (``br/v0_2/adapter.py`` vs ``reportgen/base.py``). Any disagreement is
   #564-family drift: one of the paths changed and the other did not.
"""

from __future__ import annotations

import re
from typing import Any, Dict, List, Optional, Sequence

from inference_perf.reportgen.validation import (
    Check,
    Finding,
    ReportSet,
    ReportSetValidator,
    Severity,
    StopValidation,
    approx_eq,
    distribution_findings,
    get_path,
    is_number,
)

from .schema import VERSION, BenchmarkReportV021

_ISO_DURATION_RE = re.compile(r"^PT(\d+(?:\.\d+)?)S$")

# run.time.duration is serialized with millisecond precision, so a wall-clock
# comparison against the float benchmark window needs millisecond slack.
_DURATION_ABS_TOL = 2e-3

# (path into the BR aggregate, path into the stage lifecycle file)
_AGREEMENT_PATHS: List[tuple[tuple[str, ...], tuple[str, ...]]] = [
    (("requests", "total"), ("load_summary", "count")),
    (("requests", "failures"), ("failures", "count")),
    (("latency", "request_latency", "mean"), ("successes", "latency", "request_latency", "mean")),
    (("latency", "request_latency", "min"), ("successes", "latency", "request_latency", "min")),
    (("latency", "request_latency", "max"), ("successes", "latency", "request_latency", "max")),
    (("latency", "request_latency", "p50"), ("successes", "latency", "request_latency", "median")),
    (("latency", "time_to_first_token", "mean"), ("successes", "latency", "time_to_first_token", "mean")),
    (("latency", "inter_token_latency", "mean"), ("successes", "latency", "inter_token_latency", "mean")),
    (
        ("latency", "normalized_time_per_output_token", "mean"),
        ("successes", "latency", "normalized_time_per_output_token", "mean"),
    ),
    (("throughput", "request_rate", "mean"), ("successes", "throughput", "requests_per_sec")),
    (("throughput", "input_token_rate", "mean"), ("successes", "throughput", "input_tokens_per_sec")),
    (("throughput", "output_token_rate", "mean"), ("successes", "throughput", "output_tokens_per_sec")),
    (("throughput", "total_token_rate", "mean"), ("successes", "throughput", "total_tokens_per_sec")),
]


def _partial_filename(stage_id: int) -> str:
    return f"inference-perf.partial.stage_{stage_id}.yaml"


def _find_nulls(node: Any, path: str = "") -> List[str]:
    paths: List[str] = []
    if node is None:
        return [path or "<root>"]
    if isinstance(node, dict):
        for key, value in node.items():
            paths.extend(_find_nulls(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            paths.extend(_find_nulls(item, f"{path}[{i}]"))
    return paths


class BrPartialValidator(ReportSetValidator):
    name = "br_partial"

    def covers(self, reports: ReportSet) -> List[str]:
        return [_partial_filename(stage_id) for stage_id in sorted(reports.br_partial_files())]

    def checks(self) -> Sequence[Check]:
        return [
            self._check_schema,
            self._check_version,
            self._check_nulls,
            self._check_distributions,
            self._check_run_time,
            self._check_lifecycle_agreement,
        ]

    def _schema_valid(self, reports: ReportSet) -> Dict[int, Any]:
        """Partials that validate against the schema; the rest are skipped."""
        valid: Dict[int, Any] = {}
        for stage_id, contents in reports.br_partial_files().items():
            try:
                BenchmarkReportV021.model_validate(contents)
            except Exception:
                continue
            valid[stage_id] = contents
        return valid

    def _check_schema(self, reports: ReportSet) -> List[Finding]:
        partials = reports.br_partial_files()
        if not partials:
            raise StopValidation()
        findings: List[Finding] = []
        for stage_id, contents in sorted(partials.items()):
            try:
                BenchmarkReportV021.model_validate(contents)
            except Exception as exc:
                findings.append(
                    Finding(
                        check=f"{self.name}.schema",
                        severity=Severity.ERROR,
                        message=f"does not validate as a BR {VERSION} document: {exc}",
                        report=_partial_filename(stage_id),
                    )
                )
        return findings

    def _check_version(self, reports: ReportSet) -> List[Finding]:
        # Runs over every partial, not just schema-valid ones: a wrong version
        # literal can itself be the schema failure, and it should still be
        # named explicitly.
        findings: List[Finding] = []
        for stage_id, contents in sorted(reports.br_partial_files().items()):
            version = get_path(contents, "version")
            if version != VERSION:
                findings.append(
                    Finding(
                        check=f"{self.name}.version",
                        severity=Severity.ERROR,
                        message=f"declares version {version!r}, expected {VERSION!r}",
                        report=_partial_filename(stage_id),
                    )
                )
        return findings

    def _check_nulls(self, reports: ReportSet) -> List[Finding]:
        """The merge contract: a partial must never carry a null value."""
        findings: List[Finding] = []
        for stage_id, contents in sorted(reports.br_partial_files().items()):
            null_paths = _find_nulls(contents)
            if null_paths:
                sample = ", ".join(null_paths[:5])
                findings.append(
                    Finding(
                        check=f"{self.name}.nulls",
                        severity=Severity.ERROR,
                        message=f"{len(null_paths)} null value(s) would clobber composer data on merge: {sample}",
                        report=_partial_filename(stage_id),
                    )
                )
        return findings

    def _check_distributions(self, reports: ReportSet) -> List[Finding]:
        findings: List[Finding] = []
        for stage_id, contents in sorted(self._schema_valid(reports).items()):
            findings += distribution_findings(
                _partial_filename(stage_id), contents, f"{self.name}.distributions", negative_ok_substrings=()
            )
        return findings

    def _check_run_time(self, reports: ReportSet) -> List[Finding]:
        """run.time.duration must match the stage's benchmark window."""
        findings: List[Finding] = []
        for stage_id, contents in sorted(self._schema_valid(reports).items()):
            duration = _parse_iso_duration(get_path(contents, "run", "time", "duration"))
            stage = reports.stage_lifecycle_files().get(stage_id)
            benchmark_time = get_path(stage, "benchmark_time_seconds") if isinstance(stage, dict) else None
            if duration is None or not is_number(benchmark_time):
                continue
            if abs(duration - benchmark_time) > _DURATION_ABS_TOL:
                findings.append(
                    Finding(
                        check=f"{self.name}.run_time",
                        severity=Severity.ERROR,
                        message=f"run.time.duration ({duration}s) disagrees with the stage lifecycle "
                        f"benchmark_time_seconds ({benchmark_time}s)",
                        report=_partial_filename(stage_id),
                    )
                )
        return findings

    def _check_lifecycle_agreement(self, reports: ReportSet) -> List[Finding]:
        """The BR aggregate must agree with the native per-stage report."""
        findings: List[Finding] = []
        stage_files = reports.stage_lifecycle_files()
        for stage_id, contents in sorted(self._schema_valid(reports).items()):
            stage = stage_files.get(stage_id)
            if not isinstance(stage, dict):
                continue
            aggregate = get_path(contents, "results", "request_performance", "aggregate")
            if not isinstance(aggregate, dict):
                continue
            for br_path, stage_path in _AGREEMENT_PATHS:
                br_value = get_path(aggregate, *br_path)
                stage_value = get_path(stage, *stage_path)
                if not (is_number(br_value) and is_number(stage_value)):
                    continue
                if not approx_eq(float(br_value), float(stage_value)):
                    findings.append(
                        Finding(
                            check=f"{self.name}.lifecycle_agreement",
                            severity=Severity.ERROR,
                            message=f"aggregate.{'.'.join(br_path)} ({br_value}) disagrees with the stage "
                            f"lifecycle {'.'.join(stage_path)} ({stage_value}) — the BR adapter and the "
                            "native report math have drifted apart",
                            report=_partial_filename(stage_id),
                        )
                    )
        return findings


def _parse_iso_duration(value: Any) -> Optional[float]:
    if not isinstance(value, str):
        return None
    match = _ISO_DURATION_RE.match(value)
    return float(match.group(1)) if match else None
