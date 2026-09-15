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
"""Validation of the final assembled report set.

After ``ReportGenerator.generate_reports`` assembles the run's report files,
the validators registered here inspect them for internal consistency and emit
findings at two severities:

- ``warning``: suspicious but not necessarily wrong (e.g. client/server token
  counts diverging within the reasoning-channel caveat).
- ``error``: the report set is internally inconsistent — the tool's own math
  does not reconcile. Errors indicate a bug in inference-perf, never a
  misbehaving model server: a run with many failed requests is a successful
  benchmark of an unhealthy server and must validate cleanly.

Findings land in two places: the run's logs, and a ``validation.json`` report
emitted alongside the other report files, shaped as::

    {
      "global": {"warnings": [...], "errors": [...]},
      "reports": {
        "<report filename>": {"warnings": [...], "errors": [...]},
        ...
      }
    }

Every file a validator inspected appears under ``reports`` (an empty group
means "validated, clean"); findings that span multiple files land under
``global``. Validation never fails the run: it does not raise, and it does not
affect the exit code. Test tiers (unit and e2e) treat ``validation.json`` as
their assertion interface.

Each report family owns its checks in a ``validation.py`` next to (or named
for) the reports it validates: ``summary/``, ``per_stage/``, ``per_request/``,
``cross_report/`` (multi-file consistency), and ``br/v0_2/`` for the BR0.2
partials. Checks run sequentially within a validator; a check may raise
:class:`StopValidation` to halt its validator's remaining checks. Session,
Prometheus, and per-adapter reports do not have validators yet.
"""

from __future__ import annotations

import logging
import re
from abc import ABC, abstractmethod
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Sequence, TypeGuard

from pydantic import BaseModel, ConfigDict, Field

from inference_perf.utils import ReportFile

logger = logging.getLogger(__name__)

# Anchored so adapter_<name>_stage_<n>_lifecycle_metrics files never match.
_STAGE_LIFECYCLE_RE = re.compile(r"^stage_(\d+)_lifecycle_metrics\.json$")
_BR_PARTIAL_RE = re.compile(r"^inference-perf\.partial\.stage_(\d+)\.yaml$")

SUMMARY_LIFECYCLE_FILENAME = "summary_lifecycle_metrics.json"
PER_REQUEST_FILENAME = "per_request_lifecycle_metrics.json"
CONFIG_FILENAME = "config.yaml"
VALIDATION_REPORT_NAME = "validation"


class Severity(str, Enum):
    WARNING = "warning"
    ERROR = "error"


class Finding(BaseModel):
    """A single validation finding.

    ``report`` names the report file the finding pertains to; ``None`` means
    the finding spans files and belongs to the ``global`` group.
    """

    check: str
    severity: Severity
    message: str
    report: Optional[str] = None


class FindingGroup(BaseModel):
    warnings: List[Finding] = Field(default_factory=list)
    errors: List[Finding] = Field(default_factory=list)

    def is_clean(self) -> bool:
        return not self.warnings and not self.errors


class ValidationReport(BaseModel):
    """The full result of validating one run's report set."""

    model_config = ConfigDict(populate_by_name=True)

    global_findings: FindingGroup = Field(default_factory=FindingGroup, alias="global")
    reports: Dict[str, FindingGroup] = Field(default_factory=dict)

    def _group_for(self, finding: Finding) -> FindingGroup:
        if finding.report is None:
            return self.global_findings
        return self.reports.setdefault(finding.report, FindingGroup())

    def add(self, finding: Finding) -> None:
        group = self._group_for(finding)
        if finding.severity == Severity.ERROR:
            group.errors.append(finding)
        else:
            group.warnings.append(finding)

    def all_errors(self) -> List[Finding]:
        return self.global_findings.errors + [f for group in self.reports.values() for f in group.errors]

    def all_warnings(self) -> List[Finding]:
        return self.global_findings.warnings + [f for group in self.reports.values() for f in group.warnings]


class ReportSet:
    """Read-only view over the assembled report files, keyed by filename."""

    def __init__(self, reports: Sequence[ReportFile]) -> None:
        self._by_filename: Dict[str, ReportFile] = {r.get_filename(): r for r in reports}

    def filenames(self) -> List[str]:
        return list(self._by_filename)

    def contents(self, filename: str) -> Any:
        """Contents of the named report, or ``None`` when absent."""
        report = self._by_filename.get(filename)
        return report.get_contents() if report is not None else None

    def stage_lifecycle_files(self) -> Dict[int, Any]:
        """Per-stage lifecycle reports as ``{stage_id: contents}``."""
        return self._match_by_stage(_STAGE_LIFECYCLE_RE)

    def br_partial_files(self) -> Dict[int, Any]:
        """BR0.2 partial reports as ``{stage_id: contents}``."""
        return self._match_by_stage(_BR_PARTIAL_RE)

    def _match_by_stage(self, pattern: re.Pattern[str]) -> Dict[int, Any]:
        found: Dict[int, Any] = {}
        for filename, report in self._by_filename.items():
            match = pattern.match(filename)
            if match:
                found[int(match.group(1))] = report.get_contents()
        return found


Check = Callable[[ReportSet], List[Finding]]

# Relative/absolute slack for comparing values that two code paths derived
# from the same inputs: generous against float noise, far below real drift.
_REL_TOL = 1e-6
_ABS_TOL = 1e-9

_PERCENTILE_KEY_RE = re.compile(r"^p(\d+(?:[.p]\d+)?)$")


def is_number(value: Any) -> TypeGuard[float]:
    """True for real numbers; also narrows the value's type for mypy."""
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def get_path(root: Any, *path: str) -> Any:
    """Walk nested dicts; ``None`` as soon as a step is missing or not a dict."""
    node = root
    for key in path:
        if not isinstance(node, dict):
            return None
        node = node.get(key)
    return node


def approx_le(a: float, b: float) -> bool:
    return a <= b + _ABS_TOL + _REL_TOL * max(abs(a), abs(b))


def approx_eq(a: float, b: float) -> bool:
    return abs(a - b) <= _ABS_TOL + _REL_TOL * max(abs(a), abs(b))


def percentile_rank(key: str) -> Optional[float]:
    """Percentile rank encoded by a distribution key, or ``None``.

    Understands the native style (``median``, ``p90``, ``p99.9``) and the
    BR0.2 style where ``p`` doubles as the decimal separator (``p99p9``).
    """
    if key == "median":
        return 50.0
    match = _PERCENTILE_KEY_RE.match(key)
    if match:
        return float(match.group(1).replace("p", "."))
    return None


def walk_distribution_blocks(node: Any, path: str = "") -> List[tuple[str, Dict[str, Any]]]:
    """Find every distribution block (a dict with numeric min and max) in a tree."""
    blocks: List[tuple[str, Dict[str, Any]]] = []
    if isinstance(node, dict):
        if is_number(node.get("min")) and is_number(node.get("max")):
            blocks.append((path, node))
        for key, value in node.items():
            blocks.extend(walk_distribution_blocks(value, f"{path}.{key}" if path else str(key)))
    elif isinstance(node, list):
        for i, item in enumerate(node):
            blocks.extend(walk_distribution_blocks(item, f"{path}[{i}]"))
    return blocks


def distribution_findings(
    filename: str,
    root: Any,
    check: str,
    *,
    negative_ok_substrings: Sequence[str] = ("schedule_delay",),
) -> List[Finding]:
    """Validate every distribution block found under ``root``.

    Asserts min <= mean/median/percentiles <= max, that percentile values are
    non-decreasing in rank, and that values are non-negative — except under
    paths matching ``negative_ok_substrings``, where a negative minimum
    downgrades to a warning (e.g. ``schedule_delay``, where a small negative
    value signals a scheduling anomaly rather than broken math).
    """
    findings: List[Finding] = []

    def error(message: str) -> None:
        findings.append(Finding(check=check, severity=Severity.ERROR, message=message, report=filename))

    for path, block in walk_distribution_blocks(root):
        low, high = float(block["min"]), float(block["max"])
        if not approx_le(low, high):
            error(f"{path}: min {low} > max {high}")

        if low < 0:
            if any(s in path for s in negative_ok_substrings):
                findings.append(
                    Finding(
                        check=check,
                        severity=Severity.WARNING,
                        message=f"{path}: negative minimum {low}",
                        report=filename,
                    )
                )
            else:
                error(f"{path}: negative minimum {low}")

        ranked: List[tuple[float, str, float]] = []
        for key, value in block.items():
            if not is_number(value):
                continue
            if key == "mean" and not (approx_le(low, float(value)) and approx_le(float(value), high)):
                error(f"{path}: mean {value} outside [min, max] = [{low}, {high}]")
            rank = percentile_rank(key)
            if rank is not None:
                if not (approx_le(low, float(value)) and approx_le(float(value), high)):
                    error(f"{path}.{key}: {value} outside [min, max] = [{low}, {high}]")
                ranked.append((rank, key, float(value)))

        ranked.sort()
        for (_, key_a, val_a), (_, key_b, val_b) in zip(ranked, ranked[1:], strict=False):
            if not approx_le(val_a, val_b):
                error(f"{path}: percentile ordering violated: {key_a}={val_a} > {key_b}={val_b}")

    return findings


class StopValidation(Exception):
    """Raised by a check to halt its validator's remaining checks.

    Any findings attached are still recorded, so a check can emit a fatal
    error and stop (e.g. the file is structurally broken and downstream checks
    would only cascade), or stop silently when its report family was not
    emitted for this run.
    """

    def __init__(self, findings: Optional[List[Finding]] = None) -> None:
        super().__init__()
        self.findings: List[Finding] = findings or []


class ReportSetValidator(ABC):
    """Validates one report family against the final assembled report set.

    ``checks()`` returns the sequence of checks to run, in order. Checks
    receive the full :class:`ReportSet` so cross-file context is available,
    but a family validator should confine its findings to its own files —
    multi-file consistency belongs to the ``cross_report`` validator.
    """

    name: str = "validator"

    @abstractmethod
    def checks(self) -> Sequence[Check]: ...

    def covers(self, reports: ReportSet) -> List[str]:
        """Filenames this validator inspects, given the assembled set.

        Covered files appear in ``validation.json`` even when clean, so their
        presence there means "validated", not just "no findings".
        """
        return []


def run_validators(validators: Sequence[ReportSetValidator], reports: Sequence[ReportFile]) -> ValidationReport:
    """Run each validator's checks sequentially and collect all findings.

    A crashing check is recorded as an error finding against the validator and
    does not stop its sibling checks; only :class:`StopValidation` halts a
    validator early. A crashing validator never propagates: report emission
    must not depend on validation succeeding.
    """
    report_set = ReportSet(reports)
    result = ValidationReport()

    for validator in validators:
        for filename in validator.covers(report_set):
            result.reports.setdefault(filename, FindingGroup())

        try:
            checks: Sequence[Check] = validator.checks()
        except Exception as exc:
            result.add(
                Finding(
                    check=f"{validator.name}.internal",
                    severity=Severity.ERROR,
                    message=f"validator failed to produce its checks: {exc!r}",
                )
            )
            continue

        for check in checks:
            try:
                for finding in check(report_set):
                    result.add(finding)
            except StopValidation as halt:
                for finding in halt.findings:
                    result.add(finding)
                break
            except Exception as exc:
                check_name = getattr(check, "__name__", repr(check))
                result.add(
                    Finding(
                        check=f"{validator.name}.internal",
                        severity=Severity.ERROR,
                        message=f"check {check_name} crashed: {exc!r}",
                    )
                )

    return result


def default_validators() -> List[ReportSetValidator]:
    # Imported here, not at module top: family modules import this framework
    # module, and base.py imports this module, so a top-level import of the
    # families here would be circular for any family needing reportgen.base.
    from inference_perf.reportgen.br.v0_2.validation import BrPartialValidator
    from inference_perf.reportgen.cross_report.validation import CrossReportValidator
    from inference_perf.reportgen.per_request.validation import PerRequestLifecycleValidator
    from inference_perf.reportgen.per_stage.validation import PerStageLifecycleValidator
    from inference_perf.reportgen.summary.validation import SummaryLifecycleValidator

    return [
        SummaryLifecycleValidator(),
        PerStageLifecycleValidator(),
        PerRequestLifecycleValidator(),
        CrossReportValidator(),
        BrPartialValidator(),
    ]


def validate_reports(reports: Sequence[ReportFile]) -> ReportFile:
    """Validate the assembled report set and return the validation report file.

    Logs every finding (warnings via ``logger.warning``, errors via
    ``logger.error``) plus a one-line summary, and returns the findings as a
    ``validation.json`` :class:`ReportFile` to emit alongside the other
    reports.
    """
    result = run_validators(default_validators(), reports)

    for finding in result.all_warnings():
        logger.warning("Validation [%s] %s: %s", finding.check, finding.report or "global", finding.message)
    for finding in result.all_errors():
        logger.error("Validation [%s] %s: %s", finding.check, finding.report or "global", finding.message)

    errors, warnings = len(result.all_errors()), len(result.all_warnings())
    if errors or warnings:
        logger.warning("Report validation finished: %d error(s), %d warning(s). See validation.json.", errors, warnings)
    else:
        logger.info("Report validation finished: no findings across %d report file(s).", len(result.reports))

    return ReportFile(
        name=VALIDATION_REPORT_NAME,
        contents=result.model_dump(mode="json", by_alias=True),
    )
