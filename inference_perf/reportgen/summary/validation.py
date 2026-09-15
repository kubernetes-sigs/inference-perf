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
"""Validation of the run-wide lifecycle summary report.

Also home to the content checks shared with the per-stage validator: a
per-stage lifecycle file is the same ``ResponsesSummary`` shape as the run
summary, so both validators run the same checks against their files.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from inference_perf.reportgen.validation import (
    SUMMARY_LIFECYCLE_FILENAME,
    Check,
    Finding,
    ReportSet,
    ReportSetValidator,
    Severity,
    StopValidation,
    approx_eq,
    approx_le,
    distribution_findings,
    get_path as _get,
    is_number,
)


def _error(check: str, filename: str, message: str) -> Finding:
    return Finding(check=check, severity=Severity.ERROR, message=message, report=filename)


def _warning(check: str, filename: str, message: str) -> Finding:
    return Finding(check=check, severity=Severity.WARNING, message=message, report=filename)


def structure_findings(filename: str, contents: Any, check: str) -> List[Finding]:
    """The file must be a dict with the three top-level summary sections."""
    if not isinstance(contents, dict):
        return [_error(check, filename, f"expected a JSON object, got {type(contents).__name__}")]
    missing = [key for key in ("load_summary", "successes", "failures") if key not in contents]
    if missing:
        return [_error(check, filename, f"missing top-level section(s): {', '.join(missing)}")]
    return []


def count_findings(filename: str, contents: Any, check: str) -> List[Finding]:
    """successes.count + failures.count must equal load_summary.count."""
    total = _get(contents, "load_summary", "count")
    successes = _get(contents, "successes", "count")
    failures = _get(contents, "failures", "count")
    if not (is_number(total) and is_number(successes) and is_number(failures)):
        return [_error(check, filename, "missing or non-numeric request counts")]
    if successes + failures != total:
        return [
            _error(
                check,
                filename,
                f"request counts do not reconcile: successes ({successes}) + failures ({failures}) "
                f"!= load_summary.count ({total})",
            )
        ]
    return []


def time_findings(filename: str, contents: Any, check: str) -> List[Finding]:
    findings: List[Finding] = []
    benchmark_time = _get(contents, "benchmark_time_seconds")
    total = _get(contents, "load_summary", "count")
    if is_number(benchmark_time):
        if benchmark_time < 0:
            findings.append(_error(check, filename, f"negative benchmark_time_seconds: {benchmark_time}"))
        elif is_number(total) and total > 0 and benchmark_time == 0:
            findings.append(_warning(check, filename, f"benchmark_time_seconds is 0 with {total} request(s)"))
    return findings


def token_findings(filename: str, contents: Any, check: str) -> List[Finding]:
    """The prompt/output token aggregates must be internally consistent.

    ``total`` is a sum over the same per-request values whose distribution is
    reported alongside it, so ``total == mean * successes.count`` up to float
    noise; the cached/uncached split must partition the prompt total.
    """
    findings: List[Finding] = []
    successes_count = _get(contents, "successes", "count")

    prompt = _get(contents, "successes", "prompt_tokens")
    if isinstance(prompt, dict):
        total, cached, uncached = prompt.get("total"), prompt.get("cached"), prompt.get("uncached")
        if is_number(total) and is_number(cached):
            if not approx_le(cached, total):
                findings.append(
                    _error(check, filename, f"prompt_tokens.cached ({cached}) exceeds prompt_tokens.total ({total})")
                )
            # The generator clamps uncached at 0; mirror the clamp.
            if is_number(uncached) and not approx_eq(uncached, max(total - cached, 0.0)):
                findings.append(
                    _error(
                        check,
                        filename,
                        f"prompt_tokens.uncached ({uncached}) != total - cached ({total} - {cached})",
                    )
                )
        mean = prompt.get("mean")
        if is_number(total) and is_number(mean) and is_number(successes_count) and successes_count > 0:
            if not approx_eq(total, mean * successes_count):
                findings.append(
                    _error(
                        check,
                        filename,
                        f"prompt_tokens.total ({total}) != mean * successes.count ({mean} * {successes_count})",
                    )
                )

    output = _get(contents, "successes", "output_tokens")
    if isinstance(output, dict):
        total, mean = output.get("total"), output.get("mean")
        if is_number(total) and is_number(mean) and is_number(successes_count) and successes_count > 0:
            if not approx_eq(total, mean * successes_count):
                findings.append(
                    _error(
                        check,
                        filename,
                        f"output_tokens.total ({total}) != mean * successes.count ({mean} * {successes_count})",
                    )
                )

    return findings


def throughput_findings(filename: str, contents: Any, check: str) -> List[Finding]:
    findings: List[Finding] = []
    throughput = _get(contents, "successes", "throughput")
    if not isinstance(throughput, dict):
        return findings

    for key, value in throughput.items():
        if is_number(value) and value < 0:
            findings.append(_error(check, filename, f"negative throughput {key}: {value}"))

    benchmark_time = _get(contents, "benchmark_time_seconds")
    successes_count = _get(contents, "successes", "count")
    requests_per_sec = throughput.get("requests_per_sec")
    if (
        is_number(benchmark_time)
        and benchmark_time > 0
        and is_number(successes_count)
        and is_number(requests_per_sec)
        and not approx_eq(requests_per_sec * benchmark_time, float(successes_count))
    ):
        findings.append(
            _error(
                check,
                filename,
                f"requests_per_sec ({requests_per_sec}) * benchmark_time_seconds ({benchmark_time}) "
                f"!= successes.count ({successes_count})",
            )
        )

    input_rate = throughput.get("input_tokens_per_sec")
    output_rate = throughput.get("output_tokens_per_sec")
    total_rate = throughput.get("total_tokens_per_sec")
    if is_number(input_rate) and is_number(output_rate) and is_number(total_rate):
        if not approx_eq(input_rate + output_rate, total_rate):
            findings.append(
                _error(
                    check,
                    filename,
                    f"total_tokens_per_sec ({total_rate}) != input_tokens_per_sec ({input_rate}) "
                    f"+ output_tokens_per_sec ({output_rate})",
                )
            )
    return findings


def goodput_findings(filename: str, contents: Any, check: str) -> List[Finding]:
    findings: List[Finding] = []
    goodput = _get(contents, "successes", "goodput_metrics")
    if not isinstance(goodput, dict):
        return findings

    good, total = goodput.get("good_requests"), goodput.get("total_requests")
    if is_number(good) and is_number(total) and good > total:
        findings.append(_error(check, filename, f"good_requests ({good}) exceeds total_requests ({total})"))

    successes_count = _get(contents, "successes", "count")
    if is_number(total) and is_number(successes_count) and total != successes_count:
        findings.append(_error(check, filename, f"goodput total_requests ({total}) != successes.count ({successes_count})"))

    for key, value in goodput.items():
        if key.endswith("percentage") and is_number(value) and not (0 <= value <= 100 + 1e-9):
            findings.append(_error(check, filename, f"goodput {key} outside [0, 100]: {value}"))
    return findings


def token_mismatch_findings(filename: str, contents: Any, check: str) -> List[Finding]:
    """Surface the in-band client/server token mismatch counter as a warning.

    A non-zero count is the #564 regression family showing up in-band:
    per-chunk re-tokenization disagreeing with the server's own count.
    """
    mismatches = _get(contents, "successes", "token_count_mismatches")
    if is_number(mismatches) and mismatches > 0:
        return [
            _warning(
                check,
                filename,
                f"{mismatches} request(s) where the client-side token count disagrees with the "
                "server-reported completion_tokens",
            )
        ]
    return []


def summary_content_findings(filename: str, contents: Any, check_prefix: str) -> List[Finding]:
    """All shared content checks against one ResponsesSummary-shaped file."""
    findings = list(count_findings(filename, contents, f"{check_prefix}.counts"))
    findings += time_findings(filename, contents, f"{check_prefix}.time")
    findings += distribution_findings(filename, contents, f"{check_prefix}.distributions")
    findings += token_findings(filename, contents, f"{check_prefix}.tokens")
    findings += throughput_findings(filename, contents, f"{check_prefix}.throughput")
    findings += goodput_findings(filename, contents, f"{check_prefix}.goodput")
    findings += token_mismatch_findings(filename, contents, f"{check_prefix}.token_mismatches")
    return findings


class SummaryLifecycleValidator(ReportSetValidator):
    name = "summary"

    def covers(self, reports: ReportSet) -> List[str]:
        return [SUMMARY_LIFECYCLE_FILENAME] if SUMMARY_LIFECYCLE_FILENAME in reports.filenames() else []

    def checks(self) -> Sequence[Check]:
        return [
            self._check_structure,
            self._check_counts,
            self._check_time,
            self._check_distributions,
            self._check_tokens,
            self._check_throughput,
            self._check_goodput,
            self._check_token_mismatches,
        ]

    def _contents(self, reports: ReportSet) -> Dict[str, Any]:
        contents = reports.contents(SUMMARY_LIFECYCLE_FILENAME)
        assert isinstance(contents, dict)  # guaranteed by _check_structure running first
        return contents

    def _check_structure(self, reports: ReportSet) -> List[Finding]:
        """Halts the validator when the summary is absent (silently: the report
        may be disabled by config) or structurally broken (fatally: every
        later check would only cascade)."""
        contents = reports.contents(SUMMARY_LIFECYCLE_FILENAME)
        if contents is None:
            raise StopValidation()
        findings = structure_findings(SUMMARY_LIFECYCLE_FILENAME, contents, f"{self.name}.structure")
        if findings:
            raise StopValidation(findings)
        return []

    def _check_counts(self, reports: ReportSet) -> List[Finding]:
        return count_findings(SUMMARY_LIFECYCLE_FILENAME, self._contents(reports), f"{self.name}.counts")

    def _check_time(self, reports: ReportSet) -> List[Finding]:
        return time_findings(SUMMARY_LIFECYCLE_FILENAME, self._contents(reports), f"{self.name}.time")

    def _check_distributions(self, reports: ReportSet) -> List[Finding]:
        return distribution_findings(SUMMARY_LIFECYCLE_FILENAME, self._contents(reports), f"{self.name}.distributions")

    def _check_tokens(self, reports: ReportSet) -> List[Finding]:
        return token_findings(SUMMARY_LIFECYCLE_FILENAME, self._contents(reports), f"{self.name}.tokens")

    def _check_throughput(self, reports: ReportSet) -> List[Finding]:
        return throughput_findings(SUMMARY_LIFECYCLE_FILENAME, self._contents(reports), f"{self.name}.throughput")

    def _check_goodput(self, reports: ReportSet) -> List[Finding]:
        return goodput_findings(SUMMARY_LIFECYCLE_FILENAME, self._contents(reports), f"{self.name}.goodput")

    def _check_token_mismatches(self, reports: ReportSet) -> List[Finding]:
        return token_mismatch_findings(SUMMARY_LIFECYCLE_FILENAME, self._contents(reports), f"{self.name}.token_mismatches")


__all__ = [
    "SummaryLifecycleValidator",
    "count_findings",
    "distribution_findings",
    "goodput_findings",
    "structure_findings",
    "summary_content_findings",
    "throughput_findings",
    "time_findings",
    "token_findings",
    "token_mismatch_findings",
]
