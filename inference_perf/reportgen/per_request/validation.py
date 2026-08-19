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
"""Validation of the per-request lifecycle report.

Per-record invariants over ``per_request_lifecycle_metrics.json``: timestamp
ordering, token-timestamp bookkeeping (every streamed token time must fall
inside the request's lifetime, in order), and the weak-form client/server
token-count agreement check. Findings are aggregated — one finding per rule
with a violation count and sample record indices — so a systematic bug in a
100k-request run does not produce 100k findings.
"""

from __future__ import annotations

from typing import Any, Dict, Iterator, List, Sequence, Tuple

from inference_perf.reportgen.validation import (
    PER_REQUEST_FILENAME,
    Check,
    Finding,
    ReportSet,
    ReportSetValidator,
    Severity,
    StopValidation,
    is_number,
)

_SAMPLE_LIMIT = 5

# Client-side re-tokenization and the server's own decode-step count may
# legitimately diverge (reasoning-channel tokens are counted server-side but
# may not appear in the visible stream), so agreement is checked loosely and
# violations are warnings, not errors.
_TOKEN_AGREEMENT_REL_TOL = 0.25


def _aggregate(check: str, severity: Severity, description: str, indices: List[int]) -> List[Finding]:
    if not indices:
        return []
    sample = ", ".join(str(i) for i in indices[:_SAMPLE_LIMIT])
    suffix = f" (record indices: {sample}{', …' if len(indices) > _SAMPLE_LIMIT else ''})"
    return [
        Finding(
            check=check,
            severity=severity,
            message=f"{len(indices)} record(s) {description}{suffix}",
            report=PER_REQUEST_FILENAME,
        )
    ]


class PerRequestLifecycleValidator(ReportSetValidator):
    name = "per_request"

    def covers(self, reports: ReportSet) -> List[str]:
        return [PER_REQUEST_FILENAME] if PER_REQUEST_FILENAME in reports.filenames() else []

    def checks(self) -> Sequence[Check]:
        return [
            self._check_structure,
            self._check_timestamps,
            self._check_response_presence,
            self._check_token_times,
            self._check_token_agreement,
        ]

    def _records(self, reports: ReportSet) -> Iterator[Tuple[int, Dict[str, Any]]]:
        contents = reports.contents(PER_REQUEST_FILENAME)
        assert isinstance(contents, list)  # guaranteed by _check_structure running first
        for i, record in enumerate(contents):
            if isinstance(record, dict):
                yield i, record

    def _check_structure(self, reports: ReportSet) -> List[Finding]:
        contents = reports.contents(PER_REQUEST_FILENAME)
        if contents is None:
            raise StopValidation()
        if not isinstance(contents, list):
            raise StopValidation(
                [
                    Finding(
                        check=f"{self.name}.structure",
                        severity=Severity.ERROR,
                        message=f"expected a JSON array of records, got {type(contents).__name__}",
                        report=PER_REQUEST_FILENAME,
                    )
                ]
            )
        malformed = [i for i, record in enumerate(contents) if not isinstance(record, dict)]
        return _aggregate(f"{self.name}.structure", Severity.ERROR, "are not JSON objects", malformed)

    def _check_timestamps(self, reports: ReportSet) -> List[Finding]:
        bad = [
            i
            for i, record in self._records(reports)
            if is_number(record.get("start_time"))
            and is_number(record.get("end_time"))
            and record["end_time"] < record["start_time"]
        ]
        return _aggregate(f"{self.name}.timestamps", Severity.ERROR, "have end_time before start_time", bad)

    def _check_response_presence(self, reports: ReportSet) -> List[Finding]:
        bad = [i for i, record in self._records(reports) if record.get("error") is None and _response_metrics(record) is None]
        return _aggregate(
            f"{self.name}.response_presence",
            Severity.WARNING,
            "are successful but carry no response metrics",
            bad,
        )

    def _check_token_times(self, reports: ReportSet) -> List[Finding]:
        unordered: List[int] = []
        outside: List[int] = []
        for i, record in self._records(reports):
            token_times = _token_times(record)
            if not token_times:
                continue
            if any(b < a for a, b in zip(token_times, token_times[1:], strict=False)):
                unordered.append(i)
            start, end = record.get("start_time"), record.get("end_time")
            if is_number(start) and is_number(end) and (token_times[0] < start or token_times[-1] > end):
                outside.append(i)
        return _aggregate(
            f"{self.name}.token_times", Severity.ERROR, "have out-of-order output token timestamps", unordered
        ) + _aggregate(
            f"{self.name}.token_times",
            Severity.ERROR,
            "have output token timestamps outside [start_time, end_time]",
            outside,
        )

    def _check_token_agreement(self, reports: ReportSet) -> List[Finding]:
        """Weak-form #630 invariant: client and server output-token counts agree."""
        bad: List[int] = []
        for i, record in self._records(reports):
            if record.get("error") is not None:
                continue
            response_metrics = _response_metrics(record)
            if not isinstance(response_metrics, dict):
                continue
            client_tokens = response_metrics.get("output_tokens")
            server_usage = response_metrics.get("server_usage")
            server_tokens = server_usage.get("completion_tokens") if isinstance(server_usage, dict) else None
            if not (is_number(client_tokens) and client_tokens > 0 and is_number(server_tokens) and server_tokens > 0):
                continue
            if abs(client_tokens - server_tokens) > _TOKEN_AGREEMENT_REL_TOL * max(client_tokens, server_tokens):
                bad.append(i)
        return _aggregate(
            f"{self.name}.token_agreement",
            Severity.WARNING,
            "have client output_tokens diverging from server completion_tokens by more than "
            f"{_TOKEN_AGREEMENT_REL_TOL:.0%} (reasoning-channel tokens can account for some divergence)",
            bad,
        )


def _response_metrics(record: Dict[str, Any]) -> Any:
    info = record.get("info")
    if not isinstance(info, dict):
        return None
    return info.get("response_metrics")


def _token_times(record: Dict[str, Any]) -> List[float]:
    response_metrics = _response_metrics(record)
    if not isinstance(response_metrics, dict):
        return []
    token_times = response_metrics.get("output_token_times")
    if not isinstance(token_times, list):
        return []
    return [t for t in token_times if is_number(t)]
