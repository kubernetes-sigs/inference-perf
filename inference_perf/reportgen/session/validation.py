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
"""Validation of the session lifecycle summary report.

Covers ``summary_session_lifecycle_metrics.json``: the KV/prefix cache hit
rate read from server-reported usage. An all-zero reading is ambiguous — a
genuinely cold cache, or usage counters normalized to 0 by a gateway or
sidecar in front of the engine (see #818) — so it is a warning ("suspicious
but not necessarily wrong"), never an error.
"""

from __future__ import annotations

from typing import Any, Dict, List, Sequence

from inference_perf.reportgen.validation import (
    SESSION_SUMMARY_FILENAME,
    Check,
    Finding,
    ReportSet,
    ReportSetValidator,
    Severity,
    StopValidation,
    is_number,
)


def _warning(check: str, filename: str, message: str) -> Finding:
    return Finding(check=check, severity=Severity.WARNING, message=message, report=filename)


def cache_findings(filename: str, contents: Any, check: str) -> List[Finding]:
    """An all-zero KV cache hit rate with reporting sessions is suspicious.

    Sessions that reported cache info all showing zero cached tokens can mean
    a genuinely cold cache, or usage counters normalized by a gateway or
    sidecar (e.g. llm-d's prefill/decode sidecar writing ``cached_tokens: 0``
    when the prefiller reported nothing). The reported 0% is kept as sent;
    this finding only flags the ambiguity.
    """
    hit = contents.get("kv_cache_hit_percent")
    info = contents.get("sessions_with_cache_info")
    if is_number(hit) and hit == 0 and is_number(info) and info > 0:
        return [
            _warning(
                check,
                filename,
                f"kv_cache_hit_percent is 0 with {info:g} session(s) reporting cache info: genuinely cold "
                "cache, or usage counters normalized by a gateway or sidecar in front of the engine "
                "(see #818). Verify against server-side cache metrics before treating 0% as fact.",
            )
        ]
    return []


class SessionLifecycleValidator(ReportSetValidator):
    name = "session"

    def covers(self, reports: ReportSet) -> List[str]:
        return [SESSION_SUMMARY_FILENAME] if SESSION_SUMMARY_FILENAME in reports.filenames() else []

    def checks(self) -> Sequence[Check]:
        return [
            self._check_structure,
            self._check_cache,
        ]

    def _contents(self, reports: ReportSet) -> Dict[str, Any]:
        contents = reports.contents(SESSION_SUMMARY_FILENAME)
        assert isinstance(contents, dict)  # guaranteed by _check_structure running first
        return contents

    def _check_structure(self, reports: ReportSet) -> List[Finding]:
        """Halts the validator when the session summary is absent (silently:
        session reports only run for session-based workloads) or structurally
        broken (fatally: the cache check would only cascade)."""
        contents = reports.contents(SESSION_SUMMARY_FILENAME)
        if contents is None:
            raise StopValidation()
        if not isinstance(contents, dict):
            raise StopValidation(
                [
                    Finding(
                        check=f"{self.name}.structure",
                        severity=Severity.ERROR,
                        message=f"expected a JSON object, got {type(contents).__name__}",
                        report=SESSION_SUMMARY_FILENAME,
                    )
                ]
            )
        return []

    def _check_cache(self, reports: ReportSet) -> List[Finding]:
        return cache_findings(SESSION_SUMMARY_FILENAME, self._contents(reports), f"{self.name}.cache")


__all__ = [
    "SessionLifecycleValidator",
    "cache_findings",
]
