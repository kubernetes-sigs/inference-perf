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
"""Tests for the session lifecycle summary validator."""

from __future__ import annotations

from typing import List, Optional

from inference_perf.apis.base import SessionLifecycleMetric
from inference_perf.reportgen.base import ReportGenerator
from inference_perf.reportgen.session.validation import SessionLifecycleValidator
from inference_perf.reportgen.validation import Finding, Severity, ValidationReport, run_validators
from inference_perf.utils import ReportFile

SESSION_FILE = "summary_session_lifecycle_metrics.json"


def _session(session_id: str, cached: Optional[int], cacheable: Optional[int]) -> SessionLifecycleMetric:
    return SessionLifecycleMetric(
        session_id=session_id,
        stage_id=0,
        file_path=f"{session_id}.json",
        start_time=0.0,
        end_time=1.0,
        duration_sec=1.0,
        num_events=1,
        num_events_completed=1,
        total_cached_tokens=cached,
        total_cacheable_input_tokens=cacheable,
        total_input_tokens=cacheable or 0,
        total_output_tokens=0,
        success=True,
    )


def _session_report(sessions: List[SessionLifecycleMetric]) -> ReportFile:
    """The session summary through the real generator, not a hand-made dict."""
    contents = ReportGenerator.summarize_sessions(None, sessions, [], [50])  # type: ignore[arg-type]
    return ReportFile(name="summary_session_lifecycle_metrics", contents=contents)


def _validate(reports: List[ReportFile]) -> ValidationReport:
    return run_validators([SessionLifecycleValidator()], reports)


def _warnings_for_check(result: ValidationReport, check: str) -> List[Finding]:
    return [f for f in result.all_warnings() if f.check == check]


def test_absent_session_summary_is_skipped_silently() -> None:
    result = _validate([])

    assert SESSION_FILE not in result.reports
    assert not result.all_errors() and not result.all_warnings()


def test_all_zero_hit_rate_with_cache_info_is_a_warning() -> None:
    """Explicit-zero readings are suspicious but not wrong (see #818)."""
    result = _validate([_session_report([_session("s1", 0, 100), _session("s2", 0, 200)])])

    assert not result.all_errors()
    warnings = _warnings_for_check(result, "session.cache")
    assert len(warnings) == 1
    assert warnings[0].severity == Severity.WARNING
    assert warnings[0].report == SESSION_FILE


def test_nonzero_hit_rate_is_clean() -> None:
    result = _validate([_session_report([_session("s1", 0, 100), _session("s2", 50, 100)])])

    assert not result.all_errors() and not result.all_warnings()
    assert result.reports[SESSION_FILE].is_clean()


def test_none_hit_rate_without_cache_info_is_clean() -> None:
    """No cache info at all already reports None: nothing suspicious."""
    result = _validate([_session_report([_session("s1", None, None)])])

    assert not result.all_errors() and not result.all_warnings()


def test_structurally_broken_session_summary_halts_with_a_single_error() -> None:
    result = _validate([ReportFile(name="summary_session_lifecycle_metrics", contents=["not", "a", "summary"])])

    errors = result.reports[SESSION_FILE].errors
    assert len(errors) == 1
    assert errors[0].check == "session.structure"
