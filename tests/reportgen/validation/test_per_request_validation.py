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
"""Tests for the per-request lifecycle validator."""

from __future__ import annotations

from typing import Any, List

import pytest

from inference_perf.reportgen.per_request.validation import PerRequestLifecycleValidator
from inference_perf.reportgen.validation import Finding, ValidationReport, run_validators
from inference_perf.utils import ReportFile

from .helpers import make_report_set, tampered

PER_REQUEST_FILE = "per_request_lifecycle_metrics.json"


def _validate(reports: List[ReportFile]) -> ValidationReport:
    return run_validators([PerRequestLifecycleValidator()], reports)


def _findings_for_check(result: ValidationReport, check: str) -> List[Finding]:
    return [f for f in result.all_errors() + result.all_warnings() if f.check == check]


def test_real_generator_output_is_clean() -> None:
    result = _validate(make_report_set())

    assert not result.all_errors(), [f.message for f in result.all_errors()]
    assert not result.all_warnings(), [f.message for f in result.all_warnings()]
    assert result.reports[PER_REQUEST_FILE].is_clean()


def test_end_time_before_start_time_is_an_error() -> None:
    reports, records = tampered(make_report_set(), PER_REQUEST_FILE)
    records[0]["end_time"] = records[0]["start_time"] - 1.0

    result = _validate(reports)
    errors = _findings_for_check(result, "per_request.timestamps")
    assert errors and "1 record(s)" in errors[0].message


def test_token_time_outside_request_lifetime_is_an_error() -> None:
    reports, records = tampered(make_report_set(), PER_REQUEST_FILE)
    records[0]["info"]["response_metrics"]["output_token_times"].append(records[0]["end_time"] + 5.0)

    result = _validate(reports)
    assert _findings_for_check(result, "per_request.token_times")


def test_out_of_order_token_times_is_an_error() -> None:
    reports, records = tampered(make_report_set(), PER_REQUEST_FILE)
    token_times = records[0]["info"]["response_metrics"]["output_token_times"]
    token_times[0], token_times[-1] = token_times[-1], token_times[0]

    result = _validate(reports)
    assert any("out-of-order" in f.message for f in _findings_for_check(result, "per_request.token_times"))


def test_success_without_response_metrics_is_a_warning() -> None:
    reports, records = tampered(make_report_set(), PER_REQUEST_FILE)
    records[0]["info"]["response_metrics"] = None

    result = _validate(reports)
    warnings = [f for f in result.all_warnings() if f.check == "per_request.response_presence"]
    assert warnings


def test_client_server_token_divergence_is_a_warning() -> None:
    reports, records = tampered(make_report_set(), PER_REQUEST_FILE)
    records[0]["info"]["response_metrics"]["server_usage"]["completion_tokens"] = (
        records[0]["info"]["response_metrics"]["output_tokens"] * 2
    )

    result = _validate(reports)
    warnings = [f for f in result.all_warnings() if f.check == "per_request.token_agreement"]
    assert warnings and not result.all_errors()


def test_client_server_token_divergence_is_a_warning_with_anthropic_spelling() -> None:
    reports, records = tampered(make_report_set(), PER_REQUEST_FILE)
    response_metrics = records[0]["info"]["response_metrics"]
    response_metrics["server_usage"] = {"output_tokens": response_metrics["output_tokens"] * 2}

    result = _validate(reports)
    warnings = [f for f in result.all_warnings() if f.check == "per_request.token_agreement"]
    assert warnings and not result.all_errors()


def test_systematic_violations_aggregate_into_one_finding() -> None:
    reports, records = tampered(make_report_set(), PER_REQUEST_FILE)
    many: List[Any] = []
    for _ in range(20):
        record = dict(records[0])
        record["end_time"] = record["start_time"] - 1.0
        many.append(record)
    records.extend(many)

    result = _validate(reports)
    errors = _findings_for_check(result, "per_request.timestamps")
    assert len(errors) == 1
    assert "20 record(s)" in errors[0].message
    assert "…" in errors[0].message  # sampled, not exhaustive


def test_absent_per_request_report_is_skipped_silently() -> None:
    reports = [r for r in make_report_set() if r.get_filename() != PER_REQUEST_FILE]
    result = _validate(reports)

    assert PER_REQUEST_FILE not in result.reports
    assert not result.all_errors() and not result.all_warnings()


@pytest.mark.parametrize("usage_key", ["completion_tokens", "output_tokens"])
@pytest.mark.parametrize("tool", [True, False])
@pytest.mark.parametrize(
    "client,server,tool_warning", [(2, 16, False), (2, 18, False), (2, 19, True), (100, 75, False), (100, 74, True)]
)
def test_tool_token_agreement_uses_summary_floor(
    usage_key: str, tool: bool, client: int, server: int, tool_warning: bool
) -> None:
    reports, records = tampered(make_report_set(), PER_REQUEST_FILE)
    metrics = records[0]["info"]["response_metrics"]
    metrics.update(output_tokens=client, server_usage={usage_key: server}, has_tool_calls=tool)
    warnings = _findings_for_check(_validate(reports), "per_request.token_agreement")
    expected = tool_warning if tool else abs(client - server) > 0.25 * max(client, server)
    assert bool(warnings) is expected
    if tool and warnings:
        assert "16 tokens" in warnings[0].message
