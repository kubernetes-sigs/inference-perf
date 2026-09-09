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

"""Tests for Time to First User Token (TFUT) computation and reporting."""

import pytest

from inference_perf.apis.base import (
    InferenceInfo,
    RequestLifecycleMetric,
    ResponseMetrics,
    SessionLifecycleMetric,
    StreamedResponseMetrics,
)
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.base import ReportGenerator


def _make_session(
    session_id: str = "s1",
    start_time: float = 100.0,
    dispatch_perf_counter: float | None = 100.0,
    user_facing_event_ids: list[str] | None = None,
    num_structured_output_excluded: int | None = None,
) -> SessionLifecycleMetric:
    return SessionLifecycleMetric(
        session_id=session_id,
        stage_id=0,
        file_path="test.json",
        start_time=start_time,
        end_time=start_time + 10.0,
        duration_sec=10.0,
        num_events=3,
        num_events_completed=3,
        user_facing_event_ids=user_facing_event_ids,
        num_structured_output_excluded=num_structured_output_excluded,
        dispatch_perf_counter=dispatch_perf_counter,
    )


def _make_request(
    session_id: str = "s1",
    event_id: str = "e1",
    streaming: bool = True,
    output_token_times: list[float] | None = None,
) -> RequestLifecycleMetric:
    response_metrics: StreamedResponseMetrics | ResponseMetrics
    if streaming:
        response_metrics = StreamedResponseMetrics(
            output_tokens=5,
            output_token_times=output_token_times or [],
        )
    else:
        response_metrics = ResponseMetrics(output_tokens=5)
    return RequestLifecycleMetric(
        scheduled_time=0.0,
        start_time=100.0,
        end_time=110.0,
        request_data="r",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=10)),
            response_metrics=response_metrics,
            graph_event_id=event_id,
        ),
        error=None,
        session_id=session_id,
    )


# ---------------------------------------------------------------------------
# _compute_tfut unit tests
# ---------------------------------------------------------------------------


class TestComputeTfut:
    def test_happy_path_single_user_facing(self) -> None:
        sm = _make_session(start_time=100.0, user_facing_event_ids=["e1"])
        reqs = {"e1": _make_request(output_token_times=[100.5, 101.0, 101.5])}
        ReportGenerator._compute_tfut(sm, reqs)
        assert sm.tfut_sec == pytest.approx(0.5)
        assert sm.tfut_none_reason is None

    def test_min_across_multiple_user_facing(self) -> None:
        sm = _make_session(start_time=100.0, user_facing_event_ids=["e1", "e2"])
        reqs = {
            "e1": _make_request(event_id="e1", output_token_times=[101.0]),
            "e2": _make_request(event_id="e2", output_token_times=[100.3]),
        }
        ReportGenerator._compute_tfut(sm, reqs)
        assert sm.tfut_sec == pytest.approx(0.3)

    def test_no_user_facing_ids_none(self) -> None:
        sm = _make_session(user_facing_event_ids=None)
        ReportGenerator._compute_tfut(sm, {})
        assert sm.tfut_sec is None
        assert sm.tfut_none_reason == "no_user_facing"

    def test_empty_user_facing_ids(self) -> None:
        sm = _make_session(user_facing_event_ids=[])
        ReportGenerator._compute_tfut(sm, {})
        assert sm.tfut_sec is None
        assert sm.tfut_none_reason == "no_user_facing"

    def test_non_streaming_sets_reason(self) -> None:
        sm = _make_session(start_time=100.0, user_facing_event_ids=["e1"])
        reqs = {"e1": _make_request(event_id="e1", streaming=False)}
        ReportGenerator._compute_tfut(sm, reqs)
        assert sm.tfut_sec is None
        assert sm.tfut_none_reason == "non_streaming"

    def test_streaming_no_output_tokens(self) -> None:
        sm = _make_session(start_time=100.0, user_facing_event_ids=["e1"])
        reqs = {"e1": _make_request(event_id="e1", output_token_times=[])}
        ReportGenerator._compute_tfut(sm, reqs)
        assert sm.tfut_sec is None
        assert sm.tfut_none_reason == "no_output_tokens"

    def test_mixed_user_facing_valid_wins(self) -> None:
        """One non-streaming + one valid streaming user-facing event → TFUT from the valid one."""
        sm = _make_session(start_time=100.0, user_facing_event_ids=["e1", "e2"])
        reqs = {
            "e1": _make_request(event_id="e1", streaming=False),
            "e2": _make_request(event_id="e2", output_token_times=[100.8]),
        }
        ReportGenerator._compute_tfut(sm, reqs)
        assert sm.tfut_sec == pytest.approx(0.8)
        assert sm.tfut_none_reason is None

    def test_user_facing_id_not_in_requests(self) -> None:
        sm = _make_session(start_time=100.0, user_facing_event_ids=["e_missing"])
        ReportGenerator._compute_tfut(sm, {})
        assert sm.tfut_sec is None
        assert sm.tfut_none_reason == "no_user_facing"

    def test_no_dispatch_perf_counter(self) -> None:
        sm = _make_session(dispatch_perf_counter=None, user_facing_event_ids=["e1"])
        reqs = {"e1": _make_request(output_token_times=[100.5])}
        ReportGenerator._compute_tfut(sm, reqs)
        assert sm.tfut_sec is None
        assert sm.tfut_none_reason == "no_dispatch_anchor"


# ---------------------------------------------------------------------------
# _enrich_sessions integration (TFUT wiring)
# ---------------------------------------------------------------------------


class TestEnrichSessionsTfut:
    def test_enrich_populates_tfut(self) -> None:
        sm = _make_session(start_time=100.0, user_facing_event_ids=["e1"])
        req = _make_request(session_id="s1", event_id="e1", output_token_times=[100.2])
        ReportGenerator._enrich_sessions(None, [sm], [req])  # type: ignore[arg-type]
        assert sm.tfut_sec == pytest.approx(0.2)

    def test_enrich_no_user_facing_yields_none_reason(self) -> None:
        sm = _make_session(start_time=100.0, user_facing_event_ids=None)
        req = _make_request(session_id="s1", event_id="e1", output_token_times=[100.5])
        ReportGenerator._enrich_sessions(None, [sm], [req])  # type: ignore[arg-type]
        assert sm.tfut_sec is None
        assert sm.tfut_none_reason == "no_user_facing"


# ---------------------------------------------------------------------------
# summarize_sessions includes TFUT fields
# ---------------------------------------------------------------------------


class TestSummarizeSessionsTfut:
    def test_summary_includes_tfut_percentiles(self) -> None:
        sessions = [
            _make_session(session_id="s1", start_time=100.0, user_facing_event_ids=["e1"]),
            _make_session(session_id="s2", start_time=200.0, user_facing_event_ids=["e2"]),
        ]
        sessions[0].tfut_sec = 0.5
        sessions[1].tfut_sec = 1.0
        summary = ReportGenerator.summarize_sessions(None, sessions, [], [50, 99], max_error_messages=5)  # type: ignore[arg-type]
        assert summary["sessions_with_tfut"] == 2
        assert summary["tfut_sec"]["mean"] == pytest.approx(0.75)
        assert summary["tfut_none_reasons"] is None

    def test_summary_reports_none_reasons(self) -> None:
        sessions = [
            _make_session(session_id="s1", user_facing_event_ids=None),
            _make_session(session_id="s2", user_facing_event_ids=["e1"]),
        ]
        sessions[0].tfut_none_reason = "no_user_facing"
        sessions[1].tfut_none_reason = "non_streaming"
        summary = ReportGenerator.summarize_sessions(None, sessions, [], [50], max_error_messages=5)  # type: ignore[arg-type]
        assert summary["sessions_with_tfut"] == 0
        assert summary["tfut_none_reasons"] == {"no_user_facing": 1, "non_streaming": 1}

    def test_summary_includes_user_facing_and_exclusion_counts(self) -> None:
        sessions = [
            _make_session(session_id="s1", user_facing_event_ids=["e1", "e2"], num_structured_output_excluded=1),
            _make_session(session_id="s2", user_facing_event_ids=["e3"], num_structured_output_excluded=0),
        ]
        summary = ReportGenerator.summarize_sessions(None, sessions, [], [50], max_error_messages=5)  # type: ignore[arg-type]
        assert summary["num_user_facing_events"]["mean"] == pytest.approx(1.5)
        assert summary["num_structured_output_excluded"]["mean"] == pytest.approx(0.5)
