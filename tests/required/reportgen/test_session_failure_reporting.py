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
"""Unit tests for session-level failure reporting.

Covers the session half of the error reporting that #601 delivered for requests:
that a session's failure cause survives to the report, that failures bucket on a
stable cause code, and that a session the worker marked failed is never counted
as a success.
"""

from unittest.mock import Mock

from inference_perf.apis.base import (
    ErrorResponseInfo,
    InferenceInfo,
    RequestLifecycleMetric,
    SessionLifecycleMetric,
)
from inference_perf.config.reportgen.config import ReportConfig
from inference_perf.datagen.replay.replay_graph_session_datagen import SessionFailureCause
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.base import ReportGenerator

PERCENTILES = [50.0, 90.0]


def _make_generator() -> ReportGenerator:
    config = Mock()
    config.report = ReportConfig()
    return ReportGenerator(metrics_client=None, metrics_collector=Mock(), config=config)


def _sess(
    *,
    session_id: str = "s1",
    success: bool | None = None,
    error: ErrorResponseInfo | None = None,
    num_events: int = 3,
    num_events_completed: int = 3,
) -> SessionLifecycleMetric:
    return SessionLifecycleMetric(
        session_id=session_id,
        stage_id=0,
        file_path="trace.json",
        start_time=0.0,
        end_time=1.0,
        duration_sec=1.0,
        num_events=num_events,
        num_events_completed=num_events_completed,
        success=success,
        error=error,
    )


def _req(*, session_id: str, error: ErrorResponseInfo | None = None, input_tokens: int = 10) -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        stage_id=0,
        session_id=session_id,
        scheduled_time=0.0,
        start_time=0.0,
        end_time=1.0,
        request_data="prompt",
        info=InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=input_tokens))),
        error=error,
    )


def _replay_error(cause: SessionFailureCause, msg: str) -> ErrorResponseInfo:
    return ErrorResponseInfo(error_type=cause.value, error_msg=msg)


class TestSessionFailuresByLabel:
    def test_all_success_empty_by_label(self) -> None:
        gen = _make_generator()
        summary = gen.summarize_sessions([_sess(success=True), _sess(session_id="s2", success=True)], PERCENTILES)
        assert summary["num_sessions_failed"] == 0
        assert summary["failures"]["count"] == 0
        assert summary["failures"]["by_label"] == {}

    def test_distinct_prose_still_shares_one_cause_bucket(self) -> None:
        """The reason for bucketing on the cause code rather than the message.

        RECORDED_FALLBACK_MALFORMED embeds the offending predecessor event id in
        its prose, so bucketing on raw message text would yield one singleton
        bucket per failure, which is the flat list #587 was filed to eliminate.
        """
        gen = _make_generator()
        metrics = [
            _sess(
                session_id=f"s{i}",
                success=False,
                error=_replay_error(
                    SessionFailureCause.RECORDED_FALLBACK_MALFORMED,
                    f"recorded fallback for evt_{i} is also malformed",
                ),
            )
            for i in range(3)
        ]
        summary = gen.summarize_sessions(metrics, PERCENTILES)

        by_label = summary["failures"]["by_label"]
        assert list(by_label.keys()) == ["recorded_fallback_malformed"]
        bucket = by_label["recorded_fallback_malformed"]
        assert bucket["count"] == 3
        # Distinct prose is retained as samples under the single bucket.
        assert len(bucket["messages"]) == 3
        assert {sid for m in bucket["messages"] for sid in m["session_ids"]} == {"s0", "s1", "s2"}

    def test_cause_code_not_rewritten_by_request_side_regexes(self) -> None:
        """Session causes must bypass make_concise_label's pattern matching.

        Those regexes exist to parse server error text. `predecessor_wait_failed`
        carries "TimeoutError" in its prose and would be relabelled the generic
        "Timeout" bucket if it went through them, silently merging with any
        unrelated cause whose message also mentions a timeout.
        """
        gen = _make_generator()
        metrics = [
            _sess(
                session_id="s1",
                success=False,
                error=_replay_error(SessionFailureCause.PREDECESSOR_WAIT_FAILED, "predecessor wait failed: TimeoutError"),
            ),
            _sess(
                session_id="s2",
                success=False,
                error=_replay_error(SessionFailureCause.REQUEST_FAILED, "ServerTimeoutError: connection timed out"),
            ),
        ]
        summary = gen.summarize_sessions(metrics, PERCENTILES)

        by_label = summary["failures"]["by_label"]
        assert set(by_label.keys()) == {"predecessor_wait_failed", "request_failed"}
        assert "Timeout" not in by_label
        assert by_label["predecessor_wait_failed"]["count"] == 1
        assert by_label["request_failed"]["count"] == 1

    def test_labels_sorted_by_descending_count(self) -> None:
        gen = _make_generator()
        metrics = [
            _sess(session_id="a1", success=False, error=_replay_error(SessionFailureCause.PREDECESSOR_FAILED, "pred")),
            _sess(session_id="a2", success=False, error=_replay_error(SessionFailureCause.PREDECESSOR_FAILED, "pred")),
            _sess(session_id="b1", success=False, error=_replay_error(SessionFailureCause.REQUEST_FAILED, "boom")),
        ]
        summary = gen.summarize_sessions(metrics, PERCENTILES)
        assert list(summary["failures"]["by_label"].keys())[0] == "predecessor_failed"

    def test_failed_session_without_error_is_surfaced_not_dropped(self) -> None:
        """by_label totals must reconcile with num_sessions_failed."""
        gen = _make_generator()
        metrics = [
            _sess(session_id="s1", success=False, error=None),
            _sess(session_id="s2", success=False, error=_replay_error(SessionFailureCause.PREDECESSOR_FAILED, "pred")),
        ]
        summary = gen.summarize_sessions(metrics, PERCENTILES)

        by_label = summary["failures"]["by_label"]
        assert summary["failures"]["count"] == summary["num_sessions_failed"] == 2
        assert sum(b["count"] for b in by_label.values()) == 2
        assert "unreported" in by_label


class TestSessionErrorPrecedence:
    def test_session_replay_error_survives_request_error(self) -> None:
        """The session's own cause is the more specific diagnosis and must win.

        Previously any request-level error for the session overwrote it, which
        replaced a precise replay cause with an incidental HTTP error.
        """
        gen = _make_generator()
        session = _sess(
            session_id="s1",
            error=_replay_error(SessionFailureCause.SUBSTITUTION_TOOL_CALL_EXPECTED, "tool call expected"),
        )
        requests = [_req(session_id="s1", error=ErrorResponseInfo(error_type="HTTP Error 400", error_msg="bad request"))]

        gen._enrich_sessions([session], requests)

        assert session.error is not None
        assert session.error.error_type == SessionFailureCause.SUBSTITUTION_TOOL_CALL_EXPECTED.value
        assert session.success is False

    def test_request_error_fills_in_when_session_has_no_cause(self) -> None:
        gen = _make_generator()
        session = _sess(session_id="s1", error=None)
        requests = [_req(session_id="s1", error=ErrorResponseInfo(error_type="HTTP Error 500", error_msg="boom"))]

        gen._enrich_sessions([session], requests)

        assert session.error is not None
        assert session.error.error_type == "HTTP Error 500"
        assert session.success is False

    def test_clean_session_stays_successful(self) -> None:
        gen = _make_generator()
        session = _sess(session_id="s1", error=None)
        gen._enrich_sessions([session], [_req(session_id="s1")])

        assert session.error is None
        assert session.success is True


class TestFailedSessionNeverCountsAsSuccess:
    def test_failed_session_with_all_events_completed_is_counted_failed(self) -> None:
        """Regression for the miscount: `failed` never reached the success predicate.

        A session whose events all completed but which the worker marked failed
        carries an error built from its cause, so the predicate derived from
        `error is None` puts it in the failed bucket where it belongs.
        """
        gen = _make_generator()
        session = _sess(
            session_id="s1",
            num_events=3,
            num_events_completed=3,
            error=_replay_error(SessionFailureCause.UNKNOWN, "session marked failed without a recorded reason"),
        )

        gen._enrich_sessions([session], [_req(session_id="s1")])
        summary = gen.summarize_sessions([session], PERCENTILES)

        assert session.success is False
        assert summary["num_sessions_succeeded"] == 0
        assert summary["num_sessions_failed"] == 1
        assert summary["failures"]["by_label"]["unknown"]["count"] == 1
