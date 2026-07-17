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
"""Unit tests for the error detail in the lifecycle reports."""

from unittest.mock import Mock

from inference_perf.apis.base import (
    ErrorResponseInfo,
    RequestLifecycleMetric,
    SessionLifecycleMetric,
)
from inference_perf.config.reportgen.config import ReportConfig
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.apis.base import InferenceInfo
from inference_perf.reportgen.base import ReportGenerator, summarize_requests


PERCENTILES = [50.0, 90.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_generator() -> ReportGenerator:
    """Minimal ReportGenerator with a real ReportConfig (no live clients)."""
    config = Mock()
    config.report = ReportConfig()
    return ReportGenerator(
        metrics_client=None,
        metrics_collector=Mock(),
        config=config,
    )


def _req(
    *, error: ErrorResponseInfo | None = None, session_id: str | None = None, stage_id: int = 0
) -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        stage_id=stage_id,
        session_id=session_id,
        scheduled_time=0.0,
        start_time=0.0,
        end_time=1.0,
        request_data="prompt",
        info=InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=10))),
        error=error,
    )


def _sess(
    *,
    session_id: str = "s1",
    success: bool | None = True,
    error: ErrorResponseInfo | None = None,
    n_recorded_substitutions: int | None = None,
    recorded_substitution_event_ids: list[str] | None = None,
) -> SessionLifecycleMetric:
    return SessionLifecycleMetric(
        session_id=session_id,
        stage_id=0,
        file_path="trace.json",
        start_time=0.0,
        end_time=1.0,
        duration_sec=1.0,
        num_events=3,
        num_events_completed=3,
        success=success,
        error=error,
        n_recorded_substitutions=n_recorded_substitutions,
        recorded_substitution_event_ids=recorded_substitution_event_ids,
    )


# ---------------------------------------------------------------------------
# Request-lifecycle failures.by_label
# ---------------------------------------------------------------------------


class TestRequestFailuresByLabel:
    def test_all_success_empty_by_label(self) -> None:
        summary = summarize_requests([_req(), _req()], PERCENTILES)
        assert summary.successes["count"] == 2
        assert summary.failures["count"] == 0
        assert summary.failures["by_label"] == {}

    def test_mixed_errors_bucketed(self) -> None:
        err = ErrorResponseInfo(error_type="HTTP Error 429", error_msg="rate limit exceeded")
        summary = summarize_requests([_req(), _req(error=err), _req(error=err)], PERCENTILES)
        assert summary.successes["count"] == 1
        assert summary.failures["count"] == 2
        bucket = summary.failures["by_label"]["429 - Rate Limit"]
        assert bucket["count"] == 2
        # The two failures share an identical message, so they merge into one entry.
        assert len(bucket["messages"]) == 1
        assert bucket["messages"][0]["message"] == "rate limit exceeded"

    def test_labels_sorted_by_descending_count(self) -> None:
        rate = ErrorResponseInfo(error_type="HTTP Error 429", error_msg="rate limit")
        server = ErrorResponseInfo(error_type="HTTP Error 500", error_msg="Internal Server Error")
        # 3 x 500, 1 x 429
        metrics = [_req(error=server), _req(error=server), _req(error=server), _req(error=rate)]
        summary = summarize_requests(metrics, PERCENTILES)
        labels = list(summary.failures["by_label"].keys())
        assert labels[0] == "500 - Internal Server Error"

    def test_messages_capped_by_max_error_messages(self) -> None:
        # Distinct messages so the cap on distinct messages is what limits the list.
        metrics = [
            _req(error=ErrorResponseInfo(error_type="HTTP Error 500", error_msg=f"Internal Server Error {i}"))
            for i in range(5)
        ]
        summary = summarize_requests(metrics, PERCENTILES, max_error_messages=2)
        bucket = list(summary.failures["by_label"].values())[0]
        assert bucket["count"] == 5
        assert len(bucket["messages"]) == 2  # capped on distinct messages

    def test_session_id_attached_to_messages(self) -> None:
        err = ErrorResponseInfo(error_type="HTTP Error 500", error_msg="boom")
        summary = summarize_requests([_req(error=err, session_id="s7")], PERCENTILES)
        bucket = list(summary.failures["by_label"].values())[0]
        assert bucket["messages"][0]["session_ids"] == ["s7"]

    def test_identical_messages_merged_with_session_ids(self) -> None:
        err = ErrorResponseInfo(error_type="HTTP Error 500", error_msg="boom")
        summary = summarize_requests(
            [_req(error=err, session_id="s1"), _req(error=err, session_id="s2"), _req(error=err, session_id="s3")],
            PERCENTILES,
        )
        bucket = list(summary.failures["by_label"].values())[0]
        assert bucket["count"] == 3
        assert len(bucket["messages"]) == 1
        assert bucket["messages"][0]["message"] == "boom"
        assert bucket["messages"][0]["session_ids"] == ["s1", "s2", "s3"]


# ---------------------------------------------------------------------------
# Session-lifecycle total_recorded_substitutions
# ---------------------------------------------------------------------------


class TestSessionSubstitutions:
    def test_substitutions_count_and_messages(self) -> None:
        gen = _make_generator()
        sessions = [
            _sess(session_id="s1", n_recorded_substitutions=2, recorded_substitution_event_ids=["e1", "e2"]),
            _sess(session_id="s2", n_recorded_substitutions=3, recorded_substitution_event_ids=["e3", "e4", "e5"]),
            _sess(session_id="s3", n_recorded_substitutions=0),
        ]
        summary = gen.summarize_sessions(sessions, PERCENTILES)
        subs = summary["total_recorded_substitutions"]
        assert subs["count"] == 5  # 2 + 3
        assert summary["sessions_with_recorded_substitution"] == 2  # s1, s2
        ids = {sid for m in subs["messages"] for sid in m["session_ids"]}
        assert ids == {"s1", "s2"}

    def test_no_substitutions_zero_count_no_messages(self) -> None:
        gen = _make_generator()
        summary = gen.summarize_sessions([_sess(session_id="s1"), _sess(session_id="s2")], PERCENTILES)
        subs = summary["total_recorded_substitutions"]
        assert subs["count"] == 0
        assert subs["messages"] == []

    def test_substitution_messages_capped(self) -> None:
        gen = _make_generator()
        sessions = [
            _sess(session_id=f"s{i}", n_recorded_substitutions=1, recorded_substitution_event_ids=[f"e{i}"]) for i in range(5)
        ]
        summary = gen.summarize_sessions(sessions, PERCENTILES, max_error_messages=2)
        subs = summary["total_recorded_substitutions"]
        assert subs["count"] == 5
        assert len(subs["messages"]) == 2  # capped
