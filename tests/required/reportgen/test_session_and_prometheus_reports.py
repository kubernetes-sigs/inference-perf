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
"""Unit tests for the two report surfaces `generate_reports` only builds behind a guard.

Session rollups run only when a session metrics collector is configured, and the
Prometheus section runs only when `report.prometheus` is set, so neither is reached
by a default request-only run. Both compute numbers that are published rather than
acted on: a bug here does not fail the run, it ships wrong results.

`summarize_sessions` failure accounting and substitution capping are covered in
`test_failure_reporting.py`; what follows is the rest of that function plus
`generate_session_reports` and `generate_prometheus_metrics_report`.
"""

from typing import Any, Dict, List, Optional
from unittest.mock import Mock, patch

import pytest

from inference_perf.apis.base import SessionLifecycleMetric
from inference_perf.client.modelserver.metrics import BaseMetrics
from inference_perf.client.modelserver.metrics.counter.base import CounterResult
from inference_perf.client.modelserver.metrics.gauge.base import GaugeResult
from inference_perf.client.modelserver.metrics.histogram.base import HistogramResult
from inference_perf.client.server_metrics.base import (
    ModelServerMetrics,
    PerfRuntimeParameters,
    StageRuntimeInfo,
    StageStatus,
)
from inference_perf.client.server_metrics.prometheus_client.base import PrometheusMetricsClient
from inference_perf.config.client.server_metrics.config import PrometheusClientConfig
from inference_perf.config.reportgen.config import (
    PrometheusMetricsReportConfig,
    ReportConfig,
    SessionLifecycleReportConfig,
)
from inference_perf.reportgen.base import ReportGenerator

PERCENTILES = [50.0, 90.0]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_generator(metrics_client: Optional[Any] = None) -> ReportGenerator:
    """Minimal ReportGenerator with a real ReportConfig (no live clients)."""
    config = Mock()
    config.report = ReportConfig()
    return ReportGenerator(
        metrics_client=metrics_client,
        metrics_collector=Mock(),
        config=config,
    )


def _sess(
    *,
    session_id: str = "s1",
    stage_id: int = 0,
    start_time: float = 0.0,
    end_time: float = 10.0,
    duration_sec: Optional[float] = None,
    num_events: int = 4,
    num_events_completed: int = 4,
    num_events_cancelled: Optional[int] = None,
    success: Optional[bool] = True,
    total_input_tokens: Optional[int] = None,
    total_output_tokens: Optional[int] = None,
) -> SessionLifecycleMetric:
    return SessionLifecycleMetric(
        session_id=session_id,
        stage_id=stage_id,
        file_path=f"{session_id}.json",
        start_time=start_time,
        end_time=end_time,
        duration_sec=end_time - start_time if duration_sec is None else duration_sec,
        num_events=num_events,
        num_events_completed=num_events_completed,
        num_events_cancelled=num_events_cancelled,
        success=success,
        total_input_tokens=total_input_tokens,
        total_output_tokens=total_output_tokens,
    )


def _runtime(stages: Dict[int, StageRuntimeInfo]) -> PerfRuntimeParameters:
    return PerfRuntimeParameters(
        start_time=0.0,
        duration=10.0,
        model_server_metrics=BaseMetrics(),
        stages=stages,
    )


def _stage_info(
    stage_id: int,
    *,
    status: StageStatus = StageStatus.COMPLETED,
    start_time: float = 0.0,
    end_time: float = 10.0,
    rate: float = 2.0,
    timeout: Optional[float] = None,
    concurrency_level: Optional[int] = None,
) -> StageRuntimeInfo:
    return StageRuntimeInfo(
        stage_id=stage_id,
        rate=rate,
        start_time=start_time,
        end_time=end_time,
        status=status,
        timeout=timeout,
        concurrency_level=concurrency_level,
    )


def _report_names(reports: List[Any]) -> List[str]:
    return [r.name for r in reports]


# ---------------------------------------------------------------------------
# summarize_sessions: throughput and distribution rollups
# ---------------------------------------------------------------------------


class TestSummarizeSessionsRollups:
    def test_sessions_per_second_spans_first_start_to_last_end(self) -> None:
        """Throughput is sessions over the wall-clock span of the whole set, not over
        the sum or the mean of the individual session durations."""
        gen = _make_generator()
        sessions = [
            _sess(session_id="s1", start_time=0.0, end_time=10.0),
            _sess(session_id="s2", start_time=5.0, end_time=20.0),
        ]

        summary = gen.summarize_sessions(sessions, PERCENTILES)

        # span = 20.0 - 0.0, so 2 sessions / 20s.
        assert summary["sessions_per_second"] == pytest.approx(0.1)

    def test_sessions_per_second_is_zero_for_a_zero_length_span(self) -> None:
        """Sessions that all start and end at the same instant must report 0.0 rather
        than raise on the division."""
        gen = _make_generator()
        sessions = [
            _sess(session_id="s1", start_time=3.0, end_time=3.0, duration_sec=0.0),
            _sess(session_id="s2", start_time=3.0, end_time=3.0, duration_sec=0.0),
        ]

        summary = gen.summarize_sessions(sessions, PERCENTILES)

        assert summary["sessions_per_second"] == 0.0

    def test_event_totals_sum_across_sessions(self) -> None:
        gen = _make_generator()
        sessions = [
            _sess(session_id="s1", num_events=4, num_events_completed=4, num_events_cancelled=0),
            _sess(session_id="s2", num_events=6, num_events_completed=5, num_events_cancelled=1),
        ]

        summary = gen.summarize_sessions(sessions, PERCENTILES)

        assert summary["num_sessions"] == 2
        assert summary["total_events"] == 10
        assert summary["total_events_completed"] == 9
        assert summary["total_events_cancelled"] == 1

    def test_duration_and_event_distributions_are_computed_over_all_sessions(self) -> None:
        gen = _make_generator()
        sessions = [
            _sess(session_id="s1", start_time=0.0, end_time=10.0, num_events=4),
            _sess(session_id="s2", start_time=0.0, end_time=20.0, num_events=6),
        ]

        summary = gen.summarize_sessions(sessions, PERCENTILES)

        assert summary["session_duration_sec"]["mean"] == pytest.approx(15.0)
        assert summary["session_duration_sec"]["min"] == pytest.approx(10.0)
        assert summary["session_duration_sec"]["max"] == pytest.approx(20.0)
        assert summary["num_events"]["mean"] == pytest.approx(5.0)
        assert summary["num_events"]["min"] == pytest.approx(4.0)
        assert summary["num_events"]["max"] == pytest.approx(6.0)

    def test_unset_optional_fields_are_skipped_not_counted_as_zero(self) -> None:
        """A session that never recorded cancellations or token totals contributes
        nothing to those rollups. Treating the missing value as 0 would silently
        drag the reported mean down."""
        gen = _make_generator()
        sessions = [
            _sess(session_id="s1", total_input_tokens=100, total_output_tokens=10, num_events_cancelled=2),
            _sess(session_id="s2", total_input_tokens=None, total_output_tokens=None, num_events_cancelled=None),
        ]

        summary = gen.summarize_sessions(sessions, PERCENTILES)

        assert summary["total_input_tokens"]["mean"] == pytest.approx(100.0)
        assert summary["total_output_tokens"]["mean"] == pytest.approx(10.0)
        assert summary["num_events_cancelled"]["mean"] == pytest.approx(2.0)

    def test_rollups_are_null_when_no_session_reported_the_field(self) -> None:
        """With the field unset everywhere the rollup is null, which is distinguishable
        in the report from a real measurement of zero."""
        gen = _make_generator()
        sessions = [_sess(session_id="s1"), _sess(session_id="s2")]

        summary = gen.summarize_sessions(sessions, PERCENTILES)

        assert summary["num_events_cancelled"] is None
        assert summary["total_input_tokens"] is None
        assert summary["total_output_tokens"] is None
        assert summary["total_events_cancelled"] == 0


# ---------------------------------------------------------------------------
# generate_session_reports: which files get emitted
# ---------------------------------------------------------------------------


class TestGenerateSessionReports:
    def test_no_sessions_emits_no_files(self) -> None:
        gen = _make_generator()

        reports = gen.generate_session_reports([], SessionLifecycleReportConfig(), PERCENTILES, _runtime({}), 100)

        assert reports == []

    def test_each_toggle_controls_its_own_file(self) -> None:
        gen = _make_generator()
        sessions = [_sess(session_id="s1", stage_id=0)]
        runtime = _runtime({0: _stage_info(0)})

        summary_only = gen.generate_session_reports(
            sessions, SessionLifecycleReportConfig(summary=True, per_stage=False, per_session=False), PERCENTILES, runtime, 100
        )
        per_session_only = gen.generate_session_reports(
            sessions, SessionLifecycleReportConfig(summary=False, per_stage=False, per_session=True), PERCENTILES, runtime, 100
        )
        all_on = gen.generate_session_reports(
            sessions, SessionLifecycleReportConfig(summary=True, per_stage=True, per_session=True), PERCENTILES, runtime, 100
        )

        assert _report_names(summary_only) == ["summary_session_lifecycle_metrics"]
        assert _report_names(per_session_only) == ["per_session_lifecycle_metrics"]
        assert _report_names(all_on) == [
            "summary_session_lifecycle_metrics",
            "stage_0_session_lifecycle_metrics",
            "per_session_lifecycle_metrics",
        ]

    def test_per_stage_buckets_sessions_by_stage_id(self) -> None:
        gen = _make_generator()
        sessions = [
            _sess(session_id="s1", stage_id=0),
            _sess(session_id="s2", stage_id=1),
            _sess(session_id="s3", stage_id=1),
        ]
        runtime = _runtime({0: _stage_info(0), 1: _stage_info(1)})

        reports = gen.generate_session_reports(
            sessions, SessionLifecycleReportConfig(summary=False, per_stage=True, per_session=False), PERCENTILES, runtime, 100
        )

        by_name = {r.name: r.contents for r in reports}
        assert by_name["stage_0_session_lifecycle_metrics"]["num_sessions"] == 1
        assert by_name["stage_1_session_lifecycle_metrics"]["num_sessions"] == 2

    def test_stage_metadata_leads_the_report_and_carries_the_run_shape(self) -> None:
        """`stage_metadata` is prepended so the report reads as configuration first,
        measurements second."""
        gen = _make_generator()
        runtime = _runtime({0: _stage_info(0, start_time=1.0, end_time=7.0, rate=2.0, timeout=30.0, concurrency_level=4)})

        reports = gen.generate_session_reports(
            [_sess(stage_id=0)],
            SessionLifecycleReportConfig(summary=False, per_stage=True, per_session=False),
            PERCENTILES,
            runtime,
            100,
        )

        contents = reports[0].contents
        assert next(iter(contents)) == "stage_metadata"
        assert contents["stage_metadata"] == {
            "stage_id": 0,
            "status": "COMPLETED",
            "timeout_configured": 30.0,
            "actual_duration": 6.0,
            "concurrent_sessions": 4,
            "session_rate": 2.0,
        }

    def test_a_stage_with_no_runtime_info_reports_measurements_without_metadata(self) -> None:
        gen = _make_generator()

        reports = gen.generate_session_reports(
            [_sess(stage_id=7)],
            SessionLifecycleReportConfig(summary=False, per_stage=True, per_session=False),
            PERCENTILES,
            _runtime({}),
            100,
        )

        assert _report_names(reports) == ["stage_7_session_lifecycle_metrics"]
        assert "stage_metadata" not in reports[0].contents
        assert reports[0].contents["num_sessions"] == 1

    def test_a_concurrency_driven_stage_reports_no_session_rate(self) -> None:
        """rate is 0 for a concurrency-driven stage; the report must say `null` rather
        than claim a measured rate of zero sessions per second."""
        gen = _make_generator()
        runtime = _runtime({0: _stage_info(0, rate=0.0, concurrency_level=8)})

        reports = gen.generate_session_reports(
            [_sess(stage_id=0)],
            SessionLifecycleReportConfig(summary=False, per_stage=True, per_session=False),
            PERCENTILES,
            runtime,
            100,
        )

        assert reports[0].contents["stage_metadata"]["session_rate"] is None
        assert reports[0].contents["stage_metadata"]["concurrent_sessions"] == 8

    @pytest.mark.parametrize(
        ("status", "start_time", "end_time", "timeout", "expected"),
        [
            (StageStatus.COMPLETED, 0.0, 10.0, 30.0, "COMPLETED"),
            # A failed stage that ran at least as long as its timeout is reported as a
            # timeout; one that failed early keeps the generic label.
            (StageStatus.FAILED, 0.0, 30.0, 30.0, "TIMED_OUT"),
            (StageStatus.FAILED, 0.0, 31.0, 30.0, "TIMED_OUT"),
            (StageStatus.FAILED, 0.0, 12.0, 30.0, "FAILED"),
            # No timeout configured: nothing to have timed out against.
            (StageStatus.FAILED, 0.0, 90.0, None, "FAILED"),
            # Neither terminal state should be published as a success.
            (StageStatus.RUNNING, 0.0, 10.0, 30.0, "FAILED"),
            (StageStatus.SKIPPED, 0.0, 10.0, 30.0, "FAILED"),
        ],
    )
    def test_stage_status_label(
        self, status: StageStatus, start_time: float, end_time: float, timeout: Optional[float], expected: str
    ) -> None:
        """The status string is derived, not stored, so it is the piece most able to
        mislabel a stage without anything else going wrong."""
        gen = _make_generator()
        runtime = _runtime({0: _stage_info(0, status=status, start_time=start_time, end_time=end_time, timeout=timeout)})

        reports = gen.generate_session_reports(
            [_sess(stage_id=0)],
            SessionLifecycleReportConfig(summary=False, per_stage=True, per_session=False),
            PERCENTILES,
            runtime,
            100,
        )

        assert reports[0].contents["stage_metadata"]["status"] == expected

    def test_per_session_report_carries_one_record_per_session(self) -> None:
        gen = _make_generator()
        sessions = [_sess(session_id="s1"), _sess(session_id="s2")]

        reports = gen.generate_session_reports(
            sessions,
            SessionLifecycleReportConfig(summary=False, per_stage=False, per_session=True),
            PERCENTILES,
            _runtime({}),
            100,
        )

        records = reports[0].contents
        assert [r["session_id"] for r in records] == ["s1", "s2"]


# ---------------------------------------------------------------------------
# generate_prometheus_metrics_report
# ---------------------------------------------------------------------------


def _prometheus_client() -> PrometheusMetricsClient:
    return PrometheusMetricsClient(PrometheusClientConfig(url="http://localhost:9090"))


def _server_metrics() -> ModelServerMetrics:
    return ModelServerMetrics(
        requests=CounterResult(total=12.0, per_second=1.2),
        prompt_tokens=CounterResult(avg=100.0, per_second=120.0),
        output_tokens=CounterResult(avg=8.0, per_second=9.6),
        queue_length=GaugeResult(avg=3.0),
        request_latency=HistogramResult(avg=0.5, median=0.4, p90=0.9, p99=1.1),
        prefix_cache_hits=CounterResult(total=25.0),
        prefix_cache_queries=CounterResult(total=100.0),
    )


class TestGeneratePrometheusMetricsReport:
    def test_no_metrics_client_emits_nothing(self, caplog: pytest.LogCaptureFixture) -> None:
        gen = _make_generator(metrics_client=None)

        reports = gen.generate_prometheus_metrics_report(_runtime({}), PrometheusMetricsReportConfig())

        assert reports == []
        assert "Prometheus Metrics Client is not configured" in caplog.text

    def test_a_non_prometheus_metrics_client_emits_nothing(self) -> None:
        """The section is Prometheus-specific; another client must be skipped rather
        than duck-typed into the Prometheus query path."""
        gen = _make_generator(metrics_client=Mock())

        reports = gen.generate_prometheus_metrics_report(_runtime({}), PrometheusMetricsReportConfig())

        assert reports == []

    def test_summary_report_waits_for_scrape_then_maps_the_collected_metrics(self) -> None:
        """`wait` covers the final scrape interval; without it the last stage's metrics
        are missing from the summary the report publishes."""
        client = _prometheus_client()
        gen = _make_generator(metrics_client=client)

        with (
            patch.object(PrometheusMetricsClient, "wait") as wait,
            patch.object(PrometheusMetricsClient, "collect_metrics_summary", return_value=_server_metrics()),
        ):
            reports = gen.generate_prometheus_metrics_report(
                _runtime({}), PrometheusMetricsReportConfig(summary=True, per_stage=False)
            )

        wait.assert_called_once()
        assert _report_names(reports) == ["summary_prometheus_metrics"]
        successes = reports[0].contents["successes"]
        assert successes["count"] == 12.0
        assert successes["rate"] == pytest.approx(1.2)
        assert successes["prompt_len"] == {"mean": 100.0, "rate": 120.0}
        assert successes["queue_len"] == {"mean": 3.0}
        assert successes["request_latency"] == {"mean": 0.5, "median": 0.4, "p90": 0.9, "p99": 1.1}
        # 25 hits over 100 queries, reported as a percentage.
        assert successes["prefix_cache_hit_percent"]["mean"] == pytest.approx(25.0)

    def test_cache_hit_percent_is_zero_when_nothing_queried_the_cache(self) -> None:
        client = _prometheus_client()
        gen = _make_generator(metrics_client=client)
        metrics = ModelServerMetrics(
            prefix_cache_hits=CounterResult(total=0.0),
            prefix_cache_queries=CounterResult(total=0.0),
        )

        with (
            patch.object(PrometheusMetricsClient, "wait"),
            patch.object(PrometheusMetricsClient, "collect_metrics_summary", return_value=metrics),
        ):
            reports = gen.generate_prometheus_metrics_report(
                _runtime({}), PrometheusMetricsReportConfig(summary=True, per_stage=False)
            )

        assert reports[0].contents["successes"]["prefix_cache_hit_percent"]["mean"] == 0.0

    def test_per_stage_emits_one_report_per_configured_stage(self) -> None:
        client = _prometheus_client()
        gen = _make_generator(metrics_client=client)
        runtime = _runtime({0: _stage_info(0), 1: _stage_info(1)})

        with (
            patch.object(PrometheusMetricsClient, "wait"),
            patch.object(PrometheusMetricsClient, "collect_metrics_for_stage", return_value=_server_metrics()) as collect,
        ):
            reports = gen.generate_prometheus_metrics_report(
                runtime, PrometheusMetricsReportConfig(summary=False, per_stage=True)
            )

        assert _report_names(reports) == ["stage_0_prometheus_metrics", "stage_1_prometheus_metrics"]
        assert [call.args[1] for call in collect.call_args_list] == [0, 1]

    def test_a_stage_with_no_collected_metrics_is_skipped_not_reported_as_zeros(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A failed collection must drop the stage's report. Emitting the default
        all-zero summary instead would publish an unmeasured stage as a measured one."""
        client = _prometheus_client()
        gen = _make_generator(metrics_client=client)
        runtime = _runtime({0: _stage_info(0), 1: _stage_info(1)})

        def collect(runtime_parameters: PerfRuntimeParameters, stage_id: int) -> Optional[ModelServerMetrics]:
            return _server_metrics() if stage_id == 0 else None

        with (
            patch.object(PrometheusMetricsClient, "wait"),
            patch.object(PrometheusMetricsClient, "collect_metrics_for_stage", side_effect=collect),
        ):
            reports = gen.generate_prometheus_metrics_report(
                runtime, PrometheusMetricsReportConfig(summary=False, per_stage=True)
            )

        assert _report_names(reports) == ["stage_0_prometheus_metrics"]
        assert "No metrics collected for Stage 1" in caplog.text

    def test_no_summary_metrics_emits_no_summary_report(self, caplog: pytest.LogCaptureFixture) -> None:
        client = _prometheus_client()
        gen = _make_generator(metrics_client=client)

        with (
            patch.object(PrometheusMetricsClient, "wait"),
            patch.object(PrometheusMetricsClient, "collect_metrics_summary", return_value=None),
        ):
            reports = gen.generate_prometheus_metrics_report(
                _runtime({}), PrometheusMetricsReportConfig(summary=True, per_stage=False)
            )

        assert reports == []
        assert "no metrics collected by metrics client" in caplog.text
