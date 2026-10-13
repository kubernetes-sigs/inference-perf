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
"""End-to-end assembly test: drive the real ``generate_reports`` over
synthesized request metrics and assert the emitted set validates cleanly.

This is the assembly-level seam #661 asks for: the report set users read is
produced by the real code path, and ``validation.json`` is the assertion
interface over it.
"""

from __future__ import annotations

import asyncio
from typing import Any, Dict, List, cast
from unittest.mock import Mock

from inference_perf.client.server_metrics import PerfRuntimeParameters
from inference_perf.client.server_metrics.base import StageRuntimeInfo, StageStatus
from inference_perf.apis.base import RequestLifecycleMetric, SessionLifecycleMetric
from inference_perf.config import Config
from inference_perf.config.reportgen.config import ReportConfig, RequestLifecycleMetricsReportConfig
from inference_perf.metrics.request_collector import RequestMetricCollector
from inference_perf.metrics.session_collector import SessionMetricsCollector
from inference_perf.reportgen import ReportGenerator
from inference_perf.reportgen.validation import validate_reports
from inference_perf.utils import ReportFile

from .helpers import STAGE_DISPATCH_LEAD, STAGE_TEARDOWN_OVERHANG, make_metric, make_stage_metrics


# Drives the real generate_reports over two stages of synthesized requests
# (stage 0: 3 successes + 1 failure, stage 1: 2 successes) and returns the
# emitted report files, validation.json included.
def _generate_reports() -> List[ReportFile]:
    stage_metrics = make_stage_metrics()
    all_metrics = [m for metrics in stage_metrics.values() for m in metrics]

    collector = Mock(spec=RequestMetricCollector)
    collector.get_metrics.return_value = all_metrics

    config = Mock(spec=Config)
    config.tokenizer = None
    config.model_dump.return_value = {"load": {"stages": [{"rate": 1.0}, {"rate": 1.0}]}}

    generator = ReportGenerator(metrics_client=None, metrics_collector=collector, config=cast(Config, config))

    # The offered-load window, not the request window: it opens before the
    # first request is dispatched and closes while the last requests are still
    # draining through teardown, which is what a real stage records.
    stages = {
        stage_id: StageRuntimeInfo(
            stage_id=stage_id,
            rate=1.0,
            start_time=min(m.start_time for m in metrics) - STAGE_DISPATCH_LEAD,
            end_time=max(m.end_time for m in metrics) - STAGE_TEARDOWN_OVERHANG,
            status=StageStatus.COMPLETED,
        )
        for stage_id, metrics in stage_metrics.items()
    }
    runtime_parameters = PerfRuntimeParameters(start_time=0.0, duration=12.0, model_server_metrics=Mock(), stages=stages)
    report_config = ReportConfig(
        request_lifecycle=RequestLifecycleMetricsReportConfig(per_request=True),
        prometheus=None,
    )
    return asyncio.run(generator.generate_reports(report_config, runtime_parameters))


def test_generate_reports_emits_validation_json_and_it_is_clean() -> None:
    reports = _generate_reports()
    filenames = {r.get_filename() for r in reports}

    assert "validation.json" in filenames
    validation = next(r for r in reports if r.get_filename() == "validation.json").get_contents()

    assert validation["global"]["errors"] == [], validation["global"]["errors"]
    all_errors = [e for group in validation["reports"].values() for e in group["errors"]]
    assert all_errors == [], all_errors
    all_warnings = [w for group in validation["reports"].values() for w in group["warnings"]]
    assert all_warnings == [], all_warnings


def test_generate_reports_validation_covers_the_emitted_files() -> None:
    reports = _generate_reports()
    validation = next(r for r in reports if r.get_filename() == "validation.json").get_contents()

    covered = set(validation["reports"].keys())
    assert "summary_lifecycle_metrics.json" in covered
    assert "stage_0_lifecycle_metrics.json" in covered
    assert "stage_1_lifecycle_metrics.json" in covered
    assert "per_request_lifecycle_metrics.json" in covered
    assert "inference-perf.partial.stage_0.yaml" in covered
    assert "inference-perf.partial.stage_1.yaml" in covered


def _all_zero_session_summary() -> ReportFile:
    """summary_session_lifecycle_metrics.json as the real generator emits it
    when every session's server usage explicitly reported zero cached tokens
    (the #818 shape: ambiguous, suspicious, not wrong)."""
    sessions = [
        SessionLifecycleMetric(
            session_id=f"s{i}",
            stage_id=0,
            file_path=f"s{i}.json",
            start_time=0.0,
            end_time=1.0,
            duration_sec=1.0,
            num_events=1,
            num_events_completed=1,
            total_cached_tokens=0,
            total_cacheable_input_tokens=100,
            total_input_tokens=100,
            total_output_tokens=0,
            success=True,
        )
        for i in range(2)
    ]
    contents = ReportGenerator.summarize_sessions(None, sessions, [], [50])  # type: ignore[arg-type]
    assert contents["kv_cache_hit_percent"] == 0.0
    return ReportFile(name="summary_session_lifecycle_metrics", contents=contents)


def test_validate_reports_flags_all_zero_session_cache_hit_rate() -> None:
    """The session validator is wired into default_validators().

    Goes through validate_reports() (the real default_validators() entry
    point) rather than the validator class directly, so removing
    SessionLifecycleValidator from default_validators() fails this test.
    """
    validation = validate_reports([_all_zero_session_summary()]).get_contents()

    group = validation["reports"]["summary_session_lifecycle_metrics.json"]
    assert group["errors"] == []
    assert [w["check"] for w in group["warnings"]] == ["session.cache"]


# ---------------------------------------------------------------------------
# Session workloads: each stage's partial carries that stage's sessions
# ---------------------------------------------------------------------------


def make_session_stage_metrics() -> Dict[int, List[RequestLifecycleMetric]]:
    """The request side of three stages of a session workload.

    Stage 0's session spans two requests (220 input / 20 output tokens in
    total) and stage 1's a single one (90 / 6), so a partial that mixed the
    stages up — or that read the per-session token totals before the request
    metrics had been folded into the sessions — reports different numbers.

    Stage 2 is the one that went wrong: three sessions, two of which the stage
    cut short before their last event.
    """
    return {
        0: [
            make_metric(0, 0.0, 1.0, input_tokens=100, output_tokens=8).model_copy(update={"session_id": "s0"}),
            make_metric(0, 1.0, 2.0, input_tokens=120, output_tokens=12).model_copy(update={"session_id": "s0"}),
        ],
        1: [
            make_metric(1, 10.0, 11.0, input_tokens=90, output_tokens=6).model_copy(update={"session_id": "s1"}),
        ],
        2: [
            make_metric(2, 20.0, 21.0, input_tokens=30, output_tokens=3).model_copy(update={"session_id": "s2a"}),
            make_metric(2, 21.0, 22.0, input_tokens=40, output_tokens=4).model_copy(update={"session_id": "s2b"}),
            make_metric(2, 22.0, 23.0, input_tokens=50, output_tokens=5).model_copy(update={"session_id": "s2c"}),
        ],
    }


def make_session_metrics() -> List[SessionLifecycleMetric]:
    """The sessions those requests belong to, as LoadGenerator records them."""
    return [
        SessionLifecycleMetric(
            session_id="s0",
            stage_id=0,
            file_path="s0.json",
            start_time=0.0,
            end_time=2.0,
            duration_sec=2.0,
            num_events=2,
            num_events_completed=2,
        ),
        SessionLifecycleMetric(
            session_id="s1",
            stage_id=1,
            file_path="s1.json",
            start_time=10.0,
            end_time=11.0,
            duration_sec=1.0,
            num_events=1,
            num_events_completed=1,
        ),
        SessionLifecycleMetric(
            session_id="s2a",
            stage_id=2,
            file_path="s2a.json",
            start_time=20.0,
            end_time=21.0,
            duration_sec=1.0,
            num_events=2,
            num_events_completed=2,
        ),
        # One event short of their full run: `enrich_metrics` is what turns
        # that into success=False, so these two count as failed and not as
        # merely incomplete.
        SessionLifecycleMetric(
            session_id="s2b",
            stage_id=2,
            file_path="s2b.json",
            start_time=21.0,
            end_time=22.0,
            duration_sec=1.0,
            num_events=2,
            num_events_completed=1,
        ),
        SessionLifecycleMetric(
            session_id="s2c",
            stage_id=2,
            file_path="s2c.json",
            start_time=22.0,
            end_time=23.0,
            duration_sec=1.0,
            num_events=2,
            num_events_completed=1,
        ),
    ]


def _reports_with_sessions() -> Dict[str, Any]:
    """Drive the real ``generate_reports`` for a session workload.

    Returns every emitted report keyed by name. The session collector is a real
    one, so ``_enrich_sessions`` runs the same ``enrich_metrics`` aggregation a
    production run does.
    """
    stage_metrics = make_session_stage_metrics()
    all_metrics = [m for metrics in stage_metrics.values() for m in metrics]

    collector = Mock(spec=RequestMetricCollector)
    collector.get_metrics.return_value = all_metrics

    config = Mock(spec=Config)
    config.tokenizer = None
    config.model_dump.return_value = {"load": {"stages": [{"rate": 1.0}] * 4}}

    generator = ReportGenerator(metrics_client=None, metrics_collector=collector, config=cast(Config, config))
    sessions = SessionMetricsCollector()
    for session in make_session_metrics():
        sessions.record_metric(session)
    generator.session_metrics_collector = sessions

    stages = {
        stage_id: StageRuntimeInfo(
            stage_id=stage_id,
            rate=1.0,
            start_time=min(m.start_time for m in metrics) - STAGE_DISPATCH_LEAD,
            end_time=max(m.end_time for m in metrics) - STAGE_TEARDOWN_OVERHANG,
            status=StageStatus.COMPLETED,
        )
        for stage_id, metrics in stage_metrics.items()
    }
    # Stage 2 ran short: its three recorded sessions are joined by a fourth the
    # stage never finished with. Stage 3 never dispatched a session at all, so
    # it has no request metric and no session metric either — the load
    # generator records the truncation on the stage instead.
    # Stage 3 does therefore draw a pre-existing
    # ``cross_report.configured_stages`` warning ("those stages recorded no
    # requests"): the request-side check only knows about per-stage *request*
    # lifecycle reports. That is the gap this partial closes for session counts.
    stages[2] = stages[2].model_copy(update={"sessions_not_completed_active": 1, "max_stage_duration": 3.0})
    stages[3] = StageRuntimeInfo(
        stage_id=3,
        rate=1.0,
        start_time=30.0,
        end_time=31.0,
        status=StageStatus.TIMED_OUT,
        max_stage_duration=1.0,
        sessions_not_completed_pending=1,
    )
    runtime_parameters = PerfRuntimeParameters(start_time=0.0, duration=12.0, model_server_metrics=Mock(), stages=stages)
    report_config = ReportConfig(
        request_lifecycle=RequestLifecycleMetricsReportConfig(per_request=True),
        prometheus=None,
    )
    reports = asyncio.run(generator.generate_reports(report_config, runtime_parameters))
    return {r.name: r.get_contents() for r in reports}


def _partial_sessions(reports: Dict[str, Any], stage_id: int) -> Dict[str, Any]:
    """The partial's session counts for one stage."""
    contents = reports[f"inference-perf.partial.stage_{stage_id}"]["results"]["session_performance"]["sessions"]
    return cast(Dict[str, Any], contents)


def test_partial_carries_each_stages_own_sessions() -> None:
    reports = _reports_with_sessions()

    stage_0 = _partial_sessions(reports, 0)
    stage_1 = _partial_sessions(reports, 1)

    assert stage_0["total"] == 1
    assert stage_0["total_events"] == 2
    assert stage_1["total"] == 1
    assert stage_1["total_events"] == 1


def test_partial_session_token_totals_come_from_the_request_enrichment() -> None:
    """Pins the ordering between enrichment and partial assembly.

    Per-session token totals only exist once the request metrics have been
    folded into the sessions, so sessions must be enriched *before* the partials
    are assembled. Building the partials first emits zeros here.
    """
    reports = _reports_with_sessions()
    sessions = _partial_sessions(reports, 0)

    assert sessions["input_tokens_per_session"]["mean"] == 220
    assert sessions["output_tokens_per_session"]["mean"] == 20


def test_partial_session_counts_agree_with_the_native_session_report() -> None:
    """BR0.2 partial and native session report must agree on the same run.

    Each count is paired with the native field that means the same thing:
    ``total`` with ``num_sessions``, not ``num_sessions_completed``. Stage 2
    tells the two apart — its native counts are 4 sessions, 3 of which
    completed, 1 succeeded and 2 failed — so neither pairing can pass by
    accident.
    """
    reports = _reports_with_sessions()

    for stage_id in (0, 1, 2):
        partial = _partial_sessions(reports, stage_id)
        native = reports[f"stage_{stage_id}_session_lifecycle_metrics"]

        assert partial["total"] == native["num_sessions"]
        assert partial["succeeded"] == native["num_sessions_succeeded"]
        assert partial["failed"] == native["num_sessions_failed"]

    partial = _partial_sessions(reports, 2)
    assert (partial["total"], partial["succeeded"], partial["failed"]) == (4, 1, 2)
    # The distinguishing numbers, pinned so a future edit cannot quietly go back
    # to comparing completed sessions with succeeded ones.
    assert reports["stage_2_session_lifecycle_metrics"]["num_sessions_completed"] == 3


def test_partial_counts_the_sessions_a_truncated_stage_never_finished() -> None:
    """A stage cut short by max_stage_duration reports what it stranded.

    Stage 2 finished 3 of its 4 sessions, stage 3 none of its 1. Counting only
    the recorded metrics would emit ``total: 3, succeeded: 1, failed: 2`` and
    ``total: 0``: both read as runs that ended cleanly.
    """
    reports = _reports_with_sessions()

    stage_2 = _partial_sessions(reports, 2)
    assert stage_2["total"] == 4
    assert stage_2["succeeded"] == 1
    assert stage_2["failed"] == 2
    assert stage_2["total"] == reports["stage_2_session_lifecycle_metrics"]["num_sessions"]

    # No request metric and no session metric: the stage still gets a partial,
    # because its stranded sessions have to be counted somewhere.
    stage_3 = _partial_sessions(reports, 3)
    assert stage_3["total"] == 1
    assert stage_3["succeeded"] == 0
    assert stage_3["failed"] == 0
    assert stage_3["total"] == reports["stage_3_session_lifecycle_metrics"]["num_sessions"]
    assert "request_performance" not in reports["inference-perf.partial.stage_3"]["results"]
