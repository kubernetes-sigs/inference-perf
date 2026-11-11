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
"""ReportGenerator.generate_stage_reports: the reports of a stage, written when it ends.

A run used to write every report once, at the end, so a run OOM-killed in stage N
lost stages 0..N-1 (#817). Each stage's files are now built as the stage ends and
the end-of-run pass writes them again. Object storage (GCS, S3) keeps the first
upload of a name and skips the second, so the copy built at stage end is the one
that stays: it must be byte-identical to what the end-of-run pass would have
written, and building it must not change what the end-of-run pass writes.

The fixture streams OpenAI-style SSE chunks that hold several words each, behind a
whitespace tokenizer, so tokenizer-corrected token timings differ from the raw
ones: any divergence in when, or on what, that in-place correction runs shows up
in the bytes compared here.
"""

import asyncio
import copy
import json
import logging
import re
import threading
from typing import Dict, List, Optional, Tuple
from unittest.mock import Mock

import pytest
import yaml

from inference_perf.apis import (
    ErrorResponseInfo,
    InferenceInfo,
    RequestLifecycleMetric,
    SessionLifecycleMetric,
    StreamedResponseMetrics,
    UnaryResponseMetrics,
)
from inference_perf.client.modelserver.metrics import BaseMetrics
from inference_perf.client.server_metrics.base import PerfRuntimeParameters, StageRuntimeInfo, StageStatus
from inference_perf.client.server_metrics.prometheus_client import PrometheusMetricsClient
from inference_perf.config import Config
from inference_perf.config.reportgen.config import (
    PerRequestFieldsConfig,
    PrometheusMetricsReportConfig,
    ReportConfig,
    RequestLifecycleMetricsReportConfig,
    SessionLifecycleReportConfig,
)
from inference_perf.metrics import LocalRequestMetricCollector, SessionMetricsCollector
from inference_perf.payloads import RequestMetrics, Text
from inference_perf.reportgen.base import ReportGenerator, correct_streamed_response_metrics
from inference_perf.utils import ReportFile
from inference_perf.utils.custom_tokenizer import CustomTokenizer

pytestmark = pytest.mark.timeout(30)

# Wall-clock (epoch) stage windows; request timestamps are small monotonic values.
_EPOCH = 1_750_000_000.0
_BR_PARTIAL_PREFIX = "inference-perf.partial.stage_"


class _WordTokenizer(CustomTokenizer):
    """Deterministic stand-in for CustomTokenizer: one token per whitespace-separated word."""

    def __init__(self) -> None:
        self.calling_threads: set[int] = set()

    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        self.calling_threads.add(threading.get_ident())
        return len(text.split())


# ---------------------------------------------------------------------------
# Fixture data
# ---------------------------------------------------------------------------


def _completion_chunk(text: str) -> str:
    return json.dumps({"choices": [{"index": 0, "text": text}]})


def _chat_chunk(text: Optional[str]) -> str:
    delta = {"role": "assistant"} if text is None else {"content": text}
    return json.dumps({"choices": [{"index": 0, "delta": delta}]})


_USAGE_CHUNK = json.dumps({"choices": [], "usage": {"prompt_tokens": 10, "completion_tokens": 6}})


def _streamed(
    stage_id: int,
    start: float,
    chunks: List[str],
    *,
    lora: Optional[str] = None,
    session_id: Optional[str] = None,
    event_id: Optional[str] = None,
    completion_tokens: Optional[int] = None,
    failed: bool = False,
) -> RequestLifecycleMetric:
    """A streamed request whose raw output_token_times hold one timestamp per chunk.

    The tokenizer correction re-derives them from the chunk text (several words, and
    so several tokens, per chunk; role and usage chunks hold none), so corrected and
    raw timings differ in both length and first-token time.
    """
    chunk_times = [start + 0.1 * (i + 1) for i in range(len(chunks))]
    words = sum(len(text.split()) for text in (_chunk_text(c) for c in chunks) if text)
    return RequestLifecycleMetric(
        stage_id=stage_id,
        session_id=session_id,
        scheduled_time=start - 0.01,
        start_time=start,
        end_time=chunk_times[-1] + 0.05,
        request_data=f"request {stage_id} {start}",
        response_data=None if failed else "response",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=10)),
            response_metrics=StreamedResponseMetrics(
                response_chunks=list(chunks),
                chunk_times=list(chunk_times),
                output_tokens=words,
                output_token_times=list(chunk_times),
                server_usage={
                    "prompt_tokens": 10,
                    "completion_tokens": words if completion_tokens is None else completion_tokens,
                },
            ),
            lora_adapter=lora,
            graph_event_id=event_id,
        ),
        error=ErrorResponseInfo(error_type="HTTP Error 500", error_msg="stream broke") if failed else None,
    )


def _chunk_text(chunk: str) -> Optional[str]:
    choices = json.loads(chunk).get("choices") or []
    if not choices:
        return None
    text = choices[0].get("text") or choices[0].get("delta", {}).get("content")
    return str(text) if text else None


def _unary(stage_id: int, start: float) -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        stage_id=stage_id,
        scheduled_time=start,
        start_time=start,
        end_time=start + 0.4,
        request_data="unary",
        response_data="unary response",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=7)),
            response_metrics=UnaryResponseMetrics(output_tokens=4, server_usage={"prompt_tokens": 7, "completion_tokens": 4}),
        ),
        error=None,
    )


def _request_metrics() -> Dict[int, List[RequestLifecycleMetric]]:
    """Request metrics by stage: -1 is the sweep's preprocessing probe, never reported.

    Stages mix completions and chat streams, LoRA and base-model requests, a unary
    request, a server/client token-count mismatch, and failures that still carry chunks.
    """
    return {
        -1: [_streamed(-1, -5.0, [_completion_chunk("probe request words")])],
        0: [
            _streamed(
                0,
                0.0,
                [_completion_chunk("one two"), _completion_chunk("three"), _completion_chunk("four five six")],
                session_id="sess-0a",
                event_id="e1",
            ),
            _streamed(
                0,
                0.5,
                [
                    _chat_chunk(None),
                    _chat_chunk("Hello there"),
                    _chat_chunk("general"),
                    _chat_chunk("kenobi you are"),
                    _USAGE_CHUNK,
                ],
                lora="adapter-a",
                session_id="sess-0a",
                event_id="e2",
            ),
            _streamed(
                0,
                0.9,
                [_completion_chunk("alpha beta gamma"), _completion_chunk("delta")],
                lora="adapter-b",
                completion_tokens=9,
            ),
            _unary(0, 1.2),
            _streamed(0, 1.4, [_chat_chunk("partial answer"), _chat_chunk("cut")], lora="adapter-a", failed=True),
        ],
        1: [
            _streamed(
                1,
                20.0,
                [_chat_chunk(None), _chat_chunk("red green"), _chat_chunk("blue")],
                lora="adapter-a",
                session_id="sess-1a",
                event_id="e1",
            ),
            _streamed(1, 20.3, [_completion_chunk("x y z"), _completion_chunk("w v")]),
            _streamed(1, 20.6, [_completion_chunk("lost")], failed=True),
        ],
    }


def _sessions() -> Dict[int, List[SessionLifecycleMetric]]:
    """Session metrics by stage. Stage 2 has none: all of its sessions were stranded."""

    def session(session_id: str, stage_id: int, start: float, *, num_events: int, completed: int) -> SessionLifecycleMetric:
        return SessionLifecycleMetric(
            session_id=session_id,
            stage_id=stage_id,
            file_path=f"{session_id}.json",
            start_time=start,
            end_time=start + 3.0,
            duration_sec=3.0,
            num_events=num_events,
            num_events_completed=completed,
            user_facing_event_ids=["e2"] if session_id == "sess-0a" else ["e1"],
            dispatch_perf_counter=start,
        )

    return {
        0: [session("sess-0a", 0, 0.0, num_events=2, completed=2), session("sess-0b", 0, 0.2, num_events=3, completed=1)],
        1: [session("sess-1a", 1, 20.0, num_events=1, completed=1)],
    }


def _stage_infos() -> Dict[int, StageRuntimeInfo]:
    return {
        0: StageRuntimeInfo(
            stage_id=0,
            rate=2.0,
            start_time=_EPOCH,
            end_time=_EPOCH + 10.0,
            status=StageStatus.COMPLETED,
            max_stage_duration=12.0,
            teardown_duration=0.5,
            dropped_requests=0,
        ),
        1: StageRuntimeInfo(
            stage_id=1,
            rate=1.0,
            start_time=_EPOCH + 20.0,
            end_time=_EPOCH + 30.0,
            status=StageStatus.TIMED_OUT,
            concurrency_level=4,
            teardown_duration=1.5,
            dropped_requests=3,
        ),
        2: StageRuntimeInfo(
            stage_id=2,
            rate=1.0,
            start_time=_EPOCH + 40.0,
            end_time=_EPOCH + 50.0,
            status=StageStatus.TIMED_OUT,
            max_stage_duration=10.0,
            sessions_not_completed_active=2,
            sessions_not_completed_pending=1,
        ),
    }


def _runtime(stage_infos: Dict[int, StageRuntimeInfo]) -> PerfRuntimeParameters:
    return PerfRuntimeParameters(
        start_time=_EPOCH,
        duration=50.0,
        model_server_metrics=BaseMetrics(),
        stages=stage_infos,
    )


def _generator(collector: Optional[LocalRequestMetricCollector] = None, *, with_sessions: bool = False) -> ReportGenerator:
    config = Mock(spec=Config)
    config.tokenizer = None
    config.model_dump = Mock(return_value={})
    generator = ReportGenerator(None, collector or LocalRequestMetricCollector(), config=config)
    generator._tokenizer = _WordTokenizer()
    if with_sessions:
        generator.session_metrics_collector = SessionMetricsCollector()
    return generator


def _lifecycle(
    *,
    summary: bool = True,
    per_stage: bool = True,
    per_request: bool = False,
    computed_metrics: bool = False,
    per_adapter: bool = True,
    per_adapter_stage: bool = False,
) -> ReportConfig:
    return ReportConfig(
        request_lifecycle=RequestLifecycleMetricsReportConfig(
            summary=summary,
            per_stage=per_stage,
            per_request=per_request,
            per_request_fields=PerRequestFieldsConfig(computed_metrics=computed_metrics),
            per_adapter=per_adapter,
            per_adapter_stage=per_adapter_stage,
            percentiles=[50.0, 90.0, 99.0],
        )
    )


# Which lifecycle reports are on decides which requests the end-of-run pass has
# corrected in place by the time it builds each stage's BR partial, and which it
# dumps uncorrected into per_request first.
REPORT_CONFIGS = [
    pytest.param(ReportConfig(), id="default"),
    pytest.param(_lifecycle(summary=False, per_stage=False), id="no_summary_no_per_stage"),
    pytest.param(_lifecycle(summary=False, per_stage=False, per_request=True), id="per_request_raw"),
    pytest.param(
        _lifecycle(summary=False, per_stage=False, per_request=True, computed_metrics=True), id="per_request_computed"
    ),
    pytest.param(_lifecycle(summary=False, per_stage=False, per_adapter_stage=True), id="lora_per_adapter_stage"),
    pytest.param(
        _lifecycle(summary=False, per_stage=False, per_adapter=False, per_adapter_stage=True, per_request=True),
        id="lora_per_adapter_stage_only_with_per_request_raw",
    ),
    pytest.param(_lifecycle(summary=False, per_stage=False, per_adapter=False), id="nothing_corrects"),
    pytest.param(
        ReportConfig(
            request_lifecycle=RequestLifecycleMetricsReportConfig(
                per_request=True,
                per_request_fields=PerRequestFieldsConfig(computed_metrics=True),
                per_adapter_stage=True,
            ),
            prometheus=PrometheusMetricsReportConfig(summary=True, per_stage=True),
            session_lifecycle=SessionLifecycleReportConfig(summary=True, per_stage=True, per_session=True),
        ),
        id="everything_on",
    ),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _serialize(report: ReportFile) -> str:
    """The bytes LocalStorageClient writes for a report."""
    if report.file_type == "yaml":
        return yaml.dump(report.get_contents(), sort_keys=False, default_flow_style=False)
    return json.dumps(report.get_contents(), indent=2)


def _masked(report: ReportFile) -> str:
    """Serialized bytes, with the BR ids that differ between generators masked."""
    if not report.name.startswith(_BR_PARTIAL_PREFIX):
        return _serialize(report)
    contents = copy.deepcopy(report.get_contents())
    contents["run"]["uid"] = "<uid>"
    contents["run"]["eid"] = "<eid>"
    return _serialize(ReportFile(name=report.name, contents=contents, file_type=report.file_type))


def _by_filename(reports: List[ReportFile]) -> Dict[str, ReportFile]:
    filenames = [r.get_filename() for r in reports]
    assert len(filenames) == len(set(filenames)), f"duplicate report names: {filenames}"
    return {r.get_filename(): r for r in reports}


def _per_stage_patterns(stage_id: int) -> List["re.Pattern[str]"]:
    return [
        re.compile(rf"stage_{stage_id}_lifecycle_metrics"),
        re.compile(rf"adapter_.+_stage_{stage_id}_lifecycle_metrics"),
        re.compile(rf"inference-perf\.partial\.stage_{stage_id}"),
        re.compile(rf"stage_{stage_id}_session_lifecycle_metrics"),
    ]


def _is_per_stage_file(name: str, stage_id: int) -> bool:
    return any(p.fullmatch(name) for p in _per_stage_patterns(stage_id))


async def _run_stages(
    generator: ReportGenerator,
    report_config: ReportConfig,
    stage_ids: List[int],
    requests: Dict[int, List[RequestLifecycleMetric]],
    sessions: Dict[int, List[SessionLifecycleMetric]],
    stage_infos: Dict[int, StageRuntimeInfo],
) -> Tuple[Dict[int, List[ReportFile]], List[ReportFile]]:
    """Replay a run: each stage's metrics arrive, the stage ends and its reports are
    built, then the end-of-run pass runs over everything."""
    for m in requests.get(-1, []):
        generator.metrics_collector.record_metric(m)
    stage_reports: Dict[int, List[ReportFile]] = {}
    for stage_id in stage_ids:
        for m in requests.get(stage_id, []):
            generator.metrics_collector.record_metric(m)
        if generator.session_metrics_collector is not None:
            for s in sessions.get(stage_id, []):
                generator.session_metrics_collector.record_metric(s)
        stage_reports[stage_id] = await asyncio.wait_for(
            generator.generate_stage_reports(report_config, stage_infos[stage_id]), timeout=10
        )
    end_of_run = await asyncio.wait_for(generator.generate_reports(report_config, _runtime(stage_infos)), timeout=10)
    return stage_reports, end_of_run


def _assert_stage_files_match_end_of_run(stage_reports: Dict[int, List[ReportFile]], end_of_run: List[ReportFile]) -> None:
    final = _by_filename(end_of_run)
    for stage_id, reports in stage_reports.items():
        assert reports, f"stage {stage_id} wrote nothing at stage end"
        for filename, report in _by_filename(reports).items():
            assert filename in final, f"{filename} is written at stage end but not by the end-of-run pass"
            assert _serialize(report) == _serialize(final[filename]), f"{filename} differs from its end-of-run copy"


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_fixture_token_timings_change_when_corrected() -> None:
    """Guard for the fixture the rest of this file relies on: tokenizer correction must
    move every successful stream's token timestamps, or a stage-end pass that corrected
    too much, too little or in place would still produce matching bytes."""
    tokenizer = _WordTokenizer()
    streamed = [
        m
        for stage_id, metrics in _request_metrics().items()
        if stage_id >= 0
        for m in metrics
        if m.error is None and isinstance(m.info.response_metrics, StreamedResponseMetrics)
    ]
    assert streamed
    for m in streamed:
        assert isinstance(m.info.response_metrics, StreamedResponseMetrics)
        raw = list(m.info.response_metrics.output_token_times)
        correct_streamed_response_metrics(m, tokenizer)
        assert m.info.response_metrics.output_token_times != raw


@pytest.mark.parametrize("report_config", REPORT_CONFIGS)
async def test_stage_reports_are_byte_identical_to_end_of_run_copies(report_config: ReportConfig) -> None:
    """Every file built at stage end has the bytes the end-of-run pass writes under the
    same name, BR run ids included. GCS and S3 skip a name that already exists, so a
    stage-end copy that differed would be the one left in the bucket."""
    generator = _generator()

    stage_reports, end_of_run = await _run_stages(generator, report_config, [0, 1], _request_metrics(), {}, _stage_infos())

    _assert_stage_files_match_end_of_run(stage_reports, end_of_run)


@pytest.mark.parametrize("report_config", REPORT_CONFIGS)
async def test_stage_reports_leave_end_of_run_output_unchanged(report_config: ReportConfig) -> None:
    """The end-of-run pass writes the same files whether or not the stage-end pass ran
    first. That pass corrects token timings and enriches sessions, both in place; done
    on the collector's own metrics it would leak into end-of-run files that dump them
    uncorrected, such as per_request when only LoRA reports correct timings."""
    requests = _request_metrics()
    sessions = _sessions()
    pristine_requests = copy.deepcopy(requests)
    pristine_sessions = copy.deepcopy(sessions)

    fresh = _generator(with_sessions=True)
    for metrics in pristine_requests.values():
        for m in metrics:
            fresh.metrics_collector.record_metric(m)
    assert fresh.session_metrics_collector is not None
    for stage_sessions in pristine_sessions.values():
        for s in stage_sessions:
            fresh.session_metrics_collector.record_metric(s)
    expected = await fresh.generate_reports(report_config, _runtime(_stage_infos()))

    staged = _generator(with_sessions=True)
    _, end_of_run = await _run_stages(staged, report_config, [0, 1, 2], requests, sessions, _stage_infos())

    assert [r.get_filename() for r in end_of_run] == [r.get_filename() for r in expected]
    for got, want in zip(end_of_run, expected, strict=True):
        assert _masked(got) == _masked(want), f"{got.get_filename()} changed because stage-end reports ran first"


@pytest.mark.parametrize("report_config", REPORT_CONFIGS)
async def test_stage_end_writes_only_and_all_of_the_stage_files(report_config: ReportConfig) -> None:
    """Stage end writes the per-stage files and nothing run-wide: summaries,
    per-request/per-session lists, run-wide adapter rollups, Prometheus and validation
    describe the whole run and would be frozen half-done on object storage. And it
    writes every per-stage file the end of the run would, or a run killed later
    loses it."""
    stage_infos = _stage_infos()
    generator = _generator(with_sessions=True)

    stage_reports, end_of_run = await _run_stages(
        generator, report_config, [0, 1, 2], _request_metrics(), _sessions(), stage_infos
    )

    final_names = [r.name for r in end_of_run]
    for stage_id, reports in stage_reports.items():
        names = [r.name for r in reports]
        for name in names:
            assert _is_per_stage_file(name, stage_id), f"stage {stage_id} wrote {name} at stage end"
            assert not name.startswith(("summary_", "per_request", "per_session"))
            assert "prometheus" not in name
            assert name not in ("validation", "config")
        expected = [n for n in final_names if _is_per_stage_file(n, stage_id)]
        if report_config.prometheus and report_config.prometheus.per_stage:
            # Left to the end of the run, where it picks up the stage's server metrics.
            expected = [n for n in expected if not n.startswith(_BR_PARTIAL_PREFIX)]
        assert sorted(names) == sorted(expected)


async def test_stage_end_does_not_query_prometheus() -> None:
    """Prometheus reports stay at the end of the run even when enabled per stage: the
    client waits out a scrape interval (32s by default) before querying, which at
    stage end would stall the run between every pair of stages."""
    report_config = ReportConfig(
        request_lifecycle=RequestLifecycleMetricsReportConfig(per_request=True, per_adapter_stage=True),
        prometheus=PrometheusMetricsReportConfig(summary=True, per_stage=True),
        session_lifecycle=SessionLifecycleReportConfig(summary=True, per_stage=True, per_session=True),
    )
    metrics_client = Mock(spec=PrometheusMetricsClient)
    generator = _generator(with_sessions=True)
    generator.metrics_client = metrics_client
    requests = _request_metrics()
    stage_infos = _stage_infos()

    for stage_id in (0, 1):
        for m in requests[stage_id]:
            generator.metrics_collector.record_metric(m)
        reports = await generator.generate_stage_reports(report_config, stage_infos[stage_id])
        assert reports
        assert not any("prometheus" in r.name for r in reports)

    assert metrics_client.mock_calls == []


async def test_stage_without_metrics_or_sessions_writes_nothing() -> None:
    """A stage that recorded no request and no session (e.g. it was cut off before
    dispatching) has nothing to write; an empty stage file would also be one the
    end-of-run pass never writes."""
    stage_info = StageRuntimeInfo(stage_id=3, rate=1.0, start_time=_EPOCH, end_time=_EPOCH + 1, status=StageStatus.COMPLETED)

    for with_sessions in (False, True):
        generator = _generator(with_sessions=with_sessions)
        for m in _request_metrics()[0]:
            generator.metrics_collector.record_metric(m)

        assert await generator.generate_stage_reports(ReportConfig(), stage_info) == []


async def test_stage_end_without_per_stage_still_writes_the_br_partial() -> None:
    """per_stage only governs stage_N_lifecycle_metrics. The BR partial is always
    written per stage, so turning per_stage off must not drop it at stage end."""
    generator = _generator()
    for m in _request_metrics()[0]:
        generator.metrics_collector.record_metric(m)

    reports = await generator.generate_stage_reports(_lifecycle(per_stage=False), _stage_infos()[0])

    assert [r.name for r in reports] == ["inference-perf.partial.stage_0"]


@pytest.mark.parametrize(
    ("prometheus", "writes_partial"),
    [
        pytest.param(None, True, id="no_prometheus"),
        pytest.param(PrometheusMetricsReportConfig(summary=True, per_stage=False), True, id="prometheus_summary_only"),
        pytest.param(PrometheusMetricsReportConfig(summary=True, per_stage=True), False, id="prometheus_per_stage"),
    ],
)
async def test_br_partial_waits_for_the_end_of_the_run_with_per_stage_prometheus(
    prometheus: Optional[PrometheusMetricsReportConfig], writes_partial: bool
) -> None:
    """With per-stage Prometheus reports on, the end-of-run BR partial carries the
    stage's server metrics, which are only queried at the end of the run. GCS and S3
    keep the first copy of a name, so a partial written at stage end would leave them out."""
    generator = _generator()
    for m in _request_metrics()[0]:
        generator.metrics_collector.record_metric(m)

    reports = await generator.generate_stage_reports(ReportConfig(prometheus=prometheus), _stage_infos()[0])

    assert ("inference-perf.partial.stage_0" in [r.name for r in reports]) == writes_partial


@pytest.mark.parametrize(
    "report_config",
    [
        pytest.param(ReportConfig(), id="default"),
        pytest.param(_lifecycle(summary=False, per_stage=False, per_adapter=False), id="nothing_corrects"),
    ],
)
async def test_stage_session_reports_match_end_of_run(report_config: ReportConfig) -> None:
    """Session workloads get stage_N_session_lifecycle_metrics at stage end, equal to
    the end-of-run copy: TFUT reads the first token time, which the lifecycle reports
    correct or not depending on config. A stage whose sessions were all stranded, with
    no session metric and no request, still gets its file and its BR partial, because
    its stranded counts are the only record that the stage ran."""
    stage_infos = _stage_infos()
    generator = _generator(with_sessions=True)

    stage_reports, end_of_run = await _run_stages(
        generator, report_config, [0, 1, 2], _request_metrics(), _sessions(), stage_infos
    )

    for stage_id in (0, 1, 2):
        assert f"stage_{stage_id}_session_lifecycle_metrics" in [r.name for r in stage_reports[stage_id]]
    assert sorted(r.name for r in stage_reports[2]) == ["inference-perf.partial.stage_2", "stage_2_session_lifecycle_metrics"]
    stranded = _by_filename(stage_reports[2])["stage_2_session_lifecycle_metrics.json"].get_contents()
    assert stranded["stage_metadata"]["stage_id"] == 2
    assert stranded["stage_metadata"]["status"] == "TIMED_OUT"
    _assert_stage_files_match_end_of_run(stage_reports, end_of_run)


async def test_br_partials_share_eid_and_keep_their_uid_across_passes() -> None:
    """run.eid groups a run's partials into one experiment and run.uid names one
    stage. Both are random, so they must be drawn once per run and per stage: a fresh
    eid at the end of the run would split one experiment across ids depending on which
    pass wrote which object."""
    generator = _generator()

    stage_reports, end_of_run = await _run_stages(generator, ReportConfig(), [0, 1], _request_metrics(), {}, _stage_infos())

    def run_ids(reports: List[ReportFile], stage_id: int) -> Dict[str, str]:
        partial = _by_filename(reports)[f"{_BR_PARTIAL_PREFIX}{stage_id}.yaml"]
        run = partial.get_contents()["run"]
        assert isinstance(run, dict)
        return run

    at_stage_end = {stage_id: run_ids(stage_reports[stage_id], stage_id) for stage_id in (0, 1)}
    at_end_of_run = {stage_id: run_ids(end_of_run, stage_id) for stage_id in (0, 1)}

    eids = {ids["eid"] for ids in [*at_stage_end.values(), *at_end_of_run.values()]}
    assert len(eids) == 1
    assert at_stage_end[0]["uid"] != at_stage_end[1]["uid"]
    for stage_id in (0, 1):
        assert at_stage_end[stage_id]["uid"] == at_end_of_run[stage_id]["uid"]
        assert at_stage_end[stage_id]["uid"].startswith(f"inference-perf-stage-{stage_id}-")


class _LateArrivalCollector(LocalRequestMetricCollector):
    """Holds a stage's last metric back until wait_for_recorded, like the multiprocess
    collector whose workers' metrics are still in the pipe when the stage ends."""

    def __init__(self, late: RequestLifecycleMetric) -> None:
        super().__init__()
        self.calls: List[str] = []
        self._late: Optional[RequestLifecycleMetric] = late

    async def wait_for_recorded(self, stage_id: int) -> bool:
        self.calls.append(f"wait_for_recorded({stage_id})")
        await asyncio.sleep(0)
        if self._late is not None:
            self.record_metric(self._late)
            self._late = None
        return True

    def snapshot(self) -> List[RequestLifecycleMetric]:
        self.calls.append("snapshot")
        return list(self.metrics)


async def test_stage_reports_wait_for_recorded_metrics_before_reading_them() -> None:
    """The stage's reports are read from the collector only after it has received
    every metric the workers recorded: a stage file built from a partial snapshot
    would be the copy left on object storage, short of the requests still in flight
    to the collector."""
    *early, late = _request_metrics()[0]
    collector = _LateArrivalCollector(late)
    for m in early:
        collector.record_metric(m)
    generator = _generator(collector)

    reports = await generator.generate_stage_reports(ReportConfig(), _stage_infos()[0])

    assert collector.calls == ["wait_for_recorded(0)", "snapshot"]
    stage_file = _by_filename(reports)["stage_0_lifecycle_metrics.json"].get_contents()
    assert stage_file["load_summary"]["count"] == len(early) + 1


class _IncompleteCollector(LocalRequestMetricCollector):
    """Gives up waiting, like the multiprocess collector when a stage's metrics stop arriving."""

    async def wait_for_recorded(self, stage_id: int) -> bool:
        return False


async def test_stage_short_of_metrics_is_left_to_the_end_of_the_run(caplog: pytest.LogCaptureFixture) -> None:
    """A stage whose recorded metrics did not all reach the collector is not written at
    stage end: object storage keeps the first copy of a file, and the end-of-run pass
    builds the stage from every metric that turned up by then."""
    collector = _IncompleteCollector()
    for m in _request_metrics()[0]:
        collector.record_metric(m)
    generator = _generator(collector)

    with caplog.at_level(logging.WARNING, logger="inference_perf.reportgen.base"):
        reports = await generator.generate_stage_reports(ReportConfig(), _stage_infos()[0])

    assert reports == []
    assert "Stage 0: its reports will be written at the end of the run" in caplog.text


async def test_stage_reports_build_off_the_event_loop() -> None:
    """Tokenizing a stage's streams can take seconds; it runs in a worker thread so
    the event loop keeps draining the metrics queue and timing the stage interval."""
    generator = _generator()
    for m in _request_metrics()[0]:
        generator.metrics_collector.record_metric(m)
    tokenizer = generator._tokenizer
    assert isinstance(tokenizer, _WordTokenizer)

    await generator.generate_stage_reports(ReportConfig(), _stage_infos()[0])

    assert tokenizer.calling_threads
    assert threading.get_ident() not in tokenizer.calling_threads
