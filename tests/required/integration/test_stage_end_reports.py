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
"""Integration tests for #817: each stage's reports are written when the stage ends.

Before #817 every report was written once, after the last stage, so a run that
was OOM-killed (or crashed) in stage N lost the reports of the stages 0..N-1 it
had already completed. InferencePerfRunner now hooks the load generator's
after_stage to save_stage_reports, which writes the stage's per-stage files and
config.yaml through every storage client before the next stage starts.

These tests drive that real wiring (InferencePerfRunner -> LoadGenerator ->
ReportGenerator -> storage clients) against MockModelServerClient, both with no
workers (in-process collector) and with two forked workers (multiprocess
collector, whose metrics reach the main process asynchronously).
"""

import json
import logging
import multiprocessing as mp
import os
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from queue import Empty
from typing import Dict, Generator, Iterator, List, Optional, Set, Union

import pytest
import yaml

from inference_perf.apis import InferenceAPIData, RequestLifecycleMetric
from inference_perf.client.filestorage import StorageClient
from inference_perf.client.filestorage.local import LocalStorageClient
from inference_perf.client.modelserver.mock_client import MockModelServerClient
from inference_perf.client.server_metrics.base import PerfRuntimeParameters, StageRuntimeInfo, StageStatus
from inference_perf.config import (
    APIConfig,
    APIType,
    Config,
    DataConfig,
    DataGenType,
    LoadConfig,
    LoadType,
    ReportConfig,
    StandardLoadStage,
    StorageConfigBase,
)
from inference_perf.datagen import MockDataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.loadgen.load_timer import LoadTimer
from inference_perf.main import InferencePerfRunner
from inference_perf.metrics.request_collector import (
    LocalRequestMetricCollector,
    MultiprocessRequestMetricCollector,
    RequestMetricCollector,
)
from inference_perf.reportgen.base import ReportGenerator
from inference_perf.utils import ReportFile

_RATE = 4
_DURATION = 1
_NUM_STAGES = 2
_CONFIG_FILE = "config.yaml"


def _stage_files(stage_id: int) -> Set[str]:
    """The per-stage files the default report config produces for one stage."""
    return {f"stage_{stage_id}_lifecycle_metrics.json", f"inference-perf.partial.stage_{stage_id}.yaml"}


@contextmanager
def _fork_start_method() -> Iterator[None]:
    # Workers must fork (matching production: Linux default, forced on macOS in
    # main_cli) so unpicklable test state is inherited rather than pickled.
    old = mp.get_start_method(allow_none=True)
    if old != "fork":
        if "fork" not in mp.get_all_start_methods():
            pytest.skip("fork start method unavailable on this platform")
        mp.set_start_method("fork", force=True)
    try:
        yield
    finally:
        if old is not None and old != "fork":
            mp.set_start_method(old, force=True)


@contextmanager
def _reaping_new_children() -> Iterator[None]:
    """Terminate any worker a failed run leaves behind, so it can't outlive the test."""
    before = set(mp.active_children())
    try:
        yield
    finally:
        for child in mp.active_children():
            if child not in before:
                child.terminate()
                child.join(timeout=5)


@pytest.fixture(autouse=True)
def _isolated_workers() -> Generator[None, None, None]:
    with _fork_start_method(), _reaping_new_children():
        yield


class _UnavailableStorageClient(StorageClient):
    """A storage backend that rejects every write, e.g. a full disk or a bucket without write access."""

    def __init__(self) -> None:
        super().__init__(StorageConfigBase(path="unavailable"))
        self.calls = 0

    def save_report(self, reports: List[ReportFile]) -> None:
        self.calls += 1
        raise OSError("storage backend unavailable")


class _FirstWriteWinsStorageClient(StorageClient):
    """In-memory stand-in for the GCS and S3 clients.

    Like them it rejects a batch with duplicate names, serializes with
    json.dumps, and keeps the first upload of a name: a later save of the same
    name is skipped. So whatever is saved at stage end is what object storage
    keeps, and the end-of-run pass can't correct it.
    """

    def __init__(self) -> None:
        super().__init__(StorageConfigBase(path="bucket"))
        self.objects: Dict[str, str] = {}

    def save_report(self, reports: List[ReportFile]) -> None:
        filenames = [report.get_filename() for report in reports]
        if len(filenames) != len(set(filenames)):
            raise ValueError("Duplicate filenames detected", filenames)
        for report in reports:
            filename = report.get_filename()
            if filename in self.objects:
                continue
            self.objects[filename] = json.dumps(report.get_contents())


@dataclass
class _Run:
    runner: InferencePerfRunner
    loadgen: LoadGenerator
    client: MockModelServerClient
    collector: RequestMetricCollector
    report_config: ReportConfig


def _make_run(num_workers: int, storage_clients: List[StorageClient]) -> _Run:
    """Assemble the runner the way main_cli does, around the mock model server."""
    api_config = APIConfig(type=APIType.Chat)
    data_config = DataConfig(type=DataGenType.Mock)
    load_config = LoadConfig(
        type=LoadType.CONSTANT,
        interval=0,
        stages=[StandardLoadStage(rate=_RATE, duration=_DURATION) for _ in range(_NUM_STAGES)],
        num_workers=num_workers,
        worker_max_concurrency=4,
    )
    config = Config(api=api_config, data=data_config, load=load_config, tokenizer=None)
    collector: RequestMetricCollector
    if num_workers > 0:
        collector = MultiprocessRequestMetricCollector()
    else:
        collector = LocalRequestMetricCollector()
    client = MockModelServerClient(collector, api_config, mock_latency=0)
    loadgen = LoadGenerator(MockDataGenerator(api_config, data_config, None), load_config)
    reportgen = ReportGenerator(None, collector, config=config)
    runner = InferencePerfRunner(client, loadgen, reportgen, storage_clients, config.report)
    return _Run(runner=runner, loadgen=loadgen, client=client, collector=collector, report_config=config.report)


def _end_of_run_reports(run: _Run, start_time: float) -> List[ReportFile]:
    """The end-of-run report pass, with the arguments main_cli gives it."""
    return run.runner.generate_reports(
        report_config=run.report_config,
        runtime_parameters=PerfRuntimeParameters(
            start_time=start_time,
            duration=time.time() - start_time,
            model_server_metrics=run.client.get_prometheus_metric_metadata(),
            stages=run.loadgen.stage_runtime_info,
        ),
    )


def _read_dir(path: Path) -> Dict[str, bytes]:
    if not path.exists():
        return {}
    return {name: (path / name).read_bytes() for name in os.listdir(path)}


@dataclass
class _StageStartState:
    """What the storage clients hold when a stage starts, before it sends any request."""

    local_files: Dict[str, bytes]
    object_names: Set[str]


def _record_state_at_each_stage_start(
    monkeypatch: pytest.MonkeyPatch, loadgen: LoadGenerator, out_dir: Path, store: _FirstWriteWinsStorageClient
) -> List[_StageStartState]:
    states: List[_StageStartState] = []
    get_timer = loadgen.get_timer

    def recording_get_timer(rate: float, duration: float) -> LoadTimer:
        # Both run() (no workers) and run_stage() (workers) build the stage's
        # timer once, at the start of the stage, before sending any of it.
        states.append(_StageStartState(local_files=_read_dir(out_dir), object_names=set(store.objects)))
        return get_timer(rate, duration)

    monkeypatch.setattr(loadgen, "get_timer", recording_get_timer)
    return states


class _RecordCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.DEBUG)
        self.records: List[logging.LogRecord] = []

    def emit(self, record: logging.LogRecord) -> None:
        self.records.append(record)


@contextmanager
def _capture_logs(logger_name: str) -> Iterator[List[logging.LogRecord]]:
    # A handler on the logger itself, with its level opened up: another test
    # calling setup_logging() can raise the root level or replace root handlers.
    logger = logging.getLogger(logger_name)
    handler = _RecordCollector()
    old_level = logger.level
    logger.addHandler(handler)
    logger.setLevel(logging.DEBUG)
    try:
        yield handler.records
    finally:
        logger.removeHandler(handler)
        logger.setLevel(old_level)


@dataclass
class _CompletedRun:
    num_workers: int
    out_dir: Path
    stage_start_states: List[_StageStartState]
    objects_after_run: Dict[str, str]
    stage_runtime_info: Dict[int, StageRuntimeInfo]
    unavailable_calls_after_run: int
    runner_log: List[logging.LogRecord]
    end_of_run_reports: List[ReportFile]
    store: _FirstWriteWinsStorageClient


@pytest.fixture(scope="module", params=[0, 2], ids=["no-workers", "two-workers"])
def completed_run(request: pytest.FixtureRequest, tmp_path_factory: pytest.TempPathFactory) -> Iterator[_CompletedRun]:
    """One two-stage run through InferencePerfRunner, then the end-of-run pass, as main_cli does them.

    The storage clients are an always-failing one (first, so the others only
    get their reports if each client's failure is isolated), a first-write-wins
    object store, and LocalStorageClient. Shared by the tests below so each
    worker count runs once.
    """
    num_workers: int = request.param
    out_dir = tmp_path_factory.mktemp(f"stage_end_reports_{num_workers}_workers") / "out"
    unavailable = _UnavailableStorageClient()
    store = _FirstWriteWinsStorageClient()
    local = LocalStorageClient(StorageConfigBase(path=str(out_dir)))
    with _fork_start_method(), _reaping_new_children(), pytest.MonkeyPatch.context() as monkeypatch:
        run = _make_run(num_workers, [unavailable, store, local])
        stage_start_states = _record_state_at_each_stage_start(monkeypatch, run.loadgen, out_dir, store)
        start_time = time.time()
        try:
            with _capture_logs("inference_perf.main") as runner_log:
                run.runner.run()
            objects_after_run = dict(store.objects)
            unavailable_calls_after_run = unavailable.calls
            reports = _end_of_run_reports(run, start_time)
            # The end-of-run save stops at the first client that raises (it
            # isn't isolated per client, unlike the stage-end save), so it
            # goes to the working clients only.
            run.runner.storage_clients = [store, local]
            run.runner.save_reports(reports)
        finally:
            run.runner.stop()
    yield _CompletedRun(
        num_workers=num_workers,
        out_dir=out_dir,
        stage_start_states=stage_start_states,
        objects_after_run=objects_after_run,
        stage_runtime_info=dict(run.loadgen.stage_runtime_info),
        unavailable_calls_after_run=unavailable_calls_after_run,
        runner_log=list(runner_log),
        end_of_run_reports=reports,
        store=store,
    )


@pytest.mark.timeout(60)
def test_completed_stage_is_saved_before_the_next_stage_starts(completed_run: _CompletedRun) -> None:
    """Stage 0's files are on disk and in object storage by the time stage 1 sends its first request.

    That is what lets a run killed during stage 1 keep stage 0. Only the files
    that describe a finished stage go out early: the run-wide summary,
    validation.json and the next stage's files would be partial or wrong, so
    they must wait for the end of the run. config.yaml goes with the stage so a
    directory a dead run leaves behind says how the run was configured.
    """
    assert len(completed_run.stage_start_states) == _NUM_STAGES
    at_stage_0, at_stage_1 = completed_run.stage_start_states
    assert at_stage_0.local_files == {}
    assert at_stage_0.object_names == set()

    expected = _stage_files(0) | {_CONFIG_FILE}
    assert set(at_stage_1.local_files) == expected
    assert at_stage_1.object_names == expected
    for name, contents in at_stage_1.local_files.items():
        if name.endswith(".json"):
            json.loads(contents)
        else:
            assert yaml.safe_load(contents) is not None


@pytest.mark.timeout(60)
def test_last_stage_is_saved_when_it_ends_before_the_end_of_run_pass(completed_run: _CompletedRun) -> None:
    """The final stage is written by the stage-end pass too, not left to the end-of-run pass.

    The end-of-run pass can itself be OOM-killed (it holds every metric and
    builds every report), so the last stage must not depend on it.
    """
    expected = set().union(*(_stage_files(i) for i in range(_NUM_STAGES))) | {_CONFIG_FILE}
    assert set(completed_run.objects_after_run) == expected


@pytest.mark.timeout(60)
def test_failing_storage_client_neither_stops_the_run_nor_the_other_clients(completed_run: _CompletedRun) -> None:
    """A storage client that raises at stage end is logged and skipped; the run and the other clients carry on.

    Before #817 a storage failure could only happen after the load was sent.
    Now it happens between stages, and letting it propagate would abort the
    benchmark and lose the end-of-run reports, which is worse than not writing
    early at all. The failing client is listed first, so the other two clients
    only hold stage 0 when stage 1 starts if its failure was contained.
    """
    assert sorted(completed_run.stage_runtime_info) == list(range(_NUM_STAGES))
    assert all(info.status == StageStatus.COMPLETED for info in completed_run.stage_runtime_info.values())
    assert completed_run.unavailable_calls_after_run == _NUM_STAGES
    at_stage_1 = completed_run.stage_start_states[1]
    assert _stage_files(0) <= at_stage_1.object_names
    assert _stage_files(0) <= set(at_stage_1.local_files)

    failures = [
        record
        for record in completed_run.runner_log
        if record.levelno >= logging.ERROR
        and record.exc_info is not None
        and isinstance(record.exc_info[1], OSError)
        and "_UnavailableStorageClient" in record.getMessage()
    ]
    assert len(failures) == _NUM_STAGES, [r.getMessage() for r in completed_run.runner_log]


@pytest.mark.timeout(60)
def test_object_storage_keeps_what_the_end_of_run_pass_would_have_written(completed_run: _CompletedRun) -> None:
    """GCS and S3 skip a name they already hold, so the stage-end copy is the one that stays.

    It must therefore equal what the end-of-run pass builds for the same file:
    same metrics, same BR run ids, same token accounting. (In these short runs
    every metric has reached the main process by the time the stage ends;
    test_stage_end_reports_wait_for_metrics_still_on_their_way_from_workers
    covers metrics still in transit.) LocalStorageClient rewrites the files at
    the end of the run; the copy a reader saw at stage end must not change
    either.
    """
    final_contents = {report.get_filename(): report.get_contents() for report in completed_run.end_of_run_reports}
    per_stage_files = set().union(*(_stage_files(i) for i in range(_NUM_STAGES))) | {_CONFIG_FILE}
    assert per_stage_files <= set(final_contents)
    for name in sorted(per_stage_files):
        # Saved at stage end, and kept there by the end-of-run save.
        assert completed_run.objects_after_run.get(name) == completed_run.store.objects[name], name
        assert completed_run.store.objects[name] == json.dumps(final_contents[name]), name

    on_disk_at_stage_1 = completed_run.stage_start_states[1].local_files
    on_disk_at_end = _read_dir(completed_run.out_dir)
    for name, contents in on_disk_at_stage_1.items():
        assert on_disk_at_end[name] == contents, name
    assert not [name for name in on_disk_at_end if name.endswith(".tmp")]


@pytest.mark.timeout(60)
def test_stage_end_report_counts_every_request_of_the_stage(completed_run: _CompletedRun) -> None:
    """Each stage's stage-end lifecycle report counts every request the stage sent.

    With workers, run_stage sends exactly rate * duration requests and the
    stage ends once all of them have finished, so a smaller count means the
    stage-end report missed some of them. Without workers the stage can stop
    short of rate * duration, so only the end-of-run comparison above applies.
    """
    for stage_id in range(_NUM_STAGES):
        stage_report = json.loads(completed_run.objects_after_run[f"stage_{stage_id}_lifecycle_metrics.json"])
        count = stage_report["load_summary"]["count"]
        if completed_run.num_workers > 0:
            assert count == _RATE * _DURATION, stage_id
        else:
            assert 0 < count <= _RATE * _DURATION, stage_id


_QueueItem = Optional[Union[RequestLifecycleMetric, List[RequestLifecycleMetric]]]


# Holds back the main process's drain of the collector's queue (get and
# get_nowait) until release is set. The patch lives on the queue object, which
# pickles without it, so workers put as usual. That reproduces, every time, the
# window in which a worker has put a request's metric and acknowledged the end
# of the stage while the metric is still in its feeder buffer or the pipe.
def _hold_back_drain(monkeypatch: pytest.MonkeyPatch, queue: "mp.JoinableQueue[_QueueItem]", release: threading.Event) -> None:
    real_get, real_get_nowait = queue.get, queue.get_nowait

    def held_get(block: bool = True, timeout: Optional[float] = None) -> _QueueItem:
        if not release.wait(timeout):
            raise Empty
        return real_get(block, timeout)

    def held_get_nowait() -> _QueueItem:
        if not release.is_set():
            raise Empty
        return real_get_nowait()

    monkeypatch.setattr(queue, "get", held_get)
    monkeypatch.setattr(queue, "get_nowait", held_get_nowait)


@pytest.mark.timeout(60)
def test_stage_end_reports_wait_for_metrics_still_on_their_way_from_workers(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Metrics a worker has put but the main process hasn't received yet still make it into the stage's report.

    A worker acknowledges the end of a stage once it has put its metrics on
    the queue, not once the main process has them. A stage-end report built
    without waiting for them would count fewer requests than the stage sent,
    and on GCS/S3 that short report is the one that stays. Here none of stage
    0's metrics are released to the main process until its stage-end pass has
    begun.
    """
    out_dir = tmp_path / "out"
    store = _FirstWriteWinsStorageClient()
    run = _make_run(2, [store, LocalStorageClient(StorageConfigBase(path=str(out_dir)))])
    assert isinstance(run.collector, MultiprocessRequestMetricCollector)
    release = threading.Event()
    _hold_back_drain(monkeypatch, run.collector.queue, release)

    after_stage = run.loadgen.after_stage
    assert after_stage is not None
    received_when_stage_ended: List[int] = []

    async def release_then_save(stage_info: StageRuntimeInfo) -> None:
        received_when_stage_ended.append(len(run.collector.snapshot()))
        release.set()
        await after_stage(stage_info)

    run.loadgen.after_stage = release_then_save
    get_timer = run.loadgen.get_timer
    objects_at_stage_start: List[Set[str]] = []

    def get_timer_releasing_by_stage_1(rate: float, duration: float) -> LoadTimer:
        objects_at_stage_start.append(set(store.objects))
        if len(objects_at_stage_start) > 1:
            # Never hold metrics past the stage boundary, even if no stage-end
            # pass ran; the collector must drain to let the run finish.
            release.set()
        return get_timer(rate, duration)

    monkeypatch.setattr(run.loadgen, "get_timer", get_timer_releasing_by_stage_1)
    try:
        run.runner.run()
    finally:
        release.set()
        run.runner.stop()

    # Stage 0 ended with none of its metrics in the main process yet.
    assert received_when_stage_ended[0] == 0
    assert objects_at_stage_start[1] == _stage_files(0) | {_CONFIG_FILE}
    for stage_id in range(_NUM_STAGES):
        stage_report = json.loads(store.objects[f"stage_{stage_id}_lifecycle_metrics.json"])
        assert stage_report["load_summary"]["count"] == _RATE * _DURATION, stage_id
    partial = yaml.safe_load((out_dir / "inference-perf.partial.stage_0.yaml").read_text())
    assert partial is not None


@pytest.mark.timeout(60)
def test_crash_in_a_later_stage_keeps_the_completed_stages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A run that dies in stage 1 leaves stage 0's reports behind, complete and parseable.

    This is #817 itself: the end-of-run pass never runs when the process dies
    (an exception here, an OOM kill in production), and before the fix the
    report directory wasn't even created, so the completed stage 0 was lost.
    """
    out_dir = tmp_path / "out"
    run = _make_run(0, [LocalStorageClient(StorageConfigBase(path=str(out_dir)))])
    process_request = run.client.process_request

    async def crash_in_stage_1(
        data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        if stage_id == 1:
            raise RuntimeError("simulated crash in stage 1")
        await process_request(data, stage_id, scheduled_time, lora_adapter)

    monkeypatch.setattr(run.client, "process_request", crash_in_stage_1)
    try:
        with pytest.raises(ExceptionGroup) as raised:
            run.runner.run()
    finally:
        run.runner.stop()
    assert raised.group_contains(RuntimeError, match="simulated crash in stage 1")

    assert set(os.listdir(out_dir)) == _stage_files(0) | {_CONFIG_FILE}
    stage_0 = json.loads((out_dir / "stage_0_lifecycle_metrics.json").read_text())
    assert stage_0["load_summary"]["count"] > 0
    assert stage_0["failures"]["count"] == 0
    partial = yaml.safe_load((out_dir / "inference-perf.partial.stage_0.yaml").read_text())
    assert partial is not None
    config = yaml.safe_load((out_dir / _CONFIG_FILE).read_text())
    assert config["load"]["stages"][0]["rate"] == _RATE
    assert not (out_dir / "summary_lifecycle_metrics.json").exists()
