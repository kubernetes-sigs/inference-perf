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
"""Every stage the run visits reaches the stage observer as one start and one end.

The overall progress bar reads ``inference_perf_stages_completed`` against
``inference_perf_stages``, and only ``on_stage_end`` advances the first. A
stage runner that exits without ending its stage leaves the bar short for the
rest of the run, so the pairing is pinned across every runner here rather
than by one example per path.
"""

from types import SimpleNamespace
from typing import Any, Awaitable, Callable, Dict, List, Optional, Set, Tuple

import pytest

from inference_perf.apis import LazyLoadInferenceAPIData
from inference_perf.client.modelserver import MockModelServerClient
from inference_perf.config import (
    APIConfig,
    APIType,
    Config,
    ConcurrentLoadStage,
    DataConfig,
    DataGenType,
    LoadConfig,
    LoadType,
    StageGenType,
    StandardLoadStage,
    SweepConfig,
)
from inference_perf.config.loadgen.config import TraceSessionReplayLoadStage
from inference_perf.datagen import BaseGenerator, MockDataGenerator, SessionGenerator
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.metrics.request_collector.local import LocalRequestMetricCollector
from inference_perf.observability.context import RunContext, StageContext
from inference_perf.observability.metrics.registry import MetricsHub, build_metrics


# A corpus of num_sessions sessions with no events. Each session counts as
# complete once activated, so a session stage runs entirely in the parent.
class _InstantSessionGenerator(SessionGenerator):
    def __init__(self, num_sessions: int) -> None:
        self._num_sessions = num_sessions
        self._activated: Set[str] = set()

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion]

    def get_session_count(self) -> int:
        return self._num_sessions

    def get_session_info(self, session_index: int) -> Dict[str, Any]:
        return {"session_id": f"session-{session_index}"}

    def get_session_event_indices(self, session_index: int) -> List[int]:
        return []

    def get_session_events(self, session_index: int) -> List[LazyLoadInferenceAPIData]:
        return []

    def activate_session(self, session_id: str) -> None:
        self._activated.add(session_id)

    def check_session_completed(self, session_id: str) -> bool:
        return session_id in self._activated

    def build_session_metric(self, session_id: str, stage_id: int, start_time: float, end_time: float) -> Any:
        return SimpleNamespace()

    def cleanup_session(self, session_id: str) -> None:
        pass

    def get_session_state(self, session_id: str) -> Any:
        return None


# Forwards stage events to the metrics hub and keeps their order, so a test
# can check the start/end pairing and the gauges the overall bar reads.
class _Recorder:
    def __init__(self, hub: MetricsHub) -> None:
        self.hub = hub
        self.events: List[Tuple[str, int]] = []
        self.ended: List[StageContext] = []

    def on_stage_start(self, context: StageContext) -> None:
        self.events.append(("start", context.stage_id))
        self.hub.on_stage_start(context)

    def on_stage_end(self, context: StageContext) -> None:
        self.events.append(("end", context.stage_id))
        self.ended.append(context)
        self.hub.on_stage_end(context)


# Builds a load generator for load_config with a metrics hub already past
# on_run_start, the way main.py wires them.
def _wired_loadgen(load_config: LoadConfig, datagen: BaseGenerator) -> Tuple[LoadGenerator, _Recorder]:
    hub = build_metrics(Config(load=load_config))
    recorder = _Recorder(hub)
    loadgen = LoadGenerator(datagen, load_config, stage_observer=recorder, metrics_registry=hub.registry)
    hub.on_run_start(
        RunContext(
            config=Config(load=load_config),
            in_flight_requests=loadgen.in_flight_requests,
            stage_count=loadgen.stage_count,
        )
    )
    return loadgen, recorder


# Reads one sample from the recorder's registry, None when the series is absent.
def _sample(recorder: _Recorder, name: str, labels: Optional[Dict[str, str]] = None) -> Optional[float]:
    return recorder.hub.registry.get_sample_value(name, labels or {})


# Two concurrent stages of 4 requests each, with rate and duration filled in
# the way main.py fills them before the run starts.
def _concurrent_stages() -> List[ConcurrentLoadStage]:
    stages = [ConcurrentLoadStage(num_requests=4, concurrency_level=2) for _ in range(2)]
    for stage in stages:
        stage.rate = stage.num_requests
        stage.duration = 1
    return stages


_TWO_STANDARD_STAGES = [StandardLoadStage(rate=2, duration=1), StandardLoadStage(rate=2, duration=1)]
_SWEEP = SweepConfig(type=StageGenType.LINEAR, num_requests=4, num_stages=2, stage_duration=1)


# Each case runs a whole load generator and expects stages_completed to equal
# stages, with start/end alternating per stage id in order. The session case is
# 3 sessions across two open-ended stages: stage 0 takes all 3, stage 1 finds
# the corpus exhausted. The sweep case skips the saturation probe and plants
# the two stages it would generate.
@pytest.mark.parametrize(
    "load_config",
    [
        pytest.param(
            LoadConfig(type=LoadType.CONSTANT, interval=0, stages=_TWO_STANDARD_STAGES, num_workers=0),
            id="in-process",
        ),
        pytest.param(
            LoadConfig(type=LoadType.CONSTANT, interval=0, stages=_TWO_STANDARD_STAGES, num_workers=2),
            id="multiprocess-constant",
        ),
        pytest.param(
            LoadConfig(type=LoadType.CONCURRENT, interval=0, stages=_concurrent_stages(), num_workers=2),
            id="multiprocess-concurrent",
        ),
        pytest.param(
            LoadConfig(type=LoadType.CONSTANT, interval=0, stages=[], sweep=_SWEEP, num_workers=2),
            id="sweep",
        ),
        pytest.param(
            LoadConfig(
                type=LoadType.TRACE_SESSION_REPLAY,
                interval=0,
                stages=[TraceSessionReplayLoadStage(concurrent_sessions=1) for _ in range(2)],
                num_workers=1,
            ),
            id="session-exhausted-corpus",
        ),
    ],
)
async def test_every_visited_stage_starts_and_ends_once(load_config: LoadConfig, monkeypatch: pytest.MonkeyPatch) -> None:
    api_config = APIConfig(type=APIType.Completion)
    datagen: BaseGenerator
    if load_config.type == LoadType.TRACE_SESSION_REPLAY:
        datagen = _InstantSessionGenerator(3)
    else:
        datagen = MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)
    if load_config.sweep is not None:

        async def _plant_generated_stages(self: LoadGenerator, *args: Any, **kwargs: Any) -> None:
            self.stages = list(_TWO_STANDARD_STAGES)

        monkeypatch.setattr(LoadGenerator, "preprocess", _plant_generated_stages)
    loadgen, recorder = _wired_loadgen(load_config, datagen)
    client = MockModelServerClient(LocalRequestMetricCollector(), api_config, mock_latency=0)

    try:
        await loadgen.run(client)
    finally:
        await loadgen.stop()

    assert len(loadgen.stages) == 2
    assert recorder.events == [("start", 0), ("end", 0), ("start", 1), ("end", 1)]
    assert _sample(recorder, "inference_perf_stages") == 2
    assert _sample(recorder, "inference_perf_stages_completed") == 2


# 3 sessions across stages asking for 2, the rest, and the rest. Expects the
# stages to plan 2, 1 and 0 sessions, each to finish what it planned, and the
# exhausted third stage to export planned 0 rather than no series at all.
async def test_session_stages_slice_the_corpus_and_report_the_slice() -> None:
    load_config = LoadConfig(
        type=LoadType.TRACE_SESSION_REPLAY,
        interval=0,
        stages=[
            TraceSessionReplayLoadStage(concurrent_sessions=1, num_sessions=2),
            TraceSessionReplayLoadStage(concurrent_sessions=1),
            TraceSessionReplayLoadStage(concurrent_sessions=1),
        ],
        num_workers=1,
    )
    loadgen, recorder = _wired_loadgen(load_config, _InstantSessionGenerator(3))
    client = MockModelServerClient(LocalRequestMetricCollector(), APIConfig(type=APIType.Completion), mock_latency=0)

    try:
        await loadgen.run(client)
    finally:
        await loadgen.stop()

    assert [c.planned_sessions for c in recorder.ended] == [2, 1, 0]
    assert [c.sessions_finished() for c in recorder.ended] == [2, 1, 0]
    for stage_id, planned in enumerate([2, 1, 0]):
        stage = {"stage": str(stage_id)}
        assert _sample(recorder, "inference_perf_stage_sessions_planned", stage) == planned
        assert _sample(recorder, "inference_perf_stage_running", stage) == 0
    assert _sample(recorder, "inference_perf_stages_completed") == 3


# A runner that raises after starting stage 0 with 5 planned requests.
async def _raises_after_starting(loadgen: LoadGenerator) -> None:
    loadgen._stage_started(StageContext(stage_id=0, planned_requests=5))
    raise RuntimeError("stage runner failed")


# A runner that returns before it ever reports stage 0.
async def _returns_before_starting(loadgen: LoadGenerator) -> None:
    return None


# Stage 0's runner either raises after starting or returns without starting.
# Either way the observer sees one start and one end for stage 0, the stage
# stops reading as running, and stages_completed reaches 1.
@pytest.mark.parametrize(
    "runner, raises",
    [
        pytest.param(_raises_after_starting, True, id="raises-after-start"),
        pytest.param(_returns_before_starting, False, id="returns-before-start"),
    ],
)
async def test_stage_is_ended_however_its_runner_exits(
    runner: Callable[[LoadGenerator], Awaitable[None]], raises: bool
) -> None:
    load_config = LoadConfig(type=LoadType.CONSTANT, stages=[StandardLoadStage(rate=1, duration=1)], num_workers=0)
    datagen = MockDataGenerator(APIConfig(type=APIType.Completion), DataConfig(type=DataGenType.Mock), None)
    loadgen, recorder = _wired_loadgen(load_config, datagen)

    if raises:
        with pytest.raises(RuntimeError, match="stage runner failed"):
            await loadgen._run_reported_stage(0, runner(loadgen))
    else:
        await loadgen._run_reported_stage(0, runner(loadgen))

    assert recorder.events == [("start", 0), ("end", 0)]
    assert _sample(recorder, "inference_perf_stage_running", {"stage": "0"}) == 0
    assert _sample(recorder, "inference_perf_stages_completed") == 1
