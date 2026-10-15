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
"""Validity rules for ``inference_perf.config.loadgen``.

Covers the per-stage validators (Standard / Concurrent / TraceSessionReplay)
and the cross-stage ``LoadConfig`` validator that ties stage shape to load
type and checks the MultiLoRA traffic split.
"""

import pytest
from pydantic import ValidationError

from inference_perf.config import (
    ConcurrentLoadStage,
    LoadConfig,
    LoadType,
    MultiLoRAConfig,
    StandardLoadStage,
    StageGenType,
    SweepConfig,
    TraceSessionReplayLoadStage,
)


# --- StandardLoadStage ---------------------------------------------------


def test_standard_load_stage_valid() -> None:
    stage = StandardLoadStage(rate=10, duration=60)
    assert stage.rate == 10
    assert stage.duration == 60


def test_standard_load_stage_rejects_num_requests() -> None:
    with pytest.raises(ValueError, match="num_requests should not be set"):
        StandardLoadStage(rate=10, duration=60, num_requests=100)


def test_standard_load_stage_rejects_concurrency_level() -> None:
    with pytest.raises(ValueError, match="concurrency_level should not be set"):
        StandardLoadStage(rate=10, duration=60, concurrency_level=5)


def test_standard_load_stage_requires_positive_rate_and_duration() -> None:
    with pytest.raises(ValidationError):
        StandardLoadStage(rate=0, duration=60)
    with pytest.raises(ValidationError):
        StandardLoadStage(rate=10, duration=0)


# --- ConcurrentLoadStage -------------------------------------------------


def test_concurrent_load_stage_valid() -> None:
    stage = ConcurrentLoadStage(num_requests=100, concurrency_level=10)
    assert stage.num_requests == 100
    assert stage.concurrency_level == 10
    # rate/duration are filled at runtime, not by config.
    assert stage.rate is None
    assert stage.duration is None


def test_concurrent_load_stage_requires_positive_values() -> None:
    with pytest.raises(ValidationError):
        ConcurrentLoadStage(num_requests=0, concurrency_level=10)
    with pytest.raises(ValidationError):
        ConcurrentLoadStage(num_requests=100, concurrency_level=0)


# --- TraceSessionReplayLoadStage ----------------------------------------


def test_trace_session_replay_stage_valid() -> None:
    stage = TraceSessionReplayLoadStage(concurrent_sessions=4, session_rate=2, num_sessions=10)
    assert stage.concurrent_sessions == 4
    assert stage.session_rate == 2


def test_trace_session_replay_stage_zero_concurrency_allowed() -> None:
    # 0 = stress-test mode (all sessions at once); explicitly permitted.
    stage = TraceSessionReplayLoadStage(concurrent_sessions=0)
    assert stage.concurrent_sessions == 0


def test_trace_session_replay_stage_rate_cannot_exceed_concurrency() -> None:
    with pytest.raises(ValueError, match="cannot exceed"):
        TraceSessionReplayLoadStage(concurrent_sessions=2, session_rate=5)


def test_trace_session_replay_stage_forbids_extra_fields() -> None:
    with pytest.raises(ValidationError):
        TraceSessionReplayLoadStage(concurrent_sessions=2, bogus_field=1)  # type: ignore[call-arg]


def test_trace_session_replay_stage_max_stage_duration_valid() -> None:
    stage = TraceSessionReplayLoadStage(concurrent_sessions=2, max_stage_duration=30.0)
    assert stage.max_stage_duration == 30.0


def test_trace_session_replay_stage_max_stage_duration_defaults_to_none() -> None:
    stage = TraceSessionReplayLoadStage(concurrent_sessions=2)
    assert stage.max_stage_duration is None


def test_trace_session_replay_stage_max_stage_duration_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        TraceSessionReplayLoadStage(concurrent_sessions=2, max_stage_duration=0)


# --- LoadConfig cross-stage validation -----------------------------------


def test_load_config_defaults_to_constant() -> None:
    assert LoadConfig().type == LoadType.CONSTANT


def test_sweep_with_concurrent_is_error() -> None:
    with pytest.raises(ValueError, match="Cannot have sweep config with CONCURRENT"):
        LoadConfig(
            type=LoadType.CONCURRENT,
            sweep=SweepConfig(type=StageGenType.GEOM),
            stages=[ConcurrentLoadStage(num_requests=10, concurrency_level=1)],
        )


def test_sweep_with_trace_session_replay_is_error() -> None:
    with pytest.raises(ValueError, match="Cannot have sweep config with TRACE_SESSION_REPLAY"):
        LoadConfig(
            type=LoadType.TRACE_SESSION_REPLAY,
            sweep=SweepConfig(type=StageGenType.GEOM),
            stages=[TraceSessionReplayLoadStage(concurrent_sessions=1)],
        )


def test_concurrent_load_type_requires_concurrent_stage() -> None:
    with pytest.raises(ValueError, match="CONCURRENT load type requires ConcurrentLoadStage"):
        LoadConfig(
            type=LoadType.CONCURRENT,
            stages=[StandardLoadStage(rate=10, duration=60)],
        )


def test_constant_load_type_requires_standard_stage() -> None:
    with pytest.raises(ValueError, match="CONSTANT load type requires StandardLoadStage"):
        LoadConfig(
            type=LoadType.CONSTANT,
            stages=[ConcurrentLoadStage(num_requests=10, concurrency_level=1)],
        )


def test_trace_session_replay_load_type_requires_session_stage() -> None:
    with pytest.raises(ValueError, match="TRACE_SESSION_REPLAY load type requires TraceSessionReplayLoadStage"):
        LoadConfig(
            type=LoadType.TRACE_SESSION_REPLAY,
            stages=[StandardLoadStage(rate=10, duration=60)],
        )


def test_multilora_traffic_split_must_sum_to_one() -> None:
    with pytest.raises(ValueError, match=r"MultiLoRA traffic split.*does not add up to 1.0"):
        LoadConfig(
            lora_traffic_split=[
                MultiLoRAConfig(name="a", split=0.5),
                MultiLoRAConfig(name="b", split=0.4),
            ]
        )


def test_multilora_traffic_split_summing_to_one_is_ok() -> None:
    cfg = LoadConfig(
        lora_traffic_split=[
            MultiLoRAConfig(name="a", split=0.5),
            MultiLoRAConfig(name="b", split=0.5),
        ]
    )
    assert cfg.lora_traffic_split is not None
    assert len(cfg.lora_traffic_split) == 2


# --- StandardLoadStage.stop_condition ------------------------------------------


# rate=10 with stop_condition 't >= 60' and no duration is valid; effective_duration is 60.0 and the parsed predicate is exposed.
def test_standard_load_stage_stop_condition_valid() -> None:
    stage = StandardLoadStage(rate=10, stop_condition="t >= 60")
    assert stage.duration is None
    assert stage.effective_duration == 60.0
    assert stage.predicate is not None
    assert stage.predicate.boundary == 60.0


# duration=60 is shorthand for stop_condition 't >= 60': the stage's predicate is Predicate('t >= 60') with
# boundary 60.0, while the user-facing fields keep what was written (duration 60, stop_condition None).
def test_standard_load_stage_duration_is_stop_condition_shorthand() -> None:
    stage = StandardLoadStage(rate=10, duration=60)
    assert stage.predicate.raw == "t >= 60"
    assert stage.effective_duration == 60.0
    assert stage.duration == 60
    assert stage.stop_condition is None


# For every integer duration 1..600 the shorthand's boundary is exactly float(duration), so a duration
# stage's window is unchanged by routing it through Predicate.
def test_duration_shorthand_boundary_is_exact() -> None:
    for d in range(1, 601):
        assert StandardLoadStage(rate=1, duration=d).effective_duration == float(d)


# Assigning duration=90 after construction (as main.py does for concurrent stages) moves the window to 90.0:
# the predicate is rebuilt from the current fields, not frozen at validation time.
def test_duration_assignment_after_construction_rebuilds_predicate() -> None:
    stage = StandardLoadStage(rate=10, duration=60)
    stage.duration = 90
    assert stage.predicate.raw == "t >= 90"
    assert stage.effective_duration == 90.0


# 't >= 60.5' is allowed even though duration must be an int; effective_duration is 60.5.
def test_standard_load_stage_stop_condition_fractional_boundary() -> None:
    assert StandardLoadStage(rate=10, stop_condition="t >= 60.5").effective_duration == 60.5


# rate=10 with neither duration nor stop_condition is rejected naming both fields.
def test_standard_load_stage_requires_duration_or_stop_condition() -> None:
    with pytest.raises(ValidationError, match="Exactly one of duration or stop_condition"):
        StandardLoadStage(rate=10)


# rate=10 with duration=60 AND stop_condition 't >= 60' is rejected: the two bound the same window.
def test_standard_load_stage_rejects_duration_and_stop_condition_together() -> None:
    with pytest.raises(ValidationError, match="Exactly one of duration or stop_condition"):
        StandardLoadStage(rate=10, duration=60, stop_condition="t >= 60")


# 'Eq(t, 60)' holds only at one instant; the stage is rejected with the predicate's own message.
def test_standard_load_stage_stop_condition_single_instant_rejected() -> None:
    with pytest.raises(ValidationError, match="uses equality"):
        StandardLoadStage(rate=10, stop_condition="Eq(t, 60)")


# 't < 60' already holds at t=0; the stage is rejected with the predicate's own message.
def test_standard_load_stage_stop_condition_lapsing_rejected() -> None:
    with pytest.raises(ValidationError, match=r"holds only on \[0, 60\);"):
        StandardLoadStage(rate=10, stop_condition="t < 60")


# A CONSTANT LoadConfig accepts a stop_condition stage alongside a duration stage; both are StandardLoadStage.
def test_load_config_accepts_stop_condition_stage() -> None:
    cfg = LoadConfig(
        type=LoadType.CONSTANT,
        stages=[StandardLoadStage(rate=10, duration=60), StandardLoadStage(rate=10, stop_condition="t >= 60")],
    )
    assert [s.effective_duration for s in cfg.stages if isinstance(s, StandardLoadStage)] == [60.0, 60.0]


# TRACE_REPLAY takes its request count and timing from the trace, so a stage's stop_condition would be accepted
# and then ignored. Input: a TRACE_REPLAY LoadConfig with stop_condition 't >= 60'. Expected: rejected at load,
# naming the stage. A plain duration stays accepted, since the shipped trace_replay example sets one.
def test_trace_replay_rejects_stop_condition() -> None:
    with pytest.raises(ValidationError, match="Stage 0: stop_condition has no effect under TRACE_REPLAY"):
        LoadConfig(type=LoadType.TRACE_REPLAY, stages=[StandardLoadStage(rate=1, stop_condition="t >= 60")])
    LoadConfig(type=LoadType.TRACE_REPLAY, stages=[StandardLoadStage(rate=1, duration=30)])


# Same reasoning for rate: under TRACE_REPLAY a rate expression would be accepted and ignored. Input: a
# TRACE_REPLAY LoadConfig with rate '5 + t/2'. Expected: rejected at load, naming the stage. A numeric rate
# stays accepted, since the shipped trace_replay example sets one as a placeholder.
def test_trace_replay_rejects_rate_expression() -> None:
    with pytest.raises(ValidationError, match="Stage 0: a rate expression has no effect under TRACE_REPLAY"):
        LoadConfig(type=LoadType.TRACE_REPLAY, stages=[StandardLoadStage(rate="5 + t/2", duration=30)])
    LoadConfig(type=LoadType.TRACE_REPLAY, stages=[StandardLoadStage(rate=1, duration=30)])
