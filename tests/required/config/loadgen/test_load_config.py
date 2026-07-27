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
