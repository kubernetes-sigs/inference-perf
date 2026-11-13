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

import logging
import warnings
from pathlib import Path

import pytest
from pydantic import ValidationError

from inference_perf.config import (
    BatchJobConfig,
    BatchLoadStage,
    ConcurrentLoadStage,
    LoadConfig,
    LoadType,
    MultiLoRAConfig,
    StandardLoadStage,
    StageGenType,
    SweepConfig,
    TraceSessionReplayLoadStage,
    read_config,
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


# An integer level is a schedule that never changes: 10 -> initial 10, peak 10.
def test_concurrent_load_stage_integer_is_a_constant_schedule() -> None:
    stage = ConcurrentLoadStage(num_requests=100, concurrency_level=10)
    assert stage.concurrency_schedule.is_constant
    assert (stage.concurrency_schedule.initial, stage.peak_concurrency) == (10, 10)


# A quoted whole number is an integer level, not an expression: "4" loads as 4,
# the same as an unquoted 4.
def test_concurrent_load_stage_reads_a_quoted_whole_number_as_an_integer() -> None:
    stage = ConcurrentLoadStage(num_requests=100, concurrency_level="4")
    assert stage.concurrency_level == 4
    assert stage.concurrency_schedule.is_constant


# An expression is kept as written and compiled: 'floor(Min(1 + t/6, 64))' starts at
# 1 and peaks at 64.
def test_concurrent_load_stage_accepts_an_expression() -> None:
    stage = ConcurrentLoadStage(num_requests=100, concurrency_level="floor(Min(1 + t/6, 64))")
    assert stage.concurrency_level == "floor(Min(1 + t/6, 64))"
    assert not stage.concurrency_schedule.is_constant
    assert (stage.concurrency_schedule.initial, stage.peak_concurrency) == (1, 64)


# Expressions the schedule cannot run are config errors, with the schedule's
# own message: non-linear, unbounded, or below 1.
@pytest.mark.parametrize(
    "level, message",
    [
        ("Min(1 + t**2, 64)", "not piecewise linear"),
        ("floor(1 + t)", "grows without limit"),
        ("Min(1 + t/6, 64)", "not whole numbers"),
        ("floor(Max(0.5, 10 - t))", "falls below 1 at t=9"),
        ("Normal(10, 2)", "random variable"),
    ],
)
def test_concurrent_load_stage_rejects_unrunnable_expressions(level: str, message: str) -> None:
    with pytest.raises(ValidationError, match=message):
        ConcurrentLoadStage(num_requests=100, concurrency_level=level)


# Pydantic does not re-validate on assignment, so the schedule must follow a
# reassigned level: 10 -> 'floor(Min(2 + t, 5))' gives initial 2 and peak 5.
def test_concurrent_load_stage_schedule_follows_reassignment() -> None:
    stage = ConcurrentLoadStage(num_requests=100, concurrency_level=10)
    stage.concurrency_level = "floor(Min(2 + t, 5))"
    assert (stage.concurrency_schedule.initial, stage.peak_concurrency) == (2, 5)


# A YAML config with an expression level loads through read_config.
# Input: a concurrent load block with concurrency_level 'floor(Min(1 + t/6, 64))'.
# Expected: the stage keeps the string and peaks at 64.
def test_concurrent_expression_loads_from_yaml(tmp_path: Path) -> None:
    path = tmp_path / "config.yml"
    path.write_text(
        "load:\n  type: concurrent\n  stages:\n    - num_requests: 200\n      concurrency_level: 'floor(Min(1 + t/6, 64))'\n"
    )
    stage = read_config(str(path)).load.stages[0]
    assert isinstance(stage, ConcurrentLoadStage)
    assert stage.concurrency_level == "floor(Min(1 + t/6, 64))"
    assert stage.peak_concurrency == 64


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


def test_trace_session_replay_stage_duration_defaults_to_none() -> None:
    assert TraceSessionReplayLoadStage(concurrent_sessions=2).duration is None


def test_trace_session_replay_stage_duration_alone_is_valid() -> None:
    # max_stage_duration is optional: duration on its own bounds load generation, and the
    # teardown grace bounds the wind-down that follows.
    stage = TraceSessionReplayLoadStage(concurrent_sessions=4, duration=1800)
    assert stage.duration == 1800
    assert stage.max_stage_duration is None


def test_trace_session_replay_stage_duration_shorter_than_max_stage_duration_is_valid() -> None:
    stage = TraceSessionReplayLoadStage(concurrent_sessions=4, duration=1800, max_stage_duration=2100)
    assert stage.duration == 1800
    assert stage.max_stage_duration == 2100


def test_trace_session_replay_stage_duration_cannot_exceed_max_stage_duration() -> None:
    # max_stage_duration would fire before the planned stop, reporting FAILED for a run that
    # asked to end.
    with pytest.raises(ValueError, match="must be shorter than max_stage_duration"):
        TraceSessionReplayLoadStage(concurrent_sessions=4, duration=2100, max_stage_duration=1800)


def test_trace_session_replay_stage_duration_cannot_equal_max_stage_duration() -> None:
    # Equal is not shorter: the two deadlines coincide and the failure path wins.
    with pytest.raises(ValueError, match="must be shorter than max_stage_duration"):
        TraceSessionReplayLoadStage(concurrent_sessions=4, duration=1800, max_stage_duration=1800)


def test_trace_session_replay_stage_duration_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        TraceSessionReplayLoadStage(concurrent_sessions=4, duration=0)


def test_trace_session_replay_stage_duration_and_num_sessions_are_exclusive() -> None:
    """Two stop conditions race, and a count-based win misreports the time window.

    The message has to say that, and not that duration "draws sessions for as long as it
    takes": only a cycling generator can do that, so the reason has to hold for every
    generator. Asserted on the wording because the wording is the whole point of the
    error -- it is what tells a user which field to remove and why.
    """
    with pytest.raises(ValueError, match="whichever stop condition is reached first ends the stage"):
        TraceSessionReplayLoadStage(concurrent_sessions=4, duration=1800, num_sessions=100)


def test_trace_session_replay_stage_duration_checked_against_deprecated_timeout() -> None:
    """The cap is still honoured under its old name, so the gap check cannot be bypassed.

    #786 renamed the field and kept `timeout` working, so a config written before that
    rename must still be rejected for leaving no gap -- otherwise the safety net fires
    first and every duration-bounded run reports FAILED.
    """
    with pytest.raises(ValueError, match="must be shorter than timeout"):
        TraceSessionReplayLoadStage(concurrent_sessions=4, duration=2100, timeout=1800)


# Inputs: session replay stages with concurrent_sessions=4 (and optional max_stage_duration/duration), no timeout.
# Expected: constructing them emits neither a DeprecationWarning nor a deprecation log warning.
def test_trace_session_replay_stage_without_timeout_does_not_warn(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="inference_perf.config.loadgen.config"):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            TraceSessionReplayLoadStage(concurrent_sessions=4)
            TraceSessionReplayLoadStage(concurrent_sessions=4, duration=1800, max_stage_duration=2100)

    deprecation_warnings = [w for w in caught if issubclass(w.category, DeprecationWarning)]
    assert deprecation_warnings == []
    assert caplog.records == []


# Input: TraceSessionReplayLoadStage(concurrent_sessions=4, timeout=30.0) using the deprecated field name.
# Expected: validation logs a deprecation warning and reading stage.timeout raises DeprecationWarning.
def test_trace_session_replay_stage_with_timeout_warns(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING, logger="inference_perf.config.loadgen.config"):
        stage = TraceSessionReplayLoadStage(concurrent_sessions=4, timeout=30.0)

    assert any("load.stages[].timeout is deprecated; use max_stage_duration instead." in r.message for r in caplog.records)
    with pytest.warns(DeprecationWarning, match="max_stage_duration"):
        assert stage.timeout == 30.0


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


# --- BatchLoadStage and the batch load type --------------------------------


# Input: load type batch with one stage of two jobs, {batch_size: 1, completion_window: 30m} and {batch_size: 3}.
# Expected: the stage is a BatchLoadStage, the second job's window defaults to 24h, and the stage polls every 5 s.
def test_batch_load_type_reads_its_jobs() -> None:
    config = LoadConfig.model_validate(
        {"type": "batch", "stages": [{"batches": [{"batch_size": 1, "completion_window": "30m"}, {"batch_size": 3}]}]}
    )

    (stage,) = config.stages
    assert isinstance(stage, BatchLoadStage)
    assert [(job.batch_size, job.completion_window) for job in stage.batches] == [(1, "30m"), (3, "24h")]
    assert stage.poll_interval_sec == 5.0


# Input: a batch stage with an empty batches list. Expected: ValidationError, since the stage would submit nothing.
def test_batch_stage_needs_at_least_one_job() -> None:
    with pytest.raises(ValidationError):
        BatchLoadStage(batches=[])


# Input: a job with batch_size 0. Expected: ValidationError, since a batch needs at least one line.
def test_batch_job_size_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        BatchJobConfig(batch_size=0)


# Input: a batch stage with poll_interval_sec 0. Expected: ValidationError, since 0 would poll the server in a busy loop.
def test_batch_stage_poll_interval_must_be_positive() -> None:
    with pytest.raises(ValidationError):
        BatchLoadStage(batches=[BatchJobConfig(batch_size=1)], poll_interval_sec=0)


# Input: load type batch with a constant-rate stage. Expected: ValueError naming the stage type batch needs.
def test_batch_load_type_requires_batch_stage() -> None:
    with pytest.raises(ValueError, match="BATCH load type requires BatchLoadStage"):
        LoadConfig(type=LoadType.BATCH, stages=[StandardLoadStage(rate=10, duration=60)])


# Input: load type batch with a geometric sweep. Expected: ValueError, since a sweep probes a request rate,
# which a batch run does not have.
def test_sweep_with_batch_is_error() -> None:
    with pytest.raises(ValueError, match="Cannot have sweep config with BATCH"):
        LoadConfig(
            type=LoadType.BATCH,
            sweep=SweepConfig(type=StageGenType.GEOM),
            stages=[BatchLoadStage(batches=[BatchJobConfig(batch_size=1)])],
        )
