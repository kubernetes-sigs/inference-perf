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
"""Validation of load.open_loop and load.hard_stop."""

import pytest

from inference_perf.config import (
    ConcurrentLoadStage,
    LoadConfig,
    LoadType,
    StandardLoadStage,
    TraceSessionReplayLoadStage,
)


# The defaults keep the historical behavior: bounded dispatch and drain at stage end.
def test_defaults_leave_behavior_unchanged() -> None:
    config = LoadConfig(stages=[StandardLoadStage(rate=1, duration=1)])

    assert config.open_loop is False
    assert config.hard_stop is False


# open_loop is accepted for constant, poisson and trace_replay loads.
@pytest.mark.parametrize("load_type", [LoadType.CONSTANT, LoadType.POISSON, LoadType.TRACE_REPLAY])
def test_open_loop_accepted_for_rate_based_loads(load_type: LoadType) -> None:
    config = LoadConfig(type=load_type, stages=[StandardLoadStage(rate=1, duration=1)], open_loop=True)

    assert config.open_loop is True


# open_loop is rejected for concurrent loads (closed loop by definition) and for
# trace_session_replay (dependency-driven dispatch).
def test_open_loop_rejected_for_concurrent_and_session_replay() -> None:
    with pytest.raises(ValueError, match="open_loop is not supported for the CONCURRENT"):
        LoadConfig(
            type=LoadType.CONCURRENT,
            stages=[ConcurrentLoadStage(num_requests=10, concurrency_level=2)],
            open_loop=True,
        )
    with pytest.raises(ValueError, match="open_loop is not supported for the TRACE_SESSION_REPLAY"):
        LoadConfig(
            type=LoadType.TRACE_SESSION_REPLAY,
            stages=[TraceSessionReplayLoadStage(concurrent_sessions=1)],
            open_loop=True,
        )
