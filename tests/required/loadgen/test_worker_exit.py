# Copyright 2026 The Kubernetes Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Unit tests for the bounded wait on worker exit at the end of a run (#913)."""

import logging
import time
from typing import List, Optional
from unittest.mock import Mock

import pytest

import inference_perf.loadgen.load_generator as lg_module
from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType, LoadConfig, LoadType, StandardLoadStage
from inference_perf.datagen import MockDataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator, Worker


# A LoadGenerator whose workers are the given stubs; nothing is started.
def _loadgen_with(workers: List[Worker]) -> LoadGenerator:
    api_config = APIConfig(type=APIType.Completion)
    datagen = MockDataGenerator(api_config, DataConfig(type=DataGenType.Mock), None)
    load_config = LoadConfig(type=LoadType.CONSTANT, stages=[StandardLoadStage(rate=1, duration=1)], num_workers=len(workers))
    loadgen = LoadGenerator(datagen, load_config)
    loadgen.workers = workers
    return loadgen


# A worker stub whose is_alive returns the given values in order, then repeats the last one.
# Each answer is also appended to calls when a list is given.
def _worker(*alive: bool, calls: Optional[List[bool]] = None) -> Worker:
    values = list(alive)

    def is_alive() -> bool:
        answer = values.pop(0) if len(values) > 1 else values[0]
        if calls is not None:
            calls.append(answer)
        return answer

    worker = Mock(spec=Worker)
    worker.is_alive.side_effect = is_alive
    return worker


# Two workers that have already exited: the wait returns at once and logs nothing.
async def test_returns_at_once_when_every_worker_has_exited(caplog: pytest.LogCaptureFixture) -> None:
    loadgen = _loadgen_with([_worker(False), _worker(False)])
    start = time.monotonic()
    with caplog.at_level(logging.WARNING):
        await loadgen._await_workers_exit()
    assert time.monotonic() - start < 0.5
    assert "still running" not in caplog.text


# A worker alive for two checks, then gone: the wait returns once it exits, well before a 5 s deadline, and logs nothing.
async def test_returns_when_the_last_worker_exits(monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture) -> None:
    monkeypatch.setattr(lg_module, "WORKER_EXIT_TIMEOUT_SEC", 5.0)
    calls: List[bool] = []
    loadgen = _loadgen_with([_worker(True, True, False, calls=calls)])
    start = time.monotonic()
    with caplog.at_level(logging.WARNING):
        await loadgen._await_workers_exit()
    assert time.monotonic() - start < 1.0
    assert "still running" not in caplog.text
    # The wait polled until the worker reported it had exited, not just once.
    assert calls == [True, True, False]


# One exited worker and one that never exits, with a 0.2 s deadline: the wait gives up near 0.2 s and warns about 1 worker.
async def test_gives_up_at_the_deadline_and_counts_live_workers(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(lg_module, "WORKER_EXIT_TIMEOUT_SEC", 0.2)
    loadgen = _loadgen_with([_worker(False), _worker(True)])
    start = time.monotonic()
    with caplog.at_level(logging.WARNING):
        await loadgen._await_workers_exit()
    elapsed = time.monotonic() - start
    assert 0.2 <= elapsed < 1.0
    assert "1 worker(s) still running" in caplog.text


# A deadline that has already passed and a worker that never exits: the wait warns instead of failing.
async def test_warns_when_the_deadline_has_already_passed(
    monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture
) -> None:
    monkeypatch.setattr(lg_module, "WORKER_EXIT_TIMEOUT_SEC", 0.0)
    loadgen = _loadgen_with([_worker(True)])
    with caplog.at_level(logging.WARNING):
        await loadgen._await_workers_exit()
    assert "1 worker(s) still running" in caplog.text
