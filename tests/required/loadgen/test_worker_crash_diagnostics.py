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
"""Regression tests for #593: a dying worker used to produce one context-free log line.

Before this change the only output was "A worker process died unexpectedly!" with no
worker id, no exit code and no traceback, and the run then hung on the stage barrier.
"""

import multiprocessing as mp
import os
import signal
import unittest
from typing import Any, List, Optional

import numpy as np

from inference_perf.client.modelserver.mock_client import MockModelServerClient
from inference_perf.config import (
    APIConfig,
    APIType,
    DataConfig,
    DataGenType,
    Distribution,
    LoadConfig,
    LoadType,
    StandardLoadStage,
)
from inference_perf.datagen.synthetic.random_datagen import RandomDataGenerator
from inference_perf.loadgen.load_generator import (
    LoadGenerator,
    WorkerCrash,
    WorkerFailure,
    collect_worker_failures,
)
from inference_perf.client.server_metrics.base import StageStatus
from inference_perf.metrics.request_collector import MultiprocessRequestMetricCollector
from inference_perf.utils.custom_tokenizer import CustomTokenizer


class _FakeWorker:
    # Stands in for a Worker process. is_alive()/exitcode/id are the only attributes
    # collect_worker_failures() reads, so a stub avoids spawning real processes here.
    def __init__(self, id: int, alive: bool, exitcode: Optional[int]) -> None:
        self.id = id
        self._alive = alive
        self.exitcode = exitcode

    def is_alive(self) -> bool:
        return self._alive


class _Tokenizer:
    # Space-delimited integers stand in for tokens, so "10 11 12" is 3 tokens.
    vocab_size = 1000
    all_special_ids = [1, 2, 3]

    def decode(self, tokens: List[int], **kwargs: Any) -> str:
        return " ".join(str(t) for t in tokens)

    def encode(self, text: str) -> List[int]:
        try:
            return [int(t) for t in text.split()]
        except ValueError:
            return list(range(10, 10010))


class _CustomTokenizer(CustomTokenizer):
    def __init__(self) -> None:
        pass

    def get_tokenizer(self) -> Any:
        return _Tokenizer()

    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        return len(text.split()) if text else 0


class CrashingDataGenerator(RandomDataGenerator):
    # Raises RuntimeError("induced worker crash") the first time `rng` is read in a
    # child process, and behaves normally in the parent. Worker._run() reads `rng`
    # while seeding, so this crashes the worker exactly once it has started.
    def __init__(self, *args: Any, **kwargs: Any) -> None:
        self._real_rng = np.random.default_rng(0)
        self._parent_pid = os.getpid()
        super().__init__(*args, **kwargs)

    @property
    def rng(self) -> Any:
        if os.getpid() != self._parent_pid:
            raise RuntimeError("induced worker crash")
        return self._real_rng

    @rng.setter
    def rng(self, value: Any) -> None:
        self._real_rng = value


class TestWorkerFailureDescription(unittest.TestCase):
    """The text a user sees for each way a worker can die."""

    def test_crash_with_traceback_names_worker_stage_and_exception(self) -> None:
        # A worker that raised TypeError during stage 2 with 3 requests in flight
        # must report all four facts plus the traceback body.
        failure = WorkerFailure(
            worker_id=1,
            exitcode=1,
            crash=WorkerCrash(
                worker_id=1,
                stage_id=2,
                exc_type="TypeError",
                message="cannot pickle 'generator' object",
                traceback_text="Traceback (most recent call last):\n  ...\nTypeError: nope\n",
                in_flight=3,
            ),
        )
        described = failure.describe()
        self.assertIn("Worker 1", described)
        self.assertIn("exited with code 1", described)
        self.assertIn("TypeError", described)
        self.assertIn("stage 2", described)
        # The counter behind in_flight is shared by every worker, so the text has to
        # say run-wide rather than imply the count belongs to the crashing worker.
        self.assertIn("3 request(s) in flight run-wide", described)
        self.assertIn("cannot pickle 'generator' object", described)
        self.assertIn("Traceback (most recent call last):", described)

    def test_crash_before_the_first_request_says_so_instead_of_stage_none(self) -> None:
        # A worker that dies during startup has pulled no request yet, so stage_id is
        # None. The description must read "before serving any stage" rather than the
        # bare "during stage None".
        failure = WorkerFailure(
            worker_id=0,
            exitcode=1,
            crash=WorkerCrash(
                worker_id=0,
                stage_id=None,
                exc_type="RuntimeError",
                message="induced worker crash",
                traceback_text="Traceback (most recent call last):\n  ...\n",
                in_flight=0,
            ),
        )
        described = failure.describe()
        self.assertIn("before serving any stage", described)
        self.assertNotIn("stage None", described)

    def test_signal_death_reports_the_signal_by_name(self) -> None:
        # A worker killed by SIGKILL has exitcode -9 and can never send a record,
        # so the description must name SIGKILL and say no traceback exists.
        failure = WorkerFailure(worker_id=0, exitcode=-int(signal.SIGKILL), crash=None)
        described = failure.describe()
        self.assertIn("Worker 0", described)
        self.assertIn("SIGKILL", described)
        self.assertIn("without reporting a Python traceback", described)

    def test_clean_early_exit_is_still_reported_as_a_failure(self) -> None:
        # Exit code 0 mid-stage is abnormal: the worker should have stayed alive
        # until the stop signal, so it is described rather than silently ignored.
        described = WorkerFailure(worker_id=4, exitcode=0, crash=None).describe()
        self.assertIn("Worker 4", described)
        self.assertIn("exited cleanly before the stage finished", described)


class TestCauseLabel(unittest.TestCase):
    """The bounded string that ends up as the `cause` label on inference_perf_workers_lost."""

    def test_reported_exception_class_wins(self) -> None:
        # A worker that got far enough to report names its exception class,
        # even though its exit code (1) is also available.
        failure = WorkerFailure(
            worker_id=0,
            exitcode=1,
            crash=WorkerCrash(
                worker_id=0, stage_id=0, exc_type="RuntimeError", message="boom", traceback_text="tb", in_flight=0
            ),
        )
        self.assertEqual(failure.cause_label(), "RuntimeError")

    def test_signal_death_falls_back_to_the_signal_name(self) -> None:
        # Killed at -9 with no record: SIGKILL is the only thing left to say.
        self.assertEqual(WorkerFailure(worker_id=0, exitcode=-int(signal.SIGKILL), crash=None).cause_label(), "SIGKILL")

    def test_exit_codes_fall_back_to_the_code(self) -> None:
        # No record and a positive exit code: label by the code. Exit 0 mid-stage
        # is abnormal but not a crash, so it gets its own name rather than exit_0.
        self.assertEqual(WorkerFailure(worker_id=0, exitcode=7, crash=None).cause_label(), "exit_7")
        self.assertEqual(WorkerFailure(worker_id=0, exitcode=0, crash=None).cause_label(), "clean_exit")

    def test_labels_stay_bounded(self) -> None:
        # Whatever the failure shape, the label must never carry a message or an
        # id, because it becomes a Prometheus label value.
        failure = WorkerFailure(
            worker_id=3,
            exitcode=1,
            crash=WorkerCrash(
                worker_id=3,
                stage_id=1,
                exc_type="ValueError",
                message="a very long unbounded message with an id 8f3a-11ee",
                traceback_text="tb",
                in_flight=2,
            ),
        )
        self.assertEqual(failure.cause_label(), "ValueError")
        self.assertNotIn("8f3a", failure.cause_label())


class TestCollectWorkerFailures(unittest.TestCase):
    """Pairing dead workers with the crash records drained from the queue."""

    def test_pairs_each_record_with_its_worker_and_skips_healthy_ones(self) -> None:
        # Workers 0 (alive), 1 (dead, sent a record), 2 (dead, killed by signal),
        # 3 (never started) must yield failures for 1 and 2 only, and only
        # worker 1 carries a crash record.
        queue: "mp.SimpleQueue[WorkerCrash]" = mp.SimpleQueue()
        queue.put(
            WorkerCrash(
                worker_id=1,
                stage_id=0,
                exc_type="ValueError",
                message="boom",
                traceback_text="tb",
                in_flight=1,
            )
        )
        workers = [
            _FakeWorker(0, alive=True, exitcode=None),
            _FakeWorker(1, alive=False, exitcode=1),
            _FakeWorker(2, alive=False, exitcode=-9),
            _FakeWorker(3, alive=False, exitcode=None),
        ]
        failures = collect_worker_failures(workers, queue)  # type: ignore[arg-type]

        self.assertEqual([f.worker_id for f in failures], [1, 2])
        self.assertIsNotNone(failures[0].crash)
        assert failures[0].crash is not None
        self.assertEqual(failures[0].crash.exc_type, "ValueError")
        self.assertIsNone(failures[1].crash)

    def test_no_queue_still_reports_exit_codes(self) -> None:
        # With no crash channel at all, a dead worker must still be reported using
        # its exit code alone rather than disappearing.
        failures = collect_worker_failures([_FakeWorker(0, alive=False, exitcode=3)], None)  # type: ignore[list-item]
        self.assertEqual(len(failures), 1)
        self.assertEqual(failures[0].exitcode, 3)
        self.assertIsNone(failures[0].crash)


class TestWorkerCrashEndToEnd(unittest.IsolatedAsyncioTestCase):
    """A real worker process crashes and the parent explains it without hanging."""

    def setUp(self) -> None:
        # Worker._run() reads the datagen's `rng` in the child; under fork the
        # object is inherited so the crash lands inside the worker. Python 3.14
        # defaults to forkserver on Linux, where the same object is pickled and
        # the failure moves into process startup instead (see #526).
        self._previous_start_method = mp.get_start_method()
        try:
            mp.set_start_method("fork", force=True)
        except RuntimeError:
            self.skipTest("fork start method unavailable on this platform")

    def tearDown(self) -> None:
        mp.set_start_method(self._previous_start_method, force=True)

    async def test_crash_is_attributed_and_the_stage_is_failed(self) -> None:
        # One worker, two stages, and a datagen that raises as soon as the worker
        # starts. mp_run must return instead of hanging, must record worker 0 with
        # exit code 1 and the RuntimeError traceback, and must mark the stage FAILED.
        # The worker is respawned between stages and the run continues, so the
        # second stage runs and fails the same way: two failures, two failed stages.
        api_config = APIConfig(type=APIType.Completion, streaming=False)
        data_config = DataConfig(
            type=DataGenType.Random,
            input_distribution=Distribution(min=10, max=10, mean=10.0, std_dev=0.0, total_count=4),
            output_distribution=Distribution(min=5, max=5, mean=5.0, std_dev=0.0, total_count=4),
        )
        datagen = CrashingDataGenerator(api_config, data_config, _CustomTokenizer())

        collector = MultiprocessRequestMetricCollector()
        client = MockModelServerClient(collector, api_config, mock_latency=0)
        load_config = LoadConfig(
            type=LoadType.CONSTANT,
            num_workers=1,
            worker_max_concurrency=4,
            stages=[StandardLoadStage(rate=4, duration=1), StandardLoadStage(rate=4, duration=1)],
            base_seed=42,
        )
        load_gen = LoadGenerator(datagen, load_config)

        async with collector.start():
            await load_gen.mp_run(client)

        self.assertEqual(len(load_gen.worker_failures), 2, "each stage must record the worker it lost")
        for failure in load_gen.worker_failures:
            self.assertEqual(failure.worker_id, 0)
            self.assertEqual(failure.exitcode, 1)
            self.assertIsNotNone(failure.crash, "the worker must report its traceback before dying")
            assert failure.crash is not None
            self.assertEqual(failure.crash.exc_type, "RuntimeError")
            self.assertIn("induced worker crash", failure.crash.traceback_text)
            self.assertIn("Worker 0", failure.describe())

        # Losing a worker fails the stage it happened in. The run itself continues,
        # because main respawns the dead worker at the stage boundary.
        self.assertEqual(len(load_gen.stage_runtime_info), 2, "both stages must run")
        for info in load_gen.stage_runtime_info.values():
            self.assertEqual(info.status, StageStatus.FAILED)


if __name__ == "__main__":
    unittest.main()
