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
import unittest
from unittest.mock import MagicMock
import multiprocessing as mp
import asyncio
import typing
import sys
from typing import Any

# Patch asyncio.TaskGroup for Python < 3.11
if sys.version_info < (3, 11):

    class MockTaskGroup:
        async def __aenter__(self) -> "MockTaskGroup":
            return self

        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

        def create_task(self, coro: Any) -> Any:
            return asyncio.create_task(coro)

    asyncio.TaskGroup = MockTaskGroup

# Patch typing.TypeAlias for Python < 3.10
if sys.version_info < (3, 10):
    typing.TypeAlias = typing.Any

from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.config import LoadConfig, LoadType
from inference_perf.datagen import DataGenerator


class MockWorker:
    def __init__(self, id: int, shared_max_concurrency: Any) -> None:
        self.id = id
        self.shared_max_concurrency = shared_max_concurrency

    def is_alive(self) -> bool:
        return True


class TestLoadGeneratorConcurrency(unittest.TestCase):
    def setUp(self) -> None:
        self.mock_datagen = MagicMock(spec=DataGenerator)
        self.load_config = LoadConfig(type=LoadType.CONCURRENT, num_workers=4, worker_max_concurrency=100)
        # Mocking get_circuit_breaker since LoadGenerator init calls it
        with unittest.mock.patch("inference_perf.loadgen.load_generator.get_circuit_breaker"):
            self.load_generator = LoadGenerator(self.mock_datagen, self.load_config)

    def test_set_worker_concurrency_divisible(self) -> None:
        # Setup workers
        self.load_generator.workers = []
        for i in range(4):
            shared_val = mp.Value("i", 0)
            self.load_generator.workers.append(MockWorker(i, shared_val))  # type: ignore

        # Test concurrency_level = 8 (8 / 4 = 2 per worker)
        self.load_generator._set_worker_concurrency(8)

        for worker in self.load_generator.workers:
            self.assertEqual(worker.shared_max_concurrency.value, 2, f"Worker {worker.id} should have concurrency 2")  # type: ignore

    def test_set_worker_concurrency_remainder(self) -> None:
        # Setup workers
        self.load_generator.workers = []
        for i in range(4):
            shared_val = mp.Value("i", 0)
            self.load_generator.workers.append(MockWorker(i, shared_val))  # type: ignore

        # Test concurrency_level = 10 (10 // 4 = 2, 10 % 4 = 2)
        # Workers 0, 1 should have 3
        # Workers 2, 3 should have 2
        self.load_generator._set_worker_concurrency(10)

        self.assertEqual(self.load_generator.workers[0].shared_max_concurrency.value, 3)  # type: ignore
        self.assertEqual(self.load_generator.workers[1].shared_max_concurrency.value, 3)  # type: ignore
        self.assertEqual(self.load_generator.workers[2].shared_max_concurrency.value, 2)  # type: ignore
        self.assertEqual(self.load_generator.workers[3].shared_max_concurrency.value, 2)  # type: ignore

    def test_set_worker_concurrency_less_than_workers(self) -> None:
        # Setup workers
        self.load_generator.workers = []
        for i in range(4):
            shared_val = mp.Value("i", 0)
            self.load_generator.workers.append(MockWorker(i, shared_val))  # type: ignore

        # Test concurrency_level = 3
        # Workers 0, 1 should have 1
        # Worker 3 should have 0
        self.load_generator._set_worker_concurrency(3)

        self.assertEqual(self.load_generator.workers[0].shared_max_concurrency.value, 1)  # type: ignore
        self.assertEqual(self.load_generator.workers[1].shared_max_concurrency.value, 1)  # type: ignore
        self.assertEqual(self.load_generator.workers[2].shared_max_concurrency.value, 1)  # type: ignore
        self.assertEqual(self.load_generator.workers[3].shared_max_concurrency.value, 0)  # type: ignore

    @unittest.mock.patch("inference_perf.loadgen.load_generator.sleep", new_callable=unittest.mock.AsyncMock)
    def test_run_stage_dynamic_concurrency(self, mock_sleep: Any) -> None:
        # Setup workers
        self.load_generator.workers = []
        for i in range(4):
            shared_val = mp.Value("i", 0)
            self.load_generator.workers.append(MockWorker(i, shared_val))  # type: ignore

        # Mock _set_worker_concurrency to record calls
        patcher1 = unittest.mock.patch.object(self.load_generator, "_set_worker_concurrency")
        mock_set_concurrency = patcher1.start()
        self.addCleanup(patcher1.stop)

        # Mock arguments for run_stage
        request_queue = MagicMock()
        active_requests_counter = MagicMock()
        finished_requests_counter = MagicMock()
        finished_requests_counter.get_lock.return_value.__enter__.return_value = None

        # Simulate finished_requests_counter.value changing to exit loop
        # We need enough values for setter and getter calls!
        type(finished_requests_counter).value = unittest.mock.PropertyMock(side_effect=[0, 0, 10, 10])

        request_phase = MagicMock()

        # Mock datagen and timer
        self.mock_datagen.trace = None
        self.mock_datagen.get_data.return_value = iter([MagicMock(preferred_worker_id=-1)] * 10)

        mock_timer = MagicMock()
        mock_timer.start_timer.return_value = iter([0.0] * 10)
        patcher2 = unittest.mock.patch.object(self.load_generator, "get_timer", return_value=mock_timer)
        patcher2.start()
        self.addCleanup(patcher2.stop)

        async def run() -> None:
            await self.load_generator.run_stage(
                stage_id=0,
                interval=1.0,
                duration=10,
                request_queue=request_queue,
                active_requests_counter=active_requests_counter,
                finished_requests_counter=finished_requests_counter,
                request_phase=request_phase,
                concurrency_level="10 + t",
            )

        asyncio.run(run())

        # Verify _set_worker_concurrency was called
        mock_set_concurrency.assert_called()


if __name__ == "__main__":
    unittest.main()
