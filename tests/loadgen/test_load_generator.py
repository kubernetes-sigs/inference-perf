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
from unittest.mock import MagicMock, AsyncMock, patch
import multiprocessing as mp
import asyncio
import typing
import sys
import numpy as np
from typing import Any

from inference_perf.loadgen.load_generator import LoadGenerator, RequestQueueData
from inference_perf.config import LoadConfig, LoadType, TraceConfig, TraceFormat, StandardLoadStage
from inference_perf.client.modelserver import ModelServerClient
from inference_perf.apis import InferenceAPIData
from inference_perf.utils.request_queue import RequestQueue

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

from inference_perf.datagen import DataGenerator


class MockWorker:
    def __init__(self, id: int, shared_max_concurrency: Any) -> None:
        self.id = id
        self.shared_max_concurrency = shared_max_concurrency


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


class TestLoadGenerator(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.mock_datagen = MagicMock(spec=DataGenerator)
        self.mock_datagen.get_data.return_value = iter([MagicMock(preferred_worker_id=-1) for _ in range(100)])
        self.mock_datagen.trace = None

        self.mock_client = AsyncMock(spec=ModelServerClient)

        self.load_config = LoadConfig(
            type=LoadType.CONSTANT,
            num_workers=2,
            worker_max_concurrency=10,
            interval=1,
            stages=[StandardLoadStage(rate=10, duration=1)],
            circuit_breakers=[],
            base_seed=42,
        )

        with patch("inference_perf.loadgen.load_generator.get_circuit_breaker"):
            self.load_generator = LoadGenerator(self.mock_datagen, self.load_config)

    def test_get_lora_adapter(self) -> None:
        # No config
        self.assertIsNone(self.load_generator._get_lora_adapter())

        # With config
        self.load_generator.lora_adapters = ["adapter1", "adapter2"]
        self.load_generator.lora_weights = [1.0, 0.0]

        np.random.seed(42)  # For reproducibility
        self.assertEqual(self.load_generator._get_lora_adapter(), "adapter1")

        self.load_generator.lora_weights = [0.0, 1.0]
        self.assertEqual(self.load_generator._get_lora_adapter(), "adapter2")

    def test_get_timer(self) -> None:
        from inference_perf.loadgen.load_timer import ConstantLoadTimer, PoissonLoadTimer, TraceReplayLoadTimer

        # Constant
        self.load_generator.load_type = LoadType.CONSTANT
        timer = self.load_generator.get_timer(10, 5)
        self.assertIsInstance(timer, ConstantLoadTimer)

        # Poisson
        self.load_generator.load_type = LoadType.POISSON
        timer = self.load_generator.get_timer(10, 5)
        self.assertIsInstance(timer, PoissonLoadTimer)

        # Trace (Needs trace config setup)
        self.load_generator.load_type = LoadType.TRACE_REPLAY
        self.load_generator.trace = TraceConfig(format=TraceFormat.AZURE_PUBLIC_DATASET, file="dummy.csv")
        self.load_generator.trace_reader = MagicMock()
        timer = self.load_generator.get_timer(10, 5)
        self.assertIsInstance(timer, TraceReplayLoadTimer)

    @patch("inference_perf.loadgen.load_generator.sleep", new_callable=AsyncMock)
    @patch("inference_perf.loadgen.load_generator.time")
    async def test_run_stage_timeout(self, mock_time: MagicMock, mock_sleep: AsyncMock) -> None:
        mock_time.time.return_value = 1000
        mock_time.perf_counter.side_effect = [0, 10, 20]  # simulate time passing

        request_queue = MagicMock(spec=RequestQueue)
        request_queue.drain = MagicMock()
        active_counter = MagicMock()
        active_counter.value = 0
        finished_counter = MagicMock()
        finished_counter.value = 0
        request_phase = MagicMock()
        cancel_signal = MagicMock()

        # Force a timeout
        mock_sleep.return_value = None

        # We need a timer that generates times in the future so the queue loop doesn't instantly finish
        mock_timer = MagicMock()
        mock_timer.start_timer.return_value = iter([10.0] * 10)
        with patch.object(self.load_generator, "get_timer", return_value=mock_timer):
            await self.load_generator.run_stage(
                stage_id=0,
                rate=10,
                duration=1,
                request_queue=request_queue,
                active_requests_counter=active_counter,
                finished_requests_counter=finished_counter,
                request_phase=request_phase,
                cancel_signal=cancel_signal,
                timeout=5.0,
            )

        # The timeout logic sets the cancel signal
        cancel_signal.set.assert_called_once()
        self.assertEqual(self.load_generator.stage_runtime_info[0].status.name, "FAILED")

    @patch("inference_perf.loadgen.load_generator.sleep", new_callable=AsyncMock)
    async def test_run_single_worker_mode(self, mock_sleep: AsyncMock) -> None:
        # Setup for num_workers = 0 (single worker loop)
        self.load_generator.num_workers = 0
        self.load_config.num_workers = 0
        self.load_generator.stages = [StandardLoadStage(rate=2, duration=1)]

        mock_data = MagicMock(spec=InferenceAPIData)
        mock_data.preferred_worker_id = -1
        self.mock_datagen.get_data.return_value = iter([mock_data, mock_data])

        # Patch LazyLoadDataMixin to just return the object
        with (
            patch("inference_perf.loadgen.load_generator.LazyLoadDataMixin.get_request", return_value=mock_data),
            patch("inference_perf.loadgen.load_generator.TaskGroup") as MockTaskGroup,
            patch("inference_perf.loadgen.load_generator.time.perf_counter", return_value=0.0),
        ):
            mock_tg = MagicMock()
            MockTaskGroup.return_value.__aenter__.return_value = mock_tg

            async def dummy_process_request(*args: Any, **kwargs: Any) -> None:
                pass

            self.mock_client.process_request = AsyncMock(side_effect=dummy_process_request)

            mock_timer = MagicMock()
            mock_timer.start_timer.return_value = iter([0.1, 0.2])
            with patch.object(self.load_generator, "get_timer", return_value=mock_timer):
                await self.load_generator.run(self.mock_client)

            self.assertEqual(mock_tg.create_task.call_count, 2)
            self.assertEqual(self.load_generator.stage_runtime_info[0].status.name, "COMPLETED")

    async def test_drain(self) -> None:
        q: RequestQueue[RequestQueueData] = RequestQueue(1)
        dummy_data = MagicMock(spec=InferenceAPIData)
        q.put(RequestQueueData(0, dummy_data, 0.0, None), 0)
        q.put(RequestQueueData(0, dummy_data, 0.0, None), 0)

        # Small sleep to let queue populate
        await asyncio.sleep(0.1)

        # Add tasks to queue so task_done doesn't fail
        await self.load_generator.drain(q.get_channel(0))
        self.assertTrue(q.get_channel(0).empty())

    def test_sigint_handler(self) -> None:
        import signal

        self.assertFalse(self.load_generator.interrupt_sig)
        self.load_generator._sigint_handler(signal.SIGINT, None)
        self.assertTrue(self.load_generator.interrupt_sig)


if __name__ == "__main__":
    unittest.main()
