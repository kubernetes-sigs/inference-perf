import unittest
from unittest.mock import MagicMock
import multiprocessing as mp
import asyncio
import typing

# Patch asyncio.TaskGroup for Python < 3.11
if not hasattr(asyncio, 'TaskGroup'):
    class MockTaskGroup:
        async def __aenter__(self):
            return self
        async def __aexit__(self, exc_type, exc_val, exc_tb):
            pass
        def create_task(self, coro):
            return asyncio.create_task(coro)
    asyncio.TaskGroup = MockTaskGroup

# Patch typing.TypeAlias for Python < 3.10
if not hasattr(typing, 'TypeAlias'):
    typing.TypeAlias = typing.Any

from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.config import LoadConfig, LoadType

class MockWorker:
    def __init__(self, id, shared_max_concurrency):
        self.id = id
        self.shared_max_concurrency = shared_max_concurrency

class TestLoadGeneratorConcurrency(unittest.TestCase):
    def setUp(self):
        self.mock_datagen = MagicMock()
        self.load_config = LoadConfig(
            type=LoadType.CONCURRENT,
            num_workers=4,
            worker_max_concurrency=100
        )
        # Mocking get_circuit_breaker since LoadGenerator init calls it
        with unittest.mock.patch('inference_perf.loadgen.load_generator.get_circuit_breaker'):
            self.load_generator = LoadGenerator(self.mock_datagen, self.load_config)

    def test_set_worker_concurrency_divisible(self):
        # Setup workers
        self.load_generator.workers = []
        for i in range(4):
            shared_val = mp.Value('i', 0)
            self.load_generator.workers.append(MockWorker(i, shared_val))

        # Test concurrency_level = 8 (8 / 4 = 2 per worker)
        self.load_generator._set_worker_concurrency(8)
        
        for worker in self.load_generator.workers:
            self.assertEqual(worker.shared_max_concurrency.value, 2, f"Worker {worker.id} should have concurrency 2")

    def test_set_worker_concurrency_remainder(self):
        # Setup workers
        self.load_generator.workers = []
        for i in range(4):
            shared_val = mp.Value('i', 0)
            self.load_generator.workers.append(MockWorker(i, shared_val))

        # Test concurrency_level = 10 (10 // 4 = 2, 10 % 4 = 2)
        # Workers 0, 1 should have 3
        # Workers 2, 3 should have 2
        self.load_generator._set_worker_concurrency(10)
        
        self.assertEqual(self.load_generator.workers[0].shared_max_concurrency.value, 3)
        self.assertEqual(self.load_generator.workers[1].shared_max_concurrency.value, 3)
        self.assertEqual(self.load_generator.workers[2].shared_max_concurrency.value, 2)
        self.assertEqual(self.load_generator.workers[3].shared_max_concurrency.value, 2)

    def test_set_worker_concurrency_less_than_workers(self):
        # Setup workers
        self.load_generator.workers = []
        for i in range(4):
            shared_val = mp.Value('i', 0)
            self.load_generator.workers.append(MockWorker(i, shared_val))

        # Test concurrency_level = 3
        # Workers 0, 1, 2 should have 1
        # Worker 3 should have 0
        self.load_generator._set_worker_concurrency(3)
        
        self.assertEqual(self.load_generator.workers[0].shared_max_concurrency.value, 1)
        self.assertEqual(self.load_generator.workers[1].shared_max_concurrency.value, 1)
        self.assertEqual(self.load_generator.workers[2].shared_max_concurrency.value, 1)
        self.assertEqual(self.load_generator.workers[3].shared_max_concurrency.value, 0)

if __name__ == '__main__':
    unittest.main()
