import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Any
import asyncio
import sys
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.config import LoadConfig, LoadType, SweepConfig, StageGenType
from inference_perf.client.modelserver import ModelServerClient

# Patch asyncio.TaskGroup for Python < 3.11 if needed (same as test_load_generator.py)
if sys.version_info < (3, 11):

    class MockTaskGroup:
        async def __aenter__(self) -> "MockTaskGroup":
            return self

        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

        def create_task(self, coro: Any) -> Any:
            return asyncio.create_task(coro)

    asyncio.TaskGroup = MockTaskGroup

# Patch typing.TypeAlias for Python < 3.10 if needed
if sys.version_info < (3, 10):
    import typing

    typing.TypeAlias = typing.Any


class TestLoadGeneratorPreprocess(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.mock_datagen = MagicMock()
        # Ensure datagen.trace is None to avoid attribute error in run_stage if it checks for trace
        self.mock_datagen.trace = None
        self.mock_datagen.get_request_count.return_value = 1000  # Enough requests for the test
        self.mock_datagen.get_data.return_value = iter(
            [MagicMock(prefered_worker_id=-1) for _ in range(1000)]
        )  # Mock data generator

        self.sweep_config = SweepConfig(
            type=StageGenType.LINEAR,
            num_requests=50,
            num_stages=3,
            stage_duration=2,
            timeout=10,  # Sufficient timeout
        )
        self.load_config = LoadConfig(
            type=LoadType.CONSTANT,
            num_workers=1,
            worker_max_concurrency=10,
            sweep=self.sweep_config,
            stages=[],  # Start empty, preprocess should fill this
        )

        # Patch get_circuit_breaker as it is called in __init__
        with patch("inference_perf.loadgen.load_generator.get_circuit_breaker"):
            self.load_generator = LoadGenerator(self.mock_datagen, self.load_config)

    @patch("inference_perf.loadgen.load_generator.time.perf_counter")
    @patch("inference_perf.loadgen.load_generator.sleep")
    async def test_preprocess_success(self, mock_sleep: Any, mock_perf_counter: Any) -> None:
        client = MagicMock(spec=ModelServerClient)
        request_queue = MagicMock()
        active_requests_counter = MagicMock()
        active_requests_counter.value = 0
        finished_requests_counter = MagicMock()
        request_phase = MagicMock()
        cancel_signal = MagicMock()

        # Configure Sweep for 3 steps, max 60 QPS
        self.sweep_config.num_requests = 60
        self.sweep_config.num_stages = 3

        # We need to simulate time passing.
        # aggregator sleeps 0.5s.
        # run_stage runs for 10s.

        # We can control time by side_effect of perf_counter
        # But we need it to be consistent with sleep calls.

        current_time = 0.0

        def get_time() -> float:
            nonlocal current_time
            return current_time

        mock_perf_counter.side_effect = get_time

        # When sleep is called, we behavior depends on who calls it.
        # If mock_run_stage calls it, we ADVANCE time.
        # If aggregator calls it, we WAIT for time.

        import inspect

        async def fake_sleep(seconds: float) -> None:
            nonlocal current_time
            # Check who called us
            # Stack: [0]=fake_sleep, [1]=caller (e.g. aggregator or mock_run_stage/wrapper)
            # Since mock_run_stage is a mock side effect, stack might be deeper or different.
            # But aggregator is a defined function in preprocess.

            # Helper to check if called by aggregator
            is_aggregator = False
            for frame in inspect.stack():
                if frame.function == "aggregator":
                    is_aggregator = True
                    break

            if is_aggregator:
                wake_time = current_time + seconds
                while current_time < wake_time:
                    await asyncio.sleep(0)
            else:
                # Driver (mock_run_stage)
                current_time += seconds
                await asyncio.sleep(0)

        mock_sleep.side_effect = fake_sleep

        # We also need to map rates to throughput
        # Step 1: 20 -> 20
        # Step 2: 40 -> 40
        # Step 3: 60 -> 45

        async def mock_run_stage(stage_id: int, rate: float, duration: float, *args: Any, **kwargs: Any) -> None:
            nonlocal current_time
            target_throughput = min(rate, 60.0)

            # Simulate the stage running in chunks
            # We want aggregator to sample sufficient points.
            # Aggregator samples every 0.5s.
            # Duration is 10s.
            # We can advance time in 0.5s increments

            with finished_requests_counter.get_lock():
                finished_requests_counter.value = 0

            n_steps = int(duration / 0.5)
            for _ in range(n_steps):
                await fake_sleep(0.5)
                # update requests linearly
                # finished requests = throughput * time_elapsed
                # But we reset at start of run_stage.
                # counter should be total finished since start of stage.
                # Since we reset counter.value = 0 at start:
                time_in_stage = _ * 0.5 + 0.5
                with finished_requests_counter.get_lock():
                    finished_requests_counter.value = int(target_throughput * time_in_stage)

        self.load_generator.run_stage = AsyncMock(side_effect=mock_run_stage)  # type: ignore[method-assign]

        finished_requests_counter.get_lock = MagicMock()
        finished_requests_counter.get_lock.return_value.__enter__ = MagicMock()
        finished_requests_counter.get_lock.return_value.__exit__ = MagicMock()
        finished_requests_counter.value = 0

        await self.load_generator.preprocess(
            client, request_queue, active_requests_counter, finished_requests_counter, request_phase, cancel_signal
        )

        # Check if stages were generated
        self.assertEqual(len(self.load_generator.stages), 3)
        self.assertTrue(all(s.rate is not None and s.rate > 0 for s in self.load_generator.stages))

        # Verify run_stage was called 11 times (Probe, Exp growth, Binary refinement)
        # Probe: 1.0
        # Exp: 2.0, 4.0, 8.0, 16.0, 32.0, 64.0 (Safe), 128.0 (Sat) -> 7 steps
        # Bin: 96 (Sat), 80 (Sat), 72 (Sat) -> 3 steps
        # Total: 1 + 7 + 3 = 11
        self.assertEqual(self.load_generator.run_stage.call_count, 11)

        # Max rate should be ~115 (64.0 * 1.8)
        rates = [s.rate for s in self.load_generator.stages if s.rate is not None]
        max_rate_in_stages = max(rates) if rates else 0.0
        self.assertAlmostEqual(max_rate_in_stages, 115.2, delta=5.0)

    async def test_preprocess_failure_no_data(self) -> None:
        client = MagicMock(spec=ModelServerClient)
        request_queue = MagicMock()
        active_requests_counter = MagicMock()
        active_requests_counter.value = 0
        finished_requests_counter = MagicMock()
        request_phase = MagicMock()
        cancel_signal = MagicMock()

        finished_requests_counter.value = 0
        finished_requests_counter.get_lock = MagicMock()
        finished_requests_counter.get_lock.return_value.__enter__ = MagicMock()
        finished_requests_counter.get_lock.return_value.__exit__ = MagicMock()

        # Simulate NO throughput (finished requests stays 0)
        async def mock_run_stage(*args: Any, **kwargs: Any) -> None:
            await asyncio.sleep(0.6)  # Enough for aggregator to sample
            # value remains 0

        self.load_generator.run_stage = AsyncMock(side_effect=mock_run_stage)  # type: ignore[method-assign]

        # Should raise exception now
        with self.assertRaisesRegex(Exception, "Loadgen preprocessing failed"):
            await self.load_generator.preprocess(
                client, request_queue, active_requests_counter, finished_requests_counter, request_phase, cancel_signal
            )

    async def test_preprocess_insufficient_samples_skip(self) -> None:
        client = MagicMock(spec=ModelServerClient)
        request_queue = MagicMock()
        active_requests_counter = MagicMock()
        active_requests_counter.value = 0
        finished_requests_counter = MagicMock()
        request_phase = MagicMock()
        cancel_signal = MagicMock()

        finished_requests_counter.value = 0
        finished_requests_counter.get_lock = MagicMock()
        finished_requests_counter.get_lock.return_value.__enter__ = MagicMock()
        finished_requests_counter.get_lock.return_value.__exit__ = MagicMock()

        # Simulate very short run, skipped step
        async def mock_run_stage(*args: Any, **kwargs: Any) -> None:
            await asyncio.sleep(0.0)  # Instant return

        self.load_generator.run_stage = AsyncMock(side_effect=mock_run_stage)  # type: ignore[method-assign]

        # Should raise exception
        with self.assertRaisesRegex(Exception, "Loadgen preprocessing failed"):
            await self.load_generator.preprocess(
                client, request_queue, active_requests_counter, finished_requests_counter, request_phase, cancel_signal
            )


if __name__ == "__main__":
    unittest.main()
