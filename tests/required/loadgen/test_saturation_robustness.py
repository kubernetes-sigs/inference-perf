import unittest
from unittest.mock import MagicMock, AsyncMock, patch
import asyncio
import sys
import inspect
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.datagen.random_datagen import RandomDataGenerator
from inference_perf.config import LoadConfig, LoadType, SweepConfig, StageGenType, StandardLoadStage, ConcurrentLoadStage
from inference_perf.client.modelserver import ModelServerClient
from inference_perf.circuit_breaker import CircuitBreaker

# Compatibility patches
from typing import Any

if sys.version_info < (3, 11):

    class MockTaskGroup:
        async def __aenter__(self) -> "MockTaskGroup":
            return self

        async def __aexit__(self, exc_type: Any, exc_val: Any, exc_tb: Any) -> None:
            pass

        def create_task(self, coro: Any) -> Any:
            return asyncio.create_task(coro)

    asyncio.TaskGroup = MockTaskGroup

if sys.version_info < (3, 10):
    import typing

    typing.TypeAlias = typing.Any


class TestSaturationRobustness(unittest.IsolatedAsyncioTestCase):
    async def asyncSetUp(self) -> None:
        self.mock_datagen = MagicMock(spec=RandomDataGenerator)
        self.mock_datagen.trace = None
        self.mock_datagen.get_request_count.return_value = 1000
        self.mock_datagen.get_data.return_value = iter(
            [MagicMock(prefered_worker_id=-1) for _ in range(10000)]
        )  # Plenty of data

        self.sweep_config = SweepConfig(
            type=StageGenType.LINEAR,
            num_requests=100,  # Target max QPS
            num_stages=5,  # Steps: 20, 40, 60, 80, 100
            stage_duration=5,
            timeout=10,
        )
        self.load_config = LoadConfig(
            type=LoadType.CONSTANT, num_workers=1, worker_max_concurrency=100, sweep=self.sweep_config, stages=[]
        )

        self.circuit_breaker = MagicMock(spec=CircuitBreaker)
        self.circuit_breaker.is_open.return_value = False
        self.circuit_breaker.name = "mock_cb"

        with patch("inference_perf.loadgen.load_generator.get_circuit_breaker", return_value=self.circuit_breaker):
            self.load_generator = LoadGenerator(self.mock_datagen, self.load_config)

    @patch("inference_perf.loadgen.load_generator.time.perf_counter")
    @patch("inference_perf.loadgen.load_generator.sleep")
    async def run_preprocess_with_simulation(
        self, behavior_func: Any, mock_sleep: Any, mock_perf_counter: Any, simulated_latency: float = 1.0
    ) -> None:
        """
        Helper to run preprocess with a simulated server behavior.
        behavior_func(rate) -> achieved_throughput
        """
        client = MagicMock(spec=ModelServerClient)
        request_queue = MagicMock()
        active_requests_counter = MagicMock()
        finished_requests_counter = MagicMock()
        request_phase = MagicMock()
        cancel_signal = MagicMock()

        current_time = 0.0

        def get_time() -> float:
            nonlocal current_time
            return current_time

        mock_perf_counter.side_effect = get_time

        async def fake_sleep(seconds: float) -> None:
            nonlocal current_time
            # Check stack for aggregator
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
                current_time += seconds
                await asyncio.sleep(0)

        mock_sleep.side_effect = fake_sleep

        async def mock_run_stage(stage_id: int, rate: float, duration: float, *args: Any, **kwargs: Any) -> None:
            nonlocal current_time
            # check CB in run_stage loop
            # logic: if CB open, exit early.
            # loadgen checks CB inside the loop.
            # We need to simulate that check if we want to test correct exit.
            # But here we are mocking run_stage.
            # So we must reproduce the CB check logic IF we want to test it via run_stage side effect.
            # OR we rely on loadgen.preprocess calling run_stage, and if run_stage returns, it checks measurement.

            # If behavior_func returns None, it means "crash/stop" or CB trip?
            # Let's say behavior_func determines throughput.

            achieved_throughput = behavior_func(rate, duration)
            if achieved_throughput is None:
                # Simulate CB trip or abort
                return

            # Simulate initial latency before any requests strictly finish
            await fake_sleep(simulated_latency)

            start_generating_time = current_time
            end_generating_time = start_generating_time + duration
            import math

            sent_requests = math.ceil(rate * duration)
            target_total_requests = math.ceil(achieved_throughput * duration)
            if achieved_throughput >= rate * 0.95:
                # If the system can keep up, it finishes all sent requests by the end of the duration
                target_total_requests = sent_requests

            # Linear update
            step_size = 0.5
            while current_time < end_generating_time:
                current_time += step_size
                elapsed = current_time - start_generating_time
                progress = min(1.0, elapsed / duration)
                with finished_requests_counter.get_lock():
                    finished_requests_counter.value = round(target_total_requests * progress)
                await asyncio.sleep(0)

            # Finalize
            with finished_requests_counter.get_lock():
                finished_requests_counter.value = target_total_requests
            await fake_sleep(0.5)

        self.load_generator.run_stage = AsyncMock(side_effect=mock_run_stage)  # type: ignore[method-assign]

        # Setup counters
        finished_requests_counter.get_lock = MagicMock()
        finished_requests_counter.get_lock.return_value.__enter__ = MagicMock()
        finished_requests_counter.get_lock.return_value.__exit__ = MagicMock()
        finished_requests_counter.value = 0

        await self.load_generator.preprocess(
            client, request_queue, active_requests_counter, finished_requests_counter, request_phase, cancel_signal
        )

    async def test_linear_success(self) -> None:
        """Test ideal case where server handles all load."""

        def behavior(rate: float, duration: float) -> float:
            return rate  # Server perfectly matches target rate

        await self.run_preprocess_with_simulation(behavior)

        # Should ramp up indefinitely until safety limit or large number of steps.
        # With start=10, 1.5x growth -> > 20000 after 20 steps.
        rates = [
            s.rate
            for s in self.load_generator.stages
            if isinstance(s, (StandardLoadStage, ConcurrentLoadStage)) and s.rate is not None
        ]
        max_rate = max(rates) if rates else 0.0
        self.assertGreater(max_rate, 20000.0)

    async def test_hard_saturation_limit(self) -> None:
        """Test server capping at 50 QPS."""
        # Steps: 20, 40, 60, 80, 100.
        # 20 -> 20 (OK)
        # 40 -> 40 (OK)
        # 60 -> 50 (Fail: 50 < 60*0.9=54) -> Saturation detected!
        self.sweep_config.num_stages = 5

        def behavior(rate: float, duration: float) -> float:
            return min(rate, 50.0)

        await self.run_preprocess_with_simulation(behavior)

        # Saturation detected at step 3. Measured ~50 (or 45 due to discretization).
        # Saturation 50 -> Target 90.
        rates = [
            s.rate
            for s in self.load_generator.stages
            if isinstance(s, (StandardLoadStage, ConcurrentLoadStage)) and s.rate is not None
        ]
        max_rate = max(rates) if rates else 0.0
        self.assertAlmostEqual(max_rate, 90.0, delta=12.0)

    async def test_early_saturation(self) -> None:
        """Test server saturated immediately (e.g. 10 QPS cap)."""
        # Step 1: 20 QPS.
        # Measured: 10.
        # 10 < 20*0.9 (18). Saturation detected immediately.

        def behavior(rate: float, duration: float) -> float:
            return min(rate, 10.0)

        await self.run_preprocess_with_simulation(behavior)

        # Saturation detected at step 1. Measured 10.
        # Target = 10 * 1.8 = 18.
        # With 11 as "Safe", max_rate = 11*1.8=19.8.
        rates = [
            s.rate
            for s in self.load_generator.stages
            if isinstance(s, (StandardLoadStage, ConcurrentLoadStage)) and s.rate is not None
        ]
        max_rate = max(rates) if rates else 0.0
        self.assertAlmostEqual(max_rate, 18.0, delta=2.5)

    async def test_tolerance_boundary(self) -> None:
        """Test throughput exactly at 89% (fail) and 91% (pass)."""
        # We need precise control.
        # Let's set tolerance to 0.9 in code (default is hardcoded 0.90 in preprocess).

        # Case 1: 89% -> Fail
        # Use single step to isolate.
        if self.sweep_config:
            self.sweep_config.num_stages = 1
            self.sweep_config.stage_duration = 120  # Need enough samples at 0.1 * 0.89 QPS

        def behavior_fail(rate: float, duration: float) -> float:
            if rate <= 0.2:
                return rate  # Safe at very low rates to avoid total failure
            return rate * 0.89

        await self.run_preprocess_with_simulation(behavior_fail)
        rates = [
            s.rate
            for s in self.load_generator.stages
            if isinstance(s, (StandardLoadStage, ConcurrentLoadStage)) and s.rate is not None
        ]
        max_rate_fail = max(rates) if rates else 0.0
        # Saturated at 1.0 -> Probe falls back to 0.1 -> Saturated at 0.1.
        # Max rate ~0.18 (0.1 * 1.8)
        self.assertLess(max_rate_fail, 1.0)

        # Case 2: 91% -> Pass
        # Reset load generator
        with patch("inference_perf.loadgen.load_generator.get_circuit_breaker", return_value=self.circuit_breaker):
            self.load_generator = LoadGenerator(self.mock_datagen, self.load_config)
            # Explicitly update inner config if it was copied
            if self.load_generator.sweep_config:
                self.load_generator.sweep_config.stage_duration = 120

        def behavior_pass(rate: float, duration: float) -> float:
            return rate * 0.91

        await self.run_preprocess_with_simulation(behavior_pass)
        rates = [
            s.rate
            for s in self.load_generator.stages
            if isinstance(s, (StandardLoadStage, ConcurrentLoadStage)) and s.rate is not None
        ]
        max_rate_pass = max(rates) if rates else 0.0
        # Saturation (or max measured) ~91. 91 * 1.8 = 163.8
        self.assertGreater(max_rate_pass, 150.0)

    async def test_low_throughput(self) -> None:
        """Test very low throughput (0.25 QPS)."""
        # Target: 0.25 QPS. Duration: 20s. Expect 5 requests.
        # If we measure 4 requests in ~19s, we get ~0.21 QPS. 0.21 < 0.225 (90% of 0.25).
        # We need more samples to smooth out discrete quantization error.
        # Let's target 0.5 QPS for 20s -> 10 requests.
        # Or Just increase duration for 0.25 QPS to 40s -> 10 requests.
        if self.sweep_config:
            self.sweep_config.num_requests = 1
            self.sweep_config.num_stages = 4  # 0.25, 0.5, 0.75, 1.0
            self.sweep_config.stage_duration = 100  # Increased to 100s to avoid quantization at 0.1 QPS

        # Reset load gen
        with patch("inference_perf.loadgen.load_generator.get_circuit_breaker", return_value=self.circuit_breaker):
            self.load_generator = LoadGenerator(self.mock_datagen, self.load_config)
            # Explicitly update inner config if it was copied
            if self.load_generator.sweep_config:
                self.load_generator.sweep_config.stage_duration = 100

        def behavior(rate: float, duration: float) -> float:
            return min(rate, 0.5)

        await self.run_preprocess_with_simulation(behavior)

        # Saturation detected at 0.5 (or close).
        # Max rate = 0.5 * 1.8 = 0.9.
        rates = [
            s.rate
            for s in self.load_generator.stages
            if isinstance(s, (StandardLoadStage, ConcurrentLoadStage)) and s.rate is not None
        ]
        max_rate = max(rates) if rates else 0.0
        print(f"DEBUG: Low Throughput Max Rate: {max_rate}")
        self.assertAlmostEqual(max_rate, 0.9, delta=0.5)

    async def test_high_throughput(self) -> None:
        """Test very high throughput (1000 QPS)."""
        self.sweep_config.num_requests = 1000
        self.sweep_config.num_stages = 5
        self.sweep_config.stage_duration = 5

        # Reset load gen
        with patch("inference_perf.loadgen.load_generator.get_circuit_breaker", return_value=self.circuit_breaker):
            self.load_generator = LoadGenerator(self.mock_datagen, self.load_config)

        def behavior(rate: float, duration: float) -> float:
            return rate  # Ideal

        await self.run_preprocess_with_simulation(behavior)

        # Should not saturate until very high.
        rates = [
            s.rate
            for s in self.load_generator.stages
            if isinstance(s, (StandardLoadStage, ConcurrentLoadStage)) and s.rate is not None
        ]
        max_rate = max(rates) if rates else 0.0
        self.assertGreater(max_rate, 20000.0)

    async def test_high_latency_robustness(self) -> None:
        """Test system with high latency but sufficient throughput (avoids false saturation)."""
        # A target of 1.0 QPS, perfectly met, but latency is high (e.g. 5 seconds)
        # Previous logic would truncate at step_duration + 1.0 and divide partial completions

        if self.sweep_config:
            self.sweep_config.num_stages = 3
            self.sweep_config.stage_duration = 10  # 10s duration

        # Reset load gen
        with patch("inference_perf.loadgen.load_generator.get_circuit_breaker", return_value=self.circuit_breaker):
            self.load_generator = LoadGenerator(self.mock_datagen, self.load_config)

        def behavior(rate: float, duration: float) -> float:
            return rate

        await self.run_preprocess_with_simulation(behavior, simulated_latency=5.0)

        # Ensures it didn't falsely saturate at 1.0 QPS
        rates = [
            s.rate
            for s in self.load_generator.stages
            if isinstance(s, (StandardLoadStage, ConcurrentLoadStage)) and s.rate is not None
        ]
        max_rate = max(rates) if rates else 0.0
        self.assertGreater(max_rate, 20.0)


if __name__ == "__main__":
    unittest.main()
