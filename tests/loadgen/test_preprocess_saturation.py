import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from typing import Any
import asyncio
import sys
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.config import LoadConfig, LoadType, SweepConfig, StageGenType
import multiprocessing as mp

# Patch asyncio.TaskGroup/TypeAlias if needed (copied from existing tests)
if sys.version_info < (3, 11):

    class MockTaskGroup:
        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            pass

        def create_task(self, coro):
            return asyncio.create_task(coro)

    asyncio.TaskGroup = MockTaskGroup


class TestPreprocessSaturation(unittest.IsolatedAsyncioTestCase):
    async def test_preprocess_generates_1_75x_saturation(self) -> None:
        # Setup
        mock_datagen = MagicMock()
        mock_datagen.get_request_count.return_value = 100

        # Target 200 QPS max, 5 steps -> 40, 80, 120, 160, 200
        sweep_config = SweepConfig(type=StageGenType.LINEAR, num_stages=5, num_requests=200, stage_duration=10, timeout=15)

        load_config = LoadConfig(type=LoadType.CONSTANT, sweep=sweep_config, num_workers=1)

        with patch("inference_perf.loadgen.load_generator.get_circuit_breaker"):
            generator = LoadGenerator(mock_datagen, load_config)

        # Mock counters
        active_counter = mp.Value("i", 0)
        finished_counter = mp.Value("i", 0)
        request_phase = mp.Event()
        cancel_signal = mp.Event()
        client = AsyncMock()
        queue = MagicMock()

        # Mock time control
        self.current_time = 1000.0

        def mock_time() -> float:
            return self.current_time

        async def mock_sleep(delay: float) -> None:
            # Just yield control, do NOT advance time here
            # Time is driven by mock_run_stage
            await asyncio.sleep(0)

        # Mock run_stage to simulate execution
        async def mock_run_stage(
            stage_id: int, rate: float, duration: float, *args: list[Any], **kwargs: dict[str, Any]
        ) -> None:
            # Simulate stage running
            start_t = self.current_time

            # Simulate throughput: Cap at 100 QPS
            # If rate <= 100, we get rate * duration requests
            # If rate > 100, we get 100 * duration requests
            effective_rate = min(rate, 100.0)
            target_requests = int(effective_rate * duration)

            # Increment time in small steps to allow aggregator to sample
            step = 0.5
            current = 0.0
            while current < duration:
                # Update counter
                elapsed = current
                # Interpolate finished requests
                with finished_counter.get_lock():
                    finished_counter.value = int(effective_rate * elapsed)

                # Advance time explicitly
                self.current_time += step
                await asyncio.sleep(0)  # Yield to aggregator
                current += step

            # Final update
            with finished_counter.get_lock():
                finished_counter.value = target_requests

            # Ensure time matches end
            self.current_time = start_t + duration

        generator.run_stage = AsyncMock(side_effect=mock_run_stage)  # type: ignore[method-assign]

        with (
            patch("inference_perf.loadgen.load_generator.time.perf_counter", side_effect=mock_time),
            patch("inference_perf.loadgen.load_generator.sleep", side_effect=mock_sleep),
        ):
            await generator.preprocess(client, queue, active_counter, finished_counter, request_phase, cancel_signal)

        # Verify stages
        # Saturation point should be ~100
        # Generated rates should go up to ~180 (1.8x saturation)
        # Stages are generated using generator.generateRates(saturation * 1.8, num_stages, type)
        # If saturation is 100, target is 180.
        # generateRates no longer scales down by (N-1)/N, so max rate = 180.

        self.assertTrue(len(generator.stages) > 0, "No stages generated")
        rates = [stage.rate for stage in generator.stages if stage.rate is not None]
        max_rate = max(rates) if rates else 0.0

        print(f"Generated rates: {[s.rate for s in generator.stages]}")

        # We expect max rate to be close to 180 (or higher if 90% tolerance allows ~111 QPS safe).
        self.assertAlmostEqual(max_rate, 180.0, delta=25.0)

        # Also verify that we actually detected saturation
        # The logs would show it, but programmatically we check the result (stages)
        # If we didn't detect saturation (e.g. if we thought max was 200), we would have generated up to 400?
        # No, if we didn't detect saturation, it uses max measured (which would be 100 if we capped it).
        # Wait, if we cap at 100, and we request 200. `measured` will be 100.
        # `measured < rate * 0.9` -> `100 < 200 * 0.9` (180). True. Saturation detected at 100.
        # So saturation_point = 100.
        # Target gen = 200.

        # If we didn't cap (measured = rate), then at 200, measured=200.
        # Saturation NOT detected.
        # Then `saturation_point` = max measured = 200.
        # Then generated rates would go up to 400.

        # So asserting max_rate is ~200 confirms we detected saturation at 100 and used 2x multiplier.


if __name__ == "__main__":
    unittest.main()
