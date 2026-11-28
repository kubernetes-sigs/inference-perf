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
"""Tests that the in-process run() path never sends a request before its scheduled time."""

import asyncio
import time
import unittest
from typing import List, Optional
from unittest.mock import MagicMock, patch

from inference_perf.apis import InferenceAPIData
from inference_perf.client.modelserver import MockModelServerClient
from inference_perf.config import APIConfig, APIType, LoadConfig, LoadType, StandardLoadStage
from inference_perf.datagen import DataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.metrics.request_collector.local import LocalRequestMetricCollector


class _ClockClient(MockModelServerClient):
    """Client double that records the clock and the scheduled time of every request it receives."""

    def __init__(self) -> None:
        super().__init__(LocalRequestMetricCollector(), APIConfig(type=APIType.Completion), mock_latency=0)
        self.calls: List[tuple[float, float]] = []

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        self.calls.append((time.perf_counter(), scheduled_time))


class TestInProcessDispatchTiming(unittest.IsolatedAsyncioTestCase):
    # One stage of one request scheduled 100 ms after the stage starts, with num_workers=0.
    # Every sleep wakes 5 ms before the delay it was given, as a coarse timer can.
    # Expects the client to be called at or after the scheduled time, never before it.
    async def test_early_sleep_wake_does_not_dispatch_before_scheduled_time(self) -> None:
        datagen = MagicMock(spec=DataGenerator)
        datagen.is_preferred_worker_requested.return_value = False
        data = MagicMock(spec=InferenceAPIData)
        data.preferred_worker_id = -1
        datagen.get_data.return_value = [data]
        load_config = LoadConfig(
            type=LoadType.CONSTANT,
            stages=[StandardLoadStage(rate=1.0, duration=1)],
            num_workers=0,
            worker_max_concurrency=10,
        )
        with patch("inference_perf.loadgen.load_generator.get_circuit_breaker"):
            load_generator = LoadGenerator(datagen, load_config)

        timer = MagicMock()
        timer.start_timer.side_effect = lambda start_time: iter([start_time + 0.1])
        real_sleep = asyncio.sleep

        # Wraps asyncio.sleep with a delay 5 ms shorter than asked, floored at 0.
        async def early_sleep(delay: float) -> None:
            await real_sleep(max(delay - 0.005, 0.0))

        client = _ClockClient()
        with (
            patch.object(load_generator, "get_timer", return_value=timer),
            patch("inference_perf.loadgen.load_generator.sleep", early_sleep),
        ):
            await load_generator.run(client)

        self.assertEqual(len(client.calls), 1)
        called_at, scheduled_time = client.calls[0]
        self.assertGreaterEqual(called_at, scheduled_time, "request reached the client before its scheduled time")


if __name__ == "__main__":
    unittest.main()
