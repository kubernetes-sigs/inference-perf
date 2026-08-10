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
"""Tests for run_stage's dispatch semantics: how many requests are enqueued,
onto which worker channel, and how the stage exits on worker death."""

import multiprocessing as mp
import unittest
from typing import Any, List, Optional, Tuple
from unittest.mock import AsyncMock, MagicMock, patch

from inference_perf.config import LoadConfig, LoadType, StandardLoadStage
from inference_perf.datagen import DataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator, RequestQueueData
from inference_perf.utils.request_queue import RequestQueue


def _make_load_generator(num_workers: int = 4) -> LoadGenerator:
    datagen = MagicMock(spec=DataGenerator)
    datagen.trace = None
    load_config = LoadConfig(
        type=LoadType.CONSTANT,
        num_workers=num_workers,
        worker_max_concurrency=10,
        stages=[StandardLoadStage(rate=1, duration=1)],
        base_seed=42,
    )
    with patch("inference_perf.loadgen.load_generator.get_circuit_breaker"):
        return LoadGenerator(datagen, load_config)


def _mock_requests(preferred_worker_ids: List[int]) -> Any:
    return iter([MagicMock(preferred_worker_id=worker_id) for worker_id in preferred_worker_ids])


class TestRunStageDispatch(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.load_generator = _make_load_generator()
        self.request_queue = MagicMock(spec=RequestQueue)
        self.finished_counter = mp.Value("i", 0)
        self.active_counter = mp.Value("i", 0)
        self.request_phase = MagicMock()
        self.cancel_signal = MagicMock()

    def _put_calls(self) -> List[Tuple[RequestQueueData, int]]:
        return [(call.args[0], call.args[1]) for call in self.request_queue.put.call_args_list]

    async def _run_stage(
        self,
        num_requests: int,
        stage_id: int = 0,
        rate: float = 1,
        duration: int = 1,
        concurrency_level: Optional[int] = None,
    ) -> None:
        """Run one stage, completing the wait loop by advancing the counter."""

        async def finish_all(*_args: Any, **_kwargs: Any) -> None:
            self.finished_counter.value = num_requests

        with patch("inference_perf.loadgen.load_generator.sleep", new_callable=AsyncMock) as mock_sleep:
            mock_sleep.side_effect = finish_all
            await self.load_generator.run_stage(
                stage_id=stage_id,
                rate=rate,
                duration=duration,
                request_queue=self.request_queue,
                active_requests_counter=self.active_counter,
                finished_requests_counter=self.finished_counter,
                request_phase=self.request_phase,
                cancel_signal=self.cancel_signal,
                concurrency_level=concurrency_level,
            )

    async def test_enqueues_rate_times_duration_requests_with_increasing_times(self) -> None:
        self.load_generator.datagen.get_data.return_value = _mock_requests([-1] * 10)  # type: ignore[attr-defined]
        await self._run_stage(num_requests=10, stage_id=5, rate=5, duration=2)

        puts = self._put_calls()
        self.assertEqual(len(puts), 10)
        for item, channel in puts:
            self.assertEqual(item.stage_id, 5)
            self.assertEqual(channel, -1, "unaffined requests keep the broadcast channel id")
        times = [item.request_time for item, _ in puts]
        self.assertTrue(
            all(b > a for a, b in zip(times, times[1:], strict=False)), "dispatch times must be strictly increasing"
        )
        self.assertEqual(self.load_generator.stage_runtime_info[5].status.name, "COMPLETED")

    async def test_affinity_remaps_preferred_worker_onto_num_workers(self) -> None:
        self.load_generator.datagen.get_data.return_value = _mock_requests([0, 1, 2, 3, 4, 5])  # type: ignore[attr-defined]
        await self._run_stage(num_requests=6, rate=6, duration=1)

        channels = [channel for _, channel in self._put_calls()]
        self.assertEqual(channels, [0, 1, 2, 3, 0, 1], "preferred_worker_id must wrap modulo num_workers")

    async def test_affinity_remaps_onto_active_workers_under_concurrency_level(self) -> None:
        # concurrency_level=2 leaves only 2 workers with permits, so affinity
        # must wrap onto the active workers, not all workers.
        self.load_generator.datagen.get_data.return_value = _mock_requests([0, 1, 2, 3, 4, 5])  # type: ignore[attr-defined]
        await self._run_stage(num_requests=6, rate=6, duration=1, concurrency_level=2)

        channels = [channel for _, channel in self._put_calls()]
        self.assertEqual(channels, [0, 1, 0, 1, 0, 1], "affinity must wrap modulo min(num_workers, concurrency_level)")

    async def test_trace_bound_request_count_overrides_rate(self) -> None:
        self.load_generator.datagen.trace = MagicMock()  # type: ignore[attr-defined]
        self.load_generator.datagen.get_request_count.return_value = 3  # type: ignore[attr-defined]
        self.load_generator.datagen.get_data.return_value = _mock_requests([-1] * 3)  # type: ignore[attr-defined]
        await self._run_stage(num_requests=3, rate=100, duration=100)

        self.assertEqual(len(self._put_calls()), 3, "trace-driven stages enqueue the trace's request count")

    async def test_worker_death_fails_stage_and_cleans_up(self) -> None:
        self.load_generator.datagen.get_data.return_value = _mock_requests([-1] * 5)  # type: ignore[attr-defined]
        alive_worker = MagicMock()
        alive_worker.is_alive.return_value = True
        dead_worker = MagicMock()
        dead_worker.is_alive.return_value = False
        self.load_generator.workers = [alive_worker, dead_worker]
        self.load_generator.num_workers = 2

        with patch("inference_perf.loadgen.load_generator.sleep", new_callable=AsyncMock):
            await self.load_generator.run_stage(
                stage_id=0,
                rate=5,
                duration=1,
                request_queue=self.request_queue,
                active_requests_counter=self.active_counter,
                finished_requests_counter=self.finished_counter,
                request_phase=self.request_phase,
                cancel_signal=self.cancel_signal,
            )

        self.assertEqual(self.load_generator.stage_runtime_info[0].status.name, "FAILED")
        self.cancel_signal.set.assert_called_once()
        self.cancel_signal.clear.assert_called_once()
        self.request_queue.drain.assert_called_once()
        self.request_phase.clear.assert_called_once()


if __name__ == "__main__":
    unittest.main()
