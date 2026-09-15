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
"""Unit tests asserting that a stage's terminal StageStatus records its actual cause
(max_stage_duration exceeded vs. SIGINT)
"""

import multiprocessing as mp
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from inference_perf.apis import InferenceAPIData
from inference_perf.client.server_metrics.base import StageStatus
from inference_perf.config import LoadConfig, LoadType, StandardLoadStage
from inference_perf.datagen import DataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator


def _make_load_generator() -> LoadGenerator:
    mock_datagen = MagicMock(spec=DataGenerator)
    mock_datagen.trace = None
    mock_data = MagicMock(spec=InferenceAPIData)
    mock_data.preferred_worker_id = -1
    mock_datagen.get_data.return_value = iter([mock_data])
    mock_datagen.is_preferred_worker_requested.return_value = False

    load_config = LoadConfig(
        type=LoadType.CONSTANT,
        stages=[StandardLoadStage(rate=1.0, duration=1)],
        num_workers=1,
        worker_max_concurrency=10,
    )
    with patch("inference_perf.loadgen.load_generator.get_circuit_breaker"):
        return LoadGenerator(mock_datagen, load_config)


class TestRunStageTerminationStatus(unittest.IsolatedAsyncioTestCase):
    @patch("inference_perf.loadgen.load_generator.sleep", new_callable=AsyncMock)
    async def test_timeout_without_sigint_is_timed_out(self, mock_sleep: AsyncMock) -> None:
        """A stage that never finishes its requests before `timeout` elapses, with no
        SIGINT involved, must be recorded as TIMED_OUT rather than the generic FAILED."""
        load_generator = _make_load_generator()

        # start_time is set to perf_counter() + 1 inside run_stage; a negative timeout
        # puts the deadline in the past immediately, so the loop times out on its first
        # check without needing to mock time itself or wait out a real duration.
        await load_generator.run_stage(
            stage_id=3,
            rate=1.0,
            duration=1,
            request_queue=MagicMock(put=MagicMock(), join=MagicMock()),
            active_requests_counter=mp.Value("i", 0),
            finished_requests_counter=mp.Value("i", 0),
            request_phase=mp.Event(),
            cancel_signal=mp.Event(),
            timeout=-2.0,
            progress_ctx=None,
        )

        mock_sleep.assert_not_called()
        self.assertEqual(load_generator.stage_runtime_info[3].status, StageStatus.TIMED_OUT)

    @patch("inference_perf.loadgen.load_generator.sleep", new_callable=AsyncMock)
    async def test_sigint_without_timeout_is_interrupted(self, mock_sleep: AsyncMock) -> None:
        """SIGINT must be recorded as INTERRUPTED, distinct from a real timeout, even
        though both exit the same wait loop the same way."""
        load_generator = _make_load_generator()
        load_generator.interrupt_sig = True

        async def advance_counter(*_args: Any, **_kwargs: Any) -> None:
            pass

        mock_sleep.side_effect = advance_counter

        await load_generator.run_stage(
            stage_id=4,
            rate=1.0,
            duration=1,
            request_queue=MagicMock(put=MagicMock(), join=MagicMock()),
            active_requests_counter=mp.Value("i", 0),
            finished_requests_counter=mp.Value("i", 0),
            request_phase=mp.Event(),
            cancel_signal=mp.Event(),
            timeout=None,
            progress_ctx=None,
        )

        self.assertEqual(load_generator.stage_runtime_info[4].status, StageStatus.INTERRUPTED)

    @patch("inference_perf.loadgen.load_generator.sleep", new_callable=AsyncMock)
    async def test_completed_stage_is_not_relabeled_by_unclean_teardown_guard(self, mock_sleep: AsyncMock) -> None:
        """Sanity check that the normal completion path is untouched by the new
        cause-tracking: no timeout, no SIGINT, requests finish -> COMPLETED."""
        load_generator = _make_load_generator()
        finished_counter = mp.Value("i", 0)

        async def advance_counter(*_args: Any, **_kwargs: Any) -> None:
            finished_counter.value = 1

        mock_sleep.side_effect = advance_counter

        await load_generator.run_stage(
            stage_id=5,
            rate=1.0,
            duration=1,
            request_queue=MagicMock(put=MagicMock(), join=MagicMock()),
            active_requests_counter=mp.Value("i", 0),
            finished_requests_counter=finished_counter,
            request_phase=mp.Event(),
            cancel_signal=None,
            progress_ctx=None,
        )

        self.assertEqual(load_generator.stage_runtime_info[5].status, StageStatus.COMPLETED)


if __name__ == "__main__":
    unittest.main()
