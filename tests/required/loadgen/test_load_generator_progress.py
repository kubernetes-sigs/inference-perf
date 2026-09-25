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
"""The in-process run() path drives its bars from the exported metrics.

The bar used to count requests as they were dispatched, which meant it read
full while requests were still outstanding. It now reads the same
finished/planned metrics the multiprocess path exports, so these tests assert
on the metrics rather than on calls made to a mocked-out display.
"""

import unittest
from typing import Any, List, Tuple
from unittest.mock import MagicMock, AsyncMock, patch

from inference_perf.apis import InferenceAPIData
from inference_perf.config import Config, LoadConfig, LoadType, StandardLoadStage
from inference_perf.datagen import DataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.observability.metrics.registry import build_metrics
from inference_perf.observability.progress import BarSpec, ProgressBars


class TestLoadGeneratorProgress(unittest.IsolatedAsyncioTestCase):
    def setUp(self) -> None:
        self.mock_datagen = MagicMock(spec=DataGenerator)
        # Prepare a mock data generator that yields InferenceAPIData
        mock_data = MagicMock(spec=InferenceAPIData)
        mock_data.preferred_worker_id = -1
        self.mock_datagen.get_data.return_value = [mock_data]
        self.mock_datagen.is_preferred_worker_requested.return_value = False

        self.load_config = LoadConfig(
            type=LoadType.CONSTANT,
            stages=[StandardLoadStage(rate=1.0, duration=1)],
            num_workers=0,  # 0 workers uses run()
            worker_max_concurrency=10,
        )
        self.hub = build_metrics(Config(load=self.load_config))
        self.hub.on_run_start()
        with patch("inference_perf.loadgen.load_generator.get_circuit_breaker"):
            self.load_generator = LoadGenerator(
                self.mock_datagen,
                self.load_config,
                stage_observer=self.hub,
                metrics_registry=self.hub.registry,
            )

    async def _run(self) -> None:
        mock_client = AsyncMock()
        # Override get_timer to prevent actual sleeping
        mock_timer = MagicMock()
        mock_timer.start_timer.return_value = [0.0]
        with patch.object(self.load_generator, "get_timer", return_value=mock_timer):
            await self.load_generator.run(mock_client)

    async def test_run_exports_the_numbers_its_bars_read(self) -> None:
        # One stage of one request, run to completion on the in-process path.
        # Expects planned=1 and finished=1 for stage 0, and stages_completed=1
        # against a configured stage count of 1: the numbers a bar needs to
        # reach 100% are all in the registry.
        await self._run()

        sample = self.hub.registry.get_sample_value
        self.assertEqual(sample("inference_perf_stage_requests_planned", {"stage": "0"}), 1.0)
        self.assertEqual(sample("inference_perf_stage_requests_finished", {"stage": "0"}), 1.0)
        self.assertEqual(sample("inference_perf_stages_completed"), 1.0)
        self.assertEqual(sample("inference_perf_stages"), 1.0)

    async def test_bars_opened_are_declared_specs_scoped_to_the_stage(self) -> None:
        # Runs the same stage while recording every ProgressBars.open call.
        # Expects exactly two bars: the unlabelled overall bar, and the stage
        # bar labelled stage="0". Nothing may open a bar from a raw count,
        # because open() takes a BarSpec and no number.
        opened: List[Tuple[str, dict[str, str]]] = []
        real_open = ProgressBars.open

        def _record(self: ProgressBars, spec: BarSpec, **labels: Any) -> Any:
            opened.append((spec.name, dict(labels)))
            return real_open(self, spec, **labels)

        with patch.object(ProgressBars, "open", _record):
            await self._run()

        self.assertEqual(opened, [("overall", {}), ("stage_requests", {"stage": "0"})])


if __name__ == "__main__":
    unittest.main()
