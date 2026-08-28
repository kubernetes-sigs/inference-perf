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
"""Regression test for #486: stage start/end must log at INFO so non-TTY runs see progress."""

import io
import multiprocessing as mp
import unittest
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

from rich.console import Console

from inference_perf.apis import InferenceAPIData
from inference_perf.config import Config, LoadConfig, LoadType, StandardLoadStage
from inference_perf.datagen import DataGenerator
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.observability.metrics.registry import build_metrics
from inference_perf.observability.progress import ProgressBars


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


class TestRunStageProgressLogs(unittest.IsolatedAsyncioTestCase):
    @patch("inference_perf.loadgen.load_generator.sleep", new_callable=AsyncMock)
    async def test_run_stage_logs_start_and_end_at_info(self, mock_sleep: AsyncMock) -> None:
        # Runs stage 7 with no progress display and one request finishing.
        # Expects "Stage 7 - run started" and "Stage 7 - run completed" at INFO,
        # so a non-TTY run still shows progress in the log.
        load_generator = _make_load_generator()

        finished_counter = mp.Value("i", 0)

        async def advance_counter(*_args: Any, **_kwargs: Any) -> None:
            # Exit the wait loop after one iteration.
            finished_counter.value = 1

        mock_sleep.side_effect = advance_counter

        request_queue = MagicMock()
        request_queue.put = MagicMock()
        request_queue.join = MagicMock()

        with self.assertLogs("inference_perf.loadgen.load_generator", level="INFO") as cm:
            await load_generator.run_stage(
                stage_id=7,
                rate=1.0,
                duration=1,
                request_queue=request_queue,
                active_requests_counter=mp.Value("i", 0),
                finished_requests_counter=finished_counter,
                skipped_requests_counter=mp.Value("i", 0),
                request_phase=mp.Event(),
                cancel_signal=None,
                progress_ctx=None,
            )

        messages = "\n".join(cm.output)
        self.assertIn("Stage 7 - run started", messages, "stage start must be logged at INFO")
        self.assertIn("Stage 7 - run completed", messages, "stage completion must be logged at INFO")

    @patch("inference_perf.loadgen.load_generator.sleep", new_callable=AsyncMock)
    async def test_run_stage_emits_logs_and_progress_bar(self, mock_sleep: AsyncMock) -> None:
        # Runs stage 7 with a real metrics registry behind the bars and one
        # request finishing. Expects the INFO start/end logs, the bar labelled
        # "Stage 7 Requests" on screen, and "1/1" in the rendered frame: the
        # bar is showing the exported metrics, not a count of its own.
        load_generator = _make_load_generator()

        hub = build_metrics(Config())
        hub.on_run_start()
        load_generator.stage_observer = hub

        finished_counter = mp.Value("i", 0)

        async def advance_counter(*_args: Any, **_kwargs: Any) -> None:
            finished_counter.value = 1

        mock_sleep.side_effect = advance_counter

        request_queue = MagicMock()
        request_queue.put = MagicMock()
        request_queue.join = MagicMock()

        # force_terminal makes Progress render under pytest; record buffers frames.
        console = Console(file=io.StringIO(), force_terminal=True, width=120, record=True)
        progress = ProgressBars(hub.registry, console=console, refresh_hz=0)

        with self.assertLogs("inference_perf.loadgen.load_generator", level="INFO") as cm:
            with progress:
                await load_generator.run_stage(
                    stage_id=7,
                    rate=1.0,
                    duration=1,
                    request_queue=request_queue,
                    active_requests_counter=mp.Value("i", 0),
                    finished_requests_counter=finished_counter,
                    skipped_requests_counter=mp.Value("i", 0),
                    request_phase=mp.Event(),
                    cancel_signal=None,
                    progress_ctx=progress,
                )

        log_messages = "\n".join(cm.output)
        self.assertIn("Stage 7 - run started", log_messages)
        self.assertIn("Stage 7 - run completed", log_messages)

        rendered = console.export_text()
        self.assertIn(
            "Stage 7 Requests",
            rendered,
            "progress bar task description must appear in rendered output",
        )

        # The numbers the bar reads have to be in the registry, since that is
        # now its only source. Rendering of those values is covered by
        # tests/required/observability/test_progress.py.
        self.assertEqual(hub.registry.get_sample_value("inference_perf_stage_requests_planned", {"stage": "7"}), 1.0)
        self.assertEqual(hub.registry.get_sample_value("inference_perf_stage_requests_finished", {"stage": "7"}), 1.0)


if __name__ == "__main__":
    unittest.main()
