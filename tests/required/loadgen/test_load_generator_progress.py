import unittest
from unittest.mock import MagicMock, AsyncMock, patch
from inference_perf.loadgen.load_generator import LoadGenerator
from inference_perf.config import LoadConfig, LoadType, StandardLoadStage
from inference_perf.apis import InferenceAPIData
from inference_perf.datagen import DataGenerator


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
        with patch("inference_perf.loadgen.load_generator.get_circuit_breaker"):
            self.load_generator = LoadGenerator(self.mock_datagen, self.load_config)

    @patch("inference_perf.loadgen.load_generator.Progress")
    async def test_run_progress(self, mock_progress_class: MagicMock) -> None:
        mock_progress = mock_progress_class.return_value
        mock_progress_class.return_value.__enter__.return_value = mock_progress

        mock_overall_task = MagicMock()
        mock_stage_task = MagicMock()
        mock_progress.add_task.side_effect = [mock_overall_task, mock_stage_task]

        mock_client = AsyncMock()

        # Override get_timer to prevent actual sleeping
        mock_timer = MagicMock()
        mock_timer.start_timer.return_value = [0.0]
        with patch.object(self.load_generator, "get_timer", return_value=mock_timer):
            await self.load_generator.run(mock_client)

        # Check that Progress was used as a context manager
        mock_progress.__enter__.assert_called_once()
        mock_progress.__exit__.assert_called_once()

        # Check overall and stage tasks added
        self.assertEqual(mock_progress.add_task.call_count, 2)

        # Check that it calls completed=1.0 at the end of the stage.
        mock_progress.update.assert_any_call(mock_stage_task, completed=1.0)
        mock_progress.remove_task.assert_any_call(mock_stage_task)


if __name__ == "__main__":
    unittest.main()
