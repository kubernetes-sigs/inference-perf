import tempfile
from pathlib import Path
from unittest.mock import Mock
from inference_perf.datagen.random_datagen import RandomDataGenerator
from inference_perf.config import APIConfig, DataConfig, APIType, TraceFormat, TraceConfig, DataGenType


class TestTraceReplay:
    """End-to-end test for trace replay feature."""

    def test_trace_replay_complete_flow(self) -> None:
        """Test complete flow: load trace, generate data, replay timing."""
        content = """TIMESTAMP,ContextTokens,GeneratedTokens
2023-11-16 18:15:00.00,100,50
2023-11-16 18:15:01.00,200,75
2023-11-16 18:15:02.00,150,60
"""
        with tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".csv") as f:
            f.write(content)
            f.flush()
            temp_path = Path(f.name)

        try:
            # Test 1: Load timing information
            from inference_perf.utils.trace_reader import AzurePublicDatasetReader
            from inference_perf.loadgen.load_timer import TraceReplayLoadTimer

            reader = AzurePublicDatasetReader()
            timer = TraceReplayLoadTimer(trace_reader=reader, trace_file=temp_path)

            timestamps = list(timer.start_timer(initial=0.0))
            assert len(timestamps) == 3

            # Test 2: Generate data with matching token counts
            mock_tokenizer = Mock()
            mock_tokenizer_obj = Mock()
            mock_tokenizer_obj.decode.return_value = "test prompt"
            mock_tokenizer_obj.vocab_size = 1000
            mock_tokenizer.get_tokenizer.return_value = mock_tokenizer_obj

            api_config = APIConfig(type=APIType.Completion)
            trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.AZURE_PUBLIC_DATASET)
            data_config = DataConfig(type=DataGenType.Random, trace=trace_config)

            datagen = RandomDataGenerator(api_config=api_config, config=data_config, tokenizer=mock_tokenizer)

            data_list = [datagen.get_request(i) for i in range(3)]

            # Test 3: Verify both timing and token counts are preserved
            assert len(timestamps) == len(data_list)

            # Timing preserved
            assert timestamps[1] - timestamps[0] > 0.99
            assert timestamps[2] - timestamps[1] > 0.99

            # Token counts preserved
            assert data_list[0].max_tokens == 50
            assert data_list[1].max_tokens == 75
            assert data_list[2].max_tokens == 60

        finally:
            temp_path.unlink()
