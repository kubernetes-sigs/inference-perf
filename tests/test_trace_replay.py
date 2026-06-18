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
import tempfile
from pathlib import Path
from unittest.mock import Mock
from inference_perf.apis import LazyLoadInferenceAPIData
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.datagen.random_datagen import RandomDataGenerator
from inference_perf.datagen.base import LazyLoadDataMixin
from inference_perf.datagen.replay_graph_session_datagen import (
    ReplayGraphSessionGeneratorBase,
    ReplaySession,
    ReplaySessionState,
)
from inference_perf.datagen.replay_graph_types import ReplayGraph
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
            mock_tokenizer_obj.decode.side_effect = lambda tokens, **kwargs: " ".join(map(str, tokens))
            mock_tokenizer_obj.vocab_size = 1000
            mock_tokenizer_obj.all_special_ids = []
            mock_tokenizer.get_tokenizer.return_value = mock_tokenizer_obj
            mock_tokenizer.count_tokens.side_effect = lambda text: len(text.split())

            api_config = APIConfig(type=APIType.Completion)
            trace_config = TraceConfig(file=str(temp_path), format=TraceFormat.AZURE_PUBLIC_DATASET)
            data_config = DataConfig(type=DataGenType.Random, trace=trace_config)

            datagen = RandomDataGenerator(api_config=api_config, config=data_config, tokenizer=mock_tokenizer)

            data_generator = datagen.get_data()
            lazy_load_data_list = [next(data_generator) for _ in range(3)]
            assert all(isinstance(data, LazyLoadInferenceAPIData) for data in lazy_load_data_list)

            data_list = [LazyLoadDataMixin.get_request(datagen, data) for data in lazy_load_data_list]

            # Test 3: Verify both timing and token counts are preserved
            assert len(timestamps) == len(data_list)

            # Timing preserved
            assert timestamps[1] - timestamps[0] > 0.99
            assert timestamps[2] - timestamps[1] > 0.99

            # Token counts preserved
            assert isinstance(data_list[0], CompletionAPIData)
            assert isinstance(data_list[1], CompletionAPIData)
            assert isinstance(data_list[2], CompletionAPIData)
            assert data_list[0].max_tokens == 50
            assert data_list[1].max_tokens == 75
            assert data_list[2].max_tokens == 60

        finally:
            temp_path.unlink()


class TestBuildSessionMetricFailureReason:
    """Tests that failure_reason is surfaced as an error in SessionLifecycleMetric."""

    def _make_generator(self) -> ReplayGraphSessionGeneratorBase:
        api_config = APIConfig(type=APIType.Chat)
        data_config = DataConfig(type=DataGenType.Random)
        return ReplayGraphSessionGeneratorBase(
            api_config=api_config,
            config=data_config,
            tokenizer=None,
        )

    def _make_graph(self) -> ReplayGraph:
        return ReplayGraph(
            events={"evt_1": Mock()},
            root_event_ids=["evt_1"],
            source_file="test_trace.json",
        )

    def test_failure_reason_produces_error(self) -> None:
        gen = self._make_generator()
        graph = self._make_graph()

        gen.sessions = [ReplaySession(session_id="s1", source_id="src", session_index=0, graph=graph)]
        gen.session_graph_state["s1"] = ReplaySessionState(
            session_id="s1",
            graph=graph,
            ready_events=set(),
            dispatched_events=set(),
            completed_events=set(),
            event_completion_times={},
            failed=True,
            failure_reason="predecessor wait failed: TimeoutError",
            cancelled_events=1,
        )

        metric = gen.build_session_metric(session_id="s1", stage_id=0, start_time=100.0, end_time=110.0)

        assert metric.error is not None
        assert metric.error.error_type == "SessionReplayError"
        assert metric.error.error_msg == "predecessor wait failed: TimeoutError"
        assert metric.num_events_cancelled == 1

    def test_no_failure_reason_no_error(self) -> None:
        gen = self._make_generator()
        graph = self._make_graph()

        gen.sessions = [ReplaySession(session_id="s1", source_id="src", session_index=0, graph=graph)]
        gen.session_graph_state["s1"] = ReplaySessionState(
            session_id="s1",
            graph=graph,
            ready_events=set(),
            dispatched_events=set(),
            completed_events={"evt_1"},
            event_completion_times={"evt_1": 105.0},
            failed=False,
            failure_reason=None,
        )

        metric = gen.build_session_metric(session_id="s1", stage_id=0, start_time=100.0, end_time=110.0)

        assert metric.error is None
        assert metric.num_events_completed == 1
        assert metric.num_events_cancelled == 0
