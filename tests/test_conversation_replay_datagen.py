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
"""Tests for ConversationReplayDataGenerator."""

import pytest
from unittest.mock import MagicMock
import numpy as np

from inference_perf.config import (
    APIConfig,
    APIType,
    ConversationReplayConfig,
    Distribution,
    DataConfig,
    DataGenType,
)
from inference_perf.datagen.conversation_replay_datagen import (
    ConversationReplayDataGenerator,
    _ConversationReplayAPIData,
)
from inference_perf.apis.base import LazyLoadInferenceAPIData
from inference_perf.utils.distribution import generate_distribution


def _make_mock_tokenizer(vocab_size: int = 32000) -> MagicMock:
    """Create a mock tokenizer with the expected interface."""
    mock_tokenizer = MagicMock()
    hf_tok = MagicMock()
    hf_tok.vocab_size = vocab_size
    hf_tok.decode.return_value = "mock decoded text"
    hf_tok.batch_decode.return_value = ["mock decoded text"]
    mock_tokenizer.get_tokenizer.return_value = hf_tok
    return mock_tokenizer


def _make_config(
    num_conversations: int = 5,
    seed: int = 42,
    shared_system_prompt_len: int = 100,
    turns_min: int = 3,
    turns_max: int = 5,
    turns_mean: float = 4,
) -> tuple[APIConfig, DataConfig]:
    api_config = APIConfig(type=APIType.Completion)
    cr_config = ConversationReplayConfig(
        seed=seed,
        num_conversations=num_conversations,
        shared_system_prompt_len=shared_system_prompt_len,
        dynamic_system_prompt_len=Distribution(type="normal", min=50, max=200, mean=100, std_dev=30),
        turns_per_conversation=Distribution(type="normal", min=turns_min, max=turns_max, mean=turns_mean, std_dev=1),
        input_tokens_per_turn=Distribution(type="normal", min=10, max=100, mean=50, std_dev=20),
        output_tokens_per_turn=Distribution(type="normal", min=10, max=100, mean=50, std_dev=20),
    )
    data_config = DataConfig(
        type=DataGenType.ConversationReplay,
        conversation_replay=cr_config,
    )
    return api_config, data_config


class TestConversationReplayDataGenerator:
    def test_init_creates_correct_number_of_conversations(self) -> None:
        api_config, data_config = _make_config(num_conversations=10)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        assert len(gen.blueprints) == 10
        assert len(gen.user_sessions) == 10

    def test_deterministic_with_same_seed(self) -> None:
        api_config, data_config = _make_config(seed=123)
        gen1 = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        api_config2, data_config2 = _make_config(seed=123)
        gen2 = ConversationReplayDataGenerator(api_config2, data_config2, _make_mock_tokenizer())

        assert len(gen1.blueprints) == len(gen2.blueprints)
        for bp1, bp2 in zip(gen1.blueprints, gen2.blueprints, strict=True):
            assert bp1.num_turns == bp2.num_turns
            assert bp1.turn_output_lens == bp2.turn_output_lens

    def test_different_seeds_produce_different_results(self) -> None:
        api_config1, data_config1 = _make_config(seed=1)
        gen1 = ConversationReplayDataGenerator(api_config1, data_config1, _make_mock_tokenizer())
        api_config2, data_config2 = _make_config(seed=2)
        gen2 = ConversationReplayDataGenerator(api_config2, data_config2, _make_mock_tokenizer())

        # At least some turn counts should differ
        turns1 = [bp.num_turns for bp in gen1.blueprints]
        turns2 = [bp.num_turns for bp in gen2.blueprints]
        assert turns1 != turns2

    def test_get_data_yields_lazy_load_with_preferred_worker(self) -> None:
        api_config, data_config = _make_config(num_conversations=3)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        data_iter = gen.get_data()
        items = [next(data_iter) for _ in range(9)]

        # First 3 items should cycle through conversations 0, 1, 2
        assert all(isinstance(item, LazyLoadInferenceAPIData) for item in items)
        assert items[0].preferred_worker_id == 0
        assert items[1].preferred_worker_id == 1
        assert items[2].preferred_worker_id == 2
        # Second round
        assert items[3].preferred_worker_id == 0

    def test_load_lazy_data_returns_user_session_data(self) -> None:
        api_config, data_config = _make_config(num_conversations=2)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        lazy = LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0)
        result = gen.load_lazy_data(lazy)

        assert isinstance(result, _ConversationReplayAPIData)
        assert result.user_session == gen.user_sessions[0]
        assert result.target_round == 0

    def test_turn_recycling(self) -> None:
        """When data_index exceeds total turns, it wraps around."""
        api_config, data_config = _make_config(num_conversations=2, turns_min=3, turns_max=3, turns_mean=3)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        # Conversation 0 has 3 turns. data_index=0 -> round 0, turn 0
        # data_index=2 -> conv 0, round 1, turn 1
        # data_index=6 -> conv 0, round 3, turn 0 (recycled)
        lazy = LazyLoadInferenceAPIData(data_index=6, preferred_worker_id=0)
        result = gen.load_lazy_data(lazy)
        assert isinstance(result, _ConversationReplayAPIData)
        assert result.target_round == 3  # 6 // 2 = 3

    def test_requires_tokenizer(self) -> None:
        api_config, data_config = _make_config()
        with pytest.raises(ValueError, match="Tokenizer is required"):
            ConversationReplayDataGenerator(api_config, data_config, None)

    def test_requires_conversation_replay_config(self) -> None:
        api_config = APIConfig(type=APIType.Completion)
        data_config = DataConfig(type=DataGenType.ConversationReplay)
        with pytest.raises(ValueError, match="conversation_replay config is required"):
            ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

    def test_is_preferred_worker_requested(self) -> None:
        api_config, data_config = _make_config()
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        assert gen.is_preferred_worker_requested() is True

    def test_user_session_ids(self) -> None:
        api_config, data_config = _make_config(num_conversations=3)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        ids = [s.user_session_id for s in gen.user_sessions]
        assert ids == ["conv_0", "conv_1", "conv_2"]

    def test_load_lazy_data_returns_conversation_replay_api_data(self) -> None:
        api_config, data_config = _make_config(num_conversations=2)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        lazy = LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0)
        result = gen.load_lazy_data(lazy)
        assert isinstance(result, _ConversationReplayAPIData)

    def test_tool_call_latency_not_set_gives_zero(self) -> None:
        """Without tool_call_latency_sec, all latencies are 0."""
        api_config, data_config = _make_config(num_conversations=2)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        lazy = LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0)
        result = gen.load_lazy_data(lazy)
        assert isinstance(result, _ConversationReplayAPIData)
        assert result.tool_call_latency_sec == 0.0

    def test_tool_call_latency_fixed_distribution(self) -> None:
        """Fixed tool call latency is sampled and stored per turn."""
        api_config = APIConfig(type=APIType.Completion)
        cr_config = ConversationReplayConfig(
            seed=42,
            num_conversations=2,
            shared_system_prompt_len=50,
            turns_per_conversation=Distribution(type="fixed", min=3, max=3, mean=3, std_dev=0),
            input_tokens_per_turn=Distribution(type="normal", min=10, max=50, mean=20, std_dev=5),
            output_tokens_per_turn=Distribution(type="normal", min=10, max=50, mean=20, std_dev=5),
            tool_call_latency_sec=Distribution(type="fixed", min=5, max=5, mean=5, std_dev=0),
        )
        data_config = DataConfig(type=DataGenType.ConversationReplay, conversation_replay=cr_config)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        # All turns should have latency == 5.0
        for bp in gen.blueprints:
            assert len(bp.turn_tool_call_latencies) == bp.num_turns
            assert all(lat == 5.0 for lat in bp.turn_tool_call_latencies)

        lazy = LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0)
        result = gen.load_lazy_data(lazy)
        assert isinstance(result, _ConversationReplayAPIData)
        assert result.tool_call_latency_sec == 5.0

    def test_tool_call_latency_lognormal_distribution(self) -> None:
        """Lognormal tool call latencies vary across turns."""
        api_config = APIConfig(type=APIType.Completion)
        cr_config = ConversationReplayConfig(
            seed=42,
            num_conversations=3,
            shared_system_prompt_len=50,
            turns_per_conversation=Distribution(type="fixed", min=10, max=10, mean=10, std_dev=0),
            input_tokens_per_turn=Distribution(type="normal", min=10, max=50, mean=20, std_dev=5),
            output_tokens_per_turn=Distribution(type="normal", min=10, max=50, mean=20, std_dev=5),
            tool_call_latency_sec=Distribution(type="lognormal", min=1, max=30, mean=8, std_dev=6),
        )
        data_config = DataConfig(type=DataGenType.ConversationReplay, conversation_replay=cr_config)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        for bp in gen.blueprints:
            assert len(bp.turn_tool_call_latencies) == bp.num_turns
            # Should have variation (lognormal, not fixed)
            assert not all(lat == bp.turn_tool_call_latencies[0] for lat in bp.turn_tool_call_latencies)
            # All within bounds
            assert all(1 <= lat <= 30 for lat in bp.turn_tool_call_latencies)


class TestDistributionExtensions:
    def test_lognormal_distribution(self) -> None:
        rng = np.random.default_rng(42)
        result = generate_distribution(
            min=10, max=1000, mean=100, std_dev=50, total_count=1000, dist_type="lognormal", rng=rng
        )
        assert len(result) == 1000
        assert all(10 <= v <= 1000 for v in result)

    def test_uniform_distribution(self) -> None:
        rng = np.random.default_rng(42)
        result = generate_distribution(min=10, max=100, mean=55, std_dev=0, total_count=1000, dist_type="uniform", rng=rng)
        assert len(result) == 1000
        assert all(10 <= v <= 100 for v in result)

    def test_fixed_distribution(self) -> None:
        result = generate_distribution(min=50, max=50, mean=50, std_dev=0, total_count=100, dist_type="fixed")
        assert len(result) == 100
        assert all(v == 50 for v in result)

    def test_normal_distribution_backward_compatible(self) -> None:
        """Default dist_type='normal' preserves existing behavior."""
        np.random.seed(42)
        result = generate_distribution(min=10, max=100, mean=50, std_dev=20, total_count=100)
        assert len(result) == 100
        assert all(10 <= v <= 100 for v in result)

    def test_seeded_rng_deterministic(self) -> None:
        rng1 = np.random.default_rng(99)
        result1 = generate_distribution(min=10, max=1000, mean=500, std_dev=100, total_count=50, dist_type="normal", rng=rng1)
        rng2 = np.random.default_rng(99)
        result2 = generate_distribution(min=10, max=1000, mean=500, std_dev=100, total_count=50, dist_type="normal", rng=rng2)
        assert list(result1) == list(result2)
