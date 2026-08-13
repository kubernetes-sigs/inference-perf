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

from typing import Any, Generator

import asyncio
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
from inference_perf.apis.user_session import LocalUserSession
from inference_perf.utils.numeric.distribution import generate_distribution


@pytest.fixture(autouse=True)
def _clear_user_session_registry() -> Generator[None, None, None]:
    """Isolate LocalUserSession._instances across tests."""
    LocalUserSession.clear_instances()
    yield
    LocalUserSession.clear_instances()


def _make_mock_tokenizer(vocab_size: int = 32000) -> MagicMock:
    """Create a mock tokenizer with the expected interface."""
    mock_tokenizer = MagicMock()
    mock_tokenizer.count_tokens.side_effect = lambda text, **kw: len(text.split()) * 10 if text.strip() else 0
    hf_tok = MagicMock()
    hf_tok.vocab_size = vocab_size
    hf_tok.decode.side_effect = lambda ids, **kwargs: f"decoded_{ids}"
    hf_tok.batch_decode.side_effect = lambda list_ids, **kwargs: [f"decoded_{ids}" for ids in list_ids]
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

    def test_load_lazy_data_regenerates_after_clear_instances(self) -> None:
        """After LoadGenerator clears the session registry between stages,
        load_lazy_data must regenerate the system_prompt for the slot."""
        api_config, data_config = _make_config(num_conversations=3)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        original_contexts = {bp.conversation_id: bp.system_prompt for bp in gen.blueprints}
        assert all(f"conv_{i}" in LocalUserSession._instances for i in range(3))

        # Simulate LoadGenerator's between-stage cleanup.
        LocalUserSession.clear_instances()
        assert LocalUserSession._instances == {}

        # First request after the clear should re-prime the slot it touches AND regenerate prompt.
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=1, preferred_worker_id=1, stage_id=1))

        assert "conv_1" in LocalUserSession._instances
        # It should be different from original
        assert LocalUserSession._instances["conv_1"].context != original_contexts[1]

    def test_system_prompt_regenerated_across_repeated_clears(self) -> None:
        """Across multiple stage transitions, each re-prime must regenerate a fresh system_prompt."""
        api_config, data_config = _make_config(num_conversations=3)
        # Use a tokenizer that returns different text each time to verify regeneration
        mock_tok = _make_mock_tokenizer()
        texts = [f"text_{i}" for i in range(100)]
        mock_tok.get_tokenizer().decode.side_effect = texts

        gen = ConversationReplayDataGenerator(api_config, data_config, mock_tok)

        last_contexts = {i: "" for i in range(3)}

        for stage_idx in range(3):
            LocalUserSession.clear_instances()
            for conv_idx in range(3):
                gen.load_lazy_data(
                    LazyLoadInferenceAPIData(data_index=conv_idx, preferred_worker_id=conv_idx, stage_id=stage_idx)
                )
                current_context = LocalUserSession._instances[f"conv_{conv_idx}"].context
                assert current_context != last_contexts[conv_idx]
                last_contexts[conv_idx] = current_context

    def test_load_lazy_data_does_not_replace_live_session(self) -> None:
        """When the registry still holds the session (mid-stage), load_lazy_data
        must not overwrite it — that would clobber accumulated turn context."""
        api_config, data_config = _make_config(num_conversations=2)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        live_session = LocalUserSession._instances["conv_0"]
        expected_context = f"{live_session.system_prompt} accumulated turn history"
        live_session.update_context(expected_context)

        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0))

        assert LocalUserSession._instances["conv_0"] is live_session
        assert LocalUserSession._instances["conv_0"].context == expected_context

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

    def test_reproducibility_across_runs_with_stages(self) -> None:
        """Verify that two independent runs with the same seed generate
        the same system prompt for the same stage."""
        api_config = APIConfig(type=APIType.Completion)
        cr_config = ConversationReplayConfig(
            seed=42,
            num_conversations=2,
            shared_system_prompt_len=50,
            turns_per_conversation=Distribution(type="fixed", min=1, max=1, mean=1, std_dev=0),
            input_tokens_per_turn=Distribution(type="fixed", min=10, max=10, mean=10, std_dev=0),
            output_tokens_per_turn=Distribution(type="fixed", min=10, max=10, mean=10, std_dev=0),
        )
        data_config = DataConfig(type=DataGenType.ConversationReplay, conversation_replay=cr_config)

        def make_deterministic_mock_tok() -> MagicMock:
            mock_tok = MagicMock()
            hf_tok = MagicMock()
            hf_tok.vocab_size = 32000
            hf_tok.decode.side_effect = lambda ids, **kwargs: f"decoded_{ids}"
            mock_tok.get_tokenizer.return_value = hf_tok
            return mock_tok

        # Run 1
        gen1 = ConversationReplayDataGenerator(api_config, data_config, make_deterministic_mock_tok())
        # Stage 0
        gen1.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0, stage_id=0))
        context1_stage0 = LocalUserSession._instances["conv_0"].context

        LocalUserSession.clear_instances()

        # Stage 1
        gen1.load_lazy_data(LazyLoadInferenceAPIData(data_index=2, preferred_worker_id=0, stage_id=1))
        context1_stage1 = LocalUserSession._instances["conv_0"].context

        # Isolate Run 2
        LocalUserSession.clear_instances()

        # Run 2
        gen2 = ConversationReplayDataGenerator(api_config, data_config, make_deterministic_mock_tok())
        # Stage 0
        gen2.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0, stage_id=0))
        context2_stage0 = LocalUserSession._instances["conv_0"].context

        LocalUserSession.clear_instances()

        # Stage 1
        gen2.load_lazy_data(LazyLoadInferenceAPIData(data_index=2, preferred_worker_id=0, stage_id=1))
        context2_stage1 = LocalUserSession._instances["conv_0"].context

        # Verify reproducibility across runs for Stage 0
        assert context1_stage0 == context2_stage0

        # Verify reproducibility across runs for Stage 1
        assert context1_stage1 == context2_stage1

        # Verify that prompts are DIFFERENT across stages within Run 2
        assert context2_stage0 != context2_stage1

    def test_shared_system_prompt_within_stage(self) -> None:
        """Verify that different conversations in the same stage have the same system prompt."""
        api_config = APIConfig(type=APIType.Completion)
        cr_config = ConversationReplayConfig(
            seed=42,
            num_conversations=2,
            shared_system_prompt_len=50,
            turns_per_conversation=Distribution(type="fixed", min=1, max=1, mean=1, std_dev=0),
            input_tokens_per_turn=Distribution(type="fixed", min=10, max=10, mean=10, std_dev=0),
            output_tokens_per_turn=Distribution(type="fixed", min=10, max=10, mean=10, std_dev=0),
        )
        data_config = DataConfig(type=DataGenType.ConversationReplay, conversation_replay=cr_config)

        def make_deterministic_mock_tok() -> MagicMock:
            mock_tok = MagicMock()
            hf_tok = MagicMock()
            hf_tok.vocab_size = 32000
            hf_tok.decode.side_effect = lambda ids, **kwargs: f"decoded_{ids}"
            mock_tok.get_tokenizer.return_value = hf_tok
            return mock_tok

        gen = ConversationReplayDataGenerator(api_config, data_config, make_deterministic_mock_tok())

        # Stage 0
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0, stage_id=0))
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=1, preferred_worker_id=1, stage_id=0))

        context_conv0_s0 = LocalUserSession._instances["conv_0"].context
        context_conv1_s0 = LocalUserSession._instances["conv_1"].context
        assert context_conv0_s0 == context_conv1_s0

        # Transition to Stage 1
        LocalUserSession.clear_instances()
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0, stage_id=1))
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=1, preferred_worker_id=1, stage_id=1))

        context_conv0_s1 = LocalUserSession._instances["conv_0"].context
        context_conv1_s1 = LocalUserSession._instances["conv_1"].context

        # Verify they are still identical in Stage 1
        assert context_conv0_s1 == context_conv1_s1
        # Verify Stage 1 prompt is different from Stage 0 prompt
        assert context_conv0_s1 != context_conv0_s0

    def test_unique_system_prompt_within_stage_after_clear(self) -> None:
        """Verify that different conversations with dynamic_system_prompt_len get unique system prompts after stage clear."""
        api_config, data_config = _make_config(num_conversations=2)

        def make_deterministic_mock_tok() -> MagicMock:
            mock_tok = MagicMock()
            hf_tok = MagicMock()
            hf_tok.vocab_size = 32000
            hf_tok.decode.side_effect = lambda ids, **kwargs: f"decoded_{ids}"
            mock_tok.get_tokenizer.return_value = hf_tok
            return mock_tok

        gen = ConversationReplayDataGenerator(api_config, data_config, make_deterministic_mock_tok())

        # Stage 0 runtime priming
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0, stage_id=0))
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=1, preferred_worker_id=1, stage_id=0))

        context_conv0_s0 = LocalUserSession._instances["conv_0"].context
        context_conv1_s0 = LocalUserSession._instances["conv_1"].context
        assert context_conv0_s0 != context_conv1_s0

        # Simulate stage transition by clearing instances and moving to Stage 1
        LocalUserSession.clear_instances()

        # Load lazy data for both conversations for Stage 1
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0, stage_id=1))
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=1, preferred_worker_id=1, stage_id=1))

        context_conv0_s1 = LocalUserSession._instances["conv_0"].context
        context_conv1_s1 = LocalUserSession._instances["conv_1"].context

        # 1. The system prompts in Stage 1 should still be different from each other
        assert context_conv0_s1 != context_conv1_s1

        # 2. Stage 1 prompts should also be different from their Stage 0 versions (new stage prefix)
        assert context_conv0_s1 != context_conv0_s0
        assert context_conv1_s1 != context_conv1_s0

    def test_shared_system_prompt_cached_per_stage(self) -> None:
        """Verify that the shared system prompt is only generated once per stage and cached."""
        api_config, data_config = _make_config(num_conversations=2)
        mock_tok = _make_mock_tokenizer()

        # Track how many times decode is called.
        decode_count = 0

        def counting_decode(ids: Any, **kwargs: Any) -> str:
            nonlocal decode_count
            decode_count += 1
            return f"decoded_{ids}"

        mock_tok.get_tokenizer().decode.side_effect = counting_decode

        gen = ConversationReplayDataGenerator(api_config, data_config, mock_tok)

        # Reset decode count after initialization
        decode_count = 0

        # Simulate stage transition
        LocalUserSession.clear_instances()

        # Retrieve two sessions in the same stage
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0, stage_id=5))
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=1, preferred_worker_id=1, stage_id=5))

        # Decode should have been called exactly once for the shared prompt across both slot re-primes.
        assert decode_count == 1


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


class TestSlidingWindowTruncation:
    def test_sliding_window_truncation(self) -> None:
        from inference_perf.apis.user_session import LocalUserSession

        mock_tokenizer = MagicMock()
        # count_tokens returns 100 for system prompt and 100 for each turn
        mock_tokenizer.count_tokens.side_effect = lambda text, **kw: len(text.split()) * 10

        system_prompt = "System Instruction"  # length in words = 2 -> 20 tokens
        session = LocalUserSession(
            user_session_id="test_session",
            context=system_prompt,
            system_prompt=system_prompt,
            tokenizer=mock_tokenizer,
            max_model_len=50,
        )

        # turn 1: context grows by 10 tokens
        session.update_context(system_prompt + " Turn1")
        assert session.history == ["Turn1"]
        assert session.context == "System Instruction Turn1"

        # turn 2: context grows by 10 tokens
        session.update_context(session.context + " Turn2")
        assert session.history == ["Turn1", "Turn2"]
        assert session.context == "System Instruction Turn1 Turn2"

        # turn 3: context grows by 10 tokens -> System (20) + 30 = 50 tokens
        session.update_context(session.context + " Turn3")
        assert session.history == ["Turn1", "Turn2", "Turn3"]
        assert session.context == "System Instruction Turn1 Turn2 Turn3"

        # turn 4: context exceeds 50 tokens. First turn ("Turn1") is dropped.
        session.update_context(session.context + " Turn4")
        assert session.history == ["Turn2", "Turn3", "Turn4"]
        assert session.context == "System Instruction Turn2 Turn3 Turn4"


class TestRolloverDoesNotOrphanWaiters:
    """A session id must map to exactly one object.

    `load_lazy_data` runs per request at dispatch time, so it is called
    repeatedly for the same slot and `convo_num` and asks `_new_session` for an
    id that already exists. Requests resolve their session by id late
    (`UserSessionCompletionAPIData.user_session`), so replacing the object for a
    live id splits a request across two of them: it reads context from the old
    object and writes its response to the new one.

    Two consequences, both covered below. A turn parked on the old object's
    `asyncio.Future` is never woken, because every later release targets the new
    object — the stage's finished-request counter never reaches `num_requests`
    and the run hangs silently with no traceback (the failure reported in
    llm-d-benchmark#1688 with the `interactive-chat` workload). And a straggler
    that does complete writes its response into the wrong conversation's
    history.
    """

    @staticmethod
    def _rollover_index(gen: ConversationReplayDataGenerator, conv_idx: int = 0) -> int:
        """Smallest data_index for `conv_idx` hitting `turn_idx == 0 and round_num > 0`."""
        n_conv = len(gen.blueprints)
        # round_num = data_index // n_conv; turn_idx = round_num % num_turns
        return gen.blueprints[conv_idx].num_turns * n_conv + conv_idx

    def _fixed_turn_gen(self, num_conversations: int = 2, turns: int = 3) -> ConversationReplayDataGenerator:
        api_config, data_config = _make_config(
            num_conversations=num_conversations, turns_min=turns, turns_max=turns, turns_mean=turns
        )
        return ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

    async def test_rollover_does_not_orphan_queued_waiter(self) -> None:
        """A turn parked in get_context() must still complete across a rollover.

        Turn A holds the session lock, turn B queues behind it, then another
        dispatch for the same slot rebuilds the session id. B must not hang.
        """
        gen = self._fixed_turn_gen()
        idx = self._rollover_index(gen)

        # First dispatch performs the rollover and registers slot_0_convo_1.
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=idx, preferred_worker_id=0))
        live = gen.user_sessions[0]
        sid = live.user_session_id
        assert sid == "slot_0_convo_1"

        # Turn A acquires the session; turn B queues behind it.
        await live.get_context(0)
        waiter = asyncio.ensure_future(live.get_context(1))
        await asyncio.sleep(0)
        assert live._waiting_rounds is not None and live._waiting_rounds.qsize() == 1

        # A later dispatch for the same slot/conversation rebuilds the same id
        # while A and B are still in flight.
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=idx, preferred_worker_id=0))

        # Turn A completes and releases through the registered session, which is
        # how the client path resolves it (`data.user_session`).
        LocalUserSession.get_instance(sid).update_context("A done")

        try:
            await asyncio.wait_for(waiter, timeout=0.5)
        except asyncio.TimeoutError:
            waiter.cancel()
            pytest.fail(
                f"turn parked in get_context() was orphaned by the rollover of {sid!r}: "
                "its future belongs to the replaced session object, so no later "
                "update_context() can resolve it. The request never completes and the "
                "stage hangs forever (llm-d-benchmark#1688)."
            )

    async def test_repeated_dispatch_keeps_one_object_per_session_id(self) -> None:
        """Re-dispatching the same data_index must not rebuild the session.

        This is the invariant the other two tests depend on: a request resolves
        its session by id late, so two objects for one id split it in half.
        """
        gen = self._fixed_turn_gen()
        idx = self._rollover_index(gen)

        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=idx, preferred_worker_id=0))
        live = gen.user_sessions[0]
        sid = live.user_session_id

        await live.get_context(0)  # an in-flight turn holds the session

        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=idx, preferred_worker_id=0))

        assert gen.user_sessions[0] is live, (
            f"repeated dispatch for data_index={idx} rebuilt session {sid!r}. A request "
            "already holding the previous object would read its context from that object "
            "and write its response to the new one."
        )
        assert LocalUserSession.get_instance(sid) is live, (
            f"registry entry for {sid!r} no longer points at the live session, so "
            "update_context() would resolve to a different object than get_context() did."
        )

    async def test_rollover_does_not_leak_turn_into_other_conversation(self) -> None:
        """A straggler's response must not land in another conversation's history.

        A turn parked across a rollover reads its context from the session it was
        dispatched against. If the id has since been rebound to a fresh
        conversation, that response is appended to the wrong history — polluting
        the prefix distribution this generator exists to control.
        """
        gen = self._fixed_turn_gen()
        idx = self._rollover_index(gen)

        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=idx, preferred_worker_id=0))
        session = gen.user_sessions[0]
        sid = session.user_session_id

        # Give the conversation some accumulated history.
        session.history = ["first turn", "second turn"]
        session.context = f"{session.system_prompt} {' '.join(session.history)}"

        # One turn in flight, a straggler queued behind it.
        await session.get_context(2)
        straggler = asyncio.ensure_future(session.get_context(3))
        await asyncio.sleep(0)

        # Another dispatch for the same slot arrives mid-flight.
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=idx, preferred_worker_id=0))

        # The in-flight turn completes, resolving through the registry.
        registered = LocalUserSession.get_instance(sid)
        registered.update_context(f"{registered.context} third turn")

        context_sent = await asyncio.wait_for(straggler, timeout=1.0)
        assert context_sent.startswith(session.system_prompt), (
            f"straggler was handed a context built from a different conversation's system prompt: {context_sent[:80]!r}"
        )

        # The straggler must still be talking to the object it was dispatched
        # against, so its response extends that conversation and nothing else.
        after = LocalUserSession.get_instance(sid)
        assert after is session, (
            "the straggler's context came from one object but its response would be "
            "written to another, so the two halves of the request disagree."
        )

        history_before = list(after.history)
        after.update_context(f"{context_sent} fourth turn")
        assert after.history[: len(history_before)] == history_before, (
            f"straggler's response rewrote earlier history: {history_before} -> {after.history}"
        )
        # Exactly one turn is appended, and it is the straggler's own.
        assert after.history == history_before + ["fourth turn"], (
            "straggler's response was not appended cleanly as a single new turn — the "
            "system-prompt prefix strip in update_context() was computed against a "
            f"different conversation: {after.history}"
        )

    def test_rolled_over_slot_keeps_system_prompt_across_stages(self) -> None:
        """Re-priming a rolled-over slot in a later stage must set system_prompt.

        `LoadGenerator` calls `clear_instances()` between stages, so the next
        dispatch re-primes the slot under its rolled-over id. If that session is
        built without a `system_prompt`, `update_context` takes its
        `else` branch (`user_session.py:85`) and assigns the raw response to
        `context` — silently disabling history tracking and the
        `max_model_len` sliding window for the rest of the run.
        """
        gen = self._fixed_turn_gen()
        idx = self._rollover_index(gen)

        # Stage 0: roll slot 0 over onto a fresh conversation.
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=idx, preferred_worker_id=0, stage_id=0))
        rolled_over_id = gen.user_sessions[0].user_session_id
        assert rolled_over_id == "slot_0_convo_1"

        # LoadGenerator clears the registry between stages.
        LocalUserSession.clear_instances()

        # Stage 1: the same slot is re-primed under its rolled-over id.
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=idx, preferred_worker_id=0, stage_id=1))
        session = LocalUserSession.get_instance(rolled_over_id)

        assert session.system_prompt, (
            f"session {rolled_over_id!r} was re-primed for stage 1 without a system_prompt, "
            "so update_context() falls back to assigning the raw response and stops "
            "tracking history or enforcing max_model_len."
        )
        assert session.context.startswith(session.system_prompt)

        # History tracking is live: a response is split into a new turn.
        session.update_context(f"{session.system_prompt} a new turn")
        assert session.history == ["a new turn"], f"history tracking is disabled after the stage boundary: {session.history}"
