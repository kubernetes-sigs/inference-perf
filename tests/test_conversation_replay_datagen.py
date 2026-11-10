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
from inference_perf.datagen.replay.conversation_replay_datagen import (
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

    @pytest.mark.asyncio
    async def test_first_request_separates_independently_decoded_prompts(self) -> None:
        """Conversation replay keeps its legacy boundary between prompt pieces."""
        api_config, data_config = _make_config(num_conversations=1)
        tokenizer = _make_mock_tokenizer()
        tokenizer.get_tokenizer().encode.side_effect = lambda text: list(range(len(text)))
        gen = ConversationReplayDataGenerator(api_config, data_config, tokenizer)

        result = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0))
        assert isinstance(result, _ConversationReplayAPIData)
        result.user_session.system_prompt = "SYSTEM"
        result.user_session.context = "SYSTEM"
        result.prompt = "TURN"

        body = await result.to_request_body("model", 64, False, False)
        assert body["prompt"] == "SYSTEM TURN"

    @pytest.mark.asyncio
    async def test_first_request_keeps_legacy_separator_before_leading_space(self) -> None:
        """A tokenizer-decoded leading space remains in addition to the old separator."""
        api_config, data_config = _make_config(num_conversations=1)
        tokenizer = _make_mock_tokenizer()
        tokenizer.get_tokenizer().encode.side_effect = lambda text: list(range(len(text)))
        gen = ConversationReplayDataGenerator(api_config, data_config, tokenizer)

        result = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0))
        assert isinstance(result, _ConversationReplayAPIData)
        result.user_session.system_prompt = "SYSTEM"
        result.user_session.context = "SYSTEM"
        result.prompt = " TURN"

        body = await result.to_request_body("model", 64, False, False)
        assert body["prompt"] == "SYSTEM  TURN"

    def test_turn_recycling(self) -> None:
        """A completed slot wraps to a fresh conversation after its final turn."""
        api_config, data_config = _make_config(num_conversations=1, turns_min=3, turns_max=3, turns_mean=3)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        requests = []
        for data_index in range(4):
            request = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=data_index, preferred_worker_id=0))
            assert isinstance(request, _ConversationReplayAPIData)
            requests.append(request)
            request.on_request_finished()

        assert [request.target_round for request in requests] == [0, 1, 2, 3]
        assert [request.user_session_id for request in requests] == ["conv_0"] * 3 + ["stage_0_slot_0_convo_1"]

    def test_completed_conversation_replenishes_a_busy_queue_slot(self) -> None:
        """A ready conversation advances instead of waiting behind the fixed queue order."""
        api_config, data_config = _make_config(num_conversations=3, turns_min=2, turns_max=2, turns_mean=2)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        first_round = [gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=i, preferred_worker_id=i)) for i in range(3)]
        ready = first_round[2]
        assert isinstance(ready, _ConversationReplayAPIData)
        ready.on_request_finished()

        # The next fixed token names slot 0, which is still busy. Slot 2 is ready
        # and must advance immediately instead of consuming capacity waiting on 0.
        replenished = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=3, preferred_worker_id=0))
        assert isinstance(replenished, _ConversationReplayAPIData)
        assert replenished.user_session_id == "conv_2"
        assert replenished.target_round == 1

    def test_ready_conversation_preserves_the_active_set(self) -> None:
        """A ready slot wins over an unseen slot when concurrency is below the pool size."""
        api_config, data_config = _make_config(num_conversations=3, turns_min=2, turns_max=2, turns_mean=2)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        first = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0))
        second = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=1, preferred_worker_id=1))
        assert isinstance(first, _ConversationReplayAPIData)
        assert isinstance(second, _ConversationReplayAPIData)
        first.on_request_finished()

        third = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=2, preferred_worker_id=2))
        assert isinstance(third, _ConversationReplayAPIData)
        assert third.user_session_id == "conv_0"
        assert third.target_round == 1

    # Four two-turn blueprints must all finish conversations within 32 requests
    # at concurrency 1 or 2; concurrency 4 pins the existing full-pool behavior.
    @pytest.mark.parametrize("concurrency", [1, 2, 4])
    async def test_completed_conversations_cover_pool_below_pool_size(self, concurrency: int) -> None:
        api_config, data_config = _make_config(num_conversations=4, turns_min=2, turns_max=2, turns_mean=2)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        prompt_identity: dict[str, tuple[int, int]] = {}
        for blueprint in gen.blueprints:
            blueprint.turn_prompts = [f"blueprint_{blueprint.conversation_id}_turn_{turn}" for turn in range(2)]
            for turn, prompt in enumerate(blueprint.turn_prompts):
                prompt_identity[prompt] = (blueprint.conversation_id, turn)

        completed_turns: dict[str, list[tuple[int, int]]] = {}
        completed_blueprints: set[int] = set()
        data_iter = gen.get_data()
        for _ in range(32 // concurrency):
            requests: list[_ConversationReplayAPIData] = []
            for _ in range(concurrency):
                lazy = next(data_iter)
                assert isinstance(lazy, LazyLoadInferenceAPIData)
                request = gen.load_lazy_data(lazy)
                assert isinstance(request, _ConversationReplayAPIData)
                requests.append(request)

            # Complete simulated responses through the session lifecycle before
            # admitting the next batch; coverage is recorded only after completion.
            for request in requests:
                blueprint_id, turn = prompt_identity[request.prompt]
                session = request.user_session
                context = await asyncio.wait_for(session.get_context(request.target_round), timeout=1)
                session.update_context(f"{context} {request.prompt} response")
                request.on_request_finished()
                turns = completed_turns.setdefault(request.user_session_id, [])
                turns.append((blueprint_id, turn))
                assert turns == [(blueprint_id, expected_turn) for expected_turn in range(len(turns))]
                if turn == 1:
                    completed_blueprints.add(blueprint_id)

        assert completed_blueprints == {0, 1, 2, 3}, f"Completed blueprint coverage: {sorted(completed_blueprints)}"

    # Uneven worker budgets and blueprint partitions must cover the pool without
    # cross-worker duplication; five slots over five blueprints pins the full pool.
    @pytest.mark.parametrize("pool_size,concurrency,workers", [(5, 3, 2), (7, 4, 3), (5, 5, 2)])
    async def test_pool_rotation_respects_worker_ownership(self, pool_size: int, concurrency: int, workers: int) -> None:
        all_completed: set[int] = set()
        for worker_id in range(workers):
            # Worker processes have separate registries. Run their deterministic
            # completion streams separately to preserve that isolation in process.
            LocalUserSession.clear_instances()
            api_config, data_config = _make_config(num_conversations=pool_size)
            gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
            identities: dict[str, tuple[int, int]] = {}
            for blueprint in gen.blueprints:
                blueprint.num_turns = 1 + blueprint.conversation_id % 3
                blueprint.turn_prompts = [
                    f"blueprint_{blueprint.conversation_id}_turn_{turn}" for turn in range(blueprint.num_turns)
                ]
                blueprint.turn_output_lens = [10] * blueprint.num_turns
                for turn, prompt in enumerate(blueprint.turn_prompts):
                    identities[prompt] = (blueprint.conversation_id, turn)

            capacity = concurrency // workers + int(worker_id < concurrency % workers)
            tokens = (
                LazyLoadInferenceAPIData(data_index=i, preferred_worker_id=i % pool_size, active_worker_count=workers)
                for i in range(pool_size * 60)
                if (i % pool_size) % workers == worker_id
            )
            completed: set[int] = set()
            sessions: dict[str, list[tuple[int, int]]] = {}
            for _ in range(pool_size * 6):
                requests = [gen.load_lazy_data(next(tokens)) for _ in range(capacity)]
                for request in requests:
                    assert isinstance(request, _ConversationReplayAPIData)
                    blueprint_id, turn = identities[request.prompt]
                    assert blueprint_id % workers == worker_id
                    session = request.user_session
                    context = await asyncio.wait_for(session.get_context(request.target_round), timeout=1)
                    if turn == 0:
                        assert context == gen.blueprints[blueprint_id].system_prompt
                    session.update_context(f"{context} {request.prompt} response")
                    request.on_request_finished()
                    turns = sessions.setdefault(request.user_session_id, [])
                    turns.append((blueprint_id, turn))
                    assert turns == [(blueprint_id, t) for t in range(len(turns))]
                    if turn == gen.blueprints[blueprint_id].num_turns - 1:
                        completed.add(blueprint_id)

            assert completed == set(range(worker_id, pool_size, workers))
            assert all_completed.isdisjoint(completed)
            all_completed.update(completed)
        assert all_completed == set(range(pool_size))

    def test_materialization_failure_does_not_skip_a_turn(self, monkeypatch: pytest.MonkeyPatch) -> None:
        api_config, data_config = _make_config(num_conversations=1, turns_min=1, turns_max=1, turns_mean=1)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        first = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0))
        first.on_request_finished()

        original_new_session = gen._new_session
        monkeypatch.setattr(gen, "_new_session", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))
        with pytest.raises(RuntimeError, match="boom"):
            gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=1, preferred_worker_id=0))

        monkeypatch.setattr(gen, "_new_session", original_new_session)
        retried = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=2, preferred_worker_id=0))
        assert isinstance(retried, _ConversationReplayAPIData)
        assert retried.target_round == 1

    def test_rollover_waits_for_the_previous_conversation_to_finish(self) -> None:
        """A slot gets a fresh session only after its final old turn completes."""
        api_config, data_config = _make_config(num_conversations=1, turns_min=2, turns_max=2, turns_mean=2)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

        first = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0))
        assert isinstance(first, _ConversationReplayAPIData)
        first.on_request_finished()
        final = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=1, preferred_worker_id=0))
        assert isinstance(final, _ConversationReplayAPIData)
        assert final.user_session_id == "conv_0"
        final.on_request_finished()

        rollover = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=2, preferred_worker_id=0))
        assert isinstance(rollover, _ConversationReplayAPIData)
        assert rollover.user_session_id == "stage_0_slot_0_convo_1"
        assert rollover.target_round == 2

    def test_request_finished_is_idempotent(self) -> None:
        api_config, data_config = _make_config(num_conversations=1)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        request = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0))
        assert isinstance(request, _ConversationReplayAPIData)

        request.on_request_finished()
        request.on_request_finished()

        assert list(gen._ready_slots) == [0]

    async def test_request_finished_releases_a_session_lock_after_early_failure(self) -> None:
        api_config, data_config = _make_config(num_conversations=1)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        request = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0))
        assert isinstance(request, _ConversationReplayAPIData)
        session = request.user_session
        await session.get_context(request.target_round)
        assert session._in_flight is not None and session._in_flight.locked()

        request.on_request_finished()

        assert not session._in_flight.locked()
        assert list(gen._ready_slots) == [0]

    async def test_request_finished_releases_slot_when_session_cleanup_fails(self, monkeypatch: pytest.MonkeyPatch) -> None:
        api_config, data_config = _make_config(num_conversations=1)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        request = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0))
        assert isinstance(request, _ConversationReplayAPIData)
        session = request.user_session
        await session.get_context(request.target_round)

        def fail_cleanup(context: str) -> None:
            raise RuntimeError("cleanup failed")

        monkeypatch.setattr(session, "update_context", fail_cleanup)
        with pytest.raises(RuntimeError, match="cleanup failed"):
            request.on_request_finished()

        assert session._in_flight is not None and not session._in_flight.locked()
        assert list(gen._ready_slots) == [0]

    # Stage teardown clears the session registry while scheduled tasks may still finish.
    # Resolving through the user_session property would recreate the cleared id, so the
    # next stage would see the id as live and skip its re-prime; the id must stay absent.
    def test_request_finished_does_not_resurrect_a_cleared_session(self) -> None:
        api_config, data_config = _make_config(num_conversations=1)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        request = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0))
        assert isinstance(request, _ConversationReplayAPIData)

        LocalUserSession.clear_instances()
        request.on_request_finished()

        assert request.user_session_id not in LocalUserSession._instances
        assert list(gen._ready_slots) == [0]

    # A stage-0 request that ends after stage 1 has started must not mark the stage-1
    # slot ready, and must not release the lock the stage-1 request holds: stage teardown
    # clears the registry and the next stage re-primes the same id with a new session
    # object, so the stale task must clean up only the object it was created against.
    async def test_release_slot_ignores_a_callback_from_an_earlier_stage(self) -> None:
        api_config, data_config = _make_config(num_conversations=1)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        old_stage = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0, stage_id=0))
        assert isinstance(old_stage, _ConversationReplayAPIData)
        old_session = old_stage.user_session

        LocalUserSession.clear_instances()
        current_stage = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0, stage_id=1))
        assert isinstance(current_stage, _ConversationReplayAPIData)
        re_primed = current_stage.user_session
        assert re_primed is not old_session
        await re_primed.get_context(current_stage.target_round)
        assert re_primed._in_flight is not None and re_primed._in_flight.locked()

        old_stage.on_request_finished()

        assert gen._slot_states[0].busy is True
        assert list(gen._ready_slots) == []
        assert re_primed._in_flight is not None and re_primed._in_flight.locked()

    def test_no_ready_conversation_fails_instead_of_waiting_with_capacity(self) -> None:
        api_config, data_config = _make_config(num_conversations=1)
        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0))

        with pytest.raises(RuntimeError, match="no ready conversation"):
            gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=1, preferred_worker_id=0))

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

    def test_output_distribution_leaving_no_prompt_budget_is_rejected(self) -> None:
        """An output ceiling that consumes the whole context fails at init, not mid-benchmark."""
        api_config = APIConfig(type=APIType.Completion)
        cr_config = ConversationReplayConfig(
            seed=42,
            num_conversations=1,
            shared_system_prompt_len=50,
            max_model_len=1000,
            turns_per_conversation=Distribution(type="fixed", min=1, max=1, mean=1, std_dev=0),
            input_tokens_per_turn=Distribution(type="normal", min=10, max=50, mean=20, std_dev=5),
            # 900 + 200 buffer >= 1000, so no prompt could ever fit.
            output_tokens_per_turn=Distribution(type="normal", min=10, max=900, mean=100, std_dev=50),
        )
        data_config = DataConfig(type=DataGenType.ConversationReplay, conversation_replay=cr_config)

        with pytest.raises(ValueError, match="leaves no room for a prompt"):
            ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

    def test_fixed_output_distribution_validates_mean_not_max(self) -> None:
        """Fixed sampling returns int(mean) unclipped, so mean is the real ceiling."""
        api_config = APIConfig(type=APIType.Completion)
        cr_config = ConversationReplayConfig(
            seed=42,
            num_conversations=1,
            shared_system_prompt_len=50,
            max_model_len=1000,
            turns_per_conversation=Distribution(type="fixed", min=1, max=1, mean=1, std_dev=0),
            input_tokens_per_turn=Distribution(type="normal", min=10, max=50, mean=20, std_dev=5),
            # max is well within budget, but fixed sampling ignores it and yields 5000.
            output_tokens_per_turn=Distribution(type="fixed", min=10, max=100, mean=5000, std_dev=0),
        )
        data_config = DataConfig(type=DataGenType.ConversationReplay, conversation_replay=cr_config)

        with pytest.raises(ValueError, match=r"output_tokens_per_turn\.mean \(5000\)"):
            ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

    def test_omitted_output_distribution_validates_the_default(self) -> None:
        """With no output distribution the 256-token fallback still has to fit."""
        api_config = APIConfig(type=APIType.Completion)
        cr_config = ConversationReplayConfig(
            seed=42,
            num_conversations=1,
            shared_system_prompt_len=10,
            # 256 fallback + 200 buffer >= 400, so even the default cannot fit.
            max_model_len=400,
            turns_per_conversation=Distribution(type="fixed", min=1, max=1, mean=1, std_dev=0),
            input_tokens_per_turn=Distribution(type="normal", min=10, max=50, mean=20, std_dev=5),
            output_tokens_per_turn=None,
        )
        data_config = DataConfig(type=DataGenType.ConversationReplay, conversation_replay=cr_config)

        with pytest.raises(ValueError, match=r"the default output length \(256\)"):
            ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

    def test_output_distribution_within_prompt_budget_is_accepted(self) -> None:
        """An output max that still leaves prompt room builds normally."""
        api_config = APIConfig(type=APIType.Completion)
        cr_config = ConversationReplayConfig(
            seed=42,
            num_conversations=1,
            shared_system_prompt_len=50,
            max_model_len=1000,
            turns_per_conversation=Distribution(type="fixed", min=1, max=1, mean=1, std_dev=0),
            input_tokens_per_turn=Distribution(type="normal", min=10, max=50, mean=20, std_dev=5),
            # 799 + 200 < 1000, so a prompt still fits.
            output_tokens_per_turn=Distribution(type="normal", min=10, max=799, mean=100, std_dev=50),
        )
        data_config = DataConfig(type=DataGenType.ConversationReplay, conversation_replay=cr_config)

        gen = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        assert len(gen.blueprints) == 1

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

    Requests resolve their session by id late
    (`UserSessionCompletionAPIData.user_session`), so replacing the object for a
    live id splits a request across two of them: it reads context from the old
    object and writes its response to the new one.

    In a real run the collision comes from the stage boundary. `get_data()`
    restarts `data_index` at 0 every stage while each slot keeps the id it
    rolled over to, so a stage-1 slot is re-primed as the stage-0 rolled-over
    conversation and its first rollover used to compute that same id again,
    with earlier turns still in flight.

    Consequences covered below. A turn parked on the old object's
    `asyncio.Future` is never woken, because every later release targets the new
    object — the stage's finished-request counter never reaches `num_requests`
    and the run hangs silently with no traceback (#700, and likely
    llm-d-benchmark#1688 with the `interactive-chat` workload). A straggler
    that does complete writes its response into the wrong conversation's
    history. And a rollover that reuses the live id instead of starting fresh
    makes every later stage replay conversations longer than the blueprint.
    """

    @staticmethod
    def _load(gen: ConversationReplayDataGenerator, conv_idx: int, stage_id: int = 0) -> _ConversationReplayAPIData:
        """Dispatch the next turn for one conversation slot."""
        request = gen.load_lazy_data(
            LazyLoadInferenceAPIData(data_index=conv_idx, preferred_worker_id=conv_idx, stage_id=stage_id)
        )
        assert isinstance(request, _ConversationReplayAPIData)
        return request

    @classmethod
    def _complete(cls, request: _ConversationReplayAPIData, response: str = "done") -> None:
        """Append a response to the session and return the slot to the ready queue."""
        session = LocalUserSession.get_instance(request.user_session_id)
        session.update_context(f"{session.context} {response}")
        request.on_request_finished()

    def _advance_to_rollover(
        self, gen: ConversationReplayDataGenerator, conv_idx: int = 0, stage_id: int = 0
    ) -> _ConversationReplayAPIData:
        """Complete every turn of the slot's current conversation so the next dispatch rolls over."""
        for _ in range(gen.blueprints[conv_idx].num_turns):
            self._complete(self._load(gen, conv_idx, stage_id))
        return self._load(gen, conv_idx, stage_id)

    def _fixed_turn_gen(self, num_conversations: int = 2, turns: int = 3) -> ConversationReplayDataGenerator:
        api_config, data_config = _make_config(
            num_conversations=num_conversations, turns_min=turns, turns_max=turns, turns_mean=turns
        )
        return ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())

    async def test_rollover_does_not_orphan_queued_waiter(self) -> None:
        """A turn parked in get_context() must survive a rebuild of its session id.

        Turn A holds the session lock, turn B queues behind it, then the same
        data_index is materialized again. A real run never does that within a
        stage; it pins the invariant the cross-stage tests below rely on.
        B must not hang.
        """
        gen = self._fixed_turn_gen()

        # Completing the blueprint's turns rolls the slot onto a fresh session id.
        self._advance_to_rollover(gen)
        live = gen.user_sessions[0]
        sid = live.user_session_id
        assert sid != "conv_0", "the rollover did not start a new conversation"

        # Turn A acquires the session; turn B queues behind it.
        await live.get_context(0)
        waiter = asyncio.ensure_future(live.get_context(1))
        await asyncio.sleep(0)
        assert live._waiting_rounds is not None and live._waiting_rounds.qsize() == 1

        # A duplicate dispatch while the slot is busy is rejected before it can
        # rebuild the id, so the request never splits across two session objects.
        with pytest.raises(RuntimeError, match="no ready conversation"):
            self._load(gen, 0)

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

        rolled = self._advance_to_rollover(gen)
        live = gen.user_sessions[0]
        sid = live.user_session_id

        await live.get_context(rolled.target_round)  # an in-flight turn holds the session

        # Rebuilding the same id (the rollover/re-prime path) must return the live
        # object rather than replacing it.
        rebuilt = gen._new_session(sid, context="other", system_prompt="other")

        assert rebuilt is live, (
            f"rebuilding session {sid!r} replaced the live object. A request already "
            "holding the previous object would read its context from that object and "
            "write its response to the new one."
        )
        assert gen.user_sessions[0] is live
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

        rolled = self._advance_to_rollover(gen)
        session = gen.user_sessions[0]
        sid = session.user_session_id

        # Give the conversation some accumulated history.
        session.history = ["first turn", "second turn"]
        session.context = f"{session.system_prompt} {' '.join(session.history)}"

        # One turn in flight, a straggler queued behind it.
        await session.get_context(rolled.target_round)
        straggler = asyncio.ensure_future(session.get_context(rolled.target_round + 1))
        await asyncio.sleep(0)

        # A rebuild for the live id returns that same object, so the straggler's
        # context and its response stay on one session.
        rebuilt = gen._new_session(sid, context="other", system_prompt="other")
        assert rebuilt is session

        # The in-flight turn completes, resolving through the registry.
        registered = LocalUserSession.get_instance(sid)
        registered.update_context(f"{registered.context} third turn")

        try:
            context_sent = await asyncio.wait_for(straggler, timeout=1.0)
        except asyncio.TimeoutError:
            straggler.cancel()
            pytest.fail(f"straggler parked on {sid!r} was orphaned when the id was rebuilt; the stage would hang.")
        # Both conversations on a slot share its system prompt, so check the history it carries.
        assert context_sent == f"{session.system_prompt} first turn second turn third turn", (
            f"straggler was handed a context from a different conversation: {context_sent[:120]!r}"
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
        stage's first dispatch for the slot re-primes it under its rolled-over
        id. If that session is built without a `system_prompt`,
        `update_context` takes its `else` branch and assigns the raw response
        to `context` — silently disabling history tracking and the
        `max_model_len` sliding window until the slot's next rollover.
        """
        gen = self._fixed_turn_gen()

        # Stage 0: roll slot 0 over onto a fresh conversation.
        rolled = self._advance_to_rollover(gen, stage_id=0)
        rolled_over_id = rolled.user_session_id
        assert rolled_over_id != "conv_0", "the rollover did not start a new conversation"

        # LoadGenerator clears the registry between stages.
        LocalUserSession.clear_instances()

        # Stage 1 restarts data_index at 0, so the slot is re-primed under its rolled-over id.
        first = self._load(gen, 0, stage_id=1)
        assert first.user_session_id == rolled_over_id
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

    async def test_later_stage_rollover_does_not_orphan_queued_waiter(self) -> None:
        """The real-run trigger: a stage-1 rollover must not orphan the slot's in-flight turns.

        Stage 1 re-primes the slot under its stage-0 rolled-over id. Because rollover
        is serialized after the final turn, the live id is never rebound while a
        waiter is parked behind it.
        """
        gen = self._fixed_turn_gen()
        num_turns = gen.blueprints[0].num_turns

        # Stage 0 rolls slot 0 over, then the registry is cleared between stages.
        self._advance_to_rollover(gen, stage_id=0)
        LocalUserSession.clear_instances()

        # Walk stage 1 to its final turn, leaving a waiter parked behind it.
        first = self._load(gen, 0, stage_id=1)
        sid = first.user_session_id
        held = LocalUserSession.get_instance(sid)
        for _ in range(num_turns - 1):
            self._complete(first)
            first = self._load(gen, 0, stage_id=1)
        await held.get_context(first.target_round)
        waiter = asyncio.ensure_future(held.get_context(first.target_round + 1))
        await asyncio.sleep(0)

        # Rollover cannot happen while the slot is busy; the id stays bound to `held`.
        with pytest.raises(RuntimeError, match="no ready conversation"):
            self._load(gen, 0, stage_id=1)

        # A completes through the registry, as the client path resolves it.
        LocalUserSession.get_instance(sid).update_context(f"{held.system_prompt} A done")
        first.on_request_finished()

        try:
            await asyncio.wait_for(waiter, timeout=0.5)
        except asyncio.TimeoutError:
            waiter.cancel()
            pytest.fail(
                f"turn parked on {sid!r} was orphaned by the stage-1 rollover: the stage hangs "
                "forever waiting for a request that can never complete (#700)."
            )
        assert LocalUserSession.get_instance(sid) is held

        rolled = self._load(gen, 0, stage_id=1)
        assert rolled.user_session_id != sid, f"stage-1 rollover reused the in-flight id {sid!r}"
        assert LocalUserSession.get_instance(sid) is held

    def test_later_stage_rollover_starts_a_fresh_conversation(self) -> None:
        """Every rollover, in every stage, must start the slot on an empty history.

        Reusing a live id at the stage-1 rollover would keep appending to the
        conversation already on the slot, so later stages replay conversations
        longer than the blueprint and send ever-growing contexts.
        """
        gen = self._fixed_turn_gen()
        n_conv = len(gen.blueprints)

        for stage_id in range(3):
            LocalUserSession.clear_instances()
            seen_ids: set[str] = set()
            # Three complete conversations per slot.
            for _ in range(3 * gen.blueprints[0].num_turns):
                for conv_idx in range(n_conv):
                    request = self._load(gen, conv_idx, stage_id)
                    turn_idx = request.target_round % gen.blueprints[conv_idx].num_turns
                    session = LocalUserSession.get_instance(request.user_session_id)
                    assert len(session.history) == turn_idx, (
                        f"stage {stage_id} turn {request.target_round} of {request.user_session_id!r} "
                        f"would be sent with {len(session.history)} prior turns"
                    )
                    if turn_idx == 0:
                        assert request.user_session_id not in seen_ids, (
                            f"stage {stage_id}: rollover reused {request.user_session_id!r} instead of a fresh id"
                        )
                    seen_ids.add(request.user_session_id)
                    self._complete(request)

    async def test_successive_generators_do_not_adopt_each_others_sessions(self) -> None:
        """A second generator in the same process must build its own sessions.

        The registry is process-global and initial ids are fixed (`conv_N`), so a
        generator built without a registry clear in between finds the previous
        generator's entries under its ids. Adopting them would carry over that
        generator's system prompt, history and max_model_len.
        """
        api_config, data_config = _make_config(num_conversations=2)
        first = ConversationReplayDataGenerator(api_config, data_config, _make_mock_tokenizer())
        earlier = LocalUserSession.get_instance("conv_0")
        earlier.update_context(f"{earlier.system_prompt} earlier conversation history")

        assert data_config.conversation_replay is not None
        other = data_config.conversation_replay.model_copy(update={"seed": 99, "max_model_len": 1000})
        second = ConversationReplayDataGenerator(
            api_config, data_config.model_copy(update={"conversation_replay": other}), _make_mock_tokenizer()
        )

        session = LocalUserSession.get_instance("conv_0")
        assert session is not earlier, "the second generator adopted the first generator's session"
        assert session is second.user_sessions[0]
        assert session.max_model_len == 1000
        assert session.system_prompt == second.blueprints[0].system_prompt
        assert session.history == []

        data = second.load_lazy_data(LazyLoadInferenceAPIData(data_index=0, preferred_worker_id=0))
        body = await data.to_request_body("model", 0, True, False)
        assert "earlier conversation history" not in body["prompt"]
        assert first.user_sessions[0] is earlier
