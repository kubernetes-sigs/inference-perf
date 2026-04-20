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

import pytest
from unittest.mock import MagicMock

from inference_perf.config import (
    APIConfig,
    APIType,
    DataConfig,
    DataGenType,
    Distribution,
    DistributionType,
    SharedPrefix,
    CustomTokenizerConfig,
)
from inference_perf.datagen.shared_prefix_datagen import SharedPrefixDataGenerator
from inference_perf.utils.custom_tokenizer import CustomTokenizer

# --- Tests from HEAD (Mock based) ---


def _make_mock_tokenizer(vocab_size: int = 1000) -> MagicMock:
    """Create a mock tokenizer that returns predictable text."""
    mock_tokenizer = MagicMock()
    hf_tok = MagicMock()
    hf_tok.vocab_size = vocab_size
    hf_tok.decode = MagicMock(side_effect=lambda ids, **kw: f"text_{len(ids)}")
    hf_tok.batch_decode = MagicMock(side_effect=lambda batch, **kw: [f"text_{len(ids)}" for ids in batch])
    mock_tokenizer.get_tokenizer.return_value = hf_tok
    return mock_tokenizer


def _make_generator(shared_prefix: SharedPrefix) -> SharedPrefixDataGenerator:
    api_config = APIConfig(type=APIType.Completion)
    data_config = DataConfig(type=DataGenType.SharedPrefix, shared_prefix=shared_prefix)
    return SharedPrefixDataGenerator(api_config, data_config, _make_mock_tokenizer())


class TestDeterministicSeeding:
    def test_same_seed_produces_identical_output(self) -> None:
        sp = SharedPrefix(num_groups=3, num_prompts_per_group=5, question_len=50, output_len=50, seed=42)
        gen1 = _make_generator(sp)
        gen2 = _make_generator(SharedPrefix(num_groups=3, num_prompts_per_group=5, question_len=50, output_len=50, seed=42))
        assert gen1.prompts == gen2.prompts
        assert gen1.flat_output_lens == gen2.flat_output_lens

    def test_different_seeds_produce_different_output(self) -> None:
        # Use distributions so lengths vary and differences are observable
        dist = Distribution(type=DistributionType.NORMAL, mean=100.0, min=10, max=500, std_dev=50.0)
        gen1 = _make_generator(SharedPrefix(num_groups=3, num_prompts_per_group=5, question_len=dist, output_len=50, seed=42))
        gen2 = _make_generator(SharedPrefix(num_groups=3, num_prompts_per_group=5, question_len=dist, output_len=50, seed=99))
        assert gen1.question_len_list_per_group != gen2.question_len_list_per_group

    def test_no_seed_is_nondeterministic(self) -> None:
        # Use distributions so lengths vary between unseeded generators
        dist = Distribution(type=DistributionType.NORMAL, mean=100.0, min=10, max=500, std_dev=50.0)
        gen1 = _make_generator(SharedPrefix(num_groups=3, num_prompts_per_group=10, question_len=dist, output_len=50))
        gen2 = _make_generator(SharedPrefix(num_groups=3, num_prompts_per_group=10, question_len=dist, output_len=50))
        assert gen1.question_len_list_per_group != gen2.question_len_list_per_group


class TestFixedLengths:
    def test_all_output_lens_equal_when_fixed(self) -> None:
        gen = _make_generator(SharedPrefix(num_groups=2, num_prompts_per_group=5, output_len=42, seed=1))
        assert all(v == 42 for v in gen.flat_output_lens)

    def test_prompt_count_matches_groups_times_prompts(self) -> None:
        gen = _make_generator(SharedPrefix(num_groups=4, num_prompts_per_group=7, seed=1))
        assert len(gen.prompts) == 4 * 7
        assert len(gen.flat_output_lens) == 4 * 7


class TestDistributionLengths:
    def test_question_len_distribution_within_bounds(self) -> None:
        sp = SharedPrefix(
            num_groups=2,
            num_prompts_per_group=50,
            question_len=Distribution(type=DistributionType.NORMAL, mean=200.0, min=50, max=500, std_dev=50.0),
            output_len=50,
            seed=42,
        )
        gen = _make_generator(sp)
        # Check the per-group question lengths are within bounds
        for group_lens in gen.question_len_list_per_group:
            assert all(50 <= v <= 500 for v in group_lens)

    def test_output_len_distribution_within_bounds(self) -> None:
        sp = SharedPrefix(
            num_groups=2,
            num_prompts_per_group=50,
            question_len=50,
            output_len=Distribution(type=DistributionType.LOGNORMAL, mean=150.0, min=1, max=4096, std_dev=60.0),
            seed=42,
        )
        gen = _make_generator(sp)
        assert all(1 <= v <= 4096 for v in gen.flat_output_lens)

    def test_system_prompt_len_distribution_varies_per_group(self) -> None:
        sp = SharedPrefix(
            num_groups=5,
            num_prompts_per_group=3,
            system_prompt_len=Distribution(type=DistributionType.UNIFORM, mean=100.0, min=50, max=200),
            seed=42,
        )
        gen = _make_generator(sp)
        # With uniform distribution over [50, 200], 5 groups should have varying lengths
        assert len(set(gen.system_prompt_lens_per_group)) > 1


class TestLegacyCompatibility:
    def test_legacy_question_distribution_still_works(self) -> None:
        sp = SharedPrefix(
            num_groups=2,
            num_prompts_per_group=50,
            question_len=50,
            question_distribution=Distribution(min=30, max=200, mean=100, std_dev=30),
            output_len=50,
            seed=42,
        )
        gen = _make_generator(sp)
        # Should have variation from the legacy distribution
        for group_lens in gen.question_len_list_per_group:
            assert all(30 <= v <= 200 for v in group_lens)

    def test_legacy_output_distribution_still_works(self) -> None:
        sp = SharedPrefix(
            num_groups=2,
            num_prompts_per_group=50,
            question_len=50,
            output_len=50,
            output_distribution=Distribution(min=10, max=500, mean=100, std_dev=50),
            seed=42,
        )
        gen = _make_generator(sp)
        assert all(10 <= v <= 500 for v in gen.flat_output_lens)


class TestMultiTurnChat:
    def test_user_sessions_created(self) -> None:
        sp = SharedPrefix(num_groups=2, num_prompts_per_group=5, enable_multi_turn_chat=True, seed=42)
        gen = _make_generator(sp)
        assert len(gen.user_sessions) == 10
        assert len(gen.prompts) == 10

    def test_get_data_yields_lazy_data(self) -> None:
        sp = SharedPrefix(num_groups=2, num_prompts_per_group=3, seed=42)
        gen = _make_generator(sp)
        data_iter = gen.get_data()
        items = [next(data_iter) for _ in range(10)]
        assert len(items) == 10
        assert all(hasattr(item, "data_index") for item in items)


# --- Tests from d168373 (Tokenizer based) ---


@pytest.fixture
def api_config() -> APIConfig:
    return APIConfig(
        api_type=APIType.Completion,
        model_name="gpt2",
        tokenizer_name="gpt2",
    )


@pytest.fixture
def shared_prefix_config() -> SharedPrefix:
    return SharedPrefix(
        num_groups=2,
        num_prompts_per_group=3,
        system_prompt_len=10,
        question_len=5,
        output_len=15,
    )


@pytest.fixture
def data_config(shared_prefix_config: SharedPrefix) -> DataConfig:
    return DataConfig(shared_prefix=shared_prefix_config)


@pytest.fixture
def tokenizer() -> CustomTokenizer:
    tokenizer_config = CustomTokenizerConfig(pretrained_model_name_or_path="gpt2")
    return CustomTokenizer(tokenizer_config)


def test_shared_prefix_datagen_token_counts(
    api_config: APIConfig, data_config: DataConfig, tokenizer: CustomTokenizer, shared_prefix_config: SharedPrefix
) -> None:
    generator = SharedPrefixDataGenerator(api_config, data_config, tokenizer)

    # The generator shuffles prompts, so we test all of them
    for prompt in generator.prompts:
        # It's tricky to perfectly split the prompt back into system and question
        # because of how tokenization and decoding works (e.g., spaces).
        # Instead, we'll check the total token count.

        # The expected total length is system_prompt_len + question_len.
        # There's also a space added between them, which is usually 1 token.
        assert isinstance(shared_prefix_config.system_prompt_len, int)
        assert isinstance(shared_prefix_config.question_len, int)
        expected_min_len = shared_prefix_config.system_prompt_len + shared_prefix_config.question_len

        token_ids = tokenizer.get_tokenizer().encode(prompt)

        # The actual token count might be slightly different due to the space
        # and how tokens are combined. We'll check if it's very close.
        assert abs(len(token_ids) - expected_min_len) <= 5, f"Prompt: '{prompt}', Tokens: {len(token_ids)}"
