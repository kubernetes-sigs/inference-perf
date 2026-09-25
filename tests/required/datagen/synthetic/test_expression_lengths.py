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
"""Datagens sampling their lengths from expression-string config values."""

from typing import Any
from unittest.mock import MagicMock

import pytest

from inference_perf.config import APIConfig, APIType, DataConfig, DataGenType, SharedPrefix
from inference_perf.datagen.synthetic.random_datagen import RandomDataGenerator
from inference_perf.datagen.synthetic.shared_prefix_datagen import SharedPrefixDataGenerator
from inference_perf.datagen.synthetic.synthetic_datagen import SyntheticDataGenerator
from inference_perf.utils.custom_tokenizer import CustomTokenizer


class DummyTokenizer:
    vocab_size = 1000
    all_special_ids = [1, 2, 3]

    def encode(self, text: str) -> list[int]:
        try:
            return [int(t) for t in text.split()]
        except ValueError:
            return [4, 5, 6] * 10

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
        return " ".join(str(t) for t in tokens)


class DummyCustomTokenizer(CustomTokenizer):
    def __init__(self) -> None:
        pass

    def get_tokenizer(self) -> Any:
        return DummyTokenizer()

    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        return len(text.split())


def _expression_data_config(gen_type: DataGenType) -> DataConfig:
    # Uniform supports keep the expected value ranges checkable: input lengths
    # land in [10, 20] and output lengths in [5, 10] after rounding.
    return DataConfig(
        type=gen_type,
        input_distribution="Uniform(10, 20)",
        output_distribution="Uniform(5, 10)",
    )


class TestSyntheticDatagenExpressionLengths:
    # A synthetic datagen with expression-string IO fields and total_count=7
    # pre-generates 7 input lengths in [10, 20] and 7 output lengths in [5, 10].
    def test_pregenerates_lengths_from_expressions(self) -> None:
        generator = SyntheticDataGenerator(
            APIConfig(type=APIType.Completion),
            _expression_data_config(DataGenType.Synthetic),
            DummyCustomTokenizer(),
            seed=42,
            total_count=7,
        )
        assert len(generator.input_lengths) == 7
        assert all(10 <= v <= 20 for v in generator.input_lengths)
        assert len(generator.output_lengths) == 7
        assert all(5 <= v <= 10 for v in generator.output_lengths)

    # An expression string has no total_count attribute to fall back on, so
    # constructing without the run-derived count raises the existing error.
    def test_expression_without_total_count_raises(self) -> None:
        with pytest.raises(ValueError, match="total_count"):
            SyntheticDataGenerator(
                APIConfig(type=APIType.Completion),
                _expression_data_config(DataGenType.Synthetic),
                DummyCustomTokenizer(),
                seed=42,
            )

    # Same seed, same pre-generated arrays: the expression path keeps the
    # datagen deterministic under base_seed.
    def test_expression_lengths_reproducible(self) -> None:
        def build() -> SyntheticDataGenerator:
            return SyntheticDataGenerator(
                APIConfig(type=APIType.Completion),
                _expression_data_config(DataGenType.Synthetic),
                DummyCustomTokenizer(),
                seed=42,
                total_count=20,
            )

        assert build().input_lengths.tolist() == build().input_lengths.tolist()


class TestRandomDatagenExpressionLengths:
    # The random datagen accepts the same expression-string IO fields: 5
    # input lengths in [10, 20] and 5 output lengths in [5, 10].
    def test_pregenerates_lengths_from_expressions(self) -> None:
        generator = RandomDataGenerator(
            APIConfig(type=APIType.Completion),
            _expression_data_config(DataGenType.Random),
            DummyCustomTokenizer(),
            seed=42,
            total_count=5,
        )
        assert len(generator.input_lengths) == 5
        assert all(10 <= v <= 20 for v in generator.input_lengths)
        assert len(generator.output_lengths) == 5
        assert all(5 <= v <= 10 for v in generator.output_lengths)


def _make_mock_tokenizer(vocab_size: int = 1000) -> MagicMock:
    # Mirrors the shared-prefix datagen tests: decode produces "text_<n>"
    # placeholders and count_tokens reads the token count back out of them.
    mock_tokenizer = MagicMock()
    hf_tok = MagicMock()
    hf_tok.vocab_size = vocab_size
    hf_tok.decode = MagicMock(side_effect=lambda ids, **kw: f"text_{len(ids)}")
    hf_tok.batch_decode = MagicMock(side_effect=lambda batch, **kw: [f"text_{len(ids)}" for ids in batch])
    mock_tokenizer.get_tokenizer.return_value = hf_tok

    def count_tokens(text: str) -> int:
        parts = text.split()
        total = 0
        for p in parts:
            if p.startswith("text_"):
                total += int(p[5:])
            else:
                total += 1
        return total

    mock_tokenizer.count_tokens.side_effect = count_tokens
    return mock_tokenizer


class TestSharedPrefixExpressionLengths:
    # question_len and output_len as expression strings: every sampled
    # question length lands in Uniform(5, 15)'s support [5, 15] and every
    # output length in [20, 30], across all groups.
    def test_samples_lengths_from_expressions(self) -> None:
        sp = SharedPrefix(
            num_groups=3,
            num_prompts_per_group=4,
            question_len="Uniform(5, 15)",
            output_len="Uniform(20, 30)",
            seed=42,
        )
        generator = SharedPrefixDataGenerator(
            APIConfig(type=APIType.Completion),
            DataConfig(type=DataGenType.SharedPrefix, shared_prefix=sp),
            _make_mock_tokenizer(),
        )
        assert len(generator.question_len_list_per_group) == 3
        for group in generator.question_len_list_per_group:
            assert len(group) == 4
            assert all(5 <= v <= 15 for v in group)
        for group in generator.output_len_list_per_group:
            assert all(20 <= v <= 30 for v in group)

    # Same seed, same prompts: the expression path keeps shared-prefix
    # generation deterministic.
    def test_expression_lengths_reproducible(self) -> None:
        def build() -> SharedPrefixDataGenerator:
            sp = SharedPrefix(num_groups=2, num_prompts_per_group=3, question_len="Uniform(5, 15)", seed=7)
            return SharedPrefixDataGenerator(
                APIConfig(type=APIType.Completion),
                DataConfig(type=DataGenType.SharedPrefix, shared_prefix=sp),
                _make_mock_tokenizer(),
            )

        assert build().prompts == build().prompts
