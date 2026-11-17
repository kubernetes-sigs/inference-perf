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

import numpy as np
import pytest
from typing import Any, List
from inference_perf.config import CustomTokenizerConfig
from inference_perf.datagen.datagen_utils import (
    build_word_start_token_ids,
    converge_to_exact_length_text,
    generate_random_exact_length_text,
    random_token_ids,
)
from inference_perf.datagen.synthetic.shared_prefix_datagen import SharedPrefixDataGenerator
from inference_perf.utils.custom_tokenizer import CustomTokenizer


class DummyTokenizer:
    vocab_size = 1000
    all_special_ids = [1, 2, 3]

    def encode(self, text: str) -> list[int]:
        try:
            return [int(t) for t in text.split()]
        except ValueError:
            return [4, 5, 6]

    def decode(self, tokens: list[int], **kwargs: object) -> str:
        return " ".join(str(t) for t in tokens)


class DummyCustomTokenizer(CustomTokenizer):
    def __init__(self) -> None:
        pass

    def get_tokenizer(self) -> Any:
        return DummyTokenizer()

    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        return len(text.split())


def test_generate_exact_length_text_success() -> None:
    tokenizer = DummyCustomTokenizer()
    target_len = 5
    initial_tokens = [10, 20, 30]  # len 3

    def adjust_tokens(current_tokens: List[int], current_len: int, target_len: int) -> List[int]:
        if current_len < target_len:
            current_tokens.append(40)
        return current_tokens

    result, ids = converge_to_exact_length_text(
        tokenizer=tokenizer,
        target_len=target_len,
        initial_tokens=initial_tokens,
        adjust_tokens_fn=adjust_tokens,
    )

    assert tokenizer.count_tokens(result) == target_len
    assert result == "10 20 30 40 40"  # initial 3 + 2 added
    assert ids == [10, 20, 30, 40, 40]


def test_generate_exact_length_text_failure() -> None:
    tokenizer = DummyCustomTokenizer()
    target_len = 5
    initial_tokens = [10, 20, 30]

    # Callback does nothing, so it will never converge
    def adjust_tokens(current_tokens: List[int], current_len: int, target_len: int) -> List[int]:
        return current_tokens

    with pytest.raises(ValueError, match="Could not generate a prompt of exactly 5 tokens after 20 attempts"):
        converge_to_exact_length_text(
            tokenizer=tokenizer,
            target_len=target_len,
            initial_tokens=initial_tokens,
            adjust_tokens_fn=adjust_tokens,
        )


def test_generate_exact_length_text_zero_len() -> None:
    tokenizer = DummyCustomTokenizer()
    result, ids = converge_to_exact_length_text(
        tokenizer=tokenizer,
        target_len=0,
        initial_tokens=[10],
        adjust_tokens_fn=lambda c, _, __: c,
    )
    assert result == ""
    assert ids == []


def test_generate_exact_length_text_with_wrap_fn() -> None:
    tokenizer = DummyCustomTokenizer()

    # Wrapper adds a fixed 2-word overhead, mimicking a chat template.
    def wrap(text: str) -> str:
        return f"<open> {text} <close>"

    def adjust_tokens(current_tokens: List[int], current_len: int, target_len: int) -> List[int]:
        diff = current_len - target_len
        if diff > 0:
            return current_tokens[:-diff]
        return current_tokens + [40] * -diff

    result, ids = converge_to_exact_length_text(
        tokenizer=tokenizer,
        target_len=5,
        initial_tokens=[10, 20, 30, 40, 50],
        adjust_tokens_fn=adjust_tokens,
        wrap_fn=wrap,
    )

    # target_len applies to the WRAPPED text, which is what gets returned.
    assert tokenizer.count_tokens(result, add_special_tokens=False) == 5
    assert result == "<open> 10 20 30 <close>"
    assert ids == [10, 20, 30]


# Fake tokenizer: ids are counted one per token, except that an id divisible by 5 in the
# second-to-last position merges with the token after it, so [..., 10, 7] counts 1 more.
class MergingCustomTokenizer(DummyCustomTokenizer):
    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
        ids = [int(t) for t in text.split()]
        return len(ids) + (1 if len(ids) >= 2 and ids[-2] % 5 == 0 else 0)


# Seeds 3, 4 and 14 with target_len=8: before the fix the loop bounces between 7 and 9
# tokens and raises ValueError. After it the text counts exactly 8, and the escape only drops
# and resamples the tail: the leading ids are the ones first sampled from the same rng.
@pytest.mark.parametrize("seed", [3, 4, 14])
def test_generate_random_exact_length_text_escapes_merge_oscillation(seed: int) -> None:
    tokenizer = MergingCustomTokenizer()
    valid_token_ids = np.arange(10, 500)
    target_len = 8
    initial = random_token_ids(np.random.default_rng(seed), valid_token_ids, target_len)

    text, ids = generate_random_exact_length_text(np.random.default_rng(seed), valid_token_ids, tokenizer, target_len)

    assert tokenizer.count_tokens(text) == target_len
    # At most one id is lost to a merge, so the ids are target_len or target_len - 1 long.
    assert len(ids) in (target_len - 1, target_len)
    assert ids[: target_len - 3] == initial[: target_len - 3]


# Same oscillation in the shared-prefix suffix sampler, which has its own pad/trim loop and must
# keep its word-start first id.
@pytest.mark.parametrize("seed", [3, 4, 14])
def test_sample_suffix_ids_escapes_merge_oscillation(seed: int) -> None:
    tokenizer = MergingCustomTokenizer()
    generator = object.__new__(SharedPrefixDataGenerator)
    generator.rng = np.random.default_rng(seed)
    generator.tokenizer = tokenizer
    generator.valid_token_ids = np.arange(10, 500)
    generator.word_start_token_ids = np.arange(10, 20)
    target_len = 8

    probe = np.random.default_rng(seed)
    first_id = int(probe.choice(generator.word_start_token_ids))
    initial = [first_id] + random_token_ids(probe, generator.valid_token_ids, target_len - 1)

    ids = generator._sample_suffix_ids(target_len)

    assert tokenizer.count_tokens(" ".join(str(i) for i in ids)) == target_len
    assert ids[0] == first_id
    assert ids[: target_len - 3] == initial[: target_len - 3]


# Fake tokenizer: every id decodes to " <id>" (a word start) and counts one token per
# whitespace-separated word, except ids in `split_ids` (default {13}), which decode to
# " 13 x" and so re-tokenize to two tokens, like a gpt2 id that decodes to an incomplete
# UTF-8 sequence. With word_start=False there is no leading space (" 12" -> "12",
# " 13 x" -> "13 x"), like T5, ALBERT or BERT, where no single id decodes with leading
# whitespace.
class SplittingTokenizer:
    name_or_path = "fake/splitting"

    def __init__(self, split_ids: frozenset[int], word_start: bool) -> None:
        self.split_ids = split_ids
        self.prefix = " " if word_start else ""

    def decode(self, tokens: list[int], **kwargs: object) -> str:
        return self.prefix + " ".join(f"{t} x" if t in self.split_ids else str(t) for t in tokens)


class SplittingCustomTokenizer(DummyCustomTokenizer):
    def __init__(self, split_ids: frozenset[int] = frozenset({13}), word_start: bool = True) -> None:
        self.hf_tokenizer = SplittingTokenizer(split_ids, word_start)

    def get_tokenizer(self) -> Any:
        return self.hf_tokenizer


# valid ids [12, 13, 14]: all three decode with a leading space, but " 13 x" counts 2 tokens.
# The word-start pool is [12, 14]; before the fix it also held 13.
def test_build_word_start_token_ids_drops_ids_that_split_on_retokenize() -> None:
    pool = build_word_start_token_ids(SplittingCustomTokenizer(), np.array([12, 13, 14]))

    assert pool.tolist() == [12, 14]


# valid ids [13, 14] and a length-1 suffix: with 13 in the word-start pool the sampler
# draws it for seed 1, the suffix counts 2 tokens and the loop raises ValueError after
# 20 attempts. With the pool built by build_word_start_token_ids only 14 is left, so the
# suffix is [14] and counts exactly 1 token.
def test_sample_suffix_ids_length_one_converges_with_built_pool() -> None:
    tokenizer = SplittingCustomTokenizer()
    generator = object.__new__(SharedPrefixDataGenerator)
    generator.rng = np.random.default_rng(1)
    generator.tokenizer = tokenizer
    generator.valid_token_ids = np.array([13, 14])
    generator.word_start_token_ids = build_word_start_token_ids(tokenizer, generator.valid_token_ids)

    ids = generator._sample_suffix_ids(1)

    assert ids == [14]
    assert tokenizer.count_tokens(tokenizer.get_tokenizer().decode(ids)) == 1


# Real gpt2 tokenizer (#1093): ids 20174 and 34504 decode to " 裏" plus an incomplete UTF-8
# byte and re-tokenize to [6438, 4210], two tokens. They must not be in the word-start pool,
# while an ordinary word start such as 262 (" the") stays in it.
def test_build_word_start_token_ids_gpt2_excludes_split_ids() -> None:
    tokenizer = CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path="gpt2"))
    valid_token_ids = np.array([262, 20174, 34504], dtype=np.int64)

    pool = build_word_start_token_ids(tokenizer, valid_token_ids)

    assert pool.tolist() == [262]


# valid ids [12, 13, 14] with no word starts ("12", "13 x", "14"): the pool falls back to the
# ids that re-tokenize to one token, [12, 14]. Before the fix it fell back to all three.
def test_build_word_start_token_ids_fallback_drops_ids_that_split_on_retokenize() -> None:
    tokenizer = SplittingCustomTokenizer(word_start=False)

    pool = build_word_start_token_ids(tokenizer, np.array([12, 13, 14]))

    assert pool.tolist() == [12, 14]


# valid ids [12, 13] that both decode to "<id> x" (two tokens) and none with leading
# whitespace: no id can start a one-token suffix, so building the pool raises ValueError
# naming the tokenizer instead of returning [12, 13].
def test_build_word_start_token_ids_raises_when_no_id_is_one_token() -> None:
    tokenizer = SplittingCustomTokenizer(split_ids=frozenset({12, 13}), word_start=False)

    with pytest.raises(ValueError, match="'fake/splitting' has no token id"):
        build_word_start_token_ids(tokenizer, np.array([12, 13]))


# valid ids [13, 14] with no word starts and a length-1 suffix: with the old full-vocab
# fallback the sampler draws 13 for seed 1, "13 x" counts 2 tokens and the loop raises
# ValueError after 20 attempts. The fallback pool is [14], so the suffix is [14].
def test_sample_suffix_ids_length_one_converges_with_fallback_pool() -> None:
    tokenizer = SplittingCustomTokenizer(word_start=False)
    generator = object.__new__(SharedPrefixDataGenerator)
    generator.rng = np.random.default_rng(1)
    generator.tokenizer = tokenizer
    generator.valid_token_ids = np.array([13, 14])
    generator.word_start_token_ids = build_word_start_token_ids(tokenizer, generator.valid_token_ids)

    ids = generator._sample_suffix_ids(1)

    assert ids == [14]
    assert tokenizer.count_tokens(tokenizer.get_tokenizer().decode(ids)) == 1


# Real bert-base-uncased tokenizer (#1096): no id decodes with leading whitespace, so the
# fallback applies. Id 2075 decodes to "##ing", which re-tokenizes to ["#", "#", "ing"]
# (three tokens) and must be left out, while 1996 ("the") stays.
def test_build_word_start_token_ids_bert_fallback_excludes_split_ids() -> None:
    tokenizer = CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path="bert-base-uncased"))
    valid_token_ids = np.array([1996, 2075], dtype=np.int64)

    pool = build_word_start_token_ids(tokenizer, valid_token_ids)

    assert pool.tolist() == [1996]


# Real gpt2 tokenizer: the word-start candidates are re-tokenized in one batched call,
# never one count_tokens call per id, and the pool on the first 3000 ids matches the
# per-id round trip that build_word_start_token_ids used before.
def test_build_word_start_token_ids_gpt2_retokenizes_in_one_batch(monkeypatch: pytest.MonkeyPatch) -> None:
    tokenizer = CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path="gpt2"))
    hf_tokenizer = tokenizer.get_tokenizer()
    valid_token_ids = np.arange(3000, dtype=np.int64)
    expected: List[int] = []
    for tid in range(3000):
        decoded = hf_tokenizer.decode([tid], skip_special_tokens=True)
        if (
            isinstance(decoded, str)
            and decoded[:1].isspace()
            and tokenizer.count_tokens(decoded, add_special_tokens=False) == 1
        ):
            expected.append(tid)

    def fail_count_tokens(text: str, add_special_tokens: bool = True) -> int:
        raise AssertionError("count_tokens called per id")

    monkeypatch.setattr(tokenizer, "count_tokens", fail_count_tokens)
    pool = build_word_start_token_ids(tokenizer, valid_token_ids)

    assert pool.tolist() == expected
    assert len(expected) > 0
