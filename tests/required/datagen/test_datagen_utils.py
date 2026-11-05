import numpy as np
import pytest
from typing import Any, List
from inference_perf.datagen.datagen_utils import (
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

    def decode(self, tokens: list[int], **kwargs: Any) -> str:
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
