import numpy as np
import pytest
from typing import Any, List
from inference_perf.datagen.datagen_utils import converge_to_exact_length_text, generate_random_exact_length_text
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

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def test_generate_exact_length_text_success() -> None:
    tokenizer = DummyCustomTokenizer()
    target_len = 5
    initial_tokens = [10, 20, 30]  # len 3

    def adjust_tokens(current_tokens: List[int], current_len: int, target_len: int) -> List[int]:
        if current_len < target_len:
            current_tokens.append(40)
        return current_tokens

    result = converge_to_exact_length_text(
        tokenizer=tokenizer,
        target_len=target_len,
        prefix_text="",
        initial_tokens=initial_tokens,
        adjust_tokens_fn=adjust_tokens,
    )

    assert tokenizer.count_tokens(result) == target_len
    assert result == "10 20 30 40 40"  # initial 3 + 2 added


def test_generate_exact_length_text_with_prefix() -> None:
    tokenizer = DummyCustomTokenizer()
    target_len = 6
    prefix_text = "p1 p2"  # len 2
    initial_tokens = [10, 20]  # len 2

    def adjust_tokens(current_tokens: List[int], current_len: int, target_len: int) -> List[int]:
        if current_len < target_len:
            current_tokens.append(30)
        return current_tokens

    result = converge_to_exact_length_text(
        tokenizer=tokenizer,
        target_len=target_len,
        prefix_text=prefix_text,
        initial_tokens=initial_tokens,
        adjust_tokens_fn=adjust_tokens,
    )

    assert tokenizer.count_tokens(result) == target_len
    assert result == "p1 p2 10 20 30 30"  # prefix(2) + initial(2) + added(2) = 6


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
            prefix_text="",
            initial_tokens=initial_tokens,
            adjust_tokens_fn=adjust_tokens,
        )


def test_generate_exact_length_text_zero_len() -> None:
    tokenizer = DummyCustomTokenizer()
    result = converge_to_exact_length_text(
        tokenizer=tokenizer,
        target_len=0,
        prefix_text="",
        initial_tokens=[10],
        adjust_tokens_fn=lambda c, _, __: c,
    )
    assert result == ""


def test_generate_exact_length_text_prefix_already_too_long() -> None:
    tokenizer = DummyCustomTokenizer()
    prefix_text = "p1 p2 p3"  # len 3
    with pytest.raises(
        AssertionError, match=r"target_len \(2\) must be > prefix_len \(3\)\. This helper generates suffix tokens"
    ):
        converge_to_exact_length_text(
            tokenizer=tokenizer,
            target_len=2,
            prefix_text=prefix_text,
            initial_tokens=[10],
            adjust_tokens_fn=lambda c, _, __: c,
        )


def test_generate_random_exact_length_text_prefix_already_too_long() -> None:
    tokenizer = DummyCustomTokenizer()
    rng = np.random.default_rng(0)
    valid_token_ids = np.array([10, 20, 30, 40], dtype=np.int64)
    prefix_text = "p1 p2 p3"  # len 3
    with pytest.raises(
        AssertionError, match=r"target_len \(2\) must be > prefix_len \(3\)\. This helper generates suffix tokens"
    ):
        generate_random_exact_length_text(
            rng=rng,
            valid_token_ids=valid_token_ids,
            tokenizer=tokenizer,
            target_len=2,
            prefix_text=prefix_text,
        )
