import pytest
from typing import Any, List
from inference_perf.datagen.datagen_utils import converge_to_exact_length_text
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
