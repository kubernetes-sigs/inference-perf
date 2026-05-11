from inference_perf.datagen.shared_prefix_datagen import SharedPrefixDataGenerator
from inference_perf.config import APIConfig, DataConfig, SharedPrefix, APIType, DataGenType
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from typing import Any


class MockHFTokenizer:
    vocab_size = 50000

    def decode(self, token_ids: list[int], skip_special_tokens: bool = True) -> str:
        return " ".join([str(tid) for tid in token_ids])

    def batch_decode(self, token_ids_list: list[list[int]], skip_special_tokens: bool = True) -> list[str]:
        return [self.decode(ids) for ids in token_ids_list]


class MockCustomTokenizer(CustomTokenizer):
    def __init__(self) -> None:
        pass

    def get_tokenizer(self) -> Any:
        return MockHFTokenizer()

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def test_shared_prefix_length_single_turn() -> None:
    # Setup config
    api_config = APIConfig(type=APIType.Completion)

    shared_prefix_cfg = SharedPrefix(
        num_groups=1,
        num_prompts_per_group=10,
        system_prompt_len=64,
        question_len=32,
        output_len=16,
        enable_multi_turn_chat=False,
    )

    config = DataConfig(type=DataGenType.SharedPrefix, shared_prefix=shared_prefix_cfg)

    tokenizer = MockCustomTokenizer()

    # Initialize generator
    generator = SharedPrefixDataGenerator(api_config, config, tokenizer)

    # Verify prompts are strings and have correct length
    assert len(generator.prompts) == 10
    for prompt in generator.prompts:
        assert isinstance(prompt, str)
        # Length should be exactly system_prompt_len (64) + question_len (32) = 96
        assert tokenizer.count_tokens(prompt) == 96


def test_shared_prefix_length_multi_turn() -> None:
    # Setup config for multi-turn
    api_config = APIConfig(type=APIType.Completion)

    shared_prefix_cfg = SharedPrefix(
        num_groups=1,
        num_prompts_per_group=10,
        system_prompt_len=64,
        question_len=32,
        output_len=16,
        enable_multi_turn_chat=True,
    )

    config = DataConfig(type=DataGenType.SharedPrefix, shared_prefix=shared_prefix_cfg)

    tokenizer = MockCustomTokenizer()

    generator = SharedPrefixDataGenerator(api_config, config, tokenizer)

    assert len(generator.prompts) == 10
    for prompt in generator.prompts:
        assert isinstance(prompt, str)
