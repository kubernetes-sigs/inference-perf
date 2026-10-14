from itertools import islice

import pytest

from inference_perf.apis.base import LazyLoadInferenceAPIData
from inference_perf.apis.chat import ChatCompletionAPIData
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.datagen.synthetic.shared_prefix_datagen import SharedPrefixDataGenerator
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

    def count_tokens(self, text: str, add_special_tokens: bool = True) -> int:
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


def _generator(api_type: APIType, enable_multi_turn_chat: bool = False) -> SharedPrefixDataGenerator:
    shared_prefix_cfg = SharedPrefix(
        num_groups=2,
        num_prompts_per_group=2,
        system_prompt_len=8,
        question_len=4,
        output_len=4,
        enable_multi_turn_chat=enable_multi_turn_chat,
    )
    config = DataConfig(type=DataGenType.SharedPrefix, shared_prefix=shared_prefix_cfg)
    return SharedPrefixDataGenerator(APIConfig(type=api_type), config, MockCustomTokenizer())


async def test_chat_api_sends_each_prompt_as_a_chat_message() -> None:
    generator = _generator(APIType.Chat)

    for i, lazy_data in enumerate(islice(generator.get_data(), len(generator.prompts))):
        assert isinstance(lazy_data, LazyLoadInferenceAPIData)
        api_data = generator.load_lazy_data(lazy_data)
        assert isinstance(api_data, ChatCompletionAPIData)
        assert api_data.get_route() == "/v1/chat/completions"

        body = await api_data.to_request_body(effective_model_name="m", max_tokens=4, ignore_eos=False, streaming=False)
        assert body["messages"] == [{"role": "user", "content": generator.prompts[i]}]


def test_completion_api_sends_each_prompt_as_a_completion() -> None:
    generator = _generator(APIType.Completion)

    lazy_data = next(generator.get_data())
    assert isinstance(lazy_data, LazyLoadInferenceAPIData)
    api_data = generator.load_lazy_data(lazy_data)
    assert type(api_data) is CompletionAPIData
    assert api_data.get_route() == "/v1/completions"


def test_multi_turn_rejects_the_chat_api() -> None:
    with pytest.raises(ValueError, match="enable_multi_turn_chat supports only api.type completion"):
        _generator(APIType.Chat, enable_multi_turn_chat=True)
