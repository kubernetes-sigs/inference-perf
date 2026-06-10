import numpy as np

from inference_perf.apis import CompletionAPIData, LazyLoadInferenceAPIData
from inference_perf.config import APIConfig, APIType, DataConfig, Distribution, DataGenType
from inference_perf.datagen.random_datagen import RandomDataGenerator
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from typing import Any


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


def test_random_datagen_yields_string() -> None:
    api_config = APIConfig(type=APIType.Completion, streaming=True)
    data_config = DataConfig(
        type=DataGenType.Random,
        input_distribution=Distribution(min=10, max=20, mean=15, std_dev=2, total_count=5),
        output_distribution=Distribution(min=5, max=10, mean=7, std_dev=1, total_count=5),
    )
    tokenizer = DummyCustomTokenizer()

    generator = RandomDataGenerator(api_config, data_config, tokenizer)

    # RandomDataGenerator uses LazyLoadDataMixin, so get_data() yields LazyLoadInferenceAPIData
    data_gen = generator.get_data()
    lazy_data = next(data_gen)
    assert isinstance(lazy_data, LazyLoadInferenceAPIData)

    # Load the real data
    real_data = generator.load_lazy_data(lazy_data)
    assert isinstance(real_data, CompletionAPIData)

    # Verify prompt is str
    assert isinstance(real_data.prompt, str)
    assert len(real_data.prompt) > 0


def test_worker_rng_uniqueness_per_worker() -> None:
    """Workers must produce different content even when all input lengths are identical (std_dev=0)."""
    api_config = APIConfig(type=APIType.Completion, streaming=True)
    data_config = DataConfig(
        type=DataGenType.Random,
        input_distribution=Distribution(min=10, max=10, mean=10, std_dev=0.0, total_count=5),
        output_distribution=Distribution(min=5, max=5, mean=5, std_dev=0.0, total_count=5),
    )
    tokenizer = DummyCustomTokenizer()

    # Simulate what Worker.run() does: reseed the datagen's rng with a worker-specific seed.
    # Worker 0 and Worker 1 use (base_seed + id) % 2**32, so they differ.
    base_seed = 42

    generator_w0 = RandomDataGenerator(api_config, data_config, tokenizer)
    generator_w0.rng = np.random.default_rng((base_seed + 0) % 2**32)

    generator_w1 = RandomDataGenerator(api_config, data_config, tokenizer)
    generator_w1.rng = np.random.default_rng((base_seed + 1) % 2**32)

    lazy = LazyLoadInferenceAPIData(data_index=0)
    result_w0 = generator_w0.load_lazy_data(lazy)
    result_w1 = generator_w1.load_lazy_data(lazy)

    assert isinstance(result_w0, CompletionAPIData)
    assert isinstance(result_w1, CompletionAPIData)
    assert result_w0.prompt != result_w1.prompt, "Workers with different seeds must produce different prompts"


def test_random_datagen_excludes_special_tokens() -> None:
    api_config = APIConfig(type=APIType.Completion, streaming=True)
    data_config = DataConfig(
        type=DataGenType.Random,
        input_distribution=Distribution(min=10, max=20, mean=15, std_dev=2, total_count=5),
        output_distribution=Distribution(min=5, max=10, mean=7, std_dev=1, total_count=5),
    )
    tokenizer = DummyCustomTokenizer()

    generator = RandomDataGenerator(api_config, data_config, tokenizer)
    data_gen = generator.get_data()
    lazy_data = next(data_gen)
    assert isinstance(lazy_data, LazyLoadInferenceAPIData)
    real_data = generator.load_lazy_data(lazy_data)

    assert isinstance(real_data, CompletionAPIData)
    # Verify no special tokens in prompt by encoding it back
    encoded_ids = tokenizer.get_tokenizer().encode(real_data.prompt)
    for token in encoded_ids:
        assert token not in [1, 2, 3]
