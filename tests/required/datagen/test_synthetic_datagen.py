from inference_perf.apis import CompletionAPIData, LazyLoadInferenceAPIData
from inference_perf.config import APIConfig, APIType, DataConfig, Distribution, DataGenType
from inference_perf.datagen.synthetic_datagen import SyntheticDataGenerator
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from typing import Any


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

    def count_tokens(self, text: str) -> int:
        return len(text.split())


def test_synthetic_datagen_yields_string() -> None:
    api_config = APIConfig(type=APIType.Completion)
    data_config = DataConfig(
        type=DataGenType.Synthetic,
        input_distribution=Distribution(min=10, max=20, mean=15, std_dev=2, total_count=5),
        output_distribution=Distribution(min=5, max=10, mean=7, std_dev=1, total_count=5),
    )
    tokenizer = DummyCustomTokenizer()

    generator = SyntheticDataGenerator(api_config, data_config, tokenizer)

    # SyntheticDataGenerator uses LazyLoadDataMixin
    data_gen = generator.get_data()
    lazy_data = next(data_gen)
    assert isinstance(lazy_data, LazyLoadInferenceAPIData)

    real_data = generator.load_lazy_data(lazy_data)
    assert isinstance(real_data, CompletionAPIData)

    assert isinstance(real_data.prompt, str)
    assert len(real_data.prompt) > 0
