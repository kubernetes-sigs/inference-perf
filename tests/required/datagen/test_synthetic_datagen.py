import logging
from typing import Any, Iterator
from unittest.mock import patch

from inference_perf.apis import CompletionAPIData, LazyLoadInferenceAPIData
from inference_perf.config import APIConfig, APIType, DataConfig, Distribution, DataGenType
from inference_perf.datagen import synthetic_datagen
from inference_perf.datagen.synthetic_datagen import SyntheticDataGenerator
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


def test_synthetic_datagen_logs_progress_on_interval(caplog: Any) -> None:
    """Materializing prompts should emit a heartbeat log line at the configured interval."""
    api_config = APIConfig(type=APIType.Completion)
    data_config = DataConfig(
        type=DataGenType.Synthetic,
        input_distribution=Distribution(min=10, max=20, mean=15, std_dev=2, total_count=20),
        output_distribution=Distribution(min=5, max=10, mean=7, std_dev=1, total_count=20),
    )
    generator = SyntheticDataGenerator(api_config, data_config, DummyCustomTokenizer())

    # Drive time forward by the configured interval on every materialization
    # so every call crosses the heartbeat boundary.
    fake_time: Iterator[float] = iter((i * synthetic_datagen._PROGRESS_LOG_INTERVAL_SEC for i in range(1, 100)))

    caplog.set_level(logging.INFO, logger=synthetic_datagen.__name__)
    with patch("inference_perf.datagen.synthetic_datagen.time.monotonic", side_effect=lambda: next(fake_time)):
        for i in range(3):
            generator.load_lazy_data(LazyLoadInferenceAPIData(data_index=i))

    progress_messages = [r.message for r in caplog.records if "Synthetic datagen progress" in r.message]
    assert len(progress_messages) == 3
    assert "materialized 3 prompts" in progress_messages[-1]


def test_synthetic_datagen_skips_progress_log_within_interval() -> None:
    """Sub-interval materializations should only log once."""
    api_config = APIConfig(type=APIType.Completion)
    data_config = DataConfig(
        type=DataGenType.Synthetic,
        input_distribution=Distribution(min=10, max=20, mean=15, std_dev=2, total_count=20),
        output_distribution=Distribution(min=5, max=10, mean=7, std_dev=1, total_count=20),
    )
    generator = SyntheticDataGenerator(api_config, data_config, DummyCustomTokenizer())

    base_time = 1_000_000.0
    fake_time = iter([base_time, base_time + 0.1, base_time + 0.2, base_time + 0.3])

    with (
        patch.object(synthetic_datagen, "logger") as mock_logger,
        patch("inference_perf.datagen.synthetic_datagen.time.monotonic", side_effect=lambda: next(fake_time)),
    ):
        for i in range(4):
            generator.load_lazy_data(LazyLoadInferenceAPIData(data_index=i))

    # First call sets the baseline timestamp and logs; subsequent sub-interval
    # calls should be silent.
    assert mock_logger.info.call_count == 1
