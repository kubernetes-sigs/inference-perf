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
import logging
from typing import Any, Iterator
from unittest.mock import patch

import pytest

from inference_perf.apis import CompletionAPIData, EmbeddingsAPIData, LazyLoadInferenceAPIData, TemplateAPIData
from inference_perf.config import (
    APIConfig,
    APIType,
    DataConfig,
    Distribution,
    DataGenType,
    DistributionType,
    EmbeddingsConfig,
    TemplateConfig,
    TemplateResponseConfig,
)
from inference_perf.datagen.synthetic import synthetic_datagen
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
    with patch("inference_perf.datagen.synthetic.synthetic_datagen.time.monotonic", side_effect=lambda: next(fake_time)):
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
        patch("inference_perf.datagen.synthetic.synthetic_datagen.time.monotonic", side_effect=lambda: next(fake_time)),
    ):
        for i in range(4):
            generator.load_lazy_data(LazyLoadInferenceAPIData(data_index=i))

    # First call sets the baseline timestamp and logs; subsequent sub-interval
    # calls should be silent.
    assert mock_logger.info.call_count == 1


def test_synthetic_datagen_distribution_types() -> None:
    api_config = APIConfig(type=APIType.Completion)
    data_config = DataConfig(
        type=DataGenType.Synthetic,
        input_distribution=Distribution(
            type=DistributionType.FIXED,
            min=10,
            max=20,
            mean=15,
            std_dev=2,
            total_count=5,
        ),
        output_distribution=Distribution(
            type=DistributionType.FIXED,
            min=5,
            max=10,
            mean=7,
            std_dev=1,
            total_count=5,
        ),
    )
    tokenizer = DummyCustomTokenizer()

    generator = SyntheticDataGenerator(api_config, data_config, tokenizer)

    # With FIXED type, all generated lengths must be exactly equal to the mean
    assert len(generator.input_lengths) == 5
    for length in generator.input_lengths:
        assert length == 15

    assert len(generator.output_lengths) == 5
    for length in generator.output_lengths:
        assert length == 7


def test_synthetic_datagen_embeddings_batches_without_output_distribution() -> None:
    # Embeddings generate no output, so no output_distribution is needed. Each
    # request carries batch_size inputs, each with its own sampled length.
    api_config = APIConfig(type=APIType.Embeddings, embeddings=EmbeddingsConfig(batch_size=4))
    data_config = DataConfig(
        type=DataGenType.Synthetic,
        input_distribution=Distribution(min=10, max=20, mean=15, std_dev=2, total_count=5),
    )
    tokenizer = DummyCustomTokenizer()

    generator = SyntheticDataGenerator(api_config, data_config, tokenizer)

    assert len(generator.input_lengths) == 5 * 4
    data = generator.load_lazy_data(LazyLoadInferenceAPIData(data_index=1))
    assert isinstance(data, EmbeddingsAPIData)
    assert isinstance(data.input, list)
    assert [tokenizer.count_tokens(text) for text in data.input] == list(generator.input_lengths[4:8])


def test_synthetic_datagen_completion_still_requires_output_distribution() -> None:
    data_config = DataConfig(
        type=DataGenType.Synthetic,
        input_distribution=Distribution(min=10, max=20, mean=15, std_dev=2, total_count=5),
    )
    with pytest.raises(ValueError, match="IODistribution and tokenizer are required"):
        SyntheticDataGenerator(APIConfig(type=APIType.Completion), data_config, DummyCustomTokenizer())


def test_synthetic_datagen_template_carries_the_completion_prompt() -> None:
    # A template request gets the same prompt and output length as a completion request.
    template = TemplateConfig(route="/generate", body={"text": "${prompt}"}, response=TemplateResponseConfig(text_path="text"))
    data_config = DataConfig(
        type=DataGenType.Synthetic,
        input_distribution=Distribution(min=10, max=20, mean=15, std_dev=2, total_count=5),
        output_distribution=Distribution(min=5, max=10, mean=7, std_dev=1, total_count=5),
    )
    tokenizer = DummyCustomTokenizer()

    generator = SyntheticDataGenerator(APIConfig(type=APIType.Template, template=template), data_config, tokenizer)
    data = generator.load_lazy_data(LazyLoadInferenceAPIData(data_index=2))

    assert isinstance(data, TemplateAPIData)
    assert data.template == template
    assert tokenizer.count_tokens(data.prompt) == generator.input_lengths[2]
    assert data.max_tokens == generator.output_lengths[2]
