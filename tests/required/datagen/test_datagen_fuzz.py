import pytest
from unittest.mock import MagicMock
import numpy as np
from inference_perf.config import APIConfig, APIType, DataConfig, Distribution, DataGenType
from inference_perf.apis import LazyLoadInferenceAPIData
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.datagen.random_datagen import RandomDataGenerator
from inference_perf.datagen.synthetic_datagen import SyntheticDataGenerator
from inference_perf.config import SharedPrefix
from inference_perf.datagen.base import DataGenerator, LazyLoadDataMixin
from inference_perf.datagen.shared_prefix_datagen import SharedPrefixDataGenerator


def _make_mock_tokenizer(vocab_size: int = 1000) -> MagicMock:
    """Create a mock tokenizer that returns predictable text."""
    mock_tokenizer = MagicMock()
    hf_tok = MagicMock()
    hf_tok.vocab_size = vocab_size
    hf_tok.decode = MagicMock(side_effect=lambda ids, **kw: f"text_{len(ids)}")
    hf_tok.batch_decode = MagicMock(side_effect=lambda batch, **kw: [f"text_{len(ids)}" for ids in batch])
    hf_tok.encode = MagicMock(side_effect=lambda text, **kw: [1] * max(1000, len(text.split())))
    mock_tokenizer.get_tokenizer.return_value = hf_tok

    def count_tokens(text: str) -> int:
        parts = text.split()
        total = 0
        for p in parts:
            if p.startswith("text_"):
                total += int(p[5:])
            else:
                total += 1
        return total

    mock_tokenizer.count_tokens.side_effect = count_tokens
    return mock_tokenizer


@pytest.mark.parametrize("gen_type", [DataGenType.Random, DataGenType.Synthetic, DataGenType.SharedPrefix])
def test_datagen_length_fuzz(gen_type: DataGenType) -> None:
    rng = np.random.default_rng(42)
    mock_tokenizer = _make_mock_tokenizer()

    # Run 10 random configurations
    for _ in range(10):
        target_len = int(rng.integers(10, 100))

        api_config = APIConfig(type=APIType.Completion)

        if gen_type == DataGenType.Random:
            data_config = DataConfig(
                type=DataGenType.Random,
                input_distribution=Distribution(
                    min=target_len, max=target_len, mean=float(target_len), std_dev=0.0, total_count=10
                ),
                output_distribution=Distribution(min=10, max=10, mean=10.0, std_dev=0.0, total_count=10),
            )
            gen: DataGenerator = RandomDataGenerator(api_config, data_config, mock_tokenizer)
        elif gen_type == DataGenType.Synthetic:
            data_config = DataConfig(
                type=DataGenType.Synthetic,
                input_distribution=Distribution(
                    min=target_len, max=target_len, mean=float(target_len), std_dev=0.0, total_count=10
                ),
                output_distribution=Distribution(min=10, max=10, mean=10.0, std_dev=0.0, total_count=10),
            )
            gen = SyntheticDataGenerator(api_config, data_config, mock_tokenizer)
        elif gen_type == DataGenType.SharedPrefix:
            sp = SharedPrefix(
                num_groups=2,
                num_prompts_per_group=5,
                question_len=target_len,
                output_len=10,
                system_prompt_len=10,
            )
            data_config = DataConfig(type=DataGenType.SharedPrefix, shared_prefix=sp)
            gen = SharedPrefixDataGenerator(api_config, data_config, mock_tokenizer)

        # Generate prompts and check length
        assert isinstance(gen, LazyLoadDataMixin)
        prompts = []
        for i, p in enumerate(gen.get_data()):
            if isinstance(p, LazyLoadInferenceAPIData):
                p = gen.load_lazy_data(p)
            prompts.append(p)
            if i >= 9:
                break

        assert len(prompts) == 10

        for p in prompts:
            assert isinstance(p, CompletionAPIData)
            actual_len = mock_tokenizer.count_tokens(p.prompt)

            if gen_type == DataGenType.SharedPrefix:
                # For SharedPrefix, the prompt is prefix + question.
                # Expected length is system_prompt_len + question_len = 10 + target_len
                expected_len = 10 + target_len
            else:
                expected_len = target_len

            print(f"Type: {gen_type}, Expected: {expected_len}, Actual: {actual_len}")
            assert actual_len == expected_len, f"Failed for {gen_type}, expected {expected_len}, got {actual_len}"
