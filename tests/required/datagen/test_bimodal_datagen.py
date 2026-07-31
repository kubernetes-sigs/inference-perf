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
from unittest.mock import MagicMock
import pytest

from inference_perf.apis.base import LazyLoadInferenceAPIData
from inference_perf.apis.chat import ChatCompletionAPIData
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.config import APIConfig, APIType, BimodalConfig, DataConfig, DataGenType

from inference_perf.datagen.base import DataGenerator, LazyLoadDataMixin
from inference_perf.datagen.bimodal_datagen import BimodalDataGenerator


def _make_mock_tokenizer(vocab_size: int = 1000) -> MagicMock:
    """Create a mock tokenizer for bimodal generator tests."""
    mock_tokenizer = MagicMock()
    hf_tok = MagicMock()
    hf_tok.vocab_size = vocab_size
    hf_tok.decode = MagicMock(side_effect=lambda ids, **kw: f"text_{len(ids)}")
    hf_tok.batch_decode = MagicMock(side_effect=lambda batch, **kw: [f"text_{len(ids)}" for ids in batch])
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


def _make_generator(bimodal_config: BimodalConfig, api_type: APIType = APIType.Completion) -> BimodalDataGenerator:
    api_config = APIConfig(type=api_type)
    data_config = DataConfig(type=DataGenType.Bimodal, bimodal=bimodal_config)
    return BimodalDataGenerator(api_config, data_config, _make_mock_tokenizer())


class TestBimodalGeneratorInterfaces:
    def test_inheritance_and_mixins(self) -> None:
        cfg = BimodalConfig(mode_a_ratio=0.5)
        gen = _make_generator(cfg)
        assert isinstance(gen, DataGenerator)
        assert isinstance(gen, LazyLoadDataMixin)

    def test_supported_apis_and_features(self) -> None:
        cfg = BimodalConfig()
        gen = _make_generator(cfg)
        assert APIType.Completion in gen.get_supported_apis()
        assert APIType.Chat in gen.get_supported_apis()
        assert gen.is_io_distribution_supported() is True
        assert gen.is_shared_prefix_supported() is True


class TestBimodalRatioSampling:
    def test_ratio_distribution(self) -> None:
        cfg = BimodalConfig(mode_a_ratio=0.7, seed=42)
        gen = _make_generator(cfg)
        num_samples = 1000
        mode_a_count = sum(1 for i in range(num_samples) if gen._is_mode_a(i))
        ratio = mode_a_count / num_samples
        assert 0.65 <= ratio <= 0.75

    def test_deterministic_reproducibility(self) -> None:
        cfg1 = BimodalConfig(mode_a_ratio=0.5, seed=123)
        cfg2 = BimodalConfig(mode_a_ratio=0.5, seed=123)
        gen1 = _make_generator(cfg1)
        gen2 = _make_generator(cfg2)
        for i in range(50):
            assert gen1._is_mode_a(i) == gen2._is_mode_a(i)
            req1 = gen1.load_lazy_data(LazyLoadInferenceAPIData(data_index=i))
            req2 = gen2.load_lazy_data(LazyLoadInferenceAPIData(data_index=i))
            assert req1.max_tokens == req2.max_tokens


class TestBimodalRequestMaterialization:
    def test_completion_api_materialization(self) -> None:
        cfg = BimodalConfig(
            mode_a_user_prompt_len=10,
            mode_a_output_len=20,
            mode_b_user_prompt_len=100,
            mode_b_output_len=200,
            mode_a_ratio=1.0,
            seed=42,
        )
        gen = _make_generator(cfg, APIType.Completion)
        req = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0))
        assert isinstance(req, CompletionAPIData)
        assert req.max_tokens == 20

    def test_chat_api_materialization(self) -> None:
        cfg = BimodalConfig(
            mode_a_system_prompt_len=32,
            mode_a_user_prompt_len=10,
            mode_a_output_len=15,
            mode_a_ratio=1.0,
            seed=42,
        )
        gen = _make_generator(cfg, APIType.Chat)
        req = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0))
        assert isinstance(req, ChatCompletionAPIData)
        assert req.max_tokens == 15
        assert req.prefix_text is not None
        assert len(req.messages) == 1
        assert req.messages[0].role == "user"


class TestBimodalValidation:
    def test_missing_bimodal_config_raises(self) -> None:
        api_config = APIConfig(type=APIType.Completion)
        data_config = DataConfig(type=DataGenType.Bimodal, bimodal=None)
        with pytest.raises(ValueError, match="BimodalConfig must be specified"):
            BimodalDataGenerator(api_config, data_config, _make_mock_tokenizer())
