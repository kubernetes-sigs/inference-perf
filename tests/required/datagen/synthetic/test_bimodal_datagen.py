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
from typing import Any, Dict, List, Optional, Union
from unittest.mock import MagicMock
import pytest

from inference_perf.apis.base import LazyLoadInferenceAPIData
from inference_perf.apis.chat import ChatCompletionAPIData
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.config import (
    APIConfig,
    APIType,
    BimodalConfig,
    CustomTokenizerConfig,
    DataConfig,
    DataGenType,
    Distribution,
    SharedPrefix,
)
from inference_perf.datagen.base import DataGenerator, LazyLoadDataMixin
from inference_perf.datagen.synthetic.bimodal_datagen import BimodalDataGenerator
from inference_perf.utils.custom_tokenizer import CustomTokenizer


def _extract_text_content(content: Optional[Union[str, List[Dict[str, Any]]]]) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") for part in content if isinstance(part, dict) and part.get("type") in ["text", "input_text"]
    )


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


def _make_generator(
    bimodal_config: BimodalConfig,
    api_type: APIType = APIType.Completion,
    tokenizer: Optional[CustomTokenizer] = None,
    seed: Optional[int] = None,
) -> BimodalDataGenerator:
    api_config = APIConfig(type=api_type)
    data_config = DataConfig(type=DataGenType.Bimodal, bimodal=bimodal_config)
    tok = tokenizer if tokenizer is not None else _make_mock_tokenizer()
    return BimodalDataGenerator(api_config, data_config, tok, seed=seed)


class TestBimodalGeneratorInterfaces:
    def test_inheritance_and_mixins(self) -> None:
        cfg = BimodalConfig(mode_a_ratio=0.5)
        gen = _make_generator(cfg)
        assert isinstance(gen, DataGenerator)
        assert isinstance(gen, LazyLoadDataMixin)

    def test_guardrails_return_false(self) -> None:
        cfg = BimodalConfig()
        gen = _make_generator(cfg)
        assert APIType.Completion in gen.get_supported_apis()
        assert APIType.Chat in gen.get_supported_apis()
        assert gen.is_io_distribution_supported() is False
        assert gen.is_shared_prefix_supported() is False


class TestBimodalRatioSampling:
    def test_ratio_distribution(self) -> None:
        cfg = BimodalConfig(mode_a_ratio=0.7)
        gen = _make_generator(cfg, seed=42)
        num_samples = 1000
        mode_a_count = sum(1 for i in range(num_samples) if gen._is_mode_a(i))
        ratio = mode_a_count / num_samples
        assert 0.65 <= ratio <= 0.75

    def test_deterministic_reproducibility(self) -> None:
        cfg = BimodalConfig(
            mode_a_system_prompt_len=10,
            mode_a_user_prompt_len=15,
            mode_b_system_prompt_len=20,
            mode_b_user_prompt_len=25,
            mode_a_ratio=0.5,
        )
        gen1 = _make_generator(cfg, api_type=APIType.Completion, seed=123)
        gen2 = _make_generator(cfg, api_type=APIType.Completion, seed=123)
        for i in range(100):
            assert gen1._is_mode_a(i) == gen2._is_mode_a(i)
            req1 = gen1.load_lazy_data(LazyLoadInferenceAPIData(data_index=i))
            req2 = gen2.load_lazy_data(LazyLoadInferenceAPIData(data_index=i))
            assert isinstance(req1, CompletionAPIData)
            assert isinstance(req2, CompletionAPIData)
            assert req1.prompt == req2.prompt
            assert req1.max_tokens == req2.max_tokens

        gen1_chat = _make_generator(cfg, api_type=APIType.Chat, seed=123)
        gen2_chat = _make_generator(cfg, api_type=APIType.Chat, seed=123)
        for i in range(100):
            req1_chat = gen1_chat.load_lazy_data(LazyLoadInferenceAPIData(data_index=i))
            req2_chat = gen2_chat.load_lazy_data(LazyLoadInferenceAPIData(data_index=i))
            assert isinstance(req1_chat, ChatCompletionAPIData)
            assert isinstance(req2_chat, ChatCompletionAPIData)
            assert req1_chat.prefix_text == req2_chat.prefix_text
            assert req1_chat.messages == req2_chat.messages
            assert req1_chat.max_tokens == req2_chat.max_tokens


class TestBimodalRequestMaterialization:
    def test_completion_api_materialization(self) -> None:
        cfg = BimodalConfig(
            mode_a_user_prompt_len=10,
            mode_a_output_len=20,
            mode_b_user_prompt_len=100,
            mode_b_output_len=200,
            mode_a_ratio=1.0,
        )
        gen = _make_generator(cfg, api_type=APIType.Completion, seed=42)
        req = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0))
        assert isinstance(req, CompletionAPIData)
        assert req.max_tokens == 20

    def test_chat_api_materialization(self) -> None:
        cfg = BimodalConfig(
            mode_a_system_prompt_len=32,
            mode_a_user_prompt_len=10,
            mode_a_output_len=15,
            mode_a_ratio=1.0,
        )
        gen = _make_generator(cfg, api_type=APIType.Chat, seed=42)
        req = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0))
        assert isinstance(req, ChatCompletionAPIData)
        assert req.max_tokens == 15
        assert req.prefix_text is not None
        assert len(req.messages) == 1
        assert req.messages[0].role == "user"

    def test_unique_suffixes_same_group(self) -> None:
        tokenizer = CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path="gpt2"))
        cfg = BimodalConfig(
            mode_a_system_prompt_len=20,
            mode_a_user_prompt_len=15,
            mode_a_groups=1,
            mode_a_ratio=1.0,
        )
        gen = _make_generator(cfg, api_type=APIType.Completion, tokenizer=tokenizer, seed=42)
        prompts = set()
        for i in range(50):
            req = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=i))
            assert isinstance(req, CompletionAPIData)
            assert req.prompt is not None
            prompts.add(req.prompt)
        assert len(prompts) == 50

    @pytest.mark.asyncio
    async def test_chat_api_exact_token_count(self) -> None:
        tokenizer = CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path="gpt2"))
        cfg = BimodalConfig(
            mode_a_system_prompt_len=20,
            mode_a_user_prompt_len=15,
            mode_a_ratio=1.0,
        )
        gen = _make_generator(cfg, api_type=APIType.Chat, tokenizer=tokenizer, seed=42)
        req = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=0))
        assert isinstance(req, ChatCompletionAPIData)
        body = await req.to_request_body(effective_model_name="gpt2", max_tokens=10, ignore_eos=True, streaming=False)
        messages = body["messages"]
        assert len(messages) == 1
        content_text = _extract_text_content(messages[0]["content"])
        assert tokenizer.count_tokens(content_text) == 35

    @pytest.mark.asyncio
    async def test_zero_length_edge_cases(self) -> None:
        tokenizer = CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path="gpt2"))

        # Case 1: sys_len=0, u_len=0
        cfg_zero = BimodalConfig(
            mode_a_system_prompt_len=0,
            mode_a_user_prompt_len=0,
            mode_a_ratio=1.0,
        )
        gen_comp = _make_generator(cfg_zero, api_type=APIType.Completion, tokenizer=tokenizer, seed=42)
        req_comp = gen_comp.load_lazy_data(LazyLoadInferenceAPIData(data_index=0))
        assert isinstance(req_comp, CompletionAPIData)
        assert req_comp.prompt == ""

        gen_chat = _make_generator(cfg_zero, api_type=APIType.Chat, tokenizer=tokenizer, seed=42)
        req_chat = gen_chat.load_lazy_data(LazyLoadInferenceAPIData(data_index=0))
        assert isinstance(req_chat, ChatCompletionAPIData)
        body = await req_chat.to_request_body(effective_model_name="gpt2", max_tokens=10, ignore_eos=True, streaming=False)
        assert _extract_text_content(body["messages"][0]["content"]) == ""

        # Case 2: sys_len=10, u_len=0
        cfg_sys_only = BimodalConfig(
            mode_a_system_prompt_len=10,
            mode_a_user_prompt_len=0,
            mode_a_ratio=1.0,
        )
        gen_chat2 = _make_generator(cfg_sys_only, api_type=APIType.Chat, tokenizer=tokenizer, seed=42)
        req_chat2 = gen_chat2.load_lazy_data(LazyLoadInferenceAPIData(data_index=0))
        assert isinstance(req_chat2, ChatCompletionAPIData)
        body2 = await req_chat2.to_request_body(effective_model_name="gpt2", max_tokens=10, ignore_eos=True, streaming=False)
        content_text2 = _extract_text_content(body2["messages"][0]["content"])
        assert tokenizer.count_tokens(content_text2) == 10

    def test_distribution_inputs_materialization(self) -> None:
        tokenizer = CustomTokenizer(CustomTokenizerConfig(pretrained_model_name_or_path="gpt2"))
        cfg = BimodalConfig(
            mode_a_system_prompt_len=10,
            mode_a_user_prompt_len=Distribution(min=10, max=20, mean=15, std_dev=2.0),
            mode_a_output_len=Distribution(min=5, max=15, mean=10, std_dev=1.0),
            mode_b_system_prompt_len=10,
            mode_b_user_prompt_len=Distribution(min=50, max=100, mean=75, std_dev=5.0),
            mode_b_output_len=Distribution(min=20, max=40, mean=30, std_dev=2.0),
            mode_a_ratio=0.5,
        )
        gen = _make_generator(cfg, api_type=APIType.Completion, tokenizer=tokenizer, seed=42)
        for i in range(20):
            req = gen.load_lazy_data(LazyLoadInferenceAPIData(data_index=i))
            assert isinstance(req, CompletionAPIData)
            assert req.prompt is not None
            is_a = gen._is_mode_a(i)
            if is_a:
                assert 5 <= req.max_tokens <= 15
                prompt_tokens = tokenizer.count_tokens(req.prompt)
                assert 20 <= prompt_tokens <= 30  # sys_len (10) + user_len (10..20)
            else:
                assert 20 <= req.max_tokens <= 40
                prompt_tokens = tokenizer.count_tokens(req.prompt)
                assert 60 <= prompt_tokens <= 110  # sys_len (10) + user_len (50..100)


class TestBimodalValidation:
    def test_missing_bimodal_config_raises(self) -> None:
        api_config = APIConfig(type=APIType.Completion)
        data_config = DataConfig(type=DataGenType.Bimodal, bimodal=None)
        with pytest.raises(ValueError, match="BimodalConfig must be specified"):
            BimodalDataGenerator(api_config, data_config, _make_mock_tokenizer())

    def test_misplaced_config_raises(self) -> None:
        api_config = APIConfig(type=APIType.Completion)
        cfg = BimodalConfig()
        data_cfg_io = DataConfig(
            type=DataGenType.Bimodal,
            bimodal=cfg,
            input_distribution=Distribution(mean=10, min=10, max=10, std_dev=0.0),
        )
        with pytest.raises(Exception, match="IO distribution not supported for this data generator"):
            BimodalDataGenerator(api_config, data_cfg_io, _make_mock_tokenizer())

        data_cfg_sp = DataConfig(
            type=DataGenType.Bimodal,
            bimodal=cfg,
            shared_prefix=SharedPrefix(num_groups=1, system_prompt_len=10, question_len=10),
        )
        with pytest.raises(Exception, match="Shared prefix not supported for this data generator"):
            BimodalDataGenerator(api_config, data_cfg_sp, _make_mock_tokenizer())

    def test_negative_distribution_bounds_raise(self) -> None:
        with pytest.raises(ValueError, match="distribution min cannot be negative"):
            BimodalConfig(mode_a_user_prompt_len=Distribution(min=-1, max=10, mean=5, std_dev=1.0))

        with pytest.raises(ValueError, match="distribution mean cannot be negative"):
            BimodalConfig(mode_b_output_len=Distribution(min=0, max=10, mean=-5, std_dev=1.0))

    def test_negative_int_length_raises(self) -> None:
        with pytest.raises(ValueError, match="cannot be negative"):
            BimodalConfig(mode_a_user_prompt_len=-5)
