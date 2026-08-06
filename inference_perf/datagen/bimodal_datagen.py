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
import hashlib
import logging
from dataclasses import dataclass
from typing import Dict, Generator, List, Optional, Union
import numpy as np

from inference_perf.apis.base import InferenceAPIData, LazyLoadInferenceAPIData
from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.config import APIConfig, APIType, DataConfig, Distribution

from inference_perf.config.datagen.bimodal import BimodalConfig
from inference_perf.datagen.base import DataGenerator, LazyLoadDataMixin
from inference_perf.datagen.datagen_utils import (
    build_word_start_token_ids,
    converge_to_exact_length_text,
    generate_random_exact_length_text,
    init_vocab_sampling,
    random_token_ids,
)
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.utils.numeric.distribution import sample_from_distribution

logger = logging.getLogger(__name__)


def _resolve_distribution(param: Union[int, Distribution]) -> Distribution:
    """Resolve a Union[int, Distribution] into a Distribution object."""
    if isinstance(param, Distribution):
        return param
    return Distribution(mean=float(param), min=param, max=param, std_dev=0.0)


@dataclass
class PreGeneratedPrompt:
    prefix_text: Optional[str]
    user_text: str
    full_prompt: str
    output_len: int


class BimodalDataGenerator(DataGenerator, LazyLoadDataMixin):
    """Generator for bimodal prompt workloads with high worker throughput and BPE boundary safety."""

    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: Optional[CustomTokenizer] = None):
        super().__init__(api_config, config, tokenizer)
        if self.config.bimodal is None:
            raise ValueError("BimodalConfig must be specified for BimodalDataGenerator")

        self.bimodal_config: BimodalConfig = self.config.bimodal
        self.seed = self.bimodal_config.seed if self.bimodal_config.seed is not None else 42
        self.rng = np.random.default_rng(self.seed)

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for BimodalDataGenerator.")

        self.vocab_size, self.special_token_ids, self.valid_token_ids = init_vocab_sampling(self.tokenizer)
        self.word_start_token_ids = build_word_start_token_ids(self.tokenizer, self.valid_token_ids)

        self.mode_a_user_prompt_dist = _resolve_distribution(self.bimodal_config.mode_a_user_prompt_len)
        self.mode_a_output_dist = _resolve_distribution(self.bimodal_config.mode_a_output_len)
        self.mode_b_user_prompt_dist = _resolve_distribution(self.bimodal_config.mode_b_user_prompt_len)
        self.mode_b_output_dist = _resolve_distribution(self.bimodal_config.mode_b_output_len)

        self.mode_a_prompts: Dict[int, List[PreGeneratedPrompt]] = {}
        self.mode_b_prompts: Dict[int, List[PreGeneratedPrompt]] = {}

        self._pregenerate_all_prompts()

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def is_io_distribution_supported(self) -> bool:
        return True

    def is_shared_prefix_supported(self) -> bool:
        return True

    def _sample_suffix_ids(self, length: int) -> List[int]:
        if length <= 0:
            return []
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for sampling suffix IDs.")
        initial = [int(self.rng.choice(self.word_start_token_ids))]
        if length > 1:
            initial += random_token_ids(self.rng, self.valid_token_ids, length - 1)

        def adjust(current: List[int], current_len: int, target_len: int) -> List[int]:
            if current_len < target_len:
                current.extend(random_token_ids(self.rng, self.valid_token_ids, target_len - current_len))
                return current
            diff = current_len - target_len
            if diff < len(current) - 1:
                return current[:-diff]
            return current[:1]

        _, ids = converge_to_exact_length_text(
            tokenizer=self.tokenizer,
            target_len=length,
            initial_tokens=initial,
            adjust_tokens_fn=adjust,
        )
        return ids

    def _pregenerate_mode_prompts(
        self,
        sys_len: int,
        num_groups: int,
        user_prompt_dist: Distribution,
        output_dist: Distribution,
    ) -> Dict[int, List[PreGeneratedPrompt]]:
        prompts_by_group: Dict[int, List[PreGeneratedPrompt]] = {}
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for pregenerating prompts.")
        hf_tokenizer = self.tokenizer.get_tokenizer()
        num_prompts = self.bimodal_config.num_prompts_per_group

        for group_id in range(num_groups):
            prompts_by_group[group_id] = []
            prefix_text = None
            prefix_ids: List[int] = []

            if sys_len > 0:
                prefix_text, prefix_ids = generate_random_exact_length_text(
                    self.rng, self.valid_token_ids, self.tokenizer, sys_len
                )

            u_lens = sample_from_distribution(user_prompt_dist, num_prompts, self.rng)
            o_lens = sample_from_distribution(output_dist, num_prompts, self.rng)

            for p_idx in range(num_prompts):
                u_len = int(u_lens[p_idx])
                o_len = int(o_lens[p_idx])

                if sys_len > 0:
                    suffix_ids = self._sample_suffix_ids(u_len)
                    user_text = hf_tokenizer.decode(suffix_ids, skip_special_tokens=True)
                    if isinstance(user_text, list):
                        user_text = " ".join(user_text)

                    full_text = hf_tokenizer.decode(prefix_ids + suffix_ids, skip_special_tokens=True)
                    full_prompt = full_text if isinstance(full_text, str) else " ".join(full_text)
                else:
                    user_text, _ = generate_random_exact_length_text(self.rng, self.valid_token_ids, self.tokenizer, u_len)
                    full_prompt = user_text

                prompts_by_group[group_id].append(
                    PreGeneratedPrompt(
                        prefix_text=prefix_text,
                        user_text=user_text,
                        full_prompt=full_prompt,
                        output_len=o_len,
                    )
                )

        return prompts_by_group

    def _pregenerate_all_prompts(self) -> None:
        self.mode_a_prompts = self._pregenerate_mode_prompts(
            sys_len=self.bimodal_config.mode_a_system_prompt_len,
            num_groups=self.bimodal_config.mode_a_groups,
            user_prompt_dist=self.mode_a_user_prompt_dist,
            output_dist=self.mode_a_output_dist,
        )
        self.mode_b_prompts = self._pregenerate_mode_prompts(
            sys_len=self.bimodal_config.mode_b_system_prompt_len,
            num_groups=self.bimodal_config.mode_b_groups,
            user_prompt_dist=self.mode_b_user_prompt_dist,
            output_dist=self.mode_b_output_dist,
        )

    def _is_mode_a(self, index: int) -> bool:
        hash_input = f"{self.seed}-{index}".encode("utf-8")
        hash_val = int(hashlib.md5(hash_input).hexdigest()[:8], 16)
        return (hash_val / (0xFFFFFFFF + 1)) < self.bimodal_config.mode_a_ratio

    def _get_group_id(self, index: int, num_groups: int, salt: str) -> int:
        if num_groups <= 1:
            return 0
        hash_input = f"{self.seed}-{salt}-{index}".encode("utf-8")
        return int(hashlib.md5(hash_input).hexdigest()[:8], 16) % num_groups

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> InferenceAPIData:
        n = data.data_index
        is_mode_a = self._is_mode_a(n)

        num_groups = self.bimodal_config.mode_a_groups if is_mode_a else self.bimodal_config.mode_b_groups
        salt = "mode_a" if is_mode_a else "mode_b"
        group_id = self._get_group_id(n, num_groups, salt)

        pool = self.mode_a_prompts[group_id] if is_mode_a else self.mode_b_prompts[group_id]
        prompt_item = pool[n % len(pool)]

        if self.api_config.type == APIType.Chat:
            return ChatCompletionAPIData(
                messages=[ChatMessage(role="user", content=prompt_item.user_text)],
                prefix_text=prompt_item.prefix_text,
                max_tokens=prompt_item.output_len,
            )

        return CompletionAPIData(prompt=prompt_item.full_prompt, max_tokens=prompt_item.output_len)

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        i = 0
        while True:
            yield LazyLoadInferenceAPIData(data_index=i)
            i += 1
