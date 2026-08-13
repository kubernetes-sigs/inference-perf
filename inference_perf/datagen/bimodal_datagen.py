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
from typing import Dict, Generator, List, Optional, Tuple, Union
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


def _resolve_distribution(param: Union[int, Distribution]) -> Distribution:
    """Resolve a Union[int, Distribution] into a Distribution object."""
    if isinstance(param, Distribution):
        return param
    return Distribution(mean=float(param), min=param, max=param, std_dev=0.0)


class BimodalDataGenerator(DataGenerator, LazyLoadDataMixin):
    """Generator for bimodal prompt workloads with high worker throughput and BPE boundary safety.

    Operates purely in single-turn mode, statelessly generating independent requests without
    session tracking or preferred worker affinity.
    """

    def __init__(
        self,
        api_config: APIConfig,
        config: DataConfig,
        tokenizer: Optional[CustomTokenizer] = None,
        seed: Optional[int] = None,
    ):
        super().__init__(api_config, config, tokenizer)
        if self.config.bimodal is None:
            raise ValueError("BimodalConfig must be specified for BimodalDataGenerator")

        self.bimodal_config: BimodalConfig = self.config.bimodal
        self.seed = seed if seed is not None else 42
        self.rng = np.random.default_rng(self.seed)

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for BimodalDataGenerator.")

        self.vocab_size, self.special_token_ids, self.valid_token_ids = init_vocab_sampling(self.tokenizer)
        self.word_start_token_ids = build_word_start_token_ids(self.tokenizer, self.valid_token_ids)

        self.mode_a_user_prompt_dist = _resolve_distribution(self.bimodal_config.mode_a_user_prompt_len)
        self.mode_a_output_dist = _resolve_distribution(self.bimodal_config.mode_a_output_len)
        self.mode_b_user_prompt_dist = _resolve_distribution(self.bimodal_config.mode_b_user_prompt_len)
        self.mode_b_output_dist = _resolve_distribution(self.bimodal_config.mode_b_output_len)

        self.mode_a_prefixes: Dict[int, Tuple[Optional[str], List[int]]] = {}
        self.mode_b_prefixes: Dict[int, Tuple[Optional[str], List[int]]] = {}

        self._pregenerate_prefixes()

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def is_io_distribution_supported(self) -> bool:
        return False

    def is_shared_prefix_supported(self) -> bool:
        return False

    def _pregenerate_mode_prefixes(
        self,
        sys_len: int,
        num_groups: int,
    ) -> Dict[int, Tuple[Optional[str], List[int]]]:
        if sys_len <= 0:
            return {group_id: (None, []) for group_id in range(num_groups)}

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for pregenerating prefixes.")

        prefixes: Dict[int, Tuple[Optional[str], List[int]]] = {}
        for group_id in range(num_groups):
            text, ids = generate_random_exact_length_text(
                self.rng, self.valid_token_ids, self.tokenizer, sys_len
            )
            prefixes[group_id] = (text, ids)
        return prefixes

    def _pregenerate_prefixes(self) -> None:
        self.mode_a_prefixes = self._pregenerate_mode_prefixes(
            sys_len=self.bimodal_config.mode_a_system_prompt_len,
            num_groups=self.bimodal_config.mode_a_groups,
        )
        self.mode_b_prefixes = self._pregenerate_mode_prefixes(
            sys_len=self.bimodal_config.mode_b_system_prompt_len,
            num_groups=self.bimodal_config.mode_b_groups,
        )

    def _is_mode_a(self, index: int) -> bool:
        hash_input = f"{self.seed}-mode-{index}".encode("utf-8")
        hash_val = int(hashlib.md5(hash_input).hexdigest()[:8], 16)
        return (hash_val / (0xFFFFFFFF + 1)) < self.bimodal_config.mode_a_ratio

    def _get_group_id(self, index: int, num_groups: int, salt: str) -> int:
        if num_groups <= 1:
            return 0
        hash_input = f"{self.seed}-{salt}-{index}".encode("utf-8")
        return int(hashlib.md5(hash_input).hexdigest()[:8], 16) % num_groups

    def _get_request_rng(self, index: int, salt: str) -> np.random.Generator:
        hash_input = f"{self.seed}-{salt}-req-{index}".encode("utf-8")
        seed_int = int(hashlib.md5(hash_input).hexdigest()[:8], 16)
        return np.random.default_rng(seed_int)

    def _sample_suffix_ids(self, length: int, rng: np.random.Generator) -> List[int]:
        if length <= 0:
            return []
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for sampling suffix IDs.")
        initial = [int(rng.choice(self.word_start_token_ids))]
        if length > 1:
            initial += random_token_ids(rng, self.valid_token_ids, length - 1)

        def adjust(current: List[int], current_len: int, target_len: int) -> List[int]:
            if current_len < target_len:
                current.extend(random_token_ids(rng, self.valid_token_ids, target_len - current_len))
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

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> InferenceAPIData:
        n = data.data_index
        is_mode_a = self._is_mode_a(n)
        salt = "mode_a" if is_mode_a else "mode_b"
        num_groups = self.bimodal_config.mode_a_groups if is_mode_a else self.bimodal_config.mode_b_groups
        group_id = self._get_group_id(n, num_groups, salt)

        prefix_text, prefix_ids = (
            self.mode_a_prefixes[group_id] if is_mode_a else self.mode_b_prefixes[group_id]
        )
        user_prompt_dist = self.mode_a_user_prompt_dist if is_mode_a else self.mode_b_user_prompt_dist
        output_dist = self.mode_a_output_dist if is_mode_a else self.mode_b_output_dist

        req_rng = self._get_request_rng(n, salt)
        u_len = int(sample_from_distribution(user_prompt_dist, 1, req_rng)[0])
        o_len = int(sample_from_distribution(output_dist, 1, req_rng)[0])

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for BimodalDataGenerator.")
        hf_tokenizer = self.tokenizer.get_tokenizer()

        if prefix_text is not None and prefix_ids:
            suffix_ids = self._sample_suffix_ids(u_len, req_rng)
            full_text = hf_tokenizer.decode(prefix_ids + suffix_ids, skip_special_tokens=True)
            full_prompt = full_text if isinstance(full_text, str) else " ".join(full_text)
            user_text = full_prompt[len(prefix_text) + 1 :] if len(full_prompt) > len(prefix_text) else ""
        else:
            user_text, _ = generate_random_exact_length_text(req_rng, self.valid_token_ids, self.tokenizer, u_len)
            full_prompt = user_text

        if self.api_config.type == APIType.Chat:
            return ChatCompletionAPIData(
                messages=[ChatMessage(role="user", content=user_text)],
                prefix_text=prefix_text,
                max_tokens=o_len,
            )

        return CompletionAPIData(prompt=full_prompt, max_tokens=o_len)

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        i = 0
        while True:
            yield LazyLoadInferenceAPIData(data_index=i)
            i += 1
