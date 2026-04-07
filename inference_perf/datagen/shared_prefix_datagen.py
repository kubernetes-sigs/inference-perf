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
from typing import Generator, List, Optional, Union

import numpy as np

from inference_perf.apis.base import InferenceAPIData, LazyLoadInferenceAPIData
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.apis.user_session import LocalUserSession, UserSessionCompletionAPIData
from inference_perf.config import APIConfig, APIType, DataConfig, Distribution
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.utils.distribution import sample_from_distribution
from .base import DataGenerator, LazyLoadDataMixin


# Shared Prefix Generator generates shared prefix in the prompts that are sent.
# This can be used to benchmark prefix caching cases.
class SharedPrefixDataGenerator(DataGenerator, LazyLoadDataMixin):
    @staticmethod
    def _resolve_distribution(
        param: Union[int, Distribution],
        legacy_dist: Optional[Distribution] = None,
    ) -> Distribution:
        """Resolve a Union[int, Distribution] + optional legacy Distribution into a Distribution."""
        if isinstance(param, Distribution):
            return param
        # param is an int
        if legacy_dist is not None:
            return legacy_dist
        # Fixed value: min=max=mean, std_dev=0
        return Distribution(mean=float(param), min=param, max=param, std_dev=0.0)

    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: Optional[CustomTokenizer]) -> None:
        super().__init__(api_config, config, tokenizer)

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for SharedPrefixDataGenerator but was not initialized.")

        # Initialize vocab_size
        hf_tokenizer = self.tokenizer.get_tokenizer()
        if hasattr(hf_tokenizer, "vocab_size") and hf_tokenizer.vocab_size is not None:
            self.vocab_size: int = hf_tokenizer.vocab_size
        elif hasattr(hf_tokenizer, "get_vocab") and callable(hf_tokenizer.get_vocab):
            self.vocab_size = len(hf_tokenizer.get_vocab())
        else:
            try:
                self.vocab_size = len(hf_tokenizer)
            except TypeError as e:
                raise ValueError(
                    "Tokenizer does not have a 'vocab_size' attribute, 'get_vocab()' method, "
                    "or support len() for vocabulary size. Cannot use random token generation."
                ) from e
        if self.vocab_size <= 0:
            raise ValueError(f"Tokenizer vocabulary size must be positive, got {self.vocab_size}.")

        if self.shared_prefix is None:
            raise ValueError("Shared Prefix config is required for SharedPrefixDataGenerator")

        self.num_groups: int = self.shared_prefix.num_groups
        self.num_prompts_per_group: int = self.shared_prefix.num_prompts_per_group
        self.enable_multi_turn_chat: bool = self.shared_prefix.enable_multi_turn_chat

        # Deterministic seeded RNG
        self.rng: np.random.Generator = np.random.default_rng(self.shared_prefix.seed)

        # Resolve all parameters to Distribution
        system_prompt_dist = self._resolve_distribution(self.shared_prefix.system_prompt_len)
        question_dist = self._resolve_distribution(self.shared_prefix.question_len, self.shared_prefix.question_distribution)
        output_dist = self._resolve_distribution(self.shared_prefix.output_len, self.shared_prefix.output_distribution)

        # Generate per-group system prompt lengths
        self.system_prompt_lens_per_group: List[int] = sample_from_distribution(
            system_prompt_dist, self.num_groups, self.rng
        ).tolist()

        # Generate separate distributions for each group
        self.question_len_list_per_group: List[List[int]] = []
        self.output_len_list_per_group: List[List[int]] = []

        for _ in range(self.num_groups):
            question_lens = sample_from_distribution(question_dist, self.num_prompts_per_group, self.rng)
            self.question_len_list_per_group.append(question_lens.tolist())

            output_lens = sample_from_distribution(output_dist, self.num_prompts_per_group, self.rng)
            self.output_len_list_per_group.append(output_lens.tolist())

        self.prompts: List[str] = []
        self.user_sessions: List[LocalUserSession] = []
        self.flat_output_lens: List[int] = []
        self._generate_prompts()

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion]

    def is_io_distribution_supported(self) -> bool:
        return True

    def is_shared_prefix_supported(self) -> bool:
        return True

    def is_preferred_worker_requested(self) -> bool:
        return True if self.enable_multi_turn_chat else False

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> InferenceAPIData:
        i = data.data_index % len(self.prompts)
        output_len = self.flat_output_lens[i]

        if self.enable_multi_turn_chat:
            user_id = data.data_index % len(self.user_sessions)
            round = data.data_index // len(self.user_sessions)
            return UserSessionCompletionAPIData(
                prompt=self.prompts[i],
                max_tokens=output_len,
                user_session=self.user_sessions[user_id],
                target_round=round,
            )
        else:
            return CompletionAPIData(prompt=self.prompts[i], max_tokens=output_len)

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        if not self.prompts:
            return

        i = 0
        while True:
            preferred_worker_id = i % self.num_groups if self.enable_multi_turn_chat else -1
            yield LazyLoadInferenceAPIData(data_index=i, preferred_worker_id=preferred_worker_id)
            i += 1

    def _generate_random_token_ids(self, length: int) -> List[int]:
        """Generates a list of random token IDs of a specified length."""
        if length == 0:
            return []
        return self.rng.integers(0, self.vocab_size, size=length, dtype=np.int64).tolist()  # type: ignore[no-any-return]

    def _generate_prompts(self) -> None:
        """Pre-generates all prompts based on the configuration."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is not available for generating prompts.")

        if self.shared_prefix is None:
            raise ValueError("Shared prefix is not available for generating prompts.")

        hf_tokenizer = self.tokenizer.get_tokenizer()

        for group_id in range(self.num_groups):
            # Generate a shared prefix (system prompt) with per-group length
            sys_prompt_len = self.system_prompt_lens_per_group[group_id]
            shared_prefix_token_ids = self._generate_random_token_ids(sys_prompt_len)
            shared_prefix_text = hf_tokenizer.decode(shared_prefix_token_ids, skip_special_tokens=True)

            # Batch generate all question token IDs for this group
            all_question_token_ids = [
                self._generate_random_token_ids(self.question_len_list_per_group[group_id][prompt_id])
                for prompt_id in range(self.num_prompts_per_group)
            ]

            # Batch decode all questions at once (much faster than individual decode calls)
            all_question_texts = hf_tokenizer.batch_decode(all_question_token_ids, skip_special_tokens=True)

            for prompt_id in range(self.num_prompts_per_group):
                question_text = all_question_texts[prompt_id]

                if self.enable_multi_turn_chat:
                    # multi turn chat, create user to keep conversation
                    self.user_sessions.append(
                        LocalUserSession(
                            user_session_id=f"user_session_{self.num_prompts_per_group * group_id + prompt_id}",
                            context=shared_prefix_text,
                        )
                    )
                else:
                    # Single turn chat, Combine shared prefix and question
                    question_text = shared_prefix_text + " " + question_text

                self.prompts.append(question_text)

        # Flatten output lengths to match prompts ordering
        self.flat_output_lens = [
            self.output_len_list_per_group[g][p] for g in range(self.num_groups) for p in range(self.num_prompts_per_group)
        ]

        # Shuffle using seeded RNG for reproducibility
        indices = self.rng.permutation(len(self.prompts))
        self.prompts = [self.prompts[i] for i in indices]
        self.flat_output_lens = [self.flat_output_lens[i] for i in indices]
        if self.enable_multi_turn_chat:
            self.user_sessions = [self.user_sessions[i] for i in indices]
