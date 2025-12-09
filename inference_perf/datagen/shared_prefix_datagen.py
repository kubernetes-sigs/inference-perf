import random
from typing import Generator, List, Optional
import numpy as np

from inference_perf.apis.base import InferenceAPIData, LazyLoadInferenceAPIData
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.apis.user_session import LocalUserSession, UserSessionCompletionAPIData
from inference_perf.config import APIConfig, APIType, DataConfig
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from .base import DataGenerator, LazyLoadDataMixin


# Shared Prefix Generator generates shared prefix in the prompts that are sent.
# This can be used to benchmark prefix caching cases.
class SharedPrefixDataGenerator(DataGenerator, LazyLoadDataMixin):
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
        self.system_prompt_len: int = self.shared_prefix.system_prompt_len
        self.question_len: int = self.shared_prefix.question_len
        self.output_len: int = self.shared_prefix.output_len
        self.enable_multi_turn_chat: bool = self.shared_prefix.enable_multi_turn_chat
        
        self.question_len_std: float = self.shared_prefix.question_len_std
        self.output_len_std: float = self.shared_prefix.output_len_std
        
        self.question_len_min: Optional[int] = self.shared_prefix.question_len_min
        self.question_len_max: Optional[int] = self.shared_prefix.question_len_max
        self.output_len_min: Optional[int] = self.shared_prefix.output_len_min
        self.output_len_max: Optional[int] = self.shared_prefix.output_len_max


        self.prompts: List[str] = []
        self.user_sessions: List[LocalUserSession] = []
        self._generate_prompts()

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion]

    def is_io_distribution_supported(self) -> bool:
        return True

    def is_shared_prefix_supported(self) -> bool:
        return True

    def is_prefered_worker_requested(self) -> bool:
        return True if self.enable_multi_turn_chat else False

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> InferenceAPIData:
        i = data.data_index % len(self.prompts)
        output_len = self.output_len
        if self.output_len_std > 0:
                output_len = max(1, int(np.random.normal(self.output_len, self.output_len_std)))
        if self.output_len_min is not None:
                output_len = max(self.output_len_min, output_len)
        if self.output_len_max is not None:
                output_len = min(self.output_len_max, output_len)   
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
            prefered_worker_id = i % self.num_groups if self.enable_multi_turn_chat else -1
            yield LazyLoadInferenceAPIData(data_index=i, prefered_worker_id=prefered_worker_id)
            i += 1

    def _generate_random_token_ids(self, length: int) -> List[int]:
        """Generates a list of random token IDs of a specified length."""
        if length == 0:
            return []
        # np.random.randint's high parameter is exclusive
        return np.random.randint(0, self.vocab_size, size=length, dtype=np.int64).tolist()  # type: ignore[no-any-return]

    def _generate_prompts(self) -> None:
        """Pre-generates all prompts based on the configuration."""
        if self.tokenizer is None:
            # This check is defensive; __init__ should have already validated this.
            raise ValueError("Tokenizer is not available for generating prompts.")

        hf_tokenizer = self.tokenizer.get_tokenizer()

        for group_id in range(self.num_groups):
            # Generate a shared prefix (system prompt)
            shared_prefix_token_ids = self._generate_random_token_ids(self.system_prompt_len)
            shared_prefix_text = hf_tokenizer.decode(shared_prefix_token_ids, skip_special_tokens=True)

            for prompt_id in range(self.num_prompts_per_group):
                # Generate a unique question
                question_len = self.question_len
                if self.question_len_std > 0:
                    question_len = max(1, int(np.random.normal(self.question_len, self.question_len_std)))
                if self.question_len_min is not None:
                    question_len = max(self.question_len_min, question_len)
                if self.question_len_max is not None:
                    question_len = min(self.question_len_max, question_len)
                question_token_ids = self._generate_random_token_ids(question_len)
                question_text = hf_tokenizer.decode(question_token_ids, skip_special_tokens=True)

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

        # Shuffle the generated prompts to ensure randomness if served sequentially by different workers
        random.shuffle(self.prompts)
