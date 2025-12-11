import random
from typing import Generator, List, Optional
from inference_perf.utils.distribution import generate_distribution
import numpy as np

from inference_perf.apis.base import InferenceAPIData, LazyLoadInferenceAPIData
from inference_perf.apis.completion import CompletionAPIData
from inference_perf.apis.user_session import LocalUserSession, UserSessionCompletionAPIData, UserSessionChatAPIData
from inference_perf.apis.chat import ChatCompletionAPIData, ChatMessage
from inference_perf.config import APIConfig, APIType, DataConfig, Distribution
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
        self.enable_multi_turn_chat: bool = self.shared_prefix.enable_multi_turn_chat

        # Use distribution configs, or fall back to question_len/output_len with std_dev=0
        q_len = self.shared_prefix.question_len
        o_len = self.shared_prefix.output_len
        question_dist = self.shared_prefix.question_distribution or Distribution(min=q_len, max=q_len, mean=q_len, std_dev=0)
        output_dist = self.shared_prefix.output_distribution or Distribution(min=o_len, max=o_len, mean=o_len, std_dev=0)

        # Generate separate distributions for each group
        self.question_len_list_per_group: List[List[int]] = []
        self.output_len_list_per_group: List[List[int]] = []

        for _ in range(self.num_groups):
            question_lens = generate_distribution(
                question_dist.min,
                question_dist.max,
                question_dist.mean,
                question_dist.std_dev,
                self.shared_prefix.num_prompts_per_group,
            )
            self.question_len_list_per_group.append(question_lens.tolist())

            output_lens = generate_distribution(
                output_dist.min,
                output_dist.max,
                output_dist.mean,
                output_dist.std_dev,
                self.shared_prefix.num_prompts_per_group,
            )
            self.output_len_list_per_group.append(output_lens.tolist())

        self.prompts: List[str] = []  # For completion API
        self.prompt_pairs: List[tuple[str, str]] = []  # (shared_prefix, question) pairs for chat API
        self.user_sessions: List[LocalUserSession] = []
        self.flat_output_lens: List[int] = []
        self._generate_prompts()

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def is_io_distribution_supported(self) -> bool:
        return True

    def is_shared_prefix_supported(self) -> bool:
        return True

    def is_prefered_worker_requested(self) -> bool:
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
        elif self.api_config.type == APIType.Chat:
            shared_prefix, question = self.prompt_pairs[i]
            messages = [
                ChatMessage(role="system", content=shared_prefix),
                ChatMessage(role="user", content=question)
            ]
            return ChatCompletionAPIData(messages=messages, max_tokens=output_len)
        else:
            # Single-turn: use data_index directly
            i = data.data_index % len(self.prompts)
            if self.api_config.type == APIType.Chat:
                shared_prefix, question = self.prompt_pairs[i]
                messages = [
                    ChatMessage(role="system", content=shared_prefix),
                    ChatMessage(role="user", content=question)
                ]
                return ChatCompletionAPIData(messages=messages, max_tokens=self.output_len)
            else:
                return CompletionAPIData(prompt=self.prompts[i], max_tokens=self.output_len)

    def get_request(self, n: int) -> InferenceAPIData:
        i = n % len(self.prompts)
        if self.api_config.type == APIType.Chat:
            shared_prefix, question = self.prompt_pairs[i]
            messages = [
                ChatMessage(role="system", content=shared_prefix),
                ChatMessage(role="user", content=question)
            ]
            return ChatCompletionAPIData(messages=messages, max_tokens=self.output_len)
        else:
            return CompletionAPIData(prompt=self.prompts[i], max_tokens=output_len)

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        if not self.prompts:
            return

        i = 0
        while True:
            if self.enable_multi_turn_chat:
                prefered_worker_id = i % self.num_groups
                yield LazyLoadInferenceAPIData(data_index=i, prefered_worker_id=prefered_worker_id)
                i += 1
            elif self.api_config.type == APIType.Chat:
                shared_prefix, question = self.prompt_pairs[i]
                messages = [
                    ChatMessage(role="system", content=shared_prefix),
                    ChatMessage(role="user", content=question)
                ]
                yield ChatCompletionAPIData(messages=messages, max_tokens=self.output_len)
                i = (i + 1) % len(self.prompts)
            else:
                yield CompletionAPIData(prompt=self.prompts[i], max_tokens=self.output_len)
                i = (i + 1) % len(self.prompts)

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

        if self.shared_prefix is None:
            raise ValueError("Shared prefix is not available for generating prompts.")

        hf_tokenizer = self.tokenizer.get_tokenizer()

        for group_id in range(self.num_groups):
            # Generate a shared prefix (system prompt)
            shared_prefix_token_ids = self._generate_random_token_ids(self.system_prompt_len)
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

                # Combine shared prefix and question
                full_prompt_text = shared_prefix_text + " " + question_text
                self.prompts.append(full_prompt_text)
                self.prompt_pairs.append((shared_prefix_text, question_text))

                if self.enable_multi_turn_chat:
                    # multi turn chat, create user to keep conversation
                    # For Chat API, context should be a list of messages starting with system prompt
                    # For Completion API, context is a string
                    if self.api_config.type == APIType.Chat:
                        initial_context = [
                            ChatMessage(role="system", content=shared_prefix_text)
                        ]
                    else:
                        initial_context = shared_prefix_text
                    
                    self.user_sessions.append(
                        LocalUserSession(
                            user_session_id=f"user_session_{self.num_prompts_per_group * group_id + prompt_id}",
                            context=initial_context,
                        )
                    )

        # Flatten output lengths to match prompts ordering
        self.flat_output_lens = [
            self.output_len_list_per_group[g][p] for g in range(self.num_groups) for p in range(self.num_prompts_per_group)
        ]

        # Shuffle the generated prompts to ensure randomness if served sequentially by different workers
        if self.enable_multi_turn_chat:
            # no need to sync shuffles - multi-round initial prompt does not include system prompt
            random.shuffle(self.user_sessions)
        else:
            # Shuffle prompts, prompt_pairs, and output lengths in sync
            combined = list(zip(self.prompts, self.prompt_pairs, self.flat_output_lens, strict=True))
            random.shuffle(combined)
            self.prompts, self.prompt_pairs, self.flat_output_lens = [list(t) for t in zip(*combined, strict=True)]
