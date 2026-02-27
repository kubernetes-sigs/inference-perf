# Copyright 2025 The Kubernetes Authors.
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
from inference_perf.apis import InferenceAPIData, CompletionAPIData, ChatCompletionAPIData, ChatMessage
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from .base import DataGenerator
from inference_perf.config import APIConfig, APIType, DataConfig
from typing import Generator, List, Optional, Iterator, Any
from datasets import load_dataset
import os

logger = logging.getLogger(__name__)

SHAREGPT_HF_DATASET_URL = "anon8231489123/ShareGPT_Vicuna_unfiltered"
SHAREGPT_HF_DATAFILES_PATH = "ShareGPT_V3_unfiltered_cleaned_split.json"
SHAREGPT_HF_CHAT_ROLE_MAP = {"human": "user", "gpt": "assistant"}


class HFShareGPTDataGenerator(DataGenerator):
    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: Optional[CustomTokenizer]) -> None:
        super().__init__(api_config, config, tokenizer)

        # Store config for reloading dataset when exhausted
        self._dataset_config = config
        self._dataset_path = config.path
        self.sharegpt_dataset: Iterator[Any] = self._load_dataset()

        self.min_num_turns = 2
        self.data_key = "conversations"
        self.role_key = "from"
        self.content_key = "value"
        # initialize data collection
        self._get_next_data()

    def _load_dataset(self) -> Iterator[Any]:
        """Load or reload the ShareGPT dataset iterator.

        This method creates a new iterator over the dataset. It's called
        during initialization and when the dataset is exhausted to enable
        cycling through the data indefinitely.
        """
        if self._dataset_path is not None:
            # check if the path is valid
            if not os.path.exists(self._dataset_path):
                raise ValueError(f"Invalid dataset path: {self._dataset_path}. Path does not exist.")
            # depending on whether the dataset is a single file or a directory, we need to load it differently
            # TODO: add support for other file types
            if os.path.isfile(self._dataset_path) and self._dataset_path.endswith(".json"):
                return iter(load_dataset("json", data_files=self._dataset_path, streaming=True, split="train"))
            elif os.path.isdir(self._dataset_path):
                json_files = [f for f in os.listdir(self._dataset_path) if f.endswith(".json")]
                return iter(load_dataset("json", data_files=json_files, streaming=True, split="train"))
            else:
                raise ValueError(f"Invalid dataset path: {self._dataset_path}")
        else:
            return iter(load_dataset(
                SHAREGPT_HF_DATASET_URL,
                data_files=SHAREGPT_HF_DATAFILES_PATH,
                streaming=True,
                split="train",
            ))

    def _get_next_data(self) -> Any:
        """Get the next data item, reloading the dataset if exhausted.

        This method handles StopIteration by reloading the dataset iterator,
        allowing the benchmark to cycle through the data indefinitely for
        long-running tests.
        """
        try:
            return next(self.sharegpt_dataset)
        except StopIteration:
            logger.info("ShareGPT dataset exhausted, reloading for continued benchmarking")
            self.sharegpt_dataset = self._load_dataset()
            return next(self.sharegpt_dataset)

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Chat, APIType.Completion]

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        if self.sharegpt_dataset is None:
            return
        if self.api_config.type == APIType.Completion:
            yield from self.get_completion_data()
        elif self.api_config.type == APIType.Chat:
            yield from self.get_chat_data()
        raise Exception("Unsupported API type")

    def get_completion_data(self) -> Generator[InferenceAPIData, None, None]:
        if self.tokenizer is None:
            raise Exception("Tokenizer is required for completion API of HFShareGPTDataGenerator")
        while True:
            data = self._get_next_data()
            if (
                data is None
                or data[self.data_key] is None
                or len(data[self.data_key]) < self.min_num_turns
                or len(data[self.data_key]) == 0
            ):
                continue

            try:
                prompt = data[self.data_key][0].get(self.content_key)
                completion = data[self.data_key][1].get(self.content_key)
                if not prompt:
                    continue
                completion_tokens = self.tokenizer.count_tokens(completion)
                prompt_tokens = self.tokenizer.count_tokens(prompt)

                if self.input_distribution:
                    if prompt_tokens < self.input_distribution.min:
                        continue
                    if prompt_tokens > self.input_distribution.max:
                        continue
                if self.output_distribution:
                    if completion_tokens < self.output_distribution.min:
                        continue
                    if completion_tokens > self.output_distribution.max:
                        continue

                yield CompletionAPIData(prompt=prompt, max_tokens=completion_tokens)

            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping invalid completion data: {e}")
                continue

    def get_chat_data(self) -> Generator[InferenceAPIData, None, None]:
        while True:
            data = self._get_next_data()
            if (
                data is None
                or data[self.data_key] is None
                or len(data[self.data_key]) < self.min_num_turns
                or len(data[self.data_key]) == 0
            ):
                continue
            yield ChatCompletionAPIData(
                messages=[
                    ChatMessage(
                        role=SHAREGPT_HF_CHAT_ROLE_MAP.get(conversation[self.role_key], "user"),
                        content=conversation[self.content_key],
                    )
                    for conversation in data[self.data_key]
                ]
            )

    def is_io_distribution_supported(self) -> bool:
        return True

    def is_shared_prefix_supported(self) -> bool:
        return False
