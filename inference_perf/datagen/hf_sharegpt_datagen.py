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
import random
from inference_perf.apis import InferenceAPIData, CompletionAPIData, ChatCompletionAPIData, ChatMessage
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from .base import DataGenerator, DatasetSummary
from inference_perf.config import APIConfig, APIType, DataConfig
from typing import Generator, List
from datasets import load_dataset
import os

logger = logging.getLogger(__name__)


class HFShareGPTDataGenerator(DataGenerator):
    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: CustomTokenizer) -> None:
        super().__init__(api_config, config, tokenizer)

        if config.path is not None:
            # check if the path is valid
            if not os.path.exists(config.path):
                raise ValueError(f"Invalid dataset path: {config.path}. Path does not exist.")
            # depending on whether the dataset is a single file or a directory, we need to load it differently
            # TODO: add support for other file types
            if os.path.isfile(config.path) and config.path.endswith(".json"):
                dataset = iter(load_dataset("json", data_files=config.path, streaming=True, split="train"))
            elif os.path.isdir(config.path):
                json_files = [f for f in os.listdir(config.path) if f.endswith(".json")]
                dataset = iter(load_dataset("json", data_files=json_files, streaming=True, split="train"))
            else:
                raise ValueError(f"Invalid dataset path: {config.path}")
        else:
            dataset = iter(
                load_dataset(
                    "anon8231489123/ShareGPT_Vicuna_unfiltered",
                    data_files="ShareGPT_V3_unfiltered_cleaned_split.json",
                    streaming=True,
                    split="train",
                )
            )
        
        self.min_num_turns = 2
        self.data_key = "conversations"
        self.role_key = "from"
        self.content_key = "value"
        self.sharegpt_dataset = self._create_filtered_dataset(dataset)
        self.dataset_summary = self.generate_dataset_summary()

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Chat, APIType.Completion]

    def _create_filtered_dataset(self, dataset: Generator[InferenceAPIData, None, None]) -> List[InferenceAPIData]:
        # Ensured by main.py logic and __init__ type hint for this class
        assert self.tokenizer is not None

        filtered_dataset: List[InferenceAPIData] = []
        logger.info("starting to filter dataset...")
        for index, data in enumerate(dataset):
            if index % 1000 == 0 and index > 0:
                logger.info(f"processed {index} items... (valid: {len(filtered_dataset)})")
            if not isinstance(data, dict):
                continue
            if (
                data is None
                or data[self.data_key] is None
                or len(data[self.data_key]) < self.min_num_turns
                or len(data[self.data_key]) == 0
            ):
                continue
            api_data: InferenceAPIData
            if self.api_config.type == APIType.Completion:
                try:
                    prompt = data[self.data_key][0].get(self.content_key)
                    completion = data[self.data_key][1].get(self.content_key)
                    api_data = CompletionAPIData(prompt=prompt, max_tokens=self.tokenizer.count_tokens(completion))
                except (KeyError, TypeError) as e:
                    logger.warning(f"Skipping invalid completion data: {e}")
                    continue
            elif self.api_config.type == APIType.Chat:
                api_data = ChatCompletionAPIData(
                    messages=[
                        ChatMessage(role=conversation[self.role_key], content=conversation[self.content_key])
                        for conversation in data[self.data_key]
                    ]
                )
            else:
                raise Exception("Unsupported API type")
            if api_data.valid_in_distribution(self.tokenizer, self.input_distribution, self.output_distribution):
                filtered_dataset.append(api_data)
        logger.info("finished processing dataset")
        return filtered_dataset

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        if self.sharegpt_dataset:
            while True:
                yield random.choice(self.sharegpt_dataset)

    def generate_dataset_summary(self) -> DatasetSummary:
        return DatasetSummary(num_unique_prompts=len(self.sharegpt_dataset))

    def is_io_distribution_supported(self) -> bool:
        return True

    def is_shared_prefix_supported(self) -> bool:
        return False
