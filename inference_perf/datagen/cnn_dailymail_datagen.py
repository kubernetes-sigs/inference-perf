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
import asyncio
import logging
import random
from inference_perf.apis import InferenceAPIData, CompletionAPIData
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from .base import DataGenerator, DatasetSummary
from inference_perf.config import APIConfig, APIType, DataConfig
from typing import Generator, List, Optional, Self
from datasets import load_dataset
import os

logger = logging.getLogger(__name__)


class CNNDailyMailDataGenerator(DataGenerator):
    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: CustomTokenizer | None) -> None:
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
            self.cnn_dailymail_dataset = iter(
                load_dataset(
                    "abisee/cnn_dailymail",
                    "3.0.0",
                    streaming=True,
                    split="train",
                )
            )
        self.article_key = "article"
        self.highlights_key = "highlights"
        self.dataset = dataset
        self.cnn_dailymail_datasetfiltered_dataset: List[InferenceAPIData] = []

    @classmethod
    async def create(cls, api_config: APIConfig, config: DataConfig, tokenizer: Optional[CustomTokenizer]) -> Self:
        instance = cls(api_config, config, tokenizer)
        instance._dataset_filtering_task = asyncio.create_task(instance._create_filtered_dataset(instance.dataset))
        return instance

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion]

    async def _create_filtered_dataset(self, dataset: Generator[InferenceAPIData, None, None]) -> None:
        # Ensured by main.py logic and __init__ type hint for this class
        assert self.tokenizer is not None

        logger.info("starting to filter dataset...")
        for index, data in enumerate(dataset):
            if index % 1000 == 0 and index > 0:
                logger.info(f"processed {index} items... (valid: {len(self.cnn_dailymail_dataset)})")
            if not isinstance(data, dict):
                continue
            if data is None or data[self.article_key] is None or data[self.highlights_key] is None:
                continue
            api_data: InferenceAPIData
            if self.api_config.type == APIType.Completion:
                try:
                    prompt = data[self.article_key]
                    completion = data[self.highlights_key]
                    api_data = CompletionAPIData(prompt=prompt, max_tokens=self.tokenizer.count_tokens(completion))
                except (KeyError, TypeError) as e:
                    logger.warning(f"Skipping invalid completion data: {e}")
                    continue
            else:
                raise Exception("Unsupported API type")
            if api_data.valid_in_distribution(self.tokenizer, self.input_distribution, self.output_distribution):
                self.cnn_dailymail_dataset.append(api_data)
        if len(self.cnn_dailymail_dataset) == 0:
            raise Exception("filtered dataset contains no prompts compatible with the requested distributions")
        logger.info("finished processing dataset")
        self.dataset_summary = self.generate_dataset_summary()

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        if self.cnn_dailymail_dataset:
            while True:
                yield random.choice(self.cnn_dailymail_dataset)

    def generate_dataset_summary(self) -> DatasetSummary:
        return DatasetSummary(num_unique_prompts=len(self.cnn_dailymail_dataset))

    def is_io_distribution_supported(self) -> bool:
        return True

    def is_shared_prefix_supported(self) -> bool:
        return False
