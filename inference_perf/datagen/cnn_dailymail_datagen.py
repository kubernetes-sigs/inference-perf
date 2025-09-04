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
import itertools
import logging
from inference_perf.apis import InferenceAPIData, CompletionAPIData
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from .base import DataGenerator
from inference_perf.config import APIConfig, APIType, DataConfig
from typing import Generator, List, Optional
from datasets import load_dataset
import os

logger = logging.getLogger(__name__)


class CNNDailyMailDataGenerator(DataGenerator):
    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: CustomTokenizer | None) -> None:
        super().__init__(api_config, config, tokenizer)
        if tokenizer is None:
            raise ValueError("CustomTokenizer instance cannot be None")

        if config.path is not None:
            # check if the path is valid
            if not os.path.exists(config.path):
                raise ValueError(f"Invalid dataset path: {config.path}. Path does not exist.")
            # depending on whether the dataset is a single file or a directory, we need to load it differently
            # TODO: add support for other file types
            if os.path.isfile(config.path) and config.path.endswith(".json"):
                self.cnn_dailymail_dataset = itertools.cycle(
                    load_dataset("json", data_files=config.path, streaming=True, split="train")
                )
            elif os.path.isdir(config.path):
                json_files = [f for f in os.listdir(config.path) if f.endswith(".json")]
                self.cnn_dailymail_dataset = itertools.cycle(
                    load_dataset("json", data_files=json_files, streaming=True, split="train")
                )
            else:
                raise ValueError(f"Invalid dataset path: {config.path}")
        else:
            self.cnn_dailymail_dataset = itertools.cycle(
                load_dataset(
                    "abisee/cnn_dailymail",
                    "3.0.0",
                    streaming=True,
                    split="train",
                )
            )
        self.article_key = "article"
        self.highlights_key = "highlights"
        # initialize data collection
        next(self.cnn_dailymail_dataset)

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion]

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        if self.cnn_dailymail_dataset is None:
            return

        assert self.tokenizer is not None

        if self.api_config.type != APIType.Completion:
            raise Exception("Unsupported API type")

        while True:
            data = next(self.cnn_dailymail_dataset)
            if data is None or data[self.article_key] is None or data[self.highlights_key] is None:
                continue

            try:
                prompt = data[self.article_key]
                if not prompt:
                    continue

                completion = data[self.highlights_key]
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

    def is_io_distribution_supported(self) -> bool:
        return True

    def is_shared_prefix_supported(self) -> bool:
        return False
