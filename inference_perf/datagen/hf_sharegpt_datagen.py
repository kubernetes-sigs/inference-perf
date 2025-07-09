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
from typing import Generator, List
from datasets import load_dataset

logger = logging.getLogger(__name__)


class HFShareGPTDataGenerator(DataGenerator):
    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: CustomTokenizer) -> None:
        super().__init__(api_config, config, tokenizer)
        self.sharegpt_dataset = iter(
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
        # initialize data collection
        next(self.sharegpt_dataset)

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Chat, APIType.Completion]

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        if self.sharegpt_dataset is not None:
            while True:
                data = next(self.sharegpt_dataset)
                if (
                    data is None
                    or data[self.data_key] is None
                    or len(data[self.data_key]) < self.min_num_turns
                    or len(data[self.data_key]) == 0
                ):
                    continue

                if self.api_config.type == APIType.Completion:
                    try:
                        prompt = data[self.data_key][0].get(self.content_key)
                        completion = data[self.data_key][1].get(self.content_key)
                        if not prompt:
                            continue
                        # Ensured by main.py logic and __init__ type hint for this class
                        assert self.tokenizer is not None
                        completion_tokens = self.tokenizer.count_tokens(completion)
                        prompt_tokens = self.tokenizer.count_tokens(prompt)
                    
                        if self.input_distribution:
                            if prompt_tokens < self.input_distribution.min or prompt_tokens > self.input_distribution.max:
                                continue
                        if self.output_distribution:
                            if completion_tokens < self.output_distribution.min or completion_tokens > self.output_distribution.max:
                                continue

                        yield CompletionAPIData(prompt=prompt, max_tokens=completion_tokens)
                    except (KeyError, TypeError) as e:
                        logger.warning(f"Skipping invalid completion data: {e}")
                        continue
                elif self.api_config.type == APIType.Chat:
                    yield ChatCompletionAPIData(
                        messages=[
                            ChatMessage(role=conversation[self.role_key], content=conversation[self.content_key])
                            for conversation in data[self.data_key]
                        ]
                    )
                else:
                    raise Exception("Unsupported API type")

    def is_io_distribution_supported(self) -> bool:
        return True

    def is_shared_prefix_supported(self) -> bool:
        return False
