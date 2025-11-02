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
from typing import Generator, List, Optional
from datasets import load_dataset
import os

logger = logging.getLogger(__name__)


class InfinityInstructDataGenerator(DataGenerator):
    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: Optional[CustomTokenizer]) -> None:
        super().__init__(api_config, config, tokenizer)

        if config.path is not None:
            # check if the path is valid
            if not os.path.exists(config.path):
                raise ValueError(f"Invalid dataset path: {config.path}. Path does not exist.")
            # depending on whether the dataset is a single file or a directory, we need to load it differently
            # TODO: add support for other file types
            if os.path.isfile(config.path) and config.path.endswith(".json"):
                self.infinity_instruct_dataset = iter(
                    load_dataset("json", data_files=config.path, streaming=True, split="train")
                )
            elif os.path.isdir(config.path):
                json_files = [f for f in os.listdir(config.path) if f.endswith(".json")]
                self.infinity_instruct_dataset = iter(
                    load_dataset("json", data_files=json_files, streaming=True, split="train")
                )
            else:
                raise ValueError(f"Invalid dataset path: {config.path}")
        else:
            raise ValueError("path is not provided in the config")

        self.conversations_key = "conversations"
        # initialize data collection
        next(self.infinity_instruct_dataset)

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        if self.infinity_instruct_dataset is not None:
            while True:
                data = next(self.infinity_instruct_dataset)
                if data is None or self.conversations_key not in data or data[self.conversations_key] is None:
                    continue

                conversations = data[self.conversations_key]
                if not conversations:
                    continue

                if self.api_config.type == APIType.Completion:
                    try:
                        # The last message is the completion
                        completion_message = conversations[-1]
                        if completion_message.get("from") != "gpt":
                            continue
                        completion = completion_message.get("value")

                        # The rest of the messages are the prompt
                        prompt_messages = conversations[:-1]
                        prompt = "\n".join([msg.get("value", "") for msg in prompt_messages])

                        if not prompt or not completion:
                            continue

                        assert self.tokenizer is not None
                        completion_tokens = self.tokenizer.count_tokens(completion)
                        prompt_tokens = self.tokenizer.count_tokens(prompt)

                        if self.input_distribution:
                            if prompt_tokens < self.input_distribution.min or prompt_tokens > self.input_distribution.max:
                                continue
                        if self.output_distribution:
                            if (
                                completion_tokens < self.output_distribution.min
                                or completion_tokens > self.output_distribution.max
                            ):
                                continue

                        yield CompletionAPIData(prompt=prompt, max_tokens=completion_tokens)
                    except (KeyError, TypeError) as e:
                        logger.warning(f"Skipping invalid completion data: {e}")
                        continue
                elif self.api_config.type == APIType.Chat:
                    try:
                        messages: List[ChatMessage] = []
                        for conv in conversations:
                            role = "user" if conv.get("from") == "human" else "assistant"
                            content = conv.get("value")
                            if not content:
                                continue
                            messages.append(ChatMessage(role=role, content=content))

                        if not messages:
                            continue

                        yield ChatCompletionAPIData(messages=messages)
                    except (KeyError, TypeError) as e:
                        logger.warning(f"Skipping invalid chat data: {e}")
                        continue
                else:
                    raise Exception("Unsupported API type")

    def is_io_distribution_supported(self) -> bool:
        return True

    def is_shared_prefix_supported(self) -> bool:
        return False
