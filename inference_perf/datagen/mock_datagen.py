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
from typing import Generator, List, Optional
from inference_perf.config import APIConfig, APIType, DataConfig
from inference_perf.datagen.base import DataGenerator
from inference_perf.apis import InferenceAPIData, CompletionAPIData, ChatCompletionAPIData, ChatMessage
from inference_perf.utils.custom_tokenizer import CustomTokenizer


class MockDataGenerator(DataGenerator):
    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: Optional[CustomTokenizer]) -> None:
        super().__init__(api_config, config, tokenizer)

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        i = 0
        while True:
            i += 1
            if self.api_config.type == APIType.Completion:
                yield CompletionAPIData(prompt="text" + str(i))
            elif self.api_config.type == APIType.Chat:
                yield ChatCompletionAPIData(messages=[ChatMessage(role="user", content="text" + str(i))])
            else:
                raise Exception("Unsupported API type")

    def is_io_distribution_supported(self) -> bool:
        return False

    def is_shared_prefix_supported(self) -> bool:
        return False
