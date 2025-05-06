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
from .base import DataGenerator, PromptData, CompletionData
from typing import Generator, List
from inference_perf.config import APIType


class MockDataGenerator(DataGenerator):
    def __init__(self, apiType: APIType) -> None:
        super().__init__(apiType)
        pass

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion]

    def get_data(self) -> Generator[PromptData, None, None]:
        i = 0
        while True:
            i += 1
            if self.apiType == APIType.Completion:
                yield CompletionData(prompt="text" + str(i))
            else:
                raise Exception("Unsupported API type")
