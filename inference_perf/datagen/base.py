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
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from pydantic import BaseModel
from inference_perf.config import APIType, Distribution
from abc import ABC, abstractmethod
from typing import Generator, Optional, List


class CompletionData(BaseModel):
    prompt: str


class ChatMessage(BaseModel):
    role: str
    content: str


class ChatCompletionData(BaseModel):
    messages: List[ChatMessage]


class InferenceData(BaseModel):
    type: APIType = APIType.Completion
    chat: Optional[ChatCompletionData] = None
    data: Optional[CompletionData] = None


class IODistribution(BaseModel):
    input: Distribution = Distribution()
    output: Distribution = Distribution()


class DataGenerator(ABC):
    """Abstract base class for data generators."""

    apiType: APIType
    ioDistribution: Optional[IODistribution]
    tokenizer: Optional[CustomTokenizer]

    """Abstract base class for data generators."""

    def __init__(
        self, apiType: APIType, ioDistribution: Optional[IODistribution], tokenizer: Optional[CustomTokenizer]
    ) -> None:
        if apiType not in self.get_supported_apis():
            raise Exception(f"Unsupported API type {apiType}")

        if ioDistribution is not None and not self.is_io_distribution_supported():
            raise Exception("IO distribution not supported for this data generator")

        if tokenizer is not None:
            self.tokenizer = tokenizer

        self.apiType = apiType
        self.ioDistribution = ioDistribution

    @abstractmethod
    def get_supported_apis(self) -> List[APIType]:
        raise NotImplementedError

    @abstractmethod
    def get_data(self) -> Generator[InferenceData, None, None]:
        raise NotImplementedError

    @abstractmethod
    def is_io_distribution_supported(self) -> bool:
        raise NotImplementedError
