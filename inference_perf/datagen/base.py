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
from inference_perf.apis import InferenceAPIData
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.config import APIConfig, APIType, DataConfig, Distribution, SharedPrefix
from abc import ABC, abstractmethod
from typing import Generator, Optional, List


class DataGenerator(ABC):
    """Abstract base class for data generators."""

    api_config: APIConfig
    input_distribution: Optional[Distribution]
    output_distribution: Optional[Distribution]
    shared_prefix: Optional[SharedPrefix]
    tokenizer: Optional[CustomTokenizer]

    """Abstract base class for data generators."""

    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: Optional[CustomTokenizer]) -> None:
        if api_config.type not in self.get_supported_apis():
            raise Exception(f"Unsupported API type {api_config}")

        if (
            config.input_distribution is not None or config.output_distribution is not None
        ) and not self.is_io_distribution_supported():
            raise Exception("IO distribution not supported for this data generator")

        if config.shared_prefix is not None and not self.is_shared_prefix_supported():
            raise Exception("Shared prefix not supported for this data generator")

        if tokenizer is not None:
            self.tokenizer = tokenizer

        self.api_config = api_config
        self.input_distribution = config.input_distribution
        self.output_distribution = config.output_distribution
        self.shared_prefix = config.shared_prefix

    @abstractmethod
    def get_supported_apis(self) -> List[APIType]:
        raise NotImplementedError

    @abstractmethod
    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        raise NotImplementedError

    @abstractmethod
    def is_io_distribution_supported(self) -> bool:
        raise NotImplementedError

    @abstractmethod
    def is_shared_prefix_supported(self) -> bool:
        raise NotImplementedError
