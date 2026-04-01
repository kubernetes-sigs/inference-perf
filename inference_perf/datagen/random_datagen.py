# Copyright 2026 The Kubernetes Authors.
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
from pathlib import Path
from typing import Generator, List, Optional

import numpy as np

from inference_perf.apis import CompletionAPIData, InferenceAPIData, LazyLoadInferenceAPIData
from inference_perf.config import APIConfig, APIType, DataConfig, TraceFormat
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.utils.distribution import generate_distribution
from inference_perf.utils.trace_reader import AzurePublicDatasetReader
from .base import DataGenerator, LazyLoadDataMixin
from .datagen_utils import generate_random_exact_length_text, init_vocab_sampling, random_token_ids

logger = logging.getLogger(__name__)


# Random data generator generates random tokens from the model's
# vocabulary for the desired input and output distribution.
class RandomDataGenerator(DataGenerator, LazyLoadDataMixin):
    def __init__(
        self,
        api_config: APIConfig,
        config: DataConfig,
        tokenizer: Optional[CustomTokenizer],
    ) -> None:
        super().__init__(api_config, config, tokenizer)

        if self.trace is None:
            # let's read the trace file and get the input and output lengths
            if self.input_distribution is None or self.output_distribution is None:
                raise ValueError("Input and Output Distribution are required for RandomDataGenerator")

            if self.input_distribution.total_count is None or self.output_distribution.total_count is None:
                raise ValueError("IODistribution requires total_count to be set")

            self.input_lengths = generate_distribution(
                self.input_distribution.min,
                self.input_distribution.max,
                self.input_distribution.mean,
                self.input_distribution.std_dev,
                self.input_distribution.total_count,
            )
            self.output_lengths = generate_distribution(
                self.output_distribution.min,
                self.output_distribution.max,
                self.output_distribution.mean,
                self.output_distribution.std_dev,
                self.output_distribution.total_count,
            )
        else:
            # let's read the trace file and get the input and output lengths
            if self.trace.format == TraceFormat.AZURE_PUBLIC_DATASET:
                self.trace_reader = AzurePublicDatasetReader()
            else:
                raise ValueError(f"Unsupported trace format: {self.trace.format}")

            input_lengths_list: List[int] = []
            output_lengths_list: List[int] = []
            for _, input_tokens, output_tokens in self.trace_reader.load_traces(Path(self.trace.file)):
                input_lengths_list.append(input_tokens)
                output_lengths_list.append(output_tokens)

            self.input_lengths = np.array(input_lengths_list, dtype=np.int64)
            self.output_lengths = np.array(output_lengths_list, dtype=np.int64)

            logger.info(f"Ignoring input and output distributions configurations as trace file {self.trace.file} is provided")

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for RandomDataGenerator")

        self.vocab_size, self.special_token_ids, self.valid_token_ids = init_vocab_sampling(self.tokenizer)
        self.rng: np.random.Generator = np.random.default_rng()

    def _generate_random_token_ids(self, length: int) -> List[int]:
        """Generates a list of random token IDs of a specified length."""
        return random_token_ids(self.rng, self.valid_token_ids, length)

    def _generate_exact_length_text(self, target_len: int) -> str:
        """Generates a string that tokenizes to exactly target_len."""
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for generating exact length prompts.")
        return generate_random_exact_length_text(self.rng, self.valid_token_ids, self.tokenizer, target_len)

    def get_request_count(self) -> int:
        return min(len(self.input_lengths), len(self.output_lengths))

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion]

    def is_io_distribution_supported(self) -> bool:
        return True

    def is_shared_prefix_supported(self) -> bool:
        return False

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> InferenceAPIData:
        n = data.data_index

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for RandomDataGenerator")

        if self.api_config.type == APIType.Completion:
            length = self.input_lengths[n]
            text = self._generate_exact_length_text(length)
            return CompletionAPIData(prompt=text, max_tokens=self.output_lengths[n])
        else:
            raise Exception("Unsupported API type")

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        if self.api_config.type != APIType.Completion:
            raise Exception(f"Unsupported API type: {self.api_config}. RandomDataGenerator only supports Completion.")

        i = 0
        while True:
            yield LazyLoadInferenceAPIData(data_index=i)
            i += 1
