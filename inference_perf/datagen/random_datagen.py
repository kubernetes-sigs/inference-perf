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
from pathlib import Path
import numpy as np
from inference_perf.apis import InferenceAPIData, CompletionAPIData, LazyLoadInferenceAPIData
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.utils.distribution import generate_distribution
from .base import DataGenerator, LazyLoadDataMixin
from typing import Generator, List, Optional
from inference_perf.config import APIType, APIConfig, DataConfig, TraceFormat
from inference_perf.utils.trace_reader import AzurePublicDatasetReader
import logging

logger = logging.getLogger(__name__)


# Random data generator generates random tokens from the model's
# vocabulary for the desired input and output distribution.
class RandomDataGenerator(DataGenerator, LazyLoadDataMixin):
    def __init__(
        self,
        api_config: APIConfig,
        config: DataConfig,
        tokenizer: Optional[CustomTokenizer],
        seed: Optional[int] = None,
        **kwargs,
    ) -> None:
        super().__init__(api_config, config, tokenizer)
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        else:
            self.rng = np.random.default_rng()

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

        hf_tokenizer = self.tokenizer.get_tokenizer()
        if hasattr(hf_tokenizer, "vocab_size") and hf_tokenizer.vocab_size is not None:
            self.vocab_size: int = hf_tokenizer.vocab_size
        elif hasattr(hf_tokenizer, "get_vocab") and callable(hf_tokenizer.get_vocab):
            self.vocab_size = len(hf_tokenizer.get_vocab())
        else:
            try:
                self.vocab_size = len(hf_tokenizer)
            except TypeError as e:
                raise ValueError(
                    "Tokenizer does not have a 'vocab_size' attribute, 'get_vocab()' method, "
                    "or support len() for vocabulary size. Cannot use random token generation."
                ) from e
        if self.vocab_size <= 0:
            raise ValueError(f"Tokenizer vocabulary size must be positive, got {self.vocab_size}.")

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
            tokens = self.rng.integers(0, self.vocab_size, size=self.input_lengths[n], dtype=np.int64)
            prompt_text = self.tokenizer.get_tokenizer().decode(tokens.tolist())
            return CompletionAPIData(prompt=prompt_text, max_tokens=self.output_lengths[n])
        else:
            raise Exception("Unsupported API type")

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        if self.api_config.type != APIType.Completion:
            raise Exception(f"Unsupported API type: {self.api_config}. RandomDataGenerator only supports Completion.")

        i = 0
        while True:
            yield LazyLoadInferenceAPIData(data_index=i)
            i += 1
