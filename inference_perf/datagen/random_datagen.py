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
import numpy as np
from inference_perf.apis import InferenceAPIData, CompletionAPIData
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.utils.distribution import generate_distribution
from .base import DataGenerator, IODistribution
from typing import Generator, List
from inference_perf.config import APIType

# Random data generator generates random tokens from the model's
# vocabulary for the desired input and output distribution.
class RandomDataGenerator(DataGenerator):
    def __init__(
        self,
        apiType: APIType,
        ioDistribution: IODistribution,
        tokenizer: CustomTokenizer,
    ) -> None:
        super().__init__(apiType, ioDistribution, tokenizer)

        if self.ioDistribution is None:
            raise ValueError("IODistribution is required for RandomDataGenerator")

        self.input_lengths = generate_distribution(
            self.ioDistribution.input.min,
            self.ioDistribution.input.max,
            self.ioDistribution.input.mean,
            self.ioDistribution.input.std_dev,
            self.ioDistribution.input.total_count,
        )
        self.output_lengths = generate_distribution(
            self.ioDistribution.output.min,
            self.ioDistribution.output.max,
            self.ioDistribution.output.mean,
            self.ioDistribution.output.std_dev,
            self.ioDistribution.output.total_count,
        )

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

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion]

    def is_io_distribution_supported(self) -> bool:
        return True

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        i = 0

        while True:
            if self.tokenizer is None:
                raise ValueError("Tokenizer is required for RandomDataGenerator")

            if self.apiType == APIType.Completion:
                prompt_text: str
                if self.input_lengths[i] <= 0:
                    random_token_ids_list = []
                else:
                    random_token_ids = np.random.randint(0, self.vocab_size, size=self.input_lengths[i], dtype=np.int64)
                    random_token_ids_list = random_token_ids.tolist()
                prompt_text = self.tokenizer.get_tokenizer().decode(random_token_ids_list)

                yield CompletionAPIData(
                    prompt=prompt_text,
                    max_tokens=self.output_lengths[i],
                )
                i += 1
            else:
                raise Exception(f"Unsupported API type: {self.apiType}. RandomDataGenerator only supports Completion.")
