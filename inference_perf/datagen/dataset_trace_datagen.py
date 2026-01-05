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
from inference_perf.apis import InferenceAPIData, CompletionAPIData, ChatCompletionAPIData, ChatMessage, LazyLoadInferenceAPIData
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from .base import DataGenerator, LazyLoadDataMixin
from typing import Generator, List, Optional
from inference_perf.config import APIType, APIConfig, DataConfig, TraceFormat
from inference_perf.utils.trace_reader import DatasetTraceReader, DatasetTraceEntry
import logging

logger = logging.getLogger(__name__)


class DatasetTraceDataGenerator(DataGenerator, LazyLoadDataMixin):
    """
    Data generator that reads prompts from a JSONL file (DatasetTrace format).
    
    Each line in the JSONL file should contain:
    - text_input (required): The actual prompt text to send
    - output_length (optional): If provided, sets max_tokens; if omitted, uses server default
    
    Supports both Completion and Chat API types.
    """

    def __init__(
        self,
        api_config: APIConfig,
        config: DataConfig,
        tokenizer: Optional[CustomTokenizer] = None,
    ) -> None:
        super().__init__(api_config, config, tokenizer)

        if self.trace is None:
            raise ValueError("DatasetTraceDataGenerator requires a trace config to be provided")

        if self.trace.format != TraceFormat.DATASET_TRACE:
            raise ValueError(f"DatasetTraceDataGenerator requires trace format to be DatasetTrace, got {self.trace.format}")

        self.trace_reader = DatasetTraceReader()
        self.entries: List[DatasetTraceEntry] = self.trace_reader.load_entries(Path(self.trace.file))

        if len(self.entries) == 0:
            raise ValueError(f"No valid entries found in trace file: {self.trace.file}")

        logger.info(f"Loaded {len(self.entries)} prompts from {self.trace.file}")

    def get_request_count(self) -> int:
        return len(self.entries)

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def is_io_distribution_supported(self) -> bool:
        return False

    def is_shared_prefix_supported(self) -> bool:
        return False

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> InferenceAPIData:
        n = data.data_index % len(self.entries)
        entry = self.entries[n]

        if self.api_config.type == APIType.Completion:
            # max_tokens=0 means server default will be used (handled in to_payload)
            max_tokens = entry.output_length if entry.output_length is not None else 0
            return CompletionAPIData(prompt=entry.text_input, max_tokens=max_tokens)
        elif self.api_config.type == APIType.Chat:
            max_tokens = entry.output_length if entry.output_length is not None else 0
            return ChatCompletionAPIData(
                messages=[ChatMessage(role="user", content=entry.text_input)],
                max_tokens=max_tokens,
            )
        else:
            raise Exception(f"Unsupported API type: {self.api_config.type}")

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        i = 0
        while True:
            yield LazyLoadInferenceAPIData(data_index=i)
            i += 1


