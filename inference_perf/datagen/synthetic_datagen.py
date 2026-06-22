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
import os
import time
from pathlib import Path
from typing import Generator, List, Optional

from inference_perf.apis import CompletionAPIData, InferenceAPIData, LazyLoadInferenceAPIData
from inference_perf.config import APIConfig, APIType, DataConfig
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from inference_perf.utils.numeric.distribution import generate_distribution
from .base import DataGenerator, LazyLoadDataMixin
from .datagen_utils import converge_to_exact_length_text

logger = logging.getLogger(__name__)

# Heartbeat interval for synthetic datagen progress logs. Synthetic prompt
# materialization happens lazily in worker processes; on large runs with slow
# tokenizers this can take tens of minutes with no other observable signal,
# making the run look hung from outside.
_PROGRESS_LOG_INTERVAL_SEC = 10.0


class SyntheticDataGenerator(DataGenerator, LazyLoadDataMixin):
    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: Optional[CustomTokenizer]) -> None:
        super().__init__(api_config, config, tokenizer)

        if self.input_distribution is None or self.output_distribution is None or self.tokenizer is None:
            raise ValueError("IODistribution and tokenizer are required for SyntheticDataGenerator")

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
        if self.config and self.config.corpus_file_path:
            corpus_path = Path(self.config.corpus_file_path)
        else:
            corpus_path = Path(__file__).resolve().parents[1] / "assets" / "shakespeare.txt"

        if not corpus_path.is_file():
            raise FileNotFoundError(f"Prompt corpus file not found: {corpus_path}")

        corpus_text = corpus_path.read_text(encoding="utf-8")
        logger.info(f"Loaded prompt corpus from: {corpus_path} ({len(corpus_text)} chars)")

        base_prompt = "Pick as many lines as you can from these poem lines:\n"
        self.token_ids: list[int] = self.tokenizer.get_tokenizer().encode(base_prompt + corpus_text)

        # Per-process counters for the progress heartbeat emitted from
        # load_lazy_data. Each worker process tracks its own count and
        # log timestamp; aggregating across workers is not worth the
        # plumbing for a tactical visibility fix.
        self._materialized_count: int = 0
        self._last_progress_log_time: Optional[float] = None

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion]

    def is_io_distribution_supported(self) -> bool:
        return True

    def is_shared_prefix_supported(self) -> bool:
        return False

    def _generate_exact_length_text(self, target_len: int) -> str:
        """Generates a string from self.token_ids that tokenizes to exactly target_len."""
        if target_len <= 0:
            return ""

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for generating exact length prompts.")

        # Start with a slice of target_len
        current_slice_len = target_len

        initial_slice_len = min(target_len, len(self.token_ids))
        initial_tokens = self.token_ids[:initial_slice_len]

        def adjust_tokens(current_tokens: List[int], current_len: int, target_len: int) -> List[int]:
            nonlocal current_slice_len
            if current_len < target_len:
                diff = target_len - current_len
                current_slice_len += diff
            else:
                diff = current_len - target_len
                current_slice_len -= diff
                if current_slice_len <= 0:
                    current_slice_len = 1  # Keep at least one

            slice_to_use = min(current_slice_len, len(self.token_ids))
            return self.token_ids[:slice_to_use]

        text, _ = converge_to_exact_length_text(
            tokenizer=self.tokenizer,
            target_len=target_len,
            initial_tokens=initial_tokens,
            adjust_tokens_fn=adjust_tokens,
        )
        return text

    def load_lazy_data(self, data: LazyLoadInferenceAPIData) -> InferenceAPIData:
        n = data.data_index

        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for SyntheticDataGenerator")

        if self.api_config.type == APIType.Completion:
            length = self.input_lengths[n]
            prompt_text = self._generate_exact_length_text(length)
            self._log_progress()
            return CompletionAPIData(
                prompt=prompt_text,
                max_tokens=self.output_lengths[n],
            )
        else:
            raise Exception("Unsupported API type")

    def _log_progress(self) -> None:
        self._materialized_count += 1
        now = time.monotonic()
        if self._last_progress_log_time is None or (now - self._last_progress_log_time) >= _PROGRESS_LOG_INTERVAL_SEC:
            logger.info(
                "Synthetic datagen progress: materialized %d prompts (worker pid=%d)",
                self._materialized_count,
                os.getpid(),
            )
            self._last_progress_log_time = now

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        if self.tokenizer is None:
            raise ValueError("Tokenizer is required for SyntheticDataGenerator")
        if self.api_config.type != APIType.Completion:
            raise Exception("Unsupported API type")

        i = 0
        while True:
            yield LazyLoadInferenceAPIData(data_index=i)
            i += 1
