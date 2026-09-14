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
import itertools
import logging
from inference_perf.apis import InferenceAPIData, CompletionAPIData
from inference_perf.utils.custom_tokenizer import CustomTokenizer
from ..base import DataGenerator
from inference_perf.config import APIConfig, APIType, DataConfig
from typing import Any, Dict, Generator, Iterator, List, Optional
from datasets import load_dataset
import os

logger = logging.getLogger(__name__)


class CNNDailyMailDataGenerator(DataGenerator):
    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: Optional[CustomTokenizer]) -> None:
        super().__init__(api_config, config, tokenizer)
        if tokenizer is None:
            raise ValueError("CustomTokenizer instance cannot be None")

        self.article_key = "article"
        self.highlights_key = "highlights"
        self.cnn_dailymail_dataset = self._load_dataset()
        # initialize data collection
        next(self.cnn_dailymail_dataset)
        self._dataset_ready = True

    def _load_dataset(self) -> Iterator[Any]:
        config = self.config
        if config.path is not None:
            # check if the path is valid
            if not os.path.exists(config.path):
                raise ValueError(f"Invalid dataset path: {config.path}. Path does not exist.")
            # depending on whether the dataset is a single file or a directory, we need to load it differently
            # TODO: add support for other file types
            if os.path.isfile(config.path) and config.path.endswith(".json"):
                return itertools.cycle(load_dataset("json", data_files=config.path, streaming=True, split="train"))
            elif os.path.isdir(config.path):
                json_files = [f for f in os.listdir(config.path) if f.endswith(".json")]
                return itertools.cycle(load_dataset("json", data_files=json_files, streaming=True, split="train"))
            else:
                raise ValueError(f"Invalid dataset path: {config.path}")
        else:
            return itertools.cycle(
                load_dataset(
                    "abisee/cnn_dailymail",
                    "3.0.0",
                    streaming=True,
                    split="train",
                )
            )

    def __getstate__(self) -> Dict[str, Any]:
        # itertools.cycle wraps the streaming dataset's iterator, which holds an
        # unpicklable generator internally, so a spawn/forkserver worker fails to
        # unpickle the whole generator (#589). Drop it here; __setstate__ leaves
        # it unloaded, and _ensure_dataset_loaded() rebuilds it lazily the first
        # time get_data() actually needs it in this process.
        state = self.__dict__.copy()
        del state["cnn_dailymail_dataset"]
        state["_dataset_ready"] = False
        return state

    def __setstate__(self, state: Dict[str, Any]) -> None:
        self.__dict__.update(state)

    def _ensure_dataset_loaded(self) -> None:
        # A Worker process only reaches its own copy of this generator through
        # LazyLoadDataMixin.get_request(), which is a no-op for a generator
        # (like this one) that doesn't implement that mixin: the parent process
        # is the one that calls get_data() and puts the materialized requests on
        # the queue. So a worker's copy never needs the dataset at all, and
        # rebuilding it eagerly on every worker spawn (as __setstate__ used to)
        # cost every worker a redundant dataset open, network round trip
        # included for the default Hub dataset, for nothing.
        if not self._dataset_ready:
            self.cnn_dailymail_dataset = self._load_dataset()
            next(self.cnn_dailymail_dataset)
            self._dataset_ready = True

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion]

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        self._ensure_dataset_loaded()

        assert self.tokenizer is not None

        if self.api_config.type != APIType.Completion:
            raise Exception("Unsupported API type")

        while True:
            data = next(self.cnn_dailymail_dataset)
            if data is None or data[self.article_key] is None or data[self.highlights_key] is None:
                continue

            try:
                prompt = data[self.article_key]
                if not prompt:
                    continue

                completion = data[self.highlights_key]
                completion_tokens = self.tokenizer.count_tokens(completion)
                prompt_ids = self.tokenizer.get_tokenizer().encode(prompt)
                prompt_tokens = len(prompt_ids)

                if self.input_distribution:
                    if prompt_tokens < self.input_distribution.min:
                        continue
                    if prompt_tokens > self.input_distribution.max:
                        continue
                if self.output_distribution:
                    if completion_tokens < self.output_distribution.min:
                        continue
                    if completion_tokens > self.output_distribution.max:
                        continue

                yield CompletionAPIData(prompt=prompt, max_tokens=completion_tokens)

            except (KeyError, TypeError) as e:
                logger.warning(f"Skipping invalid completion data: {e}")
                continue

    def is_io_distribution_supported(self) -> bool:
        return True

    def is_shared_prefix_supported(self) -> bool:
        return False
