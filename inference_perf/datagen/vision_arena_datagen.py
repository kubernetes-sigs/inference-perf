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
from typing import Generator, List, Optional

from inference_perf.apis import InferenceAPIData
from inference_perf.config import APIConfig, APIType, DataConfig
from inference_perf.utils.custom_tokenizer import CustomTokenizer

from .base import DataGenerator

logger = logging.getLogger(__name__)

# Hugging Face dataset reference for follow-up implementation.
# https://huggingface.co/datasets/lmarena-ai/VisionArena-Chat
VISION_ARENA_HF_DATASET_URL = "lmarena-ai/VisionArena-Chat"


class VisionArenaDataGenerator(DataGenerator):
    """Scaffolding for the VisionArena dataset (image + text VQA prompts).

    Streams real user-submitted multimodal prompts from VisionArena. Implementation
    is deferred to a follow-up PR; this stub registers the data generator so the
    config surface and dispatch path are in place.
    """

    def __init__(self, api_config: APIConfig, config: DataConfig, tokenizer: Optional[CustomTokenizer]) -> None:
        super().__init__(api_config, config, tokenizer)
        self.dataset_path = config.path

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Chat]

    def get_data(self) -> Generator[InferenceAPIData, None, None]:
        raise NotImplementedError(
            "VisionArena data generator is not yet implemented. Tracked as follow-up to the multimodal benchmarking PR."
        )
        yield  # pragma: no cover - keeps generator type

    def is_io_distribution_supported(self) -> bool:
        return False

    def is_shared_prefix_supported(self) -> bool:
        return False
