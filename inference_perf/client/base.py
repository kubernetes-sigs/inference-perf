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
from abc import ABC, abstractmethod
from typing import Tuple

from inference_perf.datagen import LlmPrompt
from inference_perf.datagen.base import PromptMetricsCollector


class ModelServerClient(ABC):
    @abstractmethod
    def __init__(self, *args: Tuple[int, ...]) -> None:
        self.prompt_metrics_collector = PromptMetricsCollector()
        pass

    @abstractmethod
    async def handle_prompt(self, data: LlmPrompt, stage_id: int) -> None:
        raise NotImplementedError
