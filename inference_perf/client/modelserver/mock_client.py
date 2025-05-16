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

import asyncio
from typing import List
from inference_perf.config import APIType
from inference_perf.prompts.base import LlmPrompt
from .base import ModelServerClient, RequestMetric


class MockModelServerClient(ModelServerClient):
    def __init__(self, api_type: APIType) -> None:
        super().__init__(api_type)
        self.request_metrics: List[RequestMetric] = list()

    async def process_request(self, payload: LlmPrompt, stage_id: int) -> None:
        print("Processing mock request for stage - " + str(stage_id))
        await asyncio.sleep(3)
        self.request_metrics.append(
            RequestMetric(
                stage_id=stage_id,
                prompt_tokens=0,
                output_tokens=0,
                time_per_request=3,
            )
        )

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def get_request_metrics(self) -> List[RequestMetric]:
        return self.request_metrics
