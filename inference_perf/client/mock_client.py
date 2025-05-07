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
from .base import ClientRequestMetric, ModelServerClient, PromptData, SuccessfulResponseData
import asyncio


class MockModelServerClient(ModelServerClient):
    def __init__(self) -> None:
        super().__init__()

    async def process_request(self, promptData: PromptData, stage_id: int) -> None:
        print(
            "Processing mock request - "
            + str(promptData.to_payload(model_name="mock-model", max_tokens=0))
            + " for stage - "
            + str(stage_id)
        )
        await asyncio.sleep(3)
        ClientRequestMetric(
            stage_id=stage_id,
            request=promptData,
            response=SuccessfulResponseData(info={"res": "this is a mock response"}),
            start_time=1.23,
            end_time=3.21,
        )
