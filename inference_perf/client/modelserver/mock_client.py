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

from inference_perf.client.requestdatacollector import RequestDataCollector
from typing import List
from inference_perf.config import APIConfig, APIType
from inference_perf.apis import InferenceAPIData, InferenceInfo, RequestLifecycleMetric
from .base import ModelServerClient
import asyncio
import time
import logging

logger = logging.getLogger(__name__)


class MockModelServerClient(ModelServerClient):
    def __init__(self, metrics_collector: RequestDataCollector, api_config: APIConfig) -> None:
        super().__init__(api_config)
        self.metrics_collector = metrics_collector

    async def process_request(self, data: InferenceAPIData, stage_id: int, scheduled_time: float) -> None:
        start = time.monotonic()
        logger.debug("Processing mock request for stage %d", stage_id)
        await asyncio.sleep(3)
        self.metrics_collector.record_metric(
            RequestLifecycleMetric(
                stage_id=stage_id,
                request_data=str(data.to_payload("mock_model", 3, False, False)),
                info=InferenceInfo(
                    input_tokens=0,
                    output_tokens=0,
                ),
                error=None,
                start_time=start,
                end_time=time.monotonic(),
                scheduled_time=scheduled_time,
            )
        )

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]
