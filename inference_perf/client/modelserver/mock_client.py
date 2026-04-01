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

from inference_perf.metrics.request_collector import RequestMetricCollector
from typing import List, Optional
from inference_perf.config import APIConfig, APIType
from inference_perf.apis import (
    InferenceAPIData,
    InferenceInfo,
    RequestLifecycleMetric,
    ErrorResponseInfo,
    UnaryResponseMetrics,
)
from inference_perf.payloads import RequestMetrics, Text
from .base import ModelServerClient, BaseMetrics
import asyncio
import time
import logging

logger = logging.getLogger(__name__)


class MockModelServerClient(ModelServerClient):
    def __init__(
        self,
        metrics_collector: RequestMetricCollector,
        api_config: APIConfig,
        timeout: Optional[float] = None,
        mock_latency: float = 1,
    ) -> None:
        super().__init__(api_config, timeout)
        self.metrics_collector = metrics_collector
        self.mock_latency = mock_latency
        self.tokenizer = None

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        start = time.perf_counter()
        logger.debug("Processing mock request for stage %d", stage_id)
        effective_model_name = lora_adapter if lora_adapter else "mock_model"
        try:
            if self.timeout and self.timeout < self.mock_latency:
                await asyncio.sleep(self.timeout)
                raise asyncio.exceptions.TimeoutError()
            else:
                if self.mock_latency > 0:
                    await asyncio.sleep(self.mock_latency)

                info = InferenceInfo(
                    request_metrics=RequestMetrics(text=Text(input_tokens=0)),
                    lora_adapter=lora_adapter,
                )
                data.on_completion(info)
                self.metrics_collector.record_metric(
                    RequestLifecycleMetric(
                        stage_id=stage_id,
                        request_data=str(await data.to_request_body(effective_model_name, 3, False, False)),
                        info=InferenceInfo(
                            request_metrics=RequestMetrics(text=Text(input_tokens=0)),
                            response_metrics=UnaryResponseMetrics(output_tokens=0),
                            lora_adapter=lora_adapter,
                        ),
                        error=None,
                        start_time=start,
                        end_time=time.perf_counter(),
                        scheduled_time=scheduled_time,
                    )
                )
        except asyncio.exceptions.TimeoutError as e:
            logger.debug("Request timedout after %f seconds", self.timeout)
            self.metrics_collector.record_metric(
                RequestLifecycleMetric(
                    stage_id=stage_id,
                    request_data=str(data.to_request_body(effective_model_name, 3, False, False)),
                    info=InferenceInfo(
                        request_metrics=RequestMetrics(text=Text(input_tokens=0)),
                        response_metrics=UnaryResponseMetrics(output_tokens=0),
                        lora_adapter=lora_adapter,
                    ),
                    error=ErrorResponseInfo(
                        error_msg=str(e),
                        error_type=type(e).__name__,
                    ),
                    start_time=start,
                    end_time=time.perf_counter(),
                    scheduled_time=scheduled_time,
                )
            )

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def get_prometheus_metric_metadata(self) -> BaseMetrics:
        return BaseMetrics()
