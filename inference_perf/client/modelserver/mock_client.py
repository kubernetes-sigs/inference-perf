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
from typing import List, Optional
from inference_perf.config import APIConfig, APIType
from inference_perf.apis import InferenceAPIData, InferenceInfo, RequestLifecycleMetric, ErrorResponseInfo
from .base import ModelServerClient, ModelServerPrometheusMetric, PrometheusMetricMetadata
import asyncio
import time
import logging

logger = logging.getLogger(__name__)


class MockModelServerClient(ModelServerClient):
    def __init__(
        self,
        metrics_collector: RequestDataCollector,
        api_config: APIConfig,
        timeout: Optional[float] = None,
        mock_latency: float = 1,
    ) -> None:
        super().__init__(api_config, timeout)
        self.metrics_collector = metrics_collector
        self.mock_latency = mock_latency
        self.tokenizer = None

    async def process_request(self, data: InferenceAPIData, stage_id: int, scheduled_time: float) -> None:
        start = time.perf_counter()
        logger.debug("Processing mock request for stage %d", stage_id)
        try:
            if self.timeout and self.timeout < self.mock_latency:
                await asyncio.sleep(self.timeout)
                raise asyncio.exceptions.TimeoutError()
            else:
                if self.mock_latency > 0:
                    await asyncio.sleep(self.mock_latency)
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
                        end_time=time.perf_counter(),
                        scheduled_time=scheduled_time,
                    )
                )
        except asyncio.exceptions.TimeoutError as e:
            logger.debug("Request timedout after %f seconds", self.timeout)
            self.metrics_collector.record_metric(
                RequestLifecycleMetric(
                    stage_id=stage_id,
                    request_data=str(data.to_payload("mock_model", 3, False, False)),
                    info=InferenceInfo(
                        input_tokens=0,
                        output_tokens=0,
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

    def get_prometheus_metric_metadata(self) -> PrometheusMetricMetadata:
        mock_prometheus_metric = ModelServerPrometheusMetric(
            name="mock_metric",
            op="mean",
            type="counter",
            filters=[],
        )
        return PrometheusMetricMetadata(
            # Throughput
            prompt_tokens_per_second=mock_prometheus_metric,
            output_tokens_per_second=mock_prometheus_metric,
            requests_per_second=mock_prometheus_metric,
            # Latency
            avg_request_latency=mock_prometheus_metric,
            median_request_latency=mock_prometheus_metric,
            p90_request_latency=mock_prometheus_metric,
            p99_request_latency=mock_prometheus_metric,
            # Request
            total_requests=mock_prometheus_metric,
            avg_prompt_tokens=mock_prometheus_metric,
            avg_output_tokens=mock_prometheus_metric,
            avg_queue_length=mock_prometheus_metric,
            # Others
            avg_time_to_first_token=None,
            median_time_to_first_token=None,
            p90_time_to_first_token=None,
            p99_time_to_first_token=None,
            avg_time_per_output_token=None,
            median_time_per_output_token=None,
            p90_time_per_output_token=None,
            p99_time_per_output_token=None,
            avg_inter_token_latency=None,
            median_inter_token_latency=None,
            p90_inter_token_latency=None,
            p99_inter_token_latency=None,
            avg_kv_cache_usage=None,
            median_kv_cache_usage=None,
            p90_kv_cache_usage=None,
            p99_kv_cache_usage=None,
            num_preemptions_total=None,
            num_requests_swapped=None,
            prefix_cache_hits=None,
            prefix_cache_queries=None,
        )
