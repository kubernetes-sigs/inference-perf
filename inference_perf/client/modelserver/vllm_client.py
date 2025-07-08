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
from inference_perf.config import APIConfig, APIType
from inference_perf.apis import InferenceAPIData, InferenceInfo, RequestLifecycleMetric, ErrorResponseInfo
from inference_perf.utils import CustomTokenizer
from .base import ModelServerClient, PrometheusMetricMetadata, ModelServerPrometheusMetric
from typing import List, Optional
import aiohttp
import json
import time


class vLLMModelServerClient(ModelServerClient):
    def __init__(
        self,
        metrics_collector: RequestDataCollector,
        api_config: APIConfig,
        uri: str,
        model_name: str,
        tokenizer: CustomTokenizer,
        max_tcp_connections: int,
        additional_filters: List[str],
        ignore_eos: bool = True,
        api_key: Optional[str] = None,
    ) -> None:
        super().__init__(api_config)
        self.model_name = model_name
        self.uri = uri
        self.max_completion_tokens = 30  # default to use when not set at the request level
        self.ignore_eos = ignore_eos
        self.tokenizer = tokenizer
        self.metrics_collector = metrics_collector
        self.max_tcp_connections = max_tcp_connections
        self.additional_filters = additional_filters
        self.api_key = api_key

        filters = [f"model_name='{self.model_name}'", *self.additional_filters]
        self.prometheus_metric_metadata = PrometheusMetricMetadata(
            avg_queue_length=ModelServerPrometheusMetric("vllm:num_requests_waiting", "mean", "gauge", filters),
            avg_time_to_first_token=ModelServerPrometheusMetric(
                "vllm:time_to_first_token_seconds",
                "mean",
                "histogram",
                filters,
            ),
            median_time_to_first_token=ModelServerPrometheusMetric(
                "vllm:time_to_first_token_seconds",
                "median",
                "histogram",
                filters,
            ),
            p90_time_to_first_token=ModelServerPrometheusMetric(
                "vllm:time_to_first_token_seconds",
                "p90",
                "histogram",
                filters,
            ),
            p99_time_to_first_token=ModelServerPrometheusMetric(
                "vllm:time_to_first_token_seconds",
                "p99",
                "histogram",
                filters,
            ),
            avg_time_per_output_token=ModelServerPrometheusMetric(
                "vllm:time_per_output_token_seconds",
                "mean",
                "histogram",
                filters,
            ),
            median_time_per_output_token=ModelServerPrometheusMetric(
                "vllm:time_per_output_token_seconds",
                "median",
                "histogram",
                filters,
            ),
            p90_time_per_output_token=ModelServerPrometheusMetric(
                "vllm:time_per_output_token_seconds",
                "p90",
                "histogram",
                filters,
            ),
            p99_time_per_output_token=ModelServerPrometheusMetric(
                "vllm:time_per_output_token_seconds",
                "p99",
                "histogram",
                filters,
            ),
            avg_prompt_tokens=ModelServerPrometheusMetric("vllm:prompt_tokens_total", "mean", "counter", filters),
            prompt_tokens_per_second=ModelServerPrometheusMetric("vllm:prompt_tokens_total", "rate", "counter", filters),
            avg_output_tokens=ModelServerPrometheusMetric("vllm:generation_tokens_total", "mean", "counter", filters),
            output_tokens_per_second=ModelServerPrometheusMetric("vllm:generation_tokens_total", "rate", "counter", filters),
            total_requests=ModelServerPrometheusMetric(
                "vllm:e2e_request_latency_seconds_count",
                "increase",
                "counter",
                filters,
            ),
            requests_per_second=ModelServerPrometheusMetric(
                "vllm:e2e_request_latency_seconds_count",
                "rate",
                "counter",
                filters,
            ),
            avg_request_latency=ModelServerPrometheusMetric(
                "vllm:e2e_request_latency_seconds",
                "mean",
                "histogram",
                filters,
            ),
            median_request_latency=ModelServerPrometheusMetric(
                "vllm:e2e_request_latency_seconds",
                "median",
                "histogram",
                filters,
            ),
            p90_request_latency=ModelServerPrometheusMetric(
                "vllm:e2e_request_latency_seconds",
                "p90",
                "histogram",
                filters,
            ),
            p99_request_latency=ModelServerPrometheusMetric(
                "vllm:e2e_request_latency_seconds",
                "p99",
                "histogram",
                filters,
            ),
        )

    async def process_request(self, data: InferenceAPIData, stage_id: int, scheduled_time: float) -> None:
        payload = data.to_payload(
            model_name=self.model_name,
            max_tokens=self.max_completion_tokens,
            ignore_eos=self.ignore_eos,
            streaming=self.api_config.streaming,
        )
        headers = {"Content-Type": "application/json"}

        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"

        request_data = json.dumps(payload)

        async with aiohttp.ClientSession(connector=aiohttp.TCPConnector(limit=self.max_tcp_connections)) as session:
            start = time.perf_counter()
            try:
                async with session.post(self.uri + data.get_route(), headers=headers, data=request_data) as response:
                    response_info = await data.process_response(
                        response=response, config=self.api_config, tokenizer=self.tokenizer
                    )
                    response_content = await response.text()
                    if response.status == 200:
                        self.metrics_collector.record_metric(
                            RequestLifecycleMetric(
                                stage_id=stage_id,
                                request_data=request_data,
                                response_data=response_content,
                                info=response_info,
                                error=None,
                                start_time=start,
                                end_time=time.perf_counter(),
                                scheduled_time=scheduled_time,
                            )
                        )
                    else:
                        self.metrics_collector.record_metric(
                            RequestLifecycleMetric(
                                stage_id=stage_id,
                                request_data=request_data,
                                response_data=response_content,
                                info=response_info,
                                error=ErrorResponseInfo(error_msg=response_content, error_type="Error response"),
                                start_time=start,
                                end_time=time.perf_counter(),
                                scheduled_time=scheduled_time,
                            )
                        )
            except Exception as e:
                self.metrics_collector.record_metric(
                    RequestLifecycleMetric(
                        stage_id=stage_id,
                        request_data=request_data,
                        response_data=response_content if "response_content" in locals() else "",
                        info=response_info if "response_info" in locals() else InferenceInfo(),
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
        return self.prometheus_metric_metadata
