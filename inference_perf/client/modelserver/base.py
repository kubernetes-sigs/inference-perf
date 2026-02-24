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
from typing import List, Optional, Tuple
from inference_perf.client.metricsclient.base import MetricsMetadata
from inference_perf.config import APIConfig, APIType
from inference_perf.apis import InferenceAPIData


class ModelServerPrometheusMetric:
    def __init__(self, name: str, op: str, type: str, filters: List[str]) -> None:
        self.name = name
        self.op = op
        self.type = type
        self.filters = ",".join(filters)


# PrometheusMetricMetadata stores the mapping of metrics to their model server names and types
# and the filters to be applied to them.
# This is used to generate Prometheus query for the metrics.
class PrometheusMetricMetadata(MetricsMetadata):
    # Throughput
    prompt_tokens_per_second: ModelServerPrometheusMetric
    output_tokens_per_second: ModelServerPrometheusMetric
    requests_per_second: ModelServerPrometheusMetric

    # Latency
    avg_request_latency: ModelServerPrometheusMetric
    median_request_latency: ModelServerPrometheusMetric
    p90_request_latency: ModelServerPrometheusMetric
    p99_request_latency: ModelServerPrometheusMetric
    avg_time_to_first_token: Optional[ModelServerPrometheusMetric]
    median_time_to_first_token: Optional[ModelServerPrometheusMetric]
    p90_time_to_first_token: Optional[ModelServerPrometheusMetric]
    p99_time_to_first_token: Optional[ModelServerPrometheusMetric]
    avg_time_per_output_token: Optional[ModelServerPrometheusMetric]
    median_time_per_output_token: Optional[ModelServerPrometheusMetric]
    p90_time_per_output_token: Optional[ModelServerPrometheusMetric]
    p99_time_per_output_token: Optional[ModelServerPrometheusMetric]
    avg_inter_token_latency: Optional[ModelServerPrometheusMetric]
    median_inter_token_latency: Optional[ModelServerPrometheusMetric]
    p90_inter_token_latency: Optional[ModelServerPrometheusMetric]
    p99_inter_token_latency: Optional[ModelServerPrometheusMetric]

    # Request
    total_requests: ModelServerPrometheusMetric
    avg_prompt_tokens: ModelServerPrometheusMetric
    avg_output_tokens: ModelServerPrometheusMetric
    avg_queue_length: ModelServerPrometheusMetric

    # Usage
    avg_kv_cache_usage: Optional[ModelServerPrometheusMetric]
    median_kv_cache_usage: Optional[ModelServerPrometheusMetric]
    p90_kv_cache_usage: Optional[ModelServerPrometheusMetric]
    p99_kv_cache_usage: Optional[ModelServerPrometheusMetric]
    num_preemptions_total: Optional[ModelServerPrometheusMetric]
    num_requests_swapped: Optional[ModelServerPrometheusMetric]

    # Prefix Cache
    prefix_cache_hits: Optional[ModelServerPrometheusMetric]
    prefix_cache_queries: Optional[ModelServerPrometheusMetric]

    # Running Requests
    avg_num_requests_running: Optional[ModelServerPrometheusMetric]

    # Request Lifecycle Latency Breakdown
    avg_request_queue_time: Optional[ModelServerPrometheusMetric]
    median_request_queue_time: Optional[ModelServerPrometheusMetric]
    p90_request_queue_time: Optional[ModelServerPrometheusMetric]
    p99_request_queue_time: Optional[ModelServerPrometheusMetric]
    avg_request_inference_time: Optional[ModelServerPrometheusMetric]
    median_request_inference_time: Optional[ModelServerPrometheusMetric]
    p90_request_inference_time: Optional[ModelServerPrometheusMetric]
    p99_request_inference_time: Optional[ModelServerPrometheusMetric]
    avg_request_prefill_time: Optional[ModelServerPrometheusMetric]
    median_request_prefill_time: Optional[ModelServerPrometheusMetric]
    p90_request_prefill_time: Optional[ModelServerPrometheusMetric]
    p99_request_prefill_time: Optional[ModelServerPrometheusMetric]
    avg_request_decode_time: Optional[ModelServerPrometheusMetric]
    median_request_decode_time: Optional[ModelServerPrometheusMetric]
    p90_request_decode_time: Optional[ModelServerPrometheusMetric]
    p99_request_decode_time: Optional[ModelServerPrometheusMetric]

    # Request Metadata
    avg_request_prompt_tokens: Optional[ModelServerPrometheusMetric]
    median_request_prompt_tokens: Optional[ModelServerPrometheusMetric]
    p90_request_prompt_tokens: Optional[ModelServerPrometheusMetric]
    p99_request_prompt_tokens: Optional[ModelServerPrometheusMetric]
    avg_request_generation_tokens: Optional[ModelServerPrometheusMetric]
    median_request_generation_tokens: Optional[ModelServerPrometheusMetric]
    p90_request_generation_tokens: Optional[ModelServerPrometheusMetric]
    p99_request_generation_tokens: Optional[ModelServerPrometheusMetric]
    avg_request_max_num_generation_tokens: Optional[ModelServerPrometheusMetric]
    median_request_max_num_generation_tokens: Optional[ModelServerPrometheusMetric]
    p90_request_max_num_generation_tokens: Optional[ModelServerPrometheusMetric]
    p99_request_max_num_generation_tokens: Optional[ModelServerPrometheusMetric]
    avg_request_params_n: Optional[ModelServerPrometheusMetric]
    median_request_params_n: Optional[ModelServerPrometheusMetric]
    p90_request_params_n: Optional[ModelServerPrometheusMetric]
    p99_request_params_n: Optional[ModelServerPrometheusMetric]
    avg_request_params_max_tokens: Optional[ModelServerPrometheusMetric]
    median_request_params_max_tokens: Optional[ModelServerPrometheusMetric]
    p90_request_params_max_tokens: Optional[ModelServerPrometheusMetric]
    p99_request_params_max_tokens: Optional[ModelServerPrometheusMetric]
    request_success_count: Optional[ModelServerPrometheusMetric]

    # Iteration Stats
    avg_iteration_tokens: Optional[ModelServerPrometheusMetric]
    median_iteration_tokens: Optional[ModelServerPrometheusMetric]
    p90_iteration_tokens: Optional[ModelServerPrometheusMetric]
    p99_iteration_tokens: Optional[ModelServerPrometheusMetric]

    # Token Cache Stats
    prompt_tokens_cached: Optional[ModelServerPrometheusMetric]
    prompt_tokens_recomputed: Optional[ModelServerPrometheusMetric]
    external_prefix_cache_hits: Optional[ModelServerPrometheusMetric]
    external_prefix_cache_queries: Optional[ModelServerPrometheusMetric]
    mm_cache_hits: Optional[ModelServerPrometheusMetric]
    mm_cache_queries: Optional[ModelServerPrometheusMetric]
    corrupted_requests: Optional[ModelServerPrometheusMetric]

    # KV Block Metrics
    avg_request_prefill_kv_computed_tokens: Optional[ModelServerPrometheusMetric]
    median_request_prefill_kv_computed_tokens: Optional[ModelServerPrometheusMetric]
    p90_request_prefill_kv_computed_tokens: Optional[ModelServerPrometheusMetric]
    p99_request_prefill_kv_computed_tokens: Optional[ModelServerPrometheusMetric]
    avg_kv_block_idle_before_evict: Optional[ModelServerPrometheusMetric]
    median_kv_block_idle_before_evict: Optional[ModelServerPrometheusMetric]
    p90_kv_block_idle_before_evict: Optional[ModelServerPrometheusMetric]
    p99_kv_block_idle_before_evict: Optional[ModelServerPrometheusMetric]
    avg_kv_block_lifetime: Optional[ModelServerPrometheusMetric]
    median_kv_block_lifetime: Optional[ModelServerPrometheusMetric]
    p90_kv_block_lifetime: Optional[ModelServerPrometheusMetric]
    p99_kv_block_lifetime: Optional[ModelServerPrometheusMetric]
    avg_kv_block_reuse_gap: Optional[ModelServerPrometheusMetric]
    median_kv_block_reuse_gap: Optional[ModelServerPrometheusMetric]
    p90_kv_block_reuse_gap: Optional[ModelServerPrometheusMetric]
    p99_kv_block_reuse_gap: Optional[ModelServerPrometheusMetric]


class ModelServerClient(ABC):
    @abstractmethod
    def __init__(self, api_config: APIConfig, timeout: Optional[float] = None, *args: Tuple[int, ...]) -> None:
        if api_config.type not in self.get_supported_apis():
            raise Exception(f"Unsupported API type {api_config}")

        self.api_config = api_config
        self.timeout = timeout

    def new_session(self) -> "ModelServerClientSession":
        return ModelServerClientSession(self)

    @abstractmethod
    def get_supported_apis(self) -> List[APIType]:
        raise NotImplementedError

    @abstractmethod
    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_prometheus_metric_metadata(self) -> PrometheusMetricMetadata:
        # assumption: all metrics clients have metrics exported in Prometheus format
        raise NotImplementedError


class ModelServerClientSession:
    def __init__(self, client: ModelServerClient):
        self.client = client

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        await self.client.process_request(data, stage_id, scheduled_time, lora_adapter)

    async def close(self) -> None:  # noqa - subclasses optionally override this
        pass
