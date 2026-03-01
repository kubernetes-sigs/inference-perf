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

from inference_perf.client.modelserver.openai_client import openAIModelServerClient
from inference_perf.client.requestdatacollector import RequestDataCollector
from inference_perf.config import APIConfig, APIType, CustomTokenizerConfig, MultiLoRAConfig
from .base import PrometheusMetricMetadata, ModelServerPrometheusMetric
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class TGImodelServerClient(openAIModelServerClient):
    def __init__(
        self,
        metrics_collector: RequestDataCollector,
        api_config: APIConfig,
        uri: str,
        model_name: Optional[str],
        tokenizer_config: Optional[CustomTokenizerConfig],
        max_tcp_connections: int,
        additional_filters: List[str],
        ignore_eos: bool = True,
        api_key: Optional[str] = None,
        timeout: Optional[float] = None,
        lora_config: Optional[List[MultiLoRAConfig]] = None,
    ) -> None:
        super().__init__(
            metrics_collector,
            api_config,
            uri,
            model_name,
            tokenizer_config,
            max_tcp_connections,
            additional_filters,
            ignore_eos,
            api_key,
            timeout,
            lora_config=lora_config,
        )
        self.metric_filters = additional_filters

    def get_prometheus_metric_metadata(self) -> PrometheusMetricMetadata:
        return PrometheusMetricMetadata(
            avg_queue_length=ModelServerPrometheusMetric(
                "tgi_queue_size",
                "mean",
                "gauge",
                self.metric_filters,
            ),
            avg_time_per_output_token=ModelServerPrometheusMetric(
                "tgi_request_mean_time_per_token_duration",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_time_per_output_token=ModelServerPrometheusMetric(
                "tgi_request_mean_time_per_token_duration",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_time_per_output_token=ModelServerPrometheusMetric(
                "tgi_request_mean_time_per_token_duration",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_time_per_output_token=ModelServerPrometheusMetric(
                "tgi_request_mean_time_per_token_duration",
                "p99",
                "histogram",
                self.metric_filters,
            ),
            avg_prompt_tokens=ModelServerPrometheusMetric(
                "tgi_request_input_length", "mean", "histogram", self.metric_filters
            ),
            prompt_tokens_per_second=ModelServerPrometheusMetric(
                "tgi_request_input_length", "rate", "histogram", self.metric_filters
            ),
            avg_output_tokens=ModelServerPrometheusMetric(
                "tgi_request_generated_tokens", "mean", "histogram", self.metric_filters
            ),
            output_tokens_per_second=ModelServerPrometheusMetric(
                "tgi_request_generated_tokens", "rate", "histogram", self.metric_filters
            ),
            total_requests=ModelServerPrometheusMetric(
                "tgi_request_success",
                "increase",
                "counter",
                self.metric_filters,
            ),
            requests_per_second=ModelServerPrometheusMetric(
                "tgi_request_success",
                "rate",
                "counter",
                self.metric_filters,
            ),
            avg_request_latency=ModelServerPrometheusMetric(
                "tgi_request_duration",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_request_latency=ModelServerPrometheusMetric(
                "tgi_request_duration",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_request_latency=ModelServerPrometheusMetric(
                "tgi_request_duration",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_request_latency=ModelServerPrometheusMetric(
                "tgi_request_duration",
                "p99",
                "histogram",
                self.metric_filters,
            ),
            avg_time_to_first_token=None,
            median_time_to_first_token=None,
            p90_time_to_first_token=None,
            p99_time_to_first_token=None,
            avg_kv_cache_usage=None,
            median_kv_cache_usage=None,
            p90_kv_cache_usage=None,
            p99_kv_cache_usage=None,
            avg_inter_token_latency=None,
            median_inter_token_latency=None,
            p90_inter_token_latency=None,
            p99_inter_token_latency=None,
            num_preemptions_total=None,
            num_requests_swapped=None,
            prefix_cache_hits=None,
            prefix_cache_queries=None,
            avg_num_requests_running=None,
            avg_request_queue_time=None,
            median_request_queue_time=None,
            p90_request_queue_time=None,
            p99_request_queue_time=None,
            avg_request_inference_time=None,
            median_request_inference_time=None,
            p90_request_inference_time=None,
            p99_request_inference_time=None,
            avg_request_prefill_time=None,
            median_request_prefill_time=None,
            p90_request_prefill_time=None,
            p99_request_prefill_time=None,
            avg_request_decode_time=None,
            median_request_decode_time=None,
            p90_request_decode_time=None,
            p99_request_decode_time=None,
            avg_request_prompt_tokens=None,
            median_request_prompt_tokens=None,
            p90_request_prompt_tokens=None,
            p99_request_prompt_tokens=None,
            avg_request_generation_tokens=None,
            median_request_generation_tokens=None,
            p90_request_generation_tokens=None,
            p99_request_generation_tokens=None,
            avg_request_max_num_generation_tokens=None,
            median_request_max_num_generation_tokens=None,
            p90_request_max_num_generation_tokens=None,
            p99_request_max_num_generation_tokens=None,
            avg_request_params_n=None,
            median_request_params_n=None,
            p90_request_params_n=None,
            p99_request_params_n=None,
            avg_request_params_max_tokens=None,
            median_request_params_max_tokens=None,
            p90_request_params_max_tokens=None,
            p99_request_params_max_tokens=None,
            request_success_count=None,
            avg_iteration_tokens=None,
            median_iteration_tokens=None,
            p90_iteration_tokens=None,
            p99_iteration_tokens=None,
            prompt_tokens_cached=None,
            prompt_tokens_recomputed=None,
            external_prefix_cache_hits=None,
            external_prefix_cache_queries=None,
            mm_cache_hits=None,
            mm_cache_queries=None,
            corrupted_requests=None,
            avg_request_prefill_kv_computed_tokens=None,
            median_request_prefill_kv_computed_tokens=None,
            p90_request_prefill_kv_computed_tokens=None,
            p99_request_prefill_kv_computed_tokens=None,
            avg_kv_block_idle_before_evict=None,
            median_kv_block_idle_before_evict=None,
            p90_kv_block_idle_before_evict=None,
            p99_kv_block_idle_before_evict=None,
            avg_kv_block_lifetime=None,
            median_kv_block_lifetime=None,
            p90_kv_block_lifetime=None,
            p99_kv_block_lifetime=None,
            avg_kv_block_reuse_gap=None,
            median_kv_block_reuse_gap=None,
            p90_kv_block_reuse_gap=None,
            p99_kv_block_reuse_gap=None,
        )
