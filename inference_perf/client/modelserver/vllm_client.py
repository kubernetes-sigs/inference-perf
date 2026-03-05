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


class vLLMModelServerClient(openAIModelServerClient):
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
        cert_path: Optional[str] = None,
        key_path: Optional[str] = None,
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
            cert_path,
            key_path,
            lora_config=lora_config,
        )
        self.metric_filters = [f"model_name='{model_name}'", *additional_filters]

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def get_prometheus_metric_metadata(self) -> PrometheusMetricMetadata:
        return PrometheusMetricMetadata(
            # Queue Length
            avg_queue_length=ModelServerPrometheusMetric(
                "vllm:num_requests_waiting",
                "mean",
                "gauge",
                self.metric_filters,
            ),
            # Running Requests
            avg_num_requests_running=ModelServerPrometheusMetric(
                "vllm:num_requests_running",
                "mean",
                "gauge",
                self.metric_filters,
            ),
            # Time to First Token
            avg_time_to_first_token=ModelServerPrometheusMetric(
                "vllm:time_to_first_token_seconds",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_time_to_first_token=ModelServerPrometheusMetric(
                "vllm:time_to_first_token_seconds",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_time_to_first_token=ModelServerPrometheusMetric(
                "vllm:time_to_first_token_seconds",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_time_to_first_token=ModelServerPrometheusMetric(
                "vllm:time_to_first_token_seconds",
                "p99",
                "histogram",
                self.metric_filters,
            ),
            # Time per Output Token (renamed in v1)
            avg_time_per_output_token=ModelServerPrometheusMetric(
                "vllm:request_time_per_output_token_seconds",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_time_per_output_token=ModelServerPrometheusMetric(
                "vllm:request_time_per_output_token_seconds",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_time_per_output_token=ModelServerPrometheusMetric(
                "vllm:request_time_per_output_token_seconds",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_time_per_output_token=ModelServerPrometheusMetric(
                "vllm:request_time_per_output_token_seconds",
                "p99",
                "histogram",
                self.metric_filters,
            ),
            # Inter-Token Latency (new in v1)
            avg_inter_token_latency=ModelServerPrometheusMetric(
                "vllm:inter_token_latency_seconds",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_inter_token_latency=ModelServerPrometheusMetric(
                "vllm:inter_token_latency_seconds",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_inter_token_latency=ModelServerPrometheusMetric(
                "vllm:inter_token_latency_seconds",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_inter_token_latency=ModelServerPrometheusMetric(
                "vllm:inter_token_latency_seconds",
                "p99",
                "histogram",
                self.metric_filters,
            ),
            # Prompt Tokens (renamed in v1: prompt_tokens_total -> prompt_tokens)
            avg_prompt_tokens=ModelServerPrometheusMetric(
                "vllm:prompt_tokens",
                "mean",
                "counter",
                self.metric_filters,
            ),
            prompt_tokens_per_second=ModelServerPrometheusMetric(
                "vllm:prompt_tokens",
                "rate",
                "counter",
                self.metric_filters,
            ),
            # Generation Tokens (renamed in v1: generation_tokens_total -> generation_tokens)
            avg_output_tokens=ModelServerPrometheusMetric(
                "vllm:generation_tokens",
                "mean",
                "counter",
                self.metric_filters,
            ),
            output_tokens_per_second=ModelServerPrometheusMetric(
                "vllm:generation_tokens",
                "rate",
                "counter",
                self.metric_filters,
            ),
            # Total Requests (using request_success counter in v1)
            total_requests=ModelServerPrometheusMetric(
                "vllm:request_success",
                "increase",
                "counter",
                self.metric_filters,
            ),
            requests_per_second=ModelServerPrometheusMetric(
                "vllm:request_success",
                "rate",
                "counter",
                self.metric_filters,
            ),
            request_success_count=ModelServerPrometheusMetric(
                "vllm:request_success",
                "increase",
                "counter",
                self.metric_filters,
            ),
            # E2E Request Latency
            avg_request_latency=ModelServerPrometheusMetric(
                "vllm:e2e_request_latency_seconds",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_request_latency=ModelServerPrometheusMetric(
                "vllm:e2e_request_latency_seconds",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_request_latency=ModelServerPrometheusMetric(
                "vllm:e2e_request_latency_seconds",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_request_latency=ModelServerPrometheusMetric(
                "vllm:e2e_request_latency_seconds",
                "p99",
                "histogram",
                self.metric_filters,
            ),
            # KV Cache Usage
            avg_kv_cache_usage=ModelServerPrometheusMetric(
                "vllm:kv_cache_usage_perc",
                "mean",
                "gauge",
                self.metric_filters,
            ),
            median_kv_cache_usage=ModelServerPrometheusMetric(
                "vllm:kv_cache_usage_perc",
                "median",
                "gauge",
                self.metric_filters,
            ),
            p90_kv_cache_usage=ModelServerPrometheusMetric(
                "vllm:kv_cache_usage_perc",
                "p90",
                "gauge",
                self.metric_filters,
            ),
            p99_kv_cache_usage=ModelServerPrometheusMetric(
                "vllm:kv_cache_usage_perc",
                "p99",
                "gauge",
                self.metric_filters,
            ),
            # Preemptions (renamed in v1: num_preemptions_total -> num_preemptions, now counter)
            num_preemptions_total=ModelServerPrometheusMetric(
                "vllm:num_preemptions",
                "increase",
                "counter",
                self.metric_filters,
            ),
            # Deprecated in v1 (KV cache offloading unused)
            num_requests_swapped=None,
            # Prefix Cache (renamed in v1: dropped _total suffix)
            prefix_cache_hits=ModelServerPrometheusMetric(
                "vllm:prefix_cache_hits",
                "increase",
                "counter",
                self.metric_filters,
            ),
            prefix_cache_queries=ModelServerPrometheusMetric(
                "vllm:prefix_cache_queries",
                "increase",
                "counter",
                self.metric_filters,
            ),
            # Request Queue Time
            avg_request_queue_time=ModelServerPrometheusMetric(
                "vllm:request_queue_time_seconds",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_request_queue_time=ModelServerPrometheusMetric(
                "vllm:request_queue_time_seconds",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_request_queue_time=ModelServerPrometheusMetric(
                "vllm:request_queue_time_seconds",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_request_queue_time=ModelServerPrometheusMetric(
                "vllm:request_queue_time_seconds",
                "p99",
                "histogram",
                self.metric_filters,
            ),
            # Request Inference Time
            avg_request_inference_time=ModelServerPrometheusMetric(
                "vllm:request_inference_time_seconds",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_request_inference_time=ModelServerPrometheusMetric(
                "vllm:request_inference_time_seconds",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_request_inference_time=ModelServerPrometheusMetric(
                "vllm:request_inference_time_seconds",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_request_inference_time=ModelServerPrometheusMetric(
                "vllm:request_inference_time_seconds",
                "p99",
                "histogram",
                self.metric_filters,
            ),
            # Request Prefill Time
            avg_request_prefill_time=ModelServerPrometheusMetric(
                "vllm:request_prefill_time_seconds",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_request_prefill_time=ModelServerPrometheusMetric(
                "vllm:request_prefill_time_seconds",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_request_prefill_time=ModelServerPrometheusMetric(
                "vllm:request_prefill_time_seconds",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_request_prefill_time=ModelServerPrometheusMetric(
                "vllm:request_prefill_time_seconds",
                "p99",
                "histogram",
                self.metric_filters,
            ),
            # Request Decode Time
            avg_request_decode_time=ModelServerPrometheusMetric(
                "vllm:request_decode_time_seconds",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_request_decode_time=ModelServerPrometheusMetric(
                "vllm:request_decode_time_seconds",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_request_decode_time=ModelServerPrometheusMetric(
                "vllm:request_decode_time_seconds",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_request_decode_time=ModelServerPrometheusMetric(
                "vllm:request_decode_time_seconds",
                "p99",
                "histogram",
                self.metric_filters,
            ),
            # Request Prompt Tokens (histogram per-request distribution)
            avg_request_prompt_tokens=ModelServerPrometheusMetric(
                "vllm:request_prompt_tokens",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_request_prompt_tokens=ModelServerPrometheusMetric(
                "vllm:request_prompt_tokens",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_request_prompt_tokens=ModelServerPrometheusMetric(
                "vllm:request_prompt_tokens",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_request_prompt_tokens=ModelServerPrometheusMetric(
                "vllm:request_prompt_tokens",
                "p99",
                "histogram",
                self.metric_filters,
            ),
            # Request Generation Tokens (histogram per-request distribution)
            avg_request_generation_tokens=ModelServerPrometheusMetric(
                "vllm:request_generation_tokens",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_request_generation_tokens=ModelServerPrometheusMetric(
                "vllm:request_generation_tokens",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_request_generation_tokens=ModelServerPrometheusMetric(
                "vllm:request_generation_tokens",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_request_generation_tokens=ModelServerPrometheusMetric(
                "vllm:request_generation_tokens",
                "p99",
                "histogram",
                self.metric_filters,
            ),
            # Request Max Num Generation Tokens
            avg_request_max_num_generation_tokens=ModelServerPrometheusMetric(
                "vllm:request_max_num_generation_tokens",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_request_max_num_generation_tokens=ModelServerPrometheusMetric(
                "vllm:request_max_num_generation_tokens",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_request_max_num_generation_tokens=ModelServerPrometheusMetric(
                "vllm:request_max_num_generation_tokens",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_request_max_num_generation_tokens=ModelServerPrometheusMetric(
                "vllm:request_max_num_generation_tokens",
                "p99",
                "histogram",
                self.metric_filters,
            ),
            # Request Params N
            avg_request_params_n=ModelServerPrometheusMetric(
                "vllm:request_params_n",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_request_params_n=ModelServerPrometheusMetric(
                "vllm:request_params_n",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_request_params_n=ModelServerPrometheusMetric(
                "vllm:request_params_n",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_request_params_n=ModelServerPrometheusMetric(
                "vllm:request_params_n",
                "p99",
                "histogram",
                self.metric_filters,
            ),
            # Request Params Max Tokens
            avg_request_params_max_tokens=ModelServerPrometheusMetric(
                "vllm:request_params_max_tokens",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_request_params_max_tokens=ModelServerPrometheusMetric(
                "vllm:request_params_max_tokens",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_request_params_max_tokens=ModelServerPrometheusMetric(
                "vllm:request_params_max_tokens",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_request_params_max_tokens=ModelServerPrometheusMetric(
                "vllm:request_params_max_tokens",
                "p99",
                "histogram",
                self.metric_filters,
            ),
            # Iteration Tokens
            avg_iteration_tokens=ModelServerPrometheusMetric(
                "vllm:iteration_tokens_total",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_iteration_tokens=ModelServerPrometheusMetric(
                "vllm:iteration_tokens_total",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_iteration_tokens=ModelServerPrometheusMetric(
                "vllm:iteration_tokens_total",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_iteration_tokens=ModelServerPrometheusMetric(
                "vllm:iteration_tokens_total",
                "p99",
                "histogram",
                self.metric_filters,
            ),
            # Token Cache Stats
            prompt_tokens_cached=ModelServerPrometheusMetric(
                "vllm:prompt_tokens_cached",
                "increase",
                "counter",
                self.metric_filters,
            ),
            prompt_tokens_recomputed=ModelServerPrometheusMetric(
                "vllm:prompt_tokens_recomputed",
                "increase",
                "counter",
                self.metric_filters,
            ),
            external_prefix_cache_hits=ModelServerPrometheusMetric(
                "vllm:external_prefix_cache_hits",
                "increase",
                "counter",
                self.metric_filters,
            ),
            external_prefix_cache_queries=ModelServerPrometheusMetric(
                "vllm:external_prefix_cache_queries",
                "increase",
                "counter",
                self.metric_filters,
            ),
            mm_cache_hits=ModelServerPrometheusMetric(
                "vllm:mm_cache_hits",
                "increase",
                "counter",
                self.metric_filters,
            ),
            mm_cache_queries=ModelServerPrometheusMetric(
                "vllm:mm_cache_queries",
                "increase",
                "counter",
                self.metric_filters,
            ),
            corrupted_requests=ModelServerPrometheusMetric(
                "vllm:corrupted_requests",
                "increase",
                "counter",
                self.metric_filters,
            ),
            # Request Prefill KV Computed Tokens
            avg_request_prefill_kv_computed_tokens=ModelServerPrometheusMetric(
                "vllm:request_prefill_kv_computed_tokens",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_request_prefill_kv_computed_tokens=ModelServerPrometheusMetric(
                "vllm:request_prefill_kv_computed_tokens",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_request_prefill_kv_computed_tokens=ModelServerPrometheusMetric(
                "vllm:request_prefill_kv_computed_tokens",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_request_prefill_kv_computed_tokens=ModelServerPrometheusMetric(
                "vllm:request_prefill_kv_computed_tokens",
                "p99",
                "histogram",
                self.metric_filters,
            ),
            # KV Block Idle Before Evict
            avg_kv_block_idle_before_evict=ModelServerPrometheusMetric(
                "vllm:kv_block_idle_before_evict_seconds",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_kv_block_idle_before_evict=ModelServerPrometheusMetric(
                "vllm:kv_block_idle_before_evict_seconds",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_kv_block_idle_before_evict=ModelServerPrometheusMetric(
                "vllm:kv_block_idle_before_evict_seconds",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_kv_block_idle_before_evict=ModelServerPrometheusMetric(
                "vllm:kv_block_idle_before_evict_seconds",
                "p99",
                "histogram",
                self.metric_filters,
            ),
            # KV Block Lifetime
            avg_kv_block_lifetime=ModelServerPrometheusMetric(
                "vllm:kv_block_lifetime_seconds",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_kv_block_lifetime=ModelServerPrometheusMetric(
                "vllm:kv_block_lifetime_seconds",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_kv_block_lifetime=ModelServerPrometheusMetric(
                "vllm:kv_block_lifetime_seconds",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_kv_block_lifetime=ModelServerPrometheusMetric(
                "vllm:kv_block_lifetime_seconds",
                "p99",
                "histogram",
                self.metric_filters,
            ),
            # KV Block Reuse Gap
            avg_kv_block_reuse_gap=ModelServerPrometheusMetric(
                "vllm:kv_block_reuse_gap_seconds",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_kv_block_reuse_gap=ModelServerPrometheusMetric(
                "vllm:kv_block_reuse_gap_seconds",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_kv_block_reuse_gap=ModelServerPrometheusMetric(
                "vllm:kv_block_reuse_gap_seconds",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_kv_block_reuse_gap=ModelServerPrometheusMetric(
                "vllm:kv_block_reuse_gap_seconds",
                "p99",
                "histogram",
                self.metric_filters,
            ),
        )
