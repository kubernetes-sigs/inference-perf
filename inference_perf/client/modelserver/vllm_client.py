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

from inference_perf.client.modelserver.openai_client import openAIModelServerClient, OpenAIMetrics
from inference_perf.client.requestdatacollector import RequestDataCollector
from inference_perf.config import APIConfig, APIType, CustomTokenizerConfig, MultiLoRAConfig
from .base import GaugeMetric, CounterMetric, HistogramMetric, CustomMetric, RequestsMetric
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

    def get_prometheus_metric_metadata(self) -> OpenAIMetrics:
        return OpenAIMetrics(
            prompt_tokens=CounterMetric("prompt_tokens", "vllm:prompt_tokens", self.metric_filters),
            output_tokens=CounterMetric("output_tokens", "vllm:generation_tokens", self.metric_filters),
            requests=RequestsMetric("vllm:request_success", self.metric_filters),
            request_latency=HistogramMetric("request_latency", "vllm:e2e_request_latency_seconds", self.metric_filters),
            queue_length=GaugeMetric("queue_length", "vllm:num_requests_waiting", self.metric_filters),
            time_per_output_token=HistogramMetric(
                "time_per_output_token", "vllm:request_time_per_output_token_seconds", self.metric_filters
            ),
            custom_metrics=[
                GaugeMetric("num_requests_running", "vllm:num_requests_running", self.metric_filters),
                HistogramMetric("time_to_first_token", "vllm:time_to_first_token_seconds", self.metric_filters),
                HistogramMetric("inter_token_latency", "vllm:inter_token_latency_seconds", self.metric_filters),
                CustomMetric("request_success_count", "vllm:request_success", "increase", "counter", self.metric_filters),
                GaugeMetric("kv_cache_usage", "vllm:kv_cache_usage_perc", self.metric_filters),
                CustomMetric(
                    "num_preemptions_total",
                    '{__name__=~"vllm:num_preemptions(_total)?"}',
                    "increase",
                    "counter",
                    self.metric_filters,
                ),
                CustomMetric(
                    "prefix_cache_hits",
                    '{__name__=~"vllm:prefix_cache_hits(_total)?"}',
                    "increase",
                    "counter",
                    self.metric_filters,
                ),
                CustomMetric(
                    "prefix_cache_queries",
                    '{__name__=~"vllm:prefix_cache_queries(_total)?"}',
                    "increase",
                    "counter",
                    self.metric_filters,
                ),
                HistogramMetric("request_queue_time", "vllm:request_queue_time_seconds", self.metric_filters),
                HistogramMetric("request_inference_time", "vllm:request_inference_time_seconds", self.metric_filters),
                HistogramMetric("request_prefill_time", "vllm:request_prefill_time_seconds", self.metric_filters),
                HistogramMetric("request_decode_time", "vllm:request_decode_time_seconds", self.metric_filters),
                HistogramMetric("request_prompt_tokens", "vllm:request_prompt_tokens", self.metric_filters),
                HistogramMetric("request_generation_tokens", "vllm:request_generation_tokens", self.metric_filters),
                HistogramMetric(
                    "request_max_num_generation_tokens", "vllm:request_max_num_generation_tokens", self.metric_filters
                ),
                HistogramMetric("request_params_n", "vllm:request_params_n", self.metric_filters),
                HistogramMetric("request_params_max_tokens", "vllm:request_params_max_tokens", self.metric_filters),
                HistogramMetric("iteration_tokens", "vllm:iteration_tokens_total", self.metric_filters),
                CustomMetric(
                    "prompt_tokens_cached",
                    '{__name__=~"vllm:prompt_tokens_cached(_total)?"}',
                    "increase",
                    "counter",
                    self.metric_filters,
                ),
                CustomMetric(
                    "prompt_tokens_recomputed",
                    '{__name__=~"vllm:prompt_tokens_recomputed(_total)?"}',
                    "increase",
                    "counter",
                    self.metric_filters,
                ),
                CustomMetric(
                    "external_prefix_cache_hits",
                    '{__name__=~"vllm:external_prefix_cache_hits(_total)?"}',
                    "increase",
                    "counter",
                    self.metric_filters,
                ),
                CustomMetric(
                    "external_prefix_cache_queries",
                    '{__name__=~"vllm:external_prefix_cache_queries(_total)?"}',
                    "increase",
                    "counter",
                    self.metric_filters,
                ),
                CustomMetric(
                    "mm_cache_hits",
                    '{__name__=~"vllm:mm_cache_hits(_total)?"}',
                    "increase",
                    "counter",
                    self.metric_filters,
                ),
                CustomMetric(
                    "mm_cache_queries",
                    '{__name__=~"vllm:mm_cache_queries(_total)?"}',
                    "increase",
                    "counter",
                    self.metric_filters,
                ),
                CustomMetric("corrupted_requests", "vllm:corrupted_requests", "increase", "counter", self.metric_filters),
                HistogramMetric(
                    "request_prefill_kv_computed_tokens", "vllm:request_prefill_kv_computed_tokens", self.metric_filters
                ),
                HistogramMetric("kv_block_idle_before_evict", "vllm:kv_block_idle_before_evict_seconds", self.metric_filters),
                HistogramMetric("kv_block_lifetime", "vllm:kv_block_lifetime_seconds", self.metric_filters),
                HistogramMetric("kv_block_reuse_gap", "vllm:kv_block_reuse_gap_seconds", self.metric_filters),
            ],
        )
