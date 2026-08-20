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
from inference_perf.metrics.request_collector import RequestMetricCollector
from inference_perf.config import APIConfig, APIType, CustomTokenizerConfig, MultiLoRAConfig
from .metrics import GaugeMetric, CounterMetric, HistogramMetric
from typing import List, Optional
import logging

logger = logging.getLogger(__name__)


class vLLMModelServerClient(openAIModelServerClient):
    def __init__(
        self,
        metrics_collector: RequestMetricCollector,
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
        return [APIType.Completion, APIType.Chat, APIType.AnthropicMessages]

    def get_prometheus_metric_metadata(self) -> OpenAIMetrics:
        return OpenAIMetrics(
            filters=self.metric_filters,
            prompt_tokens=CounterMetric("vllm:prompt_tokens"),
            output_tokens=CounterMetric("vllm:generation_tokens"),
            requests=CounterMetric("vllm:request_success"),
            request_latency=HistogramMetric("vllm:e2e_request_latency_seconds"),
            queue_length=GaugeMetric("vllm:num_requests_waiting"),
            time_per_output_token=HistogramMetric("vllm:request_time_per_output_token_seconds"),
            custom_metrics={
                "num_requests_running": GaugeMetric("vllm:num_requests_running"),
                "time_to_first_token": HistogramMetric("vllm:time_to_first_token_seconds"),
                "inter_token_latency": HistogramMetric("vllm:inter_token_latency_seconds"),
                "request_success_count": CounterMetric("vllm:request_success"),
                "kv_cache_usage": GaugeMetric("vllm:kv_cache_usage_perc"),
                "num_preemptions_total": CounterMetric("vllm:num_preemptions"),
                "prefix_cache_hits": CounterMetric("vllm:prefix_cache_hits"),
                "prefix_cache_queries": CounterMetric("vllm:prefix_cache_queries"),
                "request_queue_time": HistogramMetric("vllm:request_queue_time_seconds"),
                "request_inference_time": HistogramMetric("vllm:request_inference_time_seconds"),
                "request_prefill_time": HistogramMetric("vllm:request_prefill_time_seconds"),
                "request_decode_time": HistogramMetric("vllm:request_decode_time_seconds"),
                "request_prompt_tokens": HistogramMetric("vllm:request_prompt_tokens"),
                "request_generation_tokens": HistogramMetric("vllm:request_generation_tokens"),
                "request_max_num_generation_tokens": HistogramMetric("vllm:request_max_num_generation_tokens"),
                "request_params_n": HistogramMetric("vllm:request_params_n"),
                "request_params_max_tokens": HistogramMetric("vllm:request_params_max_tokens"),
                "iteration_tokens": HistogramMetric("vllm:iteration_tokens_total"),
                "prompt_tokens_cached": CounterMetric("vllm:prompt_tokens_cached"),
                "external_prefix_cache_hits": CounterMetric("vllm:external_prefix_cache_hits"),
                "external_prefix_cache_queries": CounterMetric("vllm:external_prefix_cache_queries"),
                "mm_cache_hits": CounterMetric("vllm:mm_cache_hits"),
                "mm_cache_queries": CounterMetric("vllm:mm_cache_queries"),
                # Only exposed when the server runs with VLLM_COMPUTE_NANS_IN_LOGITS=1.
                "corrupted_requests": CounterMetric("vllm:corrupted_requests"),
                "request_prefill_kv_computed_tokens": HistogramMetric("vllm:request_prefill_kv_computed_tokens"),
                # kv_block_* are only exposed when the server runs with --kv-cache-metrics.
                "kv_block_idle_before_evict": HistogramMetric("vllm:kv_block_idle_before_evict_seconds"),
                "kv_block_lifetime": HistogramMetric("vllm:kv_block_lifetime_seconds"),
                "kv_block_reuse_gap": HistogramMetric("vllm:kv_block_reuse_gap_seconds"),
            },
        )
