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


class SGlangModelServerClient(openAIModelServerClient):
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
        lora_config: MultiLoRAConfig = None,
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
            lora_config,
        )
        self.metric_filters = [f"model_name='{model_name}'", *additional_filters]

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def get_prometheus_metric_metadata(self) -> PrometheusMetricMetadata:
        return PrometheusMetricMetadata(
            avg_queue_length=ModelServerPrometheusMetric(
                "sglang:num_queue_reqs",
                "mean",
                "gauge",
                self.metric_filters,
            ),
            avg_time_to_first_token=ModelServerPrometheusMetric(
                "sglang:time_to_first_token_seconds",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_time_to_first_token=ModelServerPrometheusMetric(
                "sglang:time_to_first_token_seconds",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_time_to_first_token=ModelServerPrometheusMetric(
                "sglang:time_to_first_token_seconds",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_time_to_first_token=ModelServerPrometheusMetric(
                "sglang:time_to_first_token_seconds",
                "p99",
                "histogram",
                self.metric_filters,
            ),
            avg_inter_token_latency=ModelServerPrometheusMetric(
                "sglang:inter_token_latency_seconds",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_inter_token_latency=ModelServerPrometheusMetric(
                "sglang:inter_token_latency_seconds",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_inter_token_latency=ModelServerPrometheusMetric(
                "sglang:inter_token_latency_seconds",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_inter_token_latency=ModelServerPrometheusMetric(
                "sglang:inter_token_latency_seconds",
                "p99",
                "histogram",
                self.metric_filters,
            ),
            avg_prompt_tokens=ModelServerPrometheusMetric(
                "sglang:prompt_tokens_total", "mean", "counter", self.metric_filters
            ),
            prompt_tokens_per_second=ModelServerPrometheusMetric(
                "sglang:prompt_tokens_total", "rate", "counter", self.metric_filters
            ),
            avg_output_tokens=ModelServerPrometheusMetric(
                "sglang:generation_tokens_total", "mean", "counter", self.metric_filters
            ),
            output_tokens_per_second=ModelServerPrometheusMetric(
                "sglang:generation_tokens_total", "rate", "counter", self.metric_filters
            ),
            total_requests=ModelServerPrometheusMetric(
                "sglang:e2e_request_latency_seconds_count",
                "increase",
                "counter",
                self.metric_filters,
            ),
            requests_per_second=ModelServerPrometheusMetric(
                "sglang:e2e_request_latency_seconds_count",
                "rate",
                "counter",
                self.metric_filters,
            ),
            avg_request_latency=ModelServerPrometheusMetric(
                "sglang:e2e_request_latency_seconds",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_request_latency=ModelServerPrometheusMetric(
                "sglang:e2e_request_latency_seconds",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_request_latency=ModelServerPrometheusMetric(
                "sglang:e2e_request_latency_seconds",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_request_latency=ModelServerPrometheusMetric(
                "sglang:e2e_request_latency_seconds",
                "p99",
                "histogram",
                self.metric_filters,
            ),
            avg_kv_cache_usage=ModelServerPrometheusMetric(
                "sglang:cache_hit_rate",
                "mean",
                "gauge",
                self.metric_filters,
            ),
            median_kv_cache_usage=ModelServerPrometheusMetric(
                "sglang:cache_hit_rate",
                "median",
                "gauge",
                self.metric_filters,
            ),
            p90_kv_cache_usage=ModelServerPrometheusMetric(
                "sglang:cache_hit_rate",
                "p90",
                "gauge",
                self.metric_filters,
            ),
            p99_kv_cache_usage=ModelServerPrometheusMetric(
                "sglang:cache_hit_rate",
                "p99",
                "gauge",
                self.metric_filters,
            ),
            avg_time_per_output_token=ModelServerPrometheusMetric(
                "sglang:time_per_output_token_seconds",
                "mean",
                "histogram",
                self.metric_filters,
            ),
            median_time_per_output_token=ModelServerPrometheusMetric(
                "sglang:time_per_output_token_seconds",
                "median",
                "histogram",
                self.metric_filters,
            ),
            p90_time_per_output_token=ModelServerPrometheusMetric(
                "sglang:time_per_output_token_seconds",
                "p90",
                "histogram",
                self.metric_filters,
            ),
            p99_time_per_output_token=ModelServerPrometheusMetric(
                "sglang:time_per_output_token_seconds",
                "p99",
                "histogram",
                self.metric_filters,
            ),
            num_preemptions_total=None,
            num_requests_swapped=None,
            prefix_cache_hits=None,
            prefix_cache_queries=None,
        )
