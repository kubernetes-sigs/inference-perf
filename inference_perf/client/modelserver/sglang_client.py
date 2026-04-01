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
from .base import GaugeMetric, HistogramMetric, CounterMetric, RequestsMetric
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
        self.metric_filters = [f"model_name='{model_name}'", *additional_filters]

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def get_prometheus_metric_metadata(self) -> OpenAIMetrics:
        return OpenAIMetrics(
            prompt_tokens=CounterMetric("prompt_tokens", "sglang:prompt_tokens_total", self.metric_filters),
            output_tokens=CounterMetric("output_tokens", "sglang:generation_tokens_total", self.metric_filters),
            requests=RequestsMetric("sglang:e2e_request_latency_seconds_count", self.metric_filters),
            request_latency=HistogramMetric("request_latency", "sglang:e2e_request_latency_seconds", self.metric_filters),
            queue_length=GaugeMetric("queue_length", "sglang:num_queue_reqs", self.metric_filters),
            time_per_output_token=HistogramMetric(
                "time_per_output_token", "sglang:time_per_output_token_seconds", self.metric_filters
            ),
            custom_metrics=[
                HistogramMetric("time_to_first_token", "sglang:time_to_first_token_seconds", self.metric_filters),
                HistogramMetric("inter_token_latency", "sglang:inter_token_latency_seconds", self.metric_filters),
                GaugeMetric("kv_cache_usage", "sglang:cache_hit_rate", self.metric_filters),
            ],
        )
