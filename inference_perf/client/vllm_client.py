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
from inference_perf.client.base import ModelServerClient
from inference_perf.client.client_interfaces.prometheus.base import (
    PrometheusEnabledModelServerClient,
    PrometheusMetricsCollector,
)
from inference_perf.client.client_interfaces.prometheus.prometheus_metrics import (
    PrometheusCounterMetric,
    PrometheusGaugeMetric,
    PrometheusHistogramMetric,
    PrometheusMetric,
)
from inference_perf.datagen import LlmPrompt
from inference_perf.config import APIType, PrometheusCollectorConfig, VLLMConfig
from inference_perf.datagen.base import FailedResponseData, PromptMetric, ResponseData
from inference_perf.utils import CustomTokenizer
from typing import Optional, List
import aiohttp
import json
import time


class vLLMModelServerClient(ModelServerClient, PrometheusEnabledModelServerClient):
    def __init__(
        self,
        config: VLLMConfig,
        prometheus_client_config: Optional[PrometheusCollectorConfig],
    ) -> None:
        super().__init__()
        self.config = config
        try:
            self.tokenizer = CustomTokenizer(
                self.config.tokenizer.pretrained_model_name_or_path,
                self.config.tokenizer.token,
                self.config.tokenizer.trust_remote_code,
            )
        except Exception as e:
            raise Exception(
                "vLLM client is configured, but it requires a custom tokenizer which was not provided or initialized successfully. "
                "Please ensure a valid tokenizer is configured in the 'tokenizer' section of your config file."
            ) from e

        if prometheus_client_config:
            filter = f"model_name='{self.config.model_name}'"
            metrics: List[PrometheusMetric] = [
                PrometheusGaugeMetric(name="avg_queue_length", metric="vllm:num_requests_waiting", filter=filter),
                PrometheusHistogramMetric(
                    name="avg_time_to_first_token", metric="vllm:time_to_first_token_seconds", filter=filter
                ),
                PrometheusHistogramMetric(
                    name="avg_time_per_output_token", metric="vllm:time_per_output_token_seconds", filter=filter
                ),
                PrometheusCounterMetric(name="avg_prompt_tokens", metric="vllm:prompt_tokens_total", filter=filter),
                PrometheusCounterMetric(name="avg_output_tokens", metric="vllm:generation_tokens_total", filter=filter),
                PrometheusCounterMetric(name="total_requests", metric="vllm:e2e_request_latency_seconds_count", filter=filter),
                PrometheusHistogramMetric(
                    name="avg_request_latency", metric="vllm:e2e_request_latency_seconds", filter=filter
                ),
            ]
            self.prometheus_collector = PrometheusMetricsCollector(config=prometheus_client_config, metrics=metrics)
        else:
            print("No prometheus client config passed, not collecting metrics")
        self.model_name = self.config.model_name
        self.uri = self.config.url + ("/v1/chat/completions" if self.config.api == APIType.Chat else "/v1/completions")
        self.max_completion_tokens = 30
        self.custom_tokenizer = CustomTokenizer(
            self.config.tokenizer.pretrained_model_name_or_path,
            self.config.tokenizer.token,
            self.config.tokenizer.trust_remote_code,
        )

    async def handle_prompt(self, prompt: LlmPrompt, stage_id: int) -> None:
        payload = prompt.to_payload(model_name=self.config.model_name, max_tokens=self.max_completion_tokens)
        headers = {"Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            start = time.monotonic()
            try:
                async with session.post(self.uri, headers=headers, data=json.dumps(payload)) as response:
                    if response.status == 200:
                        response_body = await prompt.process_response(res=response, tokenizer=self.custom_tokenizer)
                        end = time.monotonic()
                        self.prompt_metrics_collector.record_metric(
                            PromptMetric(
                                stage_id=stage_id,
                                request=prompt,
                                response=response_body,
                                start_time=start,
                                end_time=end,
                            )
                        )
                    else:
                        self.prompt_metrics_collector.record_metric(
                            PromptMetric(
                                stage_id=stage_id,
                                request=prompt,
                                response=ResponseData(
                                    info={},
                                    error=FailedResponseData(
                                        error_msg=(await response.text()), error_type="Non 200 reponse"
                                    ),
                                ),
                                start_time=start,
                                end_time=end,
                            )
                        )
            except Exception as e:
                self.prompt_metrics_collector.record_metric(
                    PromptMetric(
                        stage_id=stage_id,
                        request=prompt,
                        response=ResponseData(
                            info={}, error=FailedResponseData(error_msg=str(e), error_type=type(e).__name__)
                        ),
                        start_time=start,
                        end_time=time.monotonic(),
                    )
                )
