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

from inference_perf.collector_reporters.request_lifecycle import RequestLifecycleMetricsCollectorReporter
from inference_perf.config import APIType
from inference_perf.prompts.base import FailedResponseData, LlmPrompt, PromptLifecycleMetric, ResponseData
from inference_perf.utils import CustomTokenizer
from .base import ModelServerClient, PrometheusMetricMetadata, RequestMetric, ModelServerPrometheusMetric
from typing import List
import aiohttp
import json
import time


class vLLMModelServerClient(ModelServerClient):
    def __init__(self, api_type: APIType, uri: str, model_name: str, tokenizer: CustomTokenizer) -> None:
        super().__init__(api_type)
        self.model_name = model_name
        self.uri = uri
        self.max_completion_tokens = 30
        self.tokenizer = tokenizer
        self.request_metrics: List[RequestMetric] = list()
        self.prompt_metrics_collector_reporter = RequestLifecycleMetricsCollectorReporter()

        self.prometheus_metric_metadata: PrometheusMetricMetadata = {
            "avg_queue_length": ModelServerPrometheusMetric(
                "vllm:num_requests_waiting", "mean", "gauge", "model_name='%s'" % self.model_name
            ),
            "avg_time_to_first_token": ModelServerPrometheusMetric(
                "vllm:time_to_first_token_seconds", "mean", "histogram", "model_name='%s'" % self.model_name
            ),
            "median_time_to_first_token": ModelServerPrometheusMetric(
                "vllm:time_to_first_token_seconds", "median", "histogram", "model_name='%s'" % self.model_name
            ),
            "p90_time_to_first_token": ModelServerPrometheusMetric(
                "vllm:time_to_first_token_seconds", "p90", "histogram", "model_name='%s'" % self.model_name
            ),
            "p99_time_to_first_token": ModelServerPrometheusMetric(
                "vllm:time_to_first_token_seconds", "p99", "histogram", "model_name='%s'" % self.model_name
            ),
            "avg_time_per_output_token": ModelServerPrometheusMetric(
                "vllm:time_per_output_token_seconds", "mean", "histogram", "model_name='%s'" % self.model_name
            ),
            "median_time_per_output_token": ModelServerPrometheusMetric(
                "vllm:time_per_output_token_seconds", "median", "histogram", "model_name='%s'" % self.model_name
            ),
            "p90_time_per_output_token": ModelServerPrometheusMetric(
                "vllm:time_per_output_token_seconds", "p90", "histogram", "model_name='%s'" % self.model_name
            ),
            "p99_time_per_output_token": ModelServerPrometheusMetric(
                "vllm:time_per_output_token_seconds", "p99", "histogram", "model_name='%s'" % self.model_name
            ),
            "avg_prompt_tokens": ModelServerPrometheusMetric(
                "vllm:prompt_tokens_total", "mean", "counter", "model_name='%s'" % self.model_name
            ),
            "prompt_tokens_per_second": ModelServerPrometheusMetric(
                "vllm:prompt_tokens_total", "rate", "counter", "model_name='%s'" % self.model_name
            ),
            "avg_output_tokens": ModelServerPrometheusMetric(
                "vllm:generation_tokens_total", "mean", "counter", "model_name='%s'" % self.model_name
            ),
            "output_tokens_per_second": ModelServerPrometheusMetric(
                "vllm:generation_tokens_total", "rate", "counter", "model_name='%s'" % self.model_name
            ),
            "total_requests": ModelServerPrometheusMetric(
                "vllm:e2e_request_latency_seconds_count", "increase", "counter", "model_name='%s'" % self.model_name
            ),
            "requests_per_second": ModelServerPrometheusMetric(
                "vllm:e2e_request_latency_seconds_count", "rate", "counter", "model_name='%s'" % self.model_name
            ),
            "avg_request_latency": ModelServerPrometheusMetric(
                "vllm:e2e_request_latency_seconds", "mean", "histogram", "model_name='%s'" % self.model_name
            ),
            "median_request_latency": ModelServerPrometheusMetric(
                "vllm:e2e_request_latency_seconds", "median", "histogram", "model_name='%s'" % self.model_name
            ),
            "p90_request_latency": ModelServerPrometheusMetric(
                "vllm:e2e_request_latency_seconds", "p90", "histogram", "model_name='%s'" % self.model_name
            ),
            "p99_request_latency": ModelServerPrometheusMetric(
                "vllm:e2e_request_latency_seconds", "p99", "histogram", "model_name='%s'" % self.model_name
            ),
        }

    async def process_request(self, prompt: LlmPrompt, stage_id: int) -> None:
        payload = prompt.to_payload(model_name=self.model_name, max_tokens=self.max_completion_tokens)
        headers = {"Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            start = time.monotonic()
            try:
                async with session.post(self.uri + prompt.get_route(), headers=headers, data=json.dumps(payload)) as response:
                    if response.status == 200:
                        response_body = await prompt.process_response(res=response, tokenizer=self.tokenizer)
                        self.prompt_metrics_collector_reporter.record_metric(
                            PromptLifecycleMetric(
                                stage_id=stage_id,
                                request=prompt,
                                response=response_body,
                                start_time=start,
                                end_time=time.monotonic(),
                            )
                        )
                    else:
                        self.prompt_metrics_collector_reporter.record_metric(
                            PromptLifecycleMetric(
                                stage_id=stage_id,
                                request=prompt,
                                response=ResponseData(
                                    info={},
                                    error=FailedResponseData(error_msg=(await response.text()), error_type="Non 200 reponse"),
                                ),
                                start_time=start,
                                end_time=time.monotonic(),
                            )
                        )
            except Exception as e:
                self.prompt_metrics_collector_reporter.record_metric(
                    PromptLifecycleMetric(
                        stage_id=stage_id,
                        request=prompt,
                        response=ResponseData(
                            info={}, error=FailedResponseData(error_msg=str(e), error_type=type(e).__name__)
                        ),
                        start_time=start,
                        end_time=time.monotonic(),
                    )
                )

    def get_supported_apis(self) -> List[APIType]:
        return [APIType.Completion, APIType.Chat]

    def get_request_metrics(self) -> List[RequestMetric]:
        return self.request_metrics

    def get_prometheus_metric_metadata(self) -> PrometheusMetricMetadata:
        return self.prometheus_metric_metadata
