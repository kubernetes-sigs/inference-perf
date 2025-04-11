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
from inference_perf.datagen import InferenceData
from inference_perf.config import APIType
from .base import ModelServerClient, ModelServerPrometheusMetric, RequestMetric
from typing import Any, List
import aiohttp
import json
import time


class vLLMModelServerClient(ModelServerClient):
    def __init__(self, uri: str, model_name: str, api_type: APIType = APIType.Completion) -> None:
        self.model_name = model_name
        self.uri = uri + ("/v1/chat/completions" if api_type == APIType.Chat else "/v1/completions")
        self.max_completion_tokens = 30
        self.request_metrics: List[RequestMetric] = list()
        self.prometheusMetricMetadata = {
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

    def _create_payload(self, payload: InferenceData) -> dict[str, Any]:
        if payload.type == APIType.Completion:
            return {
                "model": self.model_name,
                "prompt": payload.data.prompt if payload.data else "",
                "max_tokens": self.max_completion_tokens,
            }
        if payload.type == APIType.Chat:
            return {
                "model": self.model_name,
                "messages": [
                    {"role": message.role, "content": message.content}
                    for message in (payload.chat.messages if payload.chat else [])
                ],
                "max_tokens": self.max_completion_tokens,
            }
        raise Exception("api type not supported - has to be completions or chat completions")

    async def process_request(self, data: InferenceData) -> None:
        payload = self._create_payload(data)
        headers = {"Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            start = time.monotonic()
            try:
                async with session.post(self.uri, headers=headers, data=json.dumps(payload)) as response:
                    if response.status == 200:
                        content = await response.json()
                        end = time.monotonic()
                        usage = content["usage"]
                        self.request_metrics.append(
                            RequestMetric(
                                prompt_tokens=usage["prompt_tokens"],
                                output_tokens=usage["completion_tokens"],
                                time_per_request=end - start,
                            )
                        )
                    else:
                        print(await response.text())
            except aiohttp.ClientConnectorError as e:
                print("vLLM Server connection error:\n", str(e))

    def get_request_metrics(self) -> List[RequestMetric]:
        return self.request_metrics

    def get_prometheus_metric_metadata(self) -> dict[str, ModelServerPrometheusMetric]:
        return self.prometheusMetricMetadata
