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
import numpy as np
from pydantic import BaseModel
from inference_perf.datagen import PromptData
from inference_perf.config import APIType, CustomTokenizerConfig
from inference_perf.datagen.base import FailedResponseData, ResponseData, ResponsesSummary
from inference_perf.utils import CustomTokenizer
from .base import (
    ClientRequestMetric,
    ModelServerClient,
    ModelServerPrometheusMetric,
    PrometheusMetricMetadata,
    RequestMetric,
    get_summarization,
)
from typing import Any, Optional, List
import aiohttp
import json
import time


class VllmCompletionPromptData(PromptData):
    prompt: str

    def to_payload(self, model_name: str, max_tokens: int) -> dict[str, Any]:
        return {
            "model": model_name,
            "prompt": self.prompt,
            "max_tokens": max_tokens,
        }

    async def process_response(self, res: aiohttp.ClientResponse, tokenizer: CustomTokenizer) -> ResponseData:
        content = await res.json()
        choices = content.get("choices", [])
        prompt_len = tokenizer.count_tokens(self.prompt)
        output_text = choices[0].get("text", "")
        output_len = tokenizer.count_tokens(output_text)
        return ResponseData(
            info={
                "prompt": self.prompt,
                "prompt_len": prompt_len,
                "output_text": output_text,
                "output_len": output_len,
            }
        )

    def get_summary_report_for_request_metrics(self, metrics: List[ClientRequestMetric]) -> ResponsesSummary:
        all_successful: List[ClientRequestMetric] = [x for x in metrics if x.response.error is None]
        all_failed: List[ClientRequestMetric] = [x for x in metrics if x.response.error is not None]

        return ResponsesSummary(
            load_summary={
                "count": len(metrics),
                "time_per_request": get_summarization([(metric.end_time - metric.start_time) for metric in metrics]).model_dump(),
            },
            successes={
                "count": len(all_successful),
                "time_per_request": get_summarization([(metric.end_time - metric.start_time) for metric in metrics]).model_dump(),
                "prompt_len": get_summarization([success.response.info.get("prompt_len") for success in all_successful]).model_dump(),
                "output_len": get_summarization(
                    [float(v) for success in all_successful if (v := success.info.get("output_len")) is not None]
                ).model_dump(),
                "per_token_latency": get_summarization(
                    [
                        (metric.end_time - metric.start_time) / float(metric.response.info.get("output_len"))
                        if isinstance(metric.response, ResponseData)
                        and metric.response.info.get("output_len") is not None
                        and float(metric.response.info.get("output_len")) != 0
                        else 0
                        for metric in all_successful
                    ]
                ).model_dump(),
            },
            failures={
                "count": len(all_failed),
                # need to filter to only the failures, currently dont do that, same for successes
                "time_per_request": get_summarization([(failed.end_time - failed.start_time) for failed in all_failed]).model_dump(),
            },
        )


class ChatMessage(BaseModel):
    role: str
    content: str


class VllmChatCompletionPromptData(PromptData):
    messages: List[ChatMessage]

    def to_payload(self, model_name: str, max_tokens: int) -> dict[str, Any]:
        return {
            "model": model_name,
            "messages": [{"role": m.role, "content": m.content} for m in self.messages],
            "max_tokens": max_tokens,
        }

    async def process_response(self, res: aiohttp.ClientResponse, tokenizer: CustomTokenizer) -> ResponseData:
        content = await res.json()
        choices = content.get("choices", [])
        output_text = choices[0].get("message", {}).get("content", "")
        output_len = tokenizer.count_tokens(output_text)
        return ResponseData(
            info={
                "output_text": output_text,
                "output_len": output_len,
            }
        )

    def get_summary_report_for_request_metrics(self, metrics: List[ClientRequestMetric]) -> ResponsesSummary:
        all_successful: List[ClientRequestMetric] = [x for x in metrics if x.response.error is None]
        all_failed: List[ClientRequestMetric] = [x for x in metrics if x.response.error is not None]

        return ResponsesSummary(
            load_summary={
                "count": len(metrics),
                "time_per_request": get_summarization([(metric.end_time - metric.start_time) for metric in metrics]).model_dump(),
            },
            successes={
                "count": len(all_successful),
                "time_per_request": get_summarization([(successful.end_time - successful.start_time) for successful in all_successful]).model_dump(),
                "output_len": get_summarization(
                    [float(v) for success in all_successful if (v := success.info.get("output_len")) is not None]
                ).model_dump(),
                "per_token_latency": get_summarization(
                    [
                        (success.end_time - success.start_time) / success.response.output_len
                        if success.response.output_len != 0
                        else 0
                        for success in all_successful
                    ]
                ).model_dump(),
            },
            failures={
                "count": len(all_failed),
                "time_per_request": get_summarization([(failed.end_time - failed.start_time) for failed in all_failed]).model_dump(),
            },
        )


class vLLMModelServerClient(ModelServerClient):
    def __init__(self, uri: str, model_name: str, tokenizer: Optional[CustomTokenizerConfig], api_type: APIType) -> None:
        self.model_name = model_name
        self.uri = uri + ("/v1/chat/completions" if api_type == APIType.Chat else "/v1/completions")
        self.max_completion_tokens = 30
        self.tokenizer_available = False

        if tokenizer and tokenizer.pretrained_model_name_or_path:
            try:
                self.custom_tokenizer = CustomTokenizer(
                    tokenizer.pretrained_model_name_or_path,
                    tokenizer.token,
                    tokenizer.trust_remote_code,
                )
                self.tokenizer_available = True
            except Exception as e:
                print(f"Tokenizer initialization failed: {e}")
                print("Falling back to usage metrics.")
        else:
            print("Tokenizer path is empty. Falling back to usage metrics.")
        self.request_metrics: List[RequestMetric] = list()

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

    def _create_payload(self, payload: PromptData) -> dict[str, Any]:
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

    async def process_request(self, data: PromptData, stage_id: int) -> None:
        payload = self._create_payload(data)
        headers = {"Content-Type": "application/json"}
        async with aiohttp.ClientSession() as session:
            start = time.monotonic()
            try:
                async with session.post(self.uri, headers=headers, data=json.dumps(payload)) as response:
                    if response.status == 200:
                        content = await response.json()
                        end = time.monotonic()
                        usage = content.get("usage", {})
                        choices = content.get("choices", [])

                        if data.type == APIType.Completion:
                            prompt = data.data.prompt if data.data else ""
                            output_text = choices[0].get("text", "")
                        elif data.type == APIType.Chat:
                            prompt = " ".join([msg.content for msg in data.chat.messages]) if data.chat else ""
                            output_text = choices[0].get("message", {}).get("content", "")
                        else:
                            raise Exception("Unsupported API type")

                        if self.tokenizer_available:
                            prompt_tokens = self.custom_tokenizer.count_tokens(prompt)
                            output_tokens = self.custom_tokenizer.count_tokens(output_text)
                        else:
                            prompt_tokens = usage.get("prompt_tokens", 0)
                            output_tokens = usage.get("completion_tokens", 0)

                        self.request_metrics.append(
                            RequestMetric(
                                stage_id=stage_id,
                                prompt_tokens=prompt_tokens,
                                output_tokens=output_tokens,
                                time_per_request=end - start,
                            )
                        )
                    else:
                        print(await response.text())
            except aiohttp.ClientConnectorError as e:
                print("vLLM Server connection error:\n", str(e))

    def get_request_metrics(self) -> List[RequestMetric]:
        return self.request_metrics

    def get_prometheus_metric_metadata(self) -> PrometheusMetricMetadata:
        return self.prometheus_metric_metadata
