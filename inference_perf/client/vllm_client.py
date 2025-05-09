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
from pydantic import BaseModel
from inference_perf.client.client_interfaces.prometheus.prometheus import PrometheusEnabledModelServerClient
from inference_perf.client.client_interfaces.prometheus.prometheus_metrics import (
    PrometheusCounterMetric,
    PrometheusGaugeMetric,
    PrometheusHistogramMetric,
    PrometheusMetric,
    PrometheusMetricsCollector,
)
from inference_perf.datagen import LlmPrompt
from inference_perf.config import APIType, PrometheusCollectorConfig, VLLMConfig
from inference_perf.datagen.base import ResponseData, ResponsesSummary
from inference_perf.utils import CustomTokenizer
from .base import (
    ClientRequestMetric,
    FailedResponseData,
    ModelServerClient,
    RequestMetric,
    summarize,
)
from typing import Any, Optional, List
import aiohttp
import json
import time


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0


class VllmCompletionPrompt(LlmPrompt):
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
            },
            error=None,
        )

    def summarize_requests(self, metrics: List[ClientRequestMetric]) -> ResponsesSummary:
        all_successful: List[ClientRequestMetric] = [x for x in metrics if x.response.error is None]
        all_failed: List[ClientRequestMetric] = [x for x in metrics if x.response.error is not None]

        return ResponsesSummary(
            load_summary={
                "count": len(metrics),
                "time_per_request": summarize([(metric.end_time - metric.start_time) for metric in metrics]).model_dump(),
            },
            successes={
                "count": len(all_successful),
                "time_per_request": summarize([(metric.end_time - metric.start_time) for metric in metrics]).model_dump(),
                "prompt_len": summarize(
                    [safe_float(success.response.info.get("prompt_len")) for success in all_successful]
                ).model_dump(),
                "output_len": summarize(
                    [float(v) for success in all_successful if (v := success.info.get("output_len")) is not None]
                ).model_dump(),
                "per_token_latency": summarize(
                    [
                        ((metric.end_time - metric.start_time) / output_len) if output_len and output_len != 0 else 0
                        for metric in all_successful
                        for output_len in [safe_float(metric.response.info.get("output_len"))]
                    ]
                ).model_dump(),
            },
            failures={
                "count": len(all_failed),
                # need to filter to only the failures, currently dont do that, same for successes
                "time_per_request": summarize([(failed.end_time - failed.start_time) for failed in all_failed]).model_dump(),
            },
        )


class ChatMessage(BaseModel):
    role: str
    content: str


class VllmChatCompletionPrompt(LlmPrompt):
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
            },
            error=None,
        )

    def summarize_requests(self, metrics: List[ClientRequestMetric]) -> ResponsesSummary:
        all_successful: List[ClientRequestMetric] = [x for x in metrics if x.response.error is None]
        all_failed: List[ClientRequestMetric] = [x for x in metrics if x.response.error is not None]

        return ResponsesSummary(
            load_summary={
                "count": len(metrics),
                "time_per_request": summarize([(metric.end_time - metric.start_time) for metric in metrics]).model_dump(),
            },
            successes={
                "count": len(all_successful),
                "time_per_request": summarize(
                    [(successful.end_time - successful.start_time) for successful in all_successful]
                ).model_dump(),
                "output_len": summarize(
                    [float(v) for success in all_successful if (v := safe_float(success.info.get("output_len"))) is not None]
                ).model_dump(),
                "per_token_latency": summarize(
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
                "time_per_request": summarize([(failed.end_time - failed.start_time) for failed in all_failed]).model_dump(),
            },
        )


class vLLMModelServerClient(ModelServerClient, PrometheusEnabledModelServerClient):
    def __init__(self, config: VLLMConfig, prometheus_client_config: Optional[PrometheusCollectorConfig]) -> None:
        self.config = config
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

        self.model_name = self.config.model_name
        self.uri = self.config.url + ("/v1/chat/completions" if self.config.api == APIType.Chat else "/v1/completions")
        self.max_completion_tokens = 30
        self.tokenizer_available = False

        if self.config.tokenizer and self.config.tokenizer.pretrained_model_name_or_path:
            try:
                self.custom_tokenizer = CustomTokenizer(
                    self.config.tokenizer.pretrained_model_name_or_path,
                    self.config.tokenizer.token,
                    self.config.tokenizer.trust_remote_code,
                )
                self.tokenizer_available = True
            except Exception as e:
                print(f"Tokenizer initialization failed: {e}")
                print("Falling back to usage metrics.")
        else:
            print("Tokenizer path is empty. Falling back to usage metrics.")
        self.request_metrics: List[RequestMetric] = list()

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
                        self.collector.record_metric(
                            ClientRequestMetric(
                                stage_id=stage_id,
                                request=prompt,
                                response=response_body,
                                start_time=start,
                                end_time=end,
                            )
                        )
                    else:
                        self.collector.record_metric(
                            ClientRequestMetric(
                                stage_id=stage_id,
                                request=prompt,
                                response=ResponseData(
                                    info={},
                                    error=FailedResponseData(
                                        error_msg=(await response_body.text()), error_type="Non 200 reponse"
                                    ),
                                ),
                                start_time=start,
                                end_time=end,
                            )
                        )
            except Exception as e:
                self.collector.record_metric(
                    ClientRequestMetric(
                        stage_id=stage_id,
                        request=prompt,
                        response=ResponseData(
                            info={}, error=FailedResponseData(error_msg=str(e), error_type=type(e).__name__)
                        ),
                        start_time=start,
                        end_time=end,
                    )
                )
