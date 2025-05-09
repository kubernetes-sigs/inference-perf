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
from aiohttp import ClientResponse
import aiohttp
from pydantic import BaseModel
from inference_perf.client.base import ClientRequestMetric, ResponseData, summarize
from inference_perf.config import APIType
from abc import ABC, abstractmethod
from typing import Any, Generator, List

from inference_perf.utils.custom_tokenizer import CustomTokenizer


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0


class ResponsesSummary(BaseModel):
    load_summary: dict[str, Any]
    successes: dict[str, Any]
    failures: dict[str, Any]


class LlmPrompt(ABC, BaseModel):
    @abstractmethod
    def to_payload(
        self, model_name: str, max_tokens: int
    ) -> dict[str, Any]:  # Defines the HTTP request body for this request type
        raise NotImplementedError

    @abstractmethod
    async def process_response(
        self, res: ClientResponse, tokenizer: CustomTokenizer
    ) -> ResponseData:  # Awaits the HTTP response and returns either a successful or failed response object once resolved
        raise NotImplementedError

    @abstractmethod
    def summarize_requests(
        self, responses: List[ClientRequestMetric]
    ) -> (
        ResponsesSummary
    ):  # Generates a summary report from all response metrics with distinct summaries for successes and failures
        raise NotImplementedError


class LlmCompletionPrompt(LlmPrompt):
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


class LlmChatCompletionPrompt(LlmPrompt):
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


class DataGenerator(ABC):
    """Abstract base class for data generators."""

    apiType: APIType

    def __init__(self, apiType: APIType) -> None:
        if apiType not in self.get_supported_apis():
            raise Exception(f"Unsupported API type {apiType}")
        self.apiType = apiType

    @abstractmethod
    def get_supported_apis(self) -> List[APIType]:
        raise NotImplementedError

    @abstractmethod
    def generate_prompt(self) -> Generator[LlmPrompt, None, None]:
        raise NotImplementedError
