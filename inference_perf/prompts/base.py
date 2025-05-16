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

from abc import ABC, abstractmethod
from typing import Any, List, Optional
import aiohttp
import numpy as np
from pydantic import BaseModel
from inference_perf.utils.custom_tokenizer import CustomTokenizer


class FailedResponseData(BaseModel):
    error_type: str
    error_msg: str


class ResponseData(BaseModel):
    info: dict[str, Any]
    error: Optional[FailedResponseData]


class PromptLifecycleMetric(BaseModel):
    """Tracks data for a request across its lifecycle"""

    stage_id: Optional[int] = None
    start_time: float
    end_time: float
    request: "LlmPrompt"
    response: ResponseData

    async def to_report(self) -> dict[str, Any]:
        return self.model_dump()


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0


def summarize(items: List[float]) -> Optional[dict[str, float]]:
    return (
        {
            "mean": float(np.mean(items)),
            "min": float(np.min(items)),
            "p10": float(np.percentile(items, 10)),
            "p50": float(np.percentile(items, 50)),
            "p90": float(np.percentile(items, 90)),
            "max": float(np.max(items)),
        }
        if len(items) != 0
        else None
    )


class ResponsesSummary(BaseModel):
    load_summary: dict[str, Any]
    successes: dict[str, Any]
    failures: dict[str, Any]


class LlmPrompt(ABC, BaseModel):
    @abstractmethod
    def get_route(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def to_payload(
        self, model_name: str, max_tokens: int
    ) -> dict[str, Any]:  # Defines the HTTP request body for this request type
        raise NotImplementedError

    @abstractmethod
    async def process_response(
        self, res: aiohttp.ClientResponse, tokenizer: CustomTokenizer
    ) -> ResponseData:  # Awaits the HTTP response and returns either a successful or failed response object once resolved
        raise NotImplementedError

    @abstractmethod
    def summarize_requests(
        self, responses: List[PromptLifecycleMetric]
    ) -> (
        ResponsesSummary
    ):  # Generates a summary report from all response metrics with distinct summaries for successes and failures
        raise NotImplementedError
