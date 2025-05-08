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
from pydantic import BaseModel
from inference_perf.client.base import ClientRequestMetric, ResponseData
from inference_perf.config import APIType
from abc import ABC, abstractmethod
from typing import Any, Generator, List

from inference_perf.utils.custom_tokenizer import CustomTokenizer


class ResponsesSummary(BaseModel):
    load_summary: dict[str, Any]
    successes: dict[str, Any]
    failures: dict[str, Any]


class PromptData(ABC, BaseModel):
    @abstractmethod
    def to_payload(self, model_name: str, max_tokens: int) -> dict[str, Any]:
        """Defines the HTTP request body for this request type."""
        raise NotImplementedError

    @abstractmethod
    async def process_response(self, res: ClientResponse, tokenizer: CustomTokenizer) -> ResponseData:
        """Parses the HTTP response and returns either a successful or failed response object."""
        raise NotImplementedError

    @abstractmethod
    def get_summary_report_for_request_metrics(self, responses: List[ClientRequestMetric]) -> ResponsesSummary:
        """Generates a summary report from all response metrics with distinct summaries for successes and failures."""
        raise NotImplementedError


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
    def get_data(self) -> Generator[PromptData, None, None]:
        raise NotImplementedError
