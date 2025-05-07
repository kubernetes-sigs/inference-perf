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
from typing import Any, List, Tuple

from aiohttp import ClientResponse
from pydantic import BaseModel
from inference_perf.metrics.base import RequestMetric
from inference_perf.reportgen import ReportGenerator
from inference_perf.utils.custom_tokenizer import CustomTokenizer


class FailedResponseData(BaseModel):
    error_type: str
    error_msg: str


class SuccessfulResponseData(BaseModel):
    info: dict[str, Any]


ResponseData = FailedResponseData | SuccessfulResponseData


class ResponsesSummary(BaseModel):
    """Regardless of the request type, successes and failures should always be categorized separately"""

    load_summary: dict[str, Any]
    successes: dict[str, Any]
    failures: dict[str, Any]


class PromptData(ABC, BaseModel):
    @abstractmethod
    def to_payload(self, model_name: str, max_tokens: int) -> dict[str, Any]:
        """What should the body for an HTTP request contain for this request type"""
        raise NotImplementedError

    @abstractmethod
    async def get_response_data(self, res: ClientResponse, tokenizer: CustomTokenizer) -> ResponseData:
        """For a given response type, what is the success criteria and what should be reported from successful responses"""
        raise NotImplementedError

    @abstractmethod
    def get_summary_report_for_request_metrics(self, responses: List[RequestMetric]) -> ResponsesSummary:
        """Given a list of responses for this request type, how will these be summarized"""
        raise NotImplementedError


class ModelServerClient(ABC):
    @abstractmethod
    def __init__(self, *args: Tuple[int, ...]) -> None:
        pass

    @abstractmethod
    def set_report_generator(self, reportgen: ReportGenerator) -> None:
        self.reportgen = reportgen

    @abstractmethod
    async def process_request(self, data: PromptData, stage_id: int) -> None:
        raise NotImplementedError
