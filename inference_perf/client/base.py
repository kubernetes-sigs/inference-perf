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
from abc import ABC, abstractmethod
from typing import Any, List, Tuple

from aiohttp import ClientResponse
from pydantic import BaseModel
from inference_perf.config import ObservedMetricsReportConfig
from inference_perf.metrics.base import Metric, MetricsCollector
from inference_perf.utils.custom_tokenizer import CustomTokenizer


def get_summarization(items: List[float]) -> dict[str, Any]:
    return {
        "mean": float(np.mean(items)),
        "min": float(np.min(items)),
        "p10": float(np.percentile(items, 10)),
        "p50": float(np.percentile(items, 50)),
        "p90": float(np.percentile(items, 90)),
        "max": float(np.max(items)),
    }


class ClientRequestMetric(Metric):
    """Tracks data for a request across its lifecycle"""

    start_time: float
    end_time: float
    request: "PromptData"
    response: "ResponseData"


class ClientRequestMetricsCollector(MetricsCollector[ClientRequestMetric]):
    """Responsible for accumulating client request metrics and generating corresponding reports"""

    def __init__(self) -> None:
        self.metrics: List[ClientRequestMetric] = []
        pass

    def record_metric(self, metric: ClientRequestMetric) -> None:
        self.metrics.append(metric)

    def get_metrics(self) -> List[ClientRequestMetric]:
        return self.metrics

    def get_report(self, config: ObservedMetricsReportConfig) -> dict[str, Any]:
        report: dict[str, Any] = {}
        if config.summary:
            request_metrics = self.get_metrics()
            # Assumes all requests are of the same type
            if len(self.get_metrics()) != 0:
                report["summary"] = (
                    request_metrics[0].request.get_summary_report_for_request_metrics(request_metrics).model_dump()
                )
        if config.per_request:
            report["per_request"] = [metric.model_dump() for metric in self.get_metrics()]
        return report


class FailedResponseData(BaseModel):
    error_type: str
    error_msg: str


class SuccessfulResponseData(BaseModel):
    info: dict[str, Any]


ResponseData = FailedResponseData | SuccessfulResponseData


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


class ModelServerClient(ABC):
    @abstractmethod
    def __init__(self, *args: Tuple[int, ...]) -> None:
        self.collector = ClientRequestMetricsCollector()

    @abstractmethod
    async def process_request(self, data: PromptData, stage_id: int) -> None:
        raise NotImplementedError
