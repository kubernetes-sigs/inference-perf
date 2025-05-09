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
from typing import Any, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel
from build.lib.inference_perf.reportgen.base import RequestMetric
from inference_perf.config import ObservedMetricsReportConfig
from inference_perf.datagen import Prompt
from inference_perf.metrics.base import Metric, MetricCollector


class FailedResponseData(BaseModel):
    error_type: str
    error_msg: str


class ResponseData(BaseModel):
    info: dict[str, Any]
    error: Optional[FailedResponseData]


class ClientRequestMetric(Metric):
    """Tracks data for a request across its lifecycle"""

    start_time: float
    end_time: float
    request: "Prompt"
    response: "ResponseData"

    async def to_report(self) -> dict[str, Any]:
        return self.model_dump()


class ClientRequestMetricsStatisticalSummary(BaseModel):
    mean: Optional[float]
    min: Optional[float]
    p10: Optional[float]
    p50: Optional[float]
    p90: Optional[float]
    max: Optional[float]


def summarize(items: List[float]) -> ClientRequestMetricsStatisticalSummary:
    return ClientRequestMetricsStatisticalSummary(
        mean=float(np.mean(items)),
        min=float(np.min(items)),
        p10=float(np.percentile(items, 10)),
        p50=float(np.percentile(items, 50)),
        p90=float(np.percentile(items, 90)),
        max=float(np.max(items)),
    )


class ClientRequestMetricsCollector(MetricCollector[ClientRequestMetric]):
    """Responsible for accumulating client request metrics and generating corresponding reports"""

    def __init__(self) -> None:
        self.metrics: List[ClientRequestMetric] = []
        pass

    def record_metric(self, metric: ClientRequestMetric) -> None:
        self.metrics.append(metric)

    def list_metrics(self) -> List[ClientRequestMetric]:
        return self.metrics

    def get_report(self, config: ObservedMetricsReportConfig) -> dict[str, Any]:
        report: dict[str, Any] = {}
        if config.summary:
            request_metrics = self.list_metrics()
            if len(self.list_metrics()) != 0:
                report["summary"] = (
                    # Assumes all requests are of the same type
                    request_metrics[0].request.summarize_requests(request_metrics).model_dump()
                )
        if config.per_request:
            report["per_request"] = [metric.model_dump() for metric in self.list_metrics()]
        return report


class ModelServerClient(ABC):
    @abstractmethod
    def __init__(self, *args: Tuple[int, ...]) -> None:
        self.collector = ClientRequestMetricsCollector()
        pass

    @abstractmethod
    async def handle_prompt(self, data: Prompt, stage_id: int) -> None:
        raise NotImplementedError
