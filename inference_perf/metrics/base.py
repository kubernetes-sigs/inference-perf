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
from typing import Any, Generic, List, Optional, TypeVar
from pydantic import BaseModel
from inference_perf.client.modelserver.base import ModelServerClient, ModelServerMetrics
from inference_perf.reportgen.base import ReportFile


class Metric(BaseModel):
    """Abstract type to track reportable (but not neccesarily summarizable) metrics"""

    stage_id: Optional[int] = None

    @abstractmethod
    async def to_report(self, duration: float) -> dict[str, Any]:
        """Create the report for this metric"""
        raise NotImplementedError


T = TypeVar("T", bound=Metric)


class MetricCollector(ABC, BaseModel, Generic[T]):
    """Anything that can collect metrics to be included in the output report"""

    metrics: List[T]

    @abstractmethod
    async def to_report(self, report_config: Any) -> List[ReportFile]:
        raise NotImplementedError


class PerfRuntimeParameters:
    def __init__(self, start_time: float, duration: float, model_server_client: ModelServerClient) -> None:
        self.start_time = start_time
        self.duration = duration
        self.model_server_client = model_server_client


class MetricsClient(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def collect_model_server_metrics(self, runtime_parameters: PerfRuntimeParameters) -> ModelServerMetrics | None:
        raise NotImplementedError

    @abstractmethod
    def wait(self) -> None:
        raise NotImplementedError
