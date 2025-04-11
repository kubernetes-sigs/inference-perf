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
from abc import ABC, abstractmethod
from typing import Tuple
from inference_perf.metrics import MetricsClient
from inference_perf.metrics.base import PerfRuntimeParameters


class RequestMetric(BaseModel):
    prompt_tokens: int
    output_tokens: int
    time_per_request: float


class ReportGenerator(ABC):
    @abstractmethod
    def __init__(self, metrics_client: MetricsClient, *args: Tuple[int, ...]) -> None:
        self.metrics_client = metrics_client
        pass

    @abstractmethod
    def collect_request_metrics(self, metric: RequestMetric) -> None:
        raise NotImplementedError

    @abstractmethod
    async def generate_report(self, runtime_parameters: PerfRuntimeParameters) -> None:
        raise NotImplementedError
