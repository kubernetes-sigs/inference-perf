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
from pydantic import BaseModel
from inference_perf.client.base import ModelServerClient


class PerfRuntimeParameters:
    def __init__(self, evaluation_time: float, duration: float, model_server_client: ModelServerClient) -> None:
        self.evaluation_time = evaluation_time
        self.duration = duration
        self.model_server_client = model_server_client


class MetricsSummary(BaseModel):
    total_requests: int
    avg_prompt_tokens: int
    avg_output_tokens: int
    avg_request_latency: float
    avg_time_to_first_token: float
    avg_time_per_output_token: float
    avg_queue_length: int


class MetricsClient(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def collect_metrics_summary(self, runtime_parameters: PerfRuntimeParameters) -> MetricsSummary | None:
        raise NotImplementedError
