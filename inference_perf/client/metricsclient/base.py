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
from inference_perf.client.modelserver.base import ModelServerClient
from inference_perf.loadgen.load_generator import StageRuntimeInfo
from pydantic import BaseModel


class PerfRuntimeParameters:
    def __init__(
        self, start_time: float, duration: float, model_server_client: ModelServerClient, stages: dict[int, StageRuntimeInfo]
    ) -> None:
        self.start_time = start_time
        self.duration = duration
        self.stages = stages
        self.model_server_client = model_server_client


class ModelServerMetrics(BaseModel):
    # Throughput
    prompt_tokens_per_second: float = 0.0
    output_tokens_per_second: float = 0.0
    requests_per_second: float = 0.0

    # Latency
    avg_request_latency: float = 0.0
    median_request_latency: float = 0.0
    p90_request_latency: float = 0.0
    p99_request_latency: float = 0.0
    avg_time_to_first_token: float = 0.0
    median_time_to_first_token: float = 0.0
    p90_time_to_first_token: float = 0.0
    p99_time_to_first_token: float = 0.0
    avg_time_per_output_token: float = 0.0
    median_time_per_output_token: float = 0.0
    p90_time_per_output_token: float = 0.0
    p99_time_per_output_token: float = 0.0

    # Request
    total_requests: int = 0
    avg_prompt_tokens: int = 0
    avg_output_tokens: int = 0
    avg_queue_length: int = 0


class MetricsClient(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def collect_metrics_summary(self, runtime_parameters: PerfRuntimeParameters) -> ModelServerMetrics | None:
        raise NotImplementedError

    @abstractmethod
    def collect_metrics_for_stage(self, runtime_parameters: PerfRuntimeParameters, stage_id: int) -> ModelServerMetrics | None:
        raise NotImplementedError

    @abstractmethod
    def wait(self) -> None:
        raise NotImplementedError
