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
from enum import Enum, auto
from typing import Optional, TypedDict
from pydantic import BaseModel


# Base class for accumulating metrics objects on
class MetricsMetadata(TypedDict):
    pass


class StageStatus(Enum):
    COMPLETED = auto()
    FAILED = auto()
    RUNNING = auto()
    SKIPPED = auto()


class StageRuntimeInfo(BaseModel):
    stage_id: int
    rate: float
    end_time: float
    start_time: float
    status: StageStatus
    concurrency_level: Optional[int] = None


class PerfRuntimeParameters:
    def __init__(
        self, start_time: float, duration: float, model_server_metrics: MetricsMetadata, stages: dict[int, StageRuntimeInfo]
    ) -> None:
        self.start_time = start_time
        self.duration = duration
        self.stages = stages
        self.model_server_metrics = model_server_metrics


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
    avg_inter_token_latency: float = 0.0
    median_inter_token_latency: float = 0.0
    p90_inter_token_latency: float = 0.0
    p99_inter_token_latency: float = 0.0

    # Request
    total_requests: int = 0
    avg_prompt_tokens: int = 0
    avg_output_tokens: int = 0
    avg_queue_length: int = 0
    num_preemptions_total: int = 0
    num_requests_swapped: int = 0

    # Usage
    avg_kv_cache_usage: float = 0.0
    median_kv_cache_usage: float = 0.0
    p90_kv_cache_usage: float = 0.0
    p99_kv_cache_usage: float = 0.0

    # Prefix Cache
    prefix_cache_hits: float = 0.0
    prefix_cache_queries: float = 0.0


class MetricsClient(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def collect_metrics_summary(self, runtime_parameters: PerfRuntimeParameters) -> Optional[ModelServerMetrics]:
        raise NotImplementedError

    @abstractmethod
    def collect_metrics_for_stage(self, runtime_parameters: PerfRuntimeParameters, stage_id: int) -> Optional[ModelServerMetrics]:
        raise NotImplementedError

    @abstractmethod
    def wait(self) -> None:
        raise NotImplementedError
