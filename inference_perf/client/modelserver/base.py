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
from typing import List, Tuple, TypedDict
from inference_perf.config import APIType

from pydantic import BaseModel

from inference_perf.prompts.base import InferenceData


class RequestMetric(BaseModel):
    stage_id: int
    prompt_tokens: int
    output_tokens: int
    time_per_request: float


class ModelServerPrometheusMetric:
    def __init__(self, name: str, op: str, type: str, filters: str) -> None:
        self.name = name
        self.op = op
        self.type = type
        self.filters = filters


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


# PrometheusMetricMetadata stores the mapping of metrics to their model server names and types
# and the filters to be applied to them.
# This is used to generate Prometheus query for the metrics.
class PrometheusMetricMetadata(TypedDict):
    # Throughput
    prompt_tokens_per_second: ModelServerPrometheusMetric
    output_tokens_per_second: ModelServerPrometheusMetric
    requests_per_second: ModelServerPrometheusMetric

    # Latency
    avg_request_latency: ModelServerPrometheusMetric
    median_request_latency: ModelServerPrometheusMetric
    p90_request_latency: ModelServerPrometheusMetric
    p99_request_latency: ModelServerPrometheusMetric
    avg_time_to_first_token: ModelServerPrometheusMetric
    median_time_to_first_token: ModelServerPrometheusMetric
    p90_time_to_first_token: ModelServerPrometheusMetric
    p99_time_to_first_token: ModelServerPrometheusMetric
    avg_time_per_output_token: ModelServerPrometheusMetric
    median_time_per_output_token: ModelServerPrometheusMetric
    p90_time_per_output_token: ModelServerPrometheusMetric
    p99_time_per_output_token: ModelServerPrometheusMetric

    # Request
    total_requests: ModelServerPrometheusMetric
    avg_prompt_tokens: ModelServerPrometheusMetric
    avg_output_tokens: ModelServerPrometheusMetric
    avg_queue_length: ModelServerPrometheusMetric


class ModelServerClient(ABC):
    @abstractmethod
    def __init__(self, api_type: APIType, *args: Tuple[int, ...]) -> None:
        if api_type not in self.get_supported_apis():
            raise Exception(f"Unsupported API type {api_type}")

    @abstractmethod
    def get_supported_apis(self) -> List[APIType]:
        raise NotImplementedError

    @abstractmethod
    async def process_request(self, data: InferenceData, stage_id: int) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_request_metrics(self) -> List[RequestMetric]:
        raise NotImplementedError

    @abstractmethod
    def get_prometheus_metric_metadata(self) -> PrometheusMetricMetadata:
        # assumption: all metrics clients have metrics exported in Prometheus format
        raise NotImplementedError
