# Copyright 2026 The Kubernetes Authors.
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
from typing import Optional
from pydantic import BaseModel, Field

from inference_perf.client.modelserver.metrics import (
    BaseMetrics,
    HistogramResult,
    GaugeResult,
    RequestsResult,
    CounterResult,
)


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
    timeout: Optional[float] = None


class PerfRuntimeParameters:
    def __init__(
        self,
        start_time: float,
        duration: float,
        model_server_metrics: BaseMetrics,
        stages: dict[int, StageRuntimeInfo],
    ) -> None:
        self.start_time = start_time
        self.duration = duration
        self.stages = stages
        self.model_server_metrics = model_server_metrics


class ModelServerMetrics(BaseModel):
    # Throughput
    prompt_tokens: HistogramResult = Field(default_factory=HistogramResult)
    output_tokens: HistogramResult = Field(default_factory=HistogramResult)
    requests: RequestsResult = Field(default_factory=RequestsResult)

    # Latency
    request_latency: HistogramResult = Field(default_factory=HistogramResult)
    time_to_first_token: HistogramResult = Field(default_factory=HistogramResult)
    time_per_output_token: HistogramResult = Field(default_factory=HistogramResult)
    inter_token_latency: HistogramResult = Field(default_factory=HistogramResult)

    # Gauges
    queue_length: GaugeResult = Field(default_factory=GaugeResult)
    num_requests_running: GaugeResult = Field(default_factory=GaugeResult)
    kv_cache_usage: GaugeResult = Field(default_factory=GaugeResult)

    # Single-value customs
    num_requests_swapped: CounterResult = Field(default_factory=CounterResult)
    num_preemptions_total: CounterResult = Field(default_factory=CounterResult)
    prefix_cache_hits: CounterResult = Field(default_factory=CounterResult)
    prefix_cache_queries: CounterResult = Field(default_factory=CounterResult)
    request_success_count: CounterResult = Field(default_factory=CounterResult)
    prompt_tokens_cached: CounterResult = Field(default_factory=CounterResult)
    prompt_tokens_recomputed: CounterResult = Field(default_factory=CounterResult)
    external_prefix_cache_hits: CounterResult = Field(default_factory=CounterResult)
    external_prefix_cache_queries: CounterResult = Field(default_factory=CounterResult)
    mm_cache_hits: CounterResult = Field(default_factory=CounterResult)
    mm_cache_queries: CounterResult = Field(default_factory=CounterResult)
    corrupted_requests: CounterResult = Field(default_factory=CounterResult)

    # Request lifecycle histograms
    request_queue_time: HistogramResult = Field(default_factory=HistogramResult)
    request_inference_time: HistogramResult = Field(default_factory=HistogramResult)
    request_prefill_time: HistogramResult = Field(default_factory=HistogramResult)
    request_decode_time: HistogramResult = Field(default_factory=HistogramResult)
    request_prompt_tokens: HistogramResult = Field(default_factory=HistogramResult)
    request_generation_tokens: HistogramResult = Field(default_factory=HistogramResult)
    request_max_num_generation_tokens: HistogramResult = Field(default_factory=HistogramResult)
    request_params_n: HistogramResult = Field(default_factory=HistogramResult)
    request_params_max_tokens: HistogramResult = Field(default_factory=HistogramResult)
    iteration_tokens: HistogramResult = Field(default_factory=HistogramResult)

    # KV block stats
    request_prefill_kv_computed_tokens: HistogramResult = Field(default_factory=HistogramResult)
    kv_block_idle_before_evict: HistogramResult = Field(default_factory=HistogramResult)
    kv_block_lifetime: HistogramResult = Field(default_factory=HistogramResult)
    kv_block_reuse_gap: HistogramResult = Field(default_factory=HistogramResult)


class ServerMetricsClient(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def collect_metrics_summary(self, runtime_parameters: PerfRuntimeParameters) -> Optional[ModelServerMetrics]:
        raise NotImplementedError

    @abstractmethod
    def collect_metrics_for_stage(
        self, runtime_parameters: PerfRuntimeParameters, stage_id: int
    ) -> Optional[ModelServerMetrics]:
        raise NotImplementedError

    @abstractmethod
    def wait(self) -> None:
        raise NotImplementedError
