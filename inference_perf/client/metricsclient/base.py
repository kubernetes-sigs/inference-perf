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
from typing import Any, TypedDict

from pydantic import BaseModel


class MetricsMetadata(TypedDict):
    pass


class StageRuntimeInfo(BaseModel):
    stage_id: int
    rate: float
    end_time: float
    start_time: float


class PerfRuntimeParameters:
    def __init__(
        self,
        start_time: float,
        duration: float,
        model_server_metrics: MetricsMetadata,
        stages: dict[int, StageRuntimeInfo],
    ) -> None:
        self.start_time = start_time
        self.duration = duration
        self.stages = stages
        self.model_server_metrics = model_server_metrics


class MetricsClient(ABC):
    @abstractmethod
    def __init__(self) -> None:
        pass

    @abstractmethod
    def collect_metrics_summary(self, runtime_parameters: PerfRuntimeParameters) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def collect_metrics_for_stage(self, runtime_parameters: PerfRuntimeParameters, stage_id: int) -> dict[str, Any] | None:
        raise NotImplementedError

    @abstractmethod
    def wait(self) -> None:
        raise NotImplementedError
