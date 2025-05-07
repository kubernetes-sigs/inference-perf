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
from abc import abstractmethod
from typing import Generic, List, Optional, TypeVar

from pydantic import BaseModel
from inference_perf.client import PromptData, ResponseData
from inference_perf.config import MetricsConfig


class Metric(BaseModel):
    """Abstract type to track individual request metrics, prometheus metrics, etc"""

    stage_id: Optional[int] = None


class RequestMetric(Metric):
    """The source of truth for all data pertaining to a particular request"""

    start_time: float
    end_time: float
    request: PromptData
    response: ResponseData


T = TypeVar("T", bound=Metric)


class MetricsSource(Generic[T]):
    """Anything that can provide a list of metrics"""

    def __init__(self, config: MetricsConfig) -> None:
        self.config = config
        pass

    @abstractmethod
    def get_metrics(self) -> List[T]:
        pass
