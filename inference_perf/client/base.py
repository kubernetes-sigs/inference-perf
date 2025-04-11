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
from typing import Tuple
from inference_perf.datagen import InferenceData
from inference_perf.reportgen import ReportGenerator


class ModelServerPrometheusMetric:
    def __init__(self, name: str, op: str, type: str, filters: str) -> None:
        self.name = name
        self.op = op
        self.type = type
        self.filters = filters


class ModelServerClient(ABC):
    @abstractmethod
    def __init__(self, *args: Tuple[int, ...]) -> None:
        pass

    @abstractmethod
    def set_report_generator(self, reportgen: ReportGenerator) -> None:
        self.reportgen = reportgen

    @abstractmethod
    async def process_request(self, data: InferenceData) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_prometheus_metric_metadata(self) -> dict[str, ModelServerPrometheusMetric]:
        # assumption: all metrics clients have metrics exported in Prometheus format
        raise NotImplementedError
