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
from typing import Any, Generic, List, TypeVar

from inference_perf.colletor_metric.base import Metric
from inference_perf.reportgen.base import ReportFile


T = TypeVar("T", bound=Metric)


class MetricsCollector(ABC, Generic[T]):
    """Anything that can collect metrics to be included in the output report"""

    def __init__(self) -> None:
        self.metrics: List[T] = []

    @abstractmethod
    async def to_reports(self, report_config: Any) -> List[ReportFile]:
        raise NotImplementedError
