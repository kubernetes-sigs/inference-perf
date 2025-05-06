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

from typing import List
from inference_perf.metrics.base import MetricsSource
from inference_perf.config import MetricsConfig, RequestMetric


class ObservedMetricsCollector(MetricsSource[RequestMetric]):
    def __init__(self, config: MetricsConfig) -> None:
        self.config = config
        self.metrics: List[RequestMetric] = []
        pass

    def record_metric(self, metric: RequestMetric) -> None:
        self.metrics.append(metric)

    def get_metrics(self) -> List[RequestMetric]:
        return self.metrics
