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

from typing import Any, List

import numpy as np
from inference_perf.metrics import MetricsSource
from inference_perf.config import MetricsConfig, RequestMetric


def get_summarization(items: List[float]) -> dict[str, Any]:
    return {
        "mean": float(np.mean(items)),
        "min": float(np.min(items)),
        "p10": float(np.percentile(items, 10)),
        "p50": float(np.percentile(items, 50)),
        "p90": float(np.percentile(items, 90)),
        "max": float(np.max(items)),
    }


class ObservedMetricsCollector(MetricsSource[RequestMetric]):
    def __init__(self, config: MetricsConfig) -> None:
        self.config = config
        self.metrics: List[RequestMetric] = []
        pass

    def record_metric(self, metric: RequestMetric) -> None:
        self.metrics.append(metric)

    def get_metrics(self) -> List[RequestMetric]:
        return self.metrics

    def get_summary_report(self) -> dict[str, Any]:
        request_metrics = self.get_metrics()
        if len(self.get_metrics()) == 0:
            return {}

        # Assumes all requests are of the same type
        return request_metrics[0].request.get_summary_report_for_request_metrics(request_metrics).model_dump()

    def get_per_request_report(self) -> List[dict[str, Any]]:
        return [metric.model_dump() for metric in self.get_metrics()]
