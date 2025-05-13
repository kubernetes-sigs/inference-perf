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
from typing import Any, List, Optional


from inference_perf.client.client_interfaces.prometheus.prometheus_metrics import PrometheusMetric
from inference_perf.config import PrometheusMetricsReportConfig
from inference_perf.metrics.base import MetricCollector


class PrometheusMetricsCollector(MetricCollector[PrometheusMetric], ABC):
    def __init__(self, metrics: List[PrometheusMetric]):
        super().__init__(metrics=metrics)

    async def to_report(self, report_config: PrometheusMetricsReportConfig, duration: float) -> dict[str, Any]:
        total_report = {}
        for metric in self.metrics:
            metric_report = {}
            queries = metric.get_query_set(duration=duration)
            for query_name, query in queries.items():
                result = await self.query_metric(query=query, duration=duration)
                if result is not None:
                    metric_report[query_name] = result
            total_report[metric.name] = metric_report
        return total_report

    @abstractmethod
    async def query_metric(self, query: str, duration: float) -> Optional[float]:
        pass


class PrometheusEnabledModelServerClient:
    """Interface for any model server that emits prometheus metrics"""

    prometheus_collector: Optional[PrometheusMetricsCollector] = None
