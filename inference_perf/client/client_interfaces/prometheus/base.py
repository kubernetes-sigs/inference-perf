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
from typing import Any, List, Optional

from pydantic import model_validator

from inference_perf.client.client_interfaces.prometheus.prometheus_metrics import PrometheusMetric
from inference_perf.config import PrometheusCollectorConfig, PrometheusMetricsReportConfig
from inference_perf.metrics.base import MetricCollector


class PrometheusMetricsCollector(MetricCollector[PrometheusMetric]):
    config: PrometheusCollectorConfig
    metrics: List[PrometheusMetric]

    @model_validator(mode="after")  # type: ignore[misc]
    def set_metric_urls(self) -> None:
        for metric in self.metrics:
            metric.set_target_url(self.config.url)

    async def to_report(self, report_config: PrometheusMetricsReportConfig, duration: float) -> dict[str, Any]:
        total_report = {}
        for metric in self.metrics:
            total_report[metric.name] = await metric.to_report(duration=duration)
        return total_report


class PrometheusEnabledModelServerClient:
    """Interface for any model server that emits prometheus metrics"""

    prometheus_collector: Optional[PrometheusMetricsCollector] = None
