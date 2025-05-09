from abc import ABC
from typing import Any, List, Optional

from inference_perf.client.client_interfaces.prometheus.prometheus_metrics import PrometheusMetric
from inference_perf.config import PrometheusCollectorConfig, PrometheusMetricsReportConfig
from inference_perf.metrics.base import MetricCollector


class PrometheusMetricsCollector(MetricCollector[PrometheusMetric]):
    def __init__(self, config: PrometheusCollectorConfig, metrics: List[PrometheusMetric]) -> None:
        self.metrics = metrics
        self.config = config
        for metric in self.metrics:
            metric.set_target_url(self.config.url)

    async def to_report(self, report_config: PrometheusMetricsReportConfig, duration: float) -> dict[str, Any]:
        total_report = {}
        for metric in self.metrics:
            total_report[metric.name] = await metric.to_report(duration=duration)
        return total_report


class PrometheusEnabledModelServerClient(ABC):
    """Interface for any model server that emits prometheus metrics"""

    prometheus_collector: Optional[PrometheusMetricsCollector]
