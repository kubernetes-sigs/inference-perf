from abc import ABC
from typing import Optional

from inference_perf.client.client_interfaces.prometheus.prometheus_client import PrometheusMetricsCollector


class PrometheusEnabledModelServerClient(ABC):
    """Interface for any model server that emits prometheus metrics"""
    prometheus_collector: Optional[PrometheusMetricsCollector]