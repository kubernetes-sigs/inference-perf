# Copyright 2026 The Kubernetes Authors.
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

"""Minimal Prometheus exposition surface for inference-perf runtime observability.

Seed of kubernetes-sigs/inference-perf#489. Exposes
``inference_perf_run_elapsed_seconds`` and
``inference_perf_requests_sent_total`` over an HTTP ``/metrics`` endpoint.
The full metric set and naming conventions are intentionally out of scope here;
expect this surface to grow once the conventions are agreed upon.
"""

from __future__ import annotations

import time
from threading import Thread
from typing import Iterator, Optional
from wsgiref.simple_server import WSGIServer

from prometheus_client import CollectorRegistry, Counter, start_http_server
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector

from inference_perf.apis import RequestLifecycleMetric

DEFAULT_PORT = 9464


class _RunElapsedCollector(Collector):
    def __init__(self) -> None:
        self._start_time: Optional[float] = None

    def mark_start(self) -> None:
        self._start_time = time.monotonic()

    def collect(self) -> Iterator[GaugeMetricFamily]:
        elapsed = 0.0 if self._start_time is None else time.monotonic() - self._start_time
        yield GaugeMetricFamily(
            "inference_perf_run_elapsed_seconds",
            "Wall-clock seconds elapsed since the metrics server started.",
            value=elapsed,
        )


class PrometheusMetricsServer:
    """Serves inference-perf's own metrics over an HTTP ``/metrics`` endpoint.

    Emits ``inference_perf_run_elapsed_seconds`` and
    ``inference_perf_requests_sent_total``. Use a fresh ``CollectorRegistry``
    per instance (not the default global one) so multiple runs in the same
    process do not collide.

    Pass ``port=0`` to bind to an ephemeral port; the actual port is then
    available via :attr:`bound_port`.
    """

    def __init__(self, port: int = DEFAULT_PORT, addr: str = "0.0.0.0") -> None:
        self.port = port
        self.addr = addr
        self.registry = CollectorRegistry()
        self._elapsed = _RunElapsedCollector()
        self.registry.register(self._elapsed)
        self._requests_sent = Counter(
            "inference_perf_requests_sent",
            "Total number of requests sent to the model server under test.",
            labelnames=["stage", "status"],
            registry=self.registry,
        )
        self._server: Optional[WSGIServer] = None
        self._thread: Optional[Thread] = None

    def observe_request(self, metric: RequestLifecycleMetric) -> None:
        """Count one request toward ``inference_perf_requests_sent_total``.

        Wire this as an observer on the request metric collector
        (``collector.add_observer(server.observe_request)``) so every collected
        request lifecycle metric increments the counter.
        """
        stage = "" if metric.stage_id is None else str(metric.stage_id)
        status = "failure" if metric.error is not None else "success"
        self._requests_sent.labels(stage, status).inc()

    def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("PrometheusMetricsServer is already running")
        self._elapsed.mark_start()
        self._server, self._thread = start_http_server(
            port=self.port,
            addr=self.addr,
            registry=self.registry,
        )

    def stop(self) -> None:
        if self._server is None:
            return
        self._server.shutdown()
        self._server = None
        self._thread = None

    @property
    def bound_port(self) -> Optional[int]:
        if self._server is None:
            return None
        return self._server.server_address[1]
