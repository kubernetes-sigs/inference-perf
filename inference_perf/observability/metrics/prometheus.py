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

"""Prometheus exposition surface for inference-perf runtime observability.

Serves a caller-provided ``CollectorRegistry`` over an HTTP ``/metrics``
endpoint. Which metrics exist is decided by
:func:`inference_perf.observability.metrics.registry.build_metrics`, not here.
"""

from __future__ import annotations

from threading import Thread
from typing import Optional
from wsgiref.simple_server import WSGIServer

from prometheus_client import CollectorRegistry, start_http_server

DEFAULT_PORT = 9464


class PrometheusMetricsServer:
    """Serves a ``CollectorRegistry`` over an HTTP ``/metrics`` endpoint.

    Pass the registry of a :class:`~inference_perf.observability.metrics.registry.MetricsHub`
    built by ``build_metrics`` (a fresh registry per run, never the global
    default one, so multiple runs in the same process do not collide).

    Pass ``port=0`` to bind to an ephemeral port; the actual port is then
    available via :attr:`bound_port`.
    """

    def __init__(self, registry: CollectorRegistry, port: int = DEFAULT_PORT, addr: str = "0.0.0.0") -> None:
        self.registry = registry
        self.port = port
        self.addr = addr
        self._server: Optional[WSGIServer] = None
        self._thread: Optional[Thread] = None

    def start(self) -> None:
        if self._server is not None:
            raise RuntimeError("PrometheusMetricsServer is already running")
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
