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

import urllib.request
from typing import Iterator

import pytest
from prometheus_client import CollectorRegistry, Counter

from inference_perf.observability.metrics import PrometheusMetricsServer
from inference_perf.observability.metrics.prometheus import DEFAULT_PORT


def _registry_with_counter() -> CollectorRegistry:
    registry = CollectorRegistry()
    Counter("test_scrapeable", "A counter the server should expose.", registry=registry)
    return registry


@pytest.fixture
def server() -> Iterator[PrometheusMetricsServer]:
    s = PrometheusMetricsServer(_registry_with_counter(), port=0)
    s.start()
    yield s
    s.stop()


def _scrape(port: int) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2) as resp:
        body: bytes = resp.read()
    return body.decode()


def test_default_port_constant() -> None:
    assert DEFAULT_PORT == 9464
    assert PrometheusMetricsServer(_registry_with_counter()).port == DEFAULT_PORT


def test_server_exposes_given_registry(server: PrometheusMetricsServer) -> None:
    assert server.bound_port is not None
    body = _scrape(server.bound_port)
    assert "test_scrapeable_total 0.0" in body


def test_double_start_raises(server: PrometheusMetricsServer) -> None:
    with pytest.raises(RuntimeError):
        server.start()


def test_stop_is_idempotent() -> None:
    s = PrometheusMetricsServer(_registry_with_counter(), port=0)
    s.start()
    s.stop()
    s.stop()
    assert s.bound_port is None
