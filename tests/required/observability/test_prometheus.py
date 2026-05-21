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

import time
import urllib.request

import pytest

from inference_perf.observability.metrics import PrometheusMetricsServer
from inference_perf.observability.metrics.prometheus import DEFAULT_PORT


@pytest.fixture
def server():
    s = PrometheusMetricsServer(port=0)
    s.start()
    yield s
    s.stop()


def _scrape(port: int) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2) as resp:
        return resp.read().decode()


def _parse_elapsed(body: str) -> float:
    for line in body.splitlines():
        if line.startswith("inference_perf_run_elapsed_seconds "):
            return float(line.split()[1])
    raise AssertionError(f"inference_perf_run_elapsed_seconds not found in:\n{body}")


def test_default_port_constant() -> None:
    assert DEFAULT_PORT == 9464


def test_server_starts_and_exposes_elapsed_gauge(server: PrometheusMetricsServer) -> None:
    assert server.bound_port is not None
    body = _scrape(server.bound_port)
    assert "inference_perf_run_elapsed_seconds" in body
    assert _parse_elapsed(body) >= 0.0


def test_elapsed_gauge_advances_between_scrapes(server: PrometheusMetricsServer) -> None:
    assert server.bound_port is not None
    first = _parse_elapsed(_scrape(server.bound_port))
    time.sleep(0.1)
    second = _parse_elapsed(_scrape(server.bound_port))
    assert second > first


def test_double_start_raises(server: PrometheusMetricsServer) -> None:
    with pytest.raises(RuntimeError):
        server.start()


def test_stop_is_idempotent() -> None:
    s = PrometheusMetricsServer(port=0)
    s.start()
    s.stop()
    s.stop()
    assert s.bound_port is None
