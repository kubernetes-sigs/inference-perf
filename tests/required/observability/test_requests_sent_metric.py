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

from inference_perf.apis.base import ErrorResponseInfo, InferenceInfo, RequestLifecycleMetric
from inference_perf.metrics.request_collector import LocalRequestMetricCollector
from inference_perf.observability.metrics import PrometheusMetricsServer
from inference_perf.payloads import RequestMetrics, Text


def _metric(stage_id: int, failed: bool = False) -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        stage_id=stage_id,
        scheduled_time=0.0,
        start_time=0.0,
        end_time=1.0,
        request_data="r",
        info=InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=1))),
        error=ErrorResponseInfo(error_type="Timeout", error_msg="boom") if failed else None,
    )


def _scrape(port: int) -> str:
    with urllib.request.urlopen(f"http://127.0.0.1:{port}/metrics", timeout=2) as resp:
        body: bytes = resp.read()
    return body.decode()


@pytest.fixture
def server() -> Iterator[PrometheusMetricsServer]:
    s = PrometheusMetricsServer(port=0)
    s.observe_request(_metric(stage_id=0, failed=False))
    s.observe_request(_metric(stage_id=0, failed=False))
    s.observe_request(_metric(stage_id=0, failed=True))
    s.start()
    yield s
    s.stop()


def test_exposes_requests_sent_total_with_labels(server: PrometheusMetricsServer) -> None:
    assert server.bound_port is not None
    body = _scrape(server.bound_port)

    assert 'inference_perf_requests_sent_total{stage="0",status="success"} 2.0' in body
    assert 'inference_perf_requests_sent_total{stage="0",status="failure"} 1.0' in body


def test_reflects_live_updates(server: PrometheusMetricsServer) -> None:
    # The counter is incremented live, so post-start observations show up.
    assert server.bound_port is not None
    server.observe_request(_metric(stage_id=1, failed=False))

    body = _scrape(server.bound_port)
    assert 'inference_perf_requests_sent_total{stage="1",status="success"} 1.0' in body


def test_collector_observer_feeds_counter() -> None:
    # End to end: recording a metric on the collector increments the served counter.
    server = PrometheusMetricsServer(port=0)
    collector = LocalRequestMetricCollector()
    collector.add_observer(server.observe_request)

    collector.record_metric(_metric(stage_id=0, failed=False))
    collector.record_metric(_metric(stage_id=0, failed=True))

    server.start()
    try:
        assert server.bound_port is not None
        body = _scrape(server.bound_port)
        assert 'inference_perf_requests_sent_total{stage="0",status="success"} 1.0' in body
        assert 'inference_perf_requests_sent_total{stage="0",status="failure"} 1.0' in body
    finally:
        server.stop()
