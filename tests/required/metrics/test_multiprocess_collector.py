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

from unittest.mock import MagicMock, patch
import pytest

from inference_perf.apis import (
    InferenceInfo,
    RequestLifecycleMetric,
    UnaryResponseMetrics,
)
from inference_perf.circuit_breaker import _initialized_circuit_breakers
from inference_perf.metrics.request_collector.multiprocess import (
    MultiprocessRequestMetricCollector,
)
from inference_perf.payloads import RequestMetrics, Text


def _create_dummy_metric(req_id: int = 1) -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        stage_id=0,
        scheduled_time=0.0,
        start_time=1.0,
        end_time=2.0,
        request_data=f"request-{req_id}",
        response_data=f"response-{req_id}",
        info=InferenceInfo(
            request_metrics=RequestMetrics(text=Text(input_tokens=10)),
            response_metrics=UnaryResponseMetrics(output_tokens=5),
        ),
        error=None,
    )


@pytest.mark.asyncio
async def test_collector_lifecycle_start_context() -> None:
    collector = MultiprocessRequestMetricCollector()

    total_metrics = 10
    async with collector.start():
        for i in range(total_metrics):
            collector.record_metric(_create_dummy_metric(i))

    collected = collector.get_metrics()
    assert len(collected) == total_metrics
    assert [m.request_data for m in collected] == [f"request-{i}" for i in range(total_metrics)]


@pytest.mark.asyncio
async def test_collector_drains_mixed_batches_and_singletons() -> None:
    collector = MultiprocessRequestMetricCollector()

    m1 = _create_dummy_metric(1)
    m2 = _create_dummy_metric(2)
    m3 = _create_dummy_metric(3)

    # Put a list batch and a singleton into the queue directly
    collector.queue.put([m1, m2])
    collector.queue.put(m3)
    collector.queue.put(None)

    collected = await collector.collect_metrics()
    assert len(collected) == 3
    assert [m.request_data for m in collected] == ["request-1", "request-2", "request-3"]


@pytest.mark.asyncio
async def test_collector_bulk_drain_capacity() -> None:
    collector = MultiprocessRequestMetricCollector()

    num_items = 100
    for i in range(num_items):
        collector.record_metric(_create_dummy_metric(i))
    collector.queue.put(None)

    collected = await collector.collect_metrics()
    assert len(collected) == num_items


@pytest.mark.asyncio
async def test_collector_feeds_circuit_breakers_when_initialized() -> None:
    collector = MultiprocessRequestMetricCollector()
    m1 = _create_dummy_metric(1)

    mock_breaker = MagicMock()
    with patch.dict(_initialized_circuit_breakers, {"test_breaker": mock_breaker}):
        async with collector.start():
            collector.record_metric(m1)

    mock_breaker.feed.assert_called_once_with(m1)
