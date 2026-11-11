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

import asyncio

from inference_perf.apis import (
    InferenceInfo,
    RequestLifecycleMetric,
    UnaryResponseMetrics,
)
from inference_perf.metrics.request_collector.local import LocalRequestMetricCollector
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


def test_snapshot_is_a_copy_of_the_metrics_recorded_so_far() -> None:
    """snapshot() returns what has been recorded, as a list the caller owns.

    With num_workers 0 this collector backs the stage-end reports (#817). A
    stage's reports are built from a snapshot while the run keeps recording,
    so the snapshot must not grow under them, and filtering or editing it must
    not change the list the end-of-run reports read through get_metrics().
    """
    collector = LocalRequestMetricCollector()
    for i in range(2):
        collector.record_metric(_create_dummy_metric(i))

    snapshot = collector.snapshot()
    assert [m.request_data for m in snapshot] == ["request-0", "request-1"]

    collector.record_metric(_create_dummy_metric(2))
    snapshot.clear()

    assert [m.request_data for m in collector.snapshot()] == ["request-0", "request-1", "request-2"]
    assert [m.request_data for m in collector.get_metrics()] == ["request-0", "request-1", "request-2"]


async def test_wait_for_recorded_is_a_no_op() -> None:
    """wait_for_recorded() returns at once: record_metric() appends in this process.

    Stage-end reports always await it before reading a stage's metrics, so it
    must exist here and must not cost the run any time at a stage boundary.
    """
    collector = LocalRequestMetricCollector()
    collector.record_metric(_create_dummy_metric(1))

    assert await asyncio.wait_for(collector.wait_for_recorded(0), timeout=1)

    assert [m.request_data for m in collector.snapshot()] == ["request-1"]
