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
import unittest

from inference_perf.apis import InferenceInfo, RequestLifecycleMetric
from inference_perf.metrics.request_collector import (
    LocalRequestMetricCollector,
    MultiprocessRequestMetricCollector,
)
from inference_perf.payloads import RequestMetrics, Text


def make_metric(stage_id: int) -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        stage_id=stage_id,
        scheduled_time=0.0,
        start_time=0.0,
        end_time=1.0,
        request_data="{}",
        info=InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=10))),
        error=None,
    )


class TestLocalCollectorSnapshot(unittest.TestCase):
    def test_snapshot_is_a_copy(self) -> None:
        collector = LocalRequestMetricCollector()
        collector.record_metric(make_metric(0))
        snapshot = collector.snapshot()
        self.assertEqual(len(snapshot), 1)
        snapshot.clear()
        self.assertEqual(len(collector.get_metrics()), 1)


class TestMultiprocessCollectorSnapshot(unittest.IsolatedAsyncioTestCase):
    async def test_snapshot_serves_partial_results_mid_run(self) -> None:
        collector = MultiprocessRequestMetricCollector()
        self.assertEqual(collector.snapshot(), [])

        async with collector.start():
            collector.record_metric(make_metric(-1))
            collector.record_metric(make_metric(-1))
            # The collector task drains the queue asynchronously; wait for it
            # the same way the sweep probe does between rungs.
            async with asyncio.timeout(10):
                while len(collector.snapshot()) < 2:
                    await asyncio.sleep(0.05)
            self.assertEqual({m.stage_id for m in collector.snapshot()}, {-1})

        self.assertEqual(len(collector.get_metrics()), 2)


if __name__ == "__main__":
    unittest.main()
