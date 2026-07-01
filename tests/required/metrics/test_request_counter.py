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
from inference_perf.apis.base import ErrorResponseInfo, InferenceInfo, RequestLifecycleMetric
from inference_perf.metrics.request_counter import RequestSentCounter
from inference_perf.payloads import RequestMetrics, Text


def _metric(stage_id=None, failed=False) -> RequestLifecycleMetric:
    return RequestLifecycleMetric(
        stage_id=stage_id,
        scheduled_time=0.0,
        start_time=0.0,
        end_time=1.0,
        request_data="r",
        info=InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=1))),
        error=ErrorResponseInfo(error_type="Timeout", error_msg="boom") if failed else None,
    )


def test_counts_total_across_success_and_failure() -> None:
    counter = RequestSentCounter()
    counter.observe(_metric(stage_id=0, failed=False))
    counter.observe(_metric(stage_id=0, failed=True))

    # "requests sent" includes failures, since a failed request was still dispatched.
    assert counter.total == 2


def test_breaks_down_by_stage_and_status() -> None:
    metrics = [
        _metric(stage_id=0, failed=False),
        _metric(stage_id=0, failed=False),
        _metric(stage_id=0, failed=True),
        _metric(stage_id=1, failed=False),
    ]

    snapshot = RequestSentCounter.from_metrics(metrics).snapshot()

    assert snapshot["total"] == 4
    assert snapshot["by_stage_status"] == [
        {"stage_id": 0, "status": "failure", "count": 1},
        {"stage_id": 0, "status": "success", "count": 2},
        {"stage_id": 1, "status": "success", "count": 1},
    ]
