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
import unittest
from typing import List, Optional

import numpy as np

from inference_perf.apis import (
    ErrorResponseInfo,
    InferenceInfo,
    RequestLifecycleMetric,
    StreamedResponseMetrics,
)
from inference_perf.loadgen.probe import LatencyProfile, SaturationSignal, latency_profile, rung_from_metrics
from inference_perf.payloads import RequestMetrics, Text


def make_metric(
    start_time: float,
    end_time: float,
    token_times: Optional[List[float]] = None,
    error: Optional[ErrorResponseInfo] = None,
    stage_id: int = -1,
) -> RequestLifecycleMetric:
    info = InferenceInfo(request_metrics=RequestMetrics(text=Text(input_tokens=10)))
    if token_times is not None:
        info.response_metrics = StreamedResponseMetrics(output_token_times=token_times)
    return RequestLifecycleMetric(
        stage_id=stage_id,
        scheduled_time=start_time,
        start_time=start_time,
        end_time=end_time,
        request_data="{}",
        info=info,
        error=error,
    )


def streamed_metric(start_time: float, ttft: float, itl: float, num_tokens: int = 3) -> RequestLifecycleMetric:
    token_times = [start_time + ttft + i * itl for i in range(num_tokens)]
    return make_metric(start_time, token_times[-1], token_times=token_times)


class TestLatencyProfile(unittest.TestCase):
    def test_medians_of_streamed_requests(self) -> None:
        metrics = [
            streamed_metric(0.0, ttft=0.1, itl=0.01),
            streamed_metric(1.0, ttft=0.2, itl=0.02),
            streamed_metric(2.0, ttft=0.3, itl=0.03),
        ]
        profile = latency_profile(metrics)
        self.assertIsNotNone(profile)
        assert profile is not None
        self.assertAlmostEqual(profile.ttft, 0.2)
        self.assertAlmostEqual(profile.itl, 0.02)

    def test_no_streamed_requests_is_none(self) -> None:
        self.assertIsNone(latency_profile([make_metric(0.0, 0.5)]))

    def test_single_token_time_is_ignored(self) -> None:
        self.assertIsNone(latency_profile([make_metric(0.0, 0.5, token_times=[0.4])]))


class TestRungFromMetrics(unittest.TestCase):
    def test_littles_law_consistency(self) -> None:
        # Closed loop at N=5 with R=0.5s implies X=10/s; the residual must vanish.
        metrics = [make_metric(end - 0.5, float(end)) for end in np.arange(0.0, 10.0, 0.1)]
        rung = rung_from_metrics(5, metrics, 0.0, 10.0)
        self.assertAlmostEqual(rung.throughput, 10.0)
        self.assertAlmostEqual(rung.latency, 0.5)
        self.assertAlmostEqual(rung.littles_law_residual, 0.0)
        self.assertIs(rung.signal, SaturationSignal.NONE)

    def test_errors_do_not_count_toward_throughput(self) -> None:
        error = ErrorResponseInfo(error_type="server", error_msg="boom")
        metrics = [make_metric(end - 0.5, float(end)) for end in np.arange(0.0, 10.0, 0.1)]
        metrics += [make_metric(end - 0.01, float(end), error=error) for end in np.arange(0.05, 10.0, 0.1)]
        rung = rung_from_metrics(5, metrics, 0.0, 10.0)
        self.assertAlmostEqual(rung.throughput, 10.0)

    def test_empty_window_raises(self) -> None:
        with self.assertRaises(ValueError):
            rung_from_metrics(1, [make_metric(99.0, 100.0)], 0.0, 10.0)

    def test_prefill_inflation_classified(self) -> None:
        baseline = LatencyProfile(ttft=0.1, itl=0.01)
        metrics = [streamed_metric(float(start), ttft=0.5, itl=0.011) for start in np.arange(0.0, 10.0, 0.1)]
        rung = rung_from_metrics(4, metrics, 0.0, 11.0, baseline=baseline)
        self.assertIs(rung.signal, SaturationSignal.PREFILL_BOUND)

    def test_decode_inflation_classified(self) -> None:
        baseline = LatencyProfile(ttft=0.1, itl=0.01)
        metrics = [streamed_metric(float(start), ttft=0.11, itl=0.05) for start in np.arange(0.0, 10.0, 0.1)]
        rung = rung_from_metrics(4, metrics, 0.0, 11.0, baseline=baseline)
        self.assertIs(rung.signal, SaturationSignal.DECODE_BOUND)

    def test_uninflated_rung_stays_none(self) -> None:
        baseline = LatencyProfile(ttft=0.1, itl=0.01)
        metrics = [streamed_metric(float(start), ttft=0.1, itl=0.01) for start in np.arange(0.0, 10.0, 0.1)]
        rung = rung_from_metrics(4, metrics, 0.0, 11.0, baseline=baseline)
        self.assertIs(rung.signal, SaturationSignal.NONE)

    def test_non_streaming_with_baseline_stays_none(self) -> None:
        # Without token timestamps the inflation ratios are unavailable, so
        # classification degrades to NONE rather than guessing.
        baseline = LatencyProfile(ttft=0.1, itl=0.01)
        metrics = [make_metric(end - 2.0, float(end)) for end in np.arange(0.0, 10.0, 0.1)]
        rung = rung_from_metrics(4, metrics, 0.0, 10.0, baseline=baseline)
        self.assertIs(rung.signal, SaturationSignal.NONE)


if __name__ == "__main__":
    unittest.main()
