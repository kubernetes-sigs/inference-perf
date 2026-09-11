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
"""Contract tests for the load timers.

These pin the dispatch-schedule contract run_stage relies on (request count,
monotonicity, span), not the timers' internal interval distributions.
"""

import itertools
import unittest
from pathlib import Path
from unittest.mock import MagicMock

from inference_perf.loadgen.load_timer import ConstantLoadTimer, PoissonLoadTimer, TraceReplayLoadTimer


class TestConstantLoadTimer(unittest.TestCase):
    def test_emits_exactly_rate_times_duration_requests(self) -> None:
        timer = ConstantLoadTimer(rate=10, duration=5)
        times = list(timer.start_timer(initial=100.0))
        self.assertEqual(len(times), 50)

    def test_truncates_fractional_request_count(self) -> None:
        timer = ConstantLoadTimer(rate=2.5, duration=1)
        self.assertEqual(len(list(timer.start_timer(initial=0.0))), 2)

    def test_zero_requests_yields_nothing(self) -> None:
        timer = ConstantLoadTimer(rate=0.5, duration=1)
        self.assertEqual(list(timer.start_timer(initial=0.0)), [])

    def test_times_are_strictly_increasing_after_initial(self) -> None:
        initial = 1000.0
        timer = ConstantLoadTimer(rate=100, duration=2)
        times = list(timer.start_timer(initial=initial))
        self.assertTrue(all(t > initial for t in times))
        self.assertTrue(all(b > a for a, b in zip(times, times[1:], strict=False)))

    def test_schedule_spans_exactly_the_duration(self) -> None:
        # Intervals are normalized so the last dispatch lands at initial + duration.
        initial, duration = 50.0, 3.0
        timer = ConstantLoadTimer(rate=20, duration=duration)
        times = list(timer.start_timer(initial=initial))
        self.assertAlmostEqual(times[-1], initial + duration, places=6)


class TestPoissonLoadTimer(unittest.TestCase):
    def test_generator_is_unbounded_and_monotonic(self) -> None:
        initial = 10.0
        timer = PoissonLoadTimer(rate=10, duration=1)
        times = list(itertools.islice(timer.start_timer(initial=initial), 50))
        self.assertEqual(len(times), 50)
        self.assertTrue(all(t > initial for t in times))
        self.assertTrue(all(b > a for a, b in zip(times, times[1:], strict=False)))


class TestTraceReplayLoadTimer(unittest.TestCase):
    def test_offsets_trace_timestamps_from_initial(self) -> None:
        trace_reader = MagicMock()
        trace_reader.load_traces.return_value = iter([(0.0, 1, 2), (1.5, 3, 4), (4.0, 5, 6)])
        timer = TraceReplayLoadTimer(trace_reader=trace_reader, trace_file=Path("dummy.csv"))
        times = list(timer.start_timer(initial=100.0))
        self.assertEqual(times, [100.0, 101.5, 104.0])
        trace_reader.load_traces.assert_called_once_with(Path("dummy.csv"))


if __name__ == "__main__":
    unittest.main()
