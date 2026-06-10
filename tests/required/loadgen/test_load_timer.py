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
import numpy as np
import pytest

from inference_perf.config import Distribution, DistributionType
from inference_perf.loadgen.load_timer import IntervalLoadTimer


class TestIntervalLoadTimer:
    def test_fixed_interval_schedule(self) -> None:
        interval = Distribution(type=DistributionType.FIXED, mean=2.0, min=0, max=10)
        timer = IntervalLoadTimer(interval=interval, duration=10.0)
        # Gaps of exactly 2s fit requests at t=2,4,6,8,10
        assert timer.num_requests == 5
        times = list(timer.start_timer(initial=0.0))
        assert times == [2.0, 4.0, 6.0, 8.0, 10.0]

    def test_uniform_gaps_within_bounds(self) -> None:
        interval = Distribution(type=DistributionType.UNIFORM, mean=5.0, min=1, max=10)
        timer = IntervalLoadTimer(interval=interval, duration=600.0, rng=np.random.default_rng(42))
        times = list(timer.start_timer(initial=0.0))
        assert len(times) == timer.num_requests
        assert len(times) > 0
        gaps = np.diff([0.0] + times)
        assert gaps.min() >= 1.0
        assert gaps.max() <= 10.0
        assert times[-1] <= 600.0

    def test_deterministic_with_seeded_rng(self) -> None:
        interval = Distribution(type=DistributionType.UNIFORM, mean=5.0, min=1, max=10)
        timer1 = IntervalLoadTimer(interval=interval, duration=120.0, rng=np.random.default_rng(7))
        timer2 = IntervalLoadTimer(interval=interval, duration=120.0, rng=np.random.default_rng(7))
        assert list(timer1.start_timer(initial=0.0)) == list(timer2.start_timer(initial=0.0))

    def test_initial_offsets_schedule(self) -> None:
        interval = Distribution(type=DistributionType.FIXED, mean=1.0, min=0, max=10)
        timer = IntervalLoadTimer(interval=interval, duration=3.0)
        times = list(timer.start_timer(initial=100.0))
        assert times == [101.0, 102.0, 103.0]

    def test_gap_larger_than_duration_yields_nothing(self) -> None:
        interval = Distribution(type=DistributionType.FIXED, mean=30.0, min=0, max=60)
        timer = IntervalLoadTimer(interval=interval, duration=10.0)
        assert timer.num_requests == 0
        assert list(timer.start_timer(initial=0.0)) == []

    def test_zero_gaps_raise(self) -> None:
        interval = Distribution(type=DistributionType.FIXED, mean=0.0, min=0, max=10)
        with pytest.raises(ValueError, match="non-positive gaps"):
            IntervalLoadTimer(interval=interval, duration=10.0)
