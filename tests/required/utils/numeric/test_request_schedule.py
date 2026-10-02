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
"""RequestSchedule: request arrival offsets drawn from a request_interval expression."""

import numpy as np
from numpy.typing import NDArray
import pytest

from inference_perf.utils.numeric.request_schedule import RequestSchedule
from inference_perf.utils.numeric.expression import Expression


# Builds a schedule from a request_interval string over `duration` seconds, drawing with a seeded rng.
def _schedule(request_interval: str, duration: float, seed: int = 0) -> RequestSchedule:
    expression = Expression(request_interval, allow_time=False, minimum=0)
    return RequestSchedule(expression, duration, rng=np.random.default_rng(seed))


# Recovers the gaps between consecutive arrivals, the first gap being measured from 0.
def _gaps(schedule: RequestSchedule) -> NDArray[np.float64]:
    return np.diff(np.concatenate(([0.0], schedule.offsets)))


# A fixed gap of 0.1s over 1s is exactly 10 requests at 0.1, 0.2, ... 1.0 (not 9, which is
# what summing 0.1 ten times in floating point would give), a mean of 10 per second.
# A gap of 0.3s over 1s fits 3 requests, at 0.3, 0.6 and 0.9.
def test_fixed_gap_is_evenly_spaced() -> None:
    tenth = _schedule("0.1", 1)
    assert tenth.expected_requests == 10
    assert tenth.offsets == pytest.approx([0.1 * k for k in range(1, 11)])
    assert tenth.mean_rate == 10.0
    assert _schedule("0.3", 1).offsets == pytest.approx([0.3, 0.6, 0.9])


# Exponential(10) gaps over 2000s (a Poisson process at 10 per second): about 20000
# requests, a mean gap near 0.1s, and a gap spread equal to the mean (ratio near 1).
# Offsets ascend, all fall inside the window, and the count matches the offsets.
def test_exponential_gaps_are_a_poisson_process() -> None:
    schedule = _schedule("Exponential(10)", 2000)
    gaps = _gaps(schedule)
    assert schedule.expected_requests == len(schedule.offsets)
    assert schedule.expected_requests == pytest.approx(20000, rel=0.05)
    assert schedule.mean_rate == pytest.approx(10, rel=0.05)
    assert float(gaps.mean()) == pytest.approx(0.1, rel=0.05)
    assert float(gaps.std() / gaps.mean()) == pytest.approx(1.0, abs=0.05)
    assert bool(np.all(gaps >= 0))
    assert 0 < float(schedule.offsets[0]) and float(schedule.offsets[-1]) <= 2000


# Gamma(0.25, 0.4) gaps have the same 0.1s mean as above but twice the spread (ratio 2):
# the same average load arriving in bursts. Uniform(0.05, 0.15) gaps never leave their bounds.
def test_gap_distribution_sets_the_burstiness() -> None:
    bursty = _gaps(_schedule("Gamma(0.25, 0.4)", 2000))
    assert float(bursty.mean()) == pytest.approx(0.1, rel=0.1)
    assert float(bursty.std() / bursty.mean()) == pytest.approx(2.0, abs=0.2)
    bounded = _gaps(_schedule("Uniform(0.05, 0.15)", 200))
    assert 0.05 <= float(bounded.min()) and float(bounded.max()) <= 0.15


# The same seed draws the same arrivals; a different seed draws different ones.
def test_draw_is_reproducible_from_the_seed() -> None:
    first = _schedule("Exponential(10)", 60, seed=3)
    assert np.array_equal(first.offsets, _schedule("Exponential(10)", 60, seed=3).offsets)
    assert not np.array_equal(first.offsets, _schedule("Exponential(10)", 60, seed=4).offsets)


# Rejected: gaps so small the stage would schedule over ten million requests
# (Exponential(1e9) over 60s), and a window that is not positive.
def test_unusable_schedules_are_rejected() -> None:
    with pytest.raises(ValueError, match="gaps are too small"):
        _schedule("Exponential(1e9)", 60)
    with pytest.raises(ValueError, match="duration must be positive"):
        _schedule("0.1", 0)
