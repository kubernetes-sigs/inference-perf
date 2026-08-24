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
"""Unit tests for the individual Trigger implementations.

Neither trigger reads the wall clock itself -- RateOverWindow keys its window
eviction off each HitSample's own `ts` -- so every test here drives state via
hand-constructed HitSample timelines instead of sleep().
"""

from datetime import datetime, timedelta

from inference_perf.circuit_breaker.triggers import HitSample
from inference_perf.circuit_breaker.triggers.consecutive import Consecutive
from inference_perf.circuit_breaker.triggers.rate_over_window import RateOverWindow

BASE = datetime(2026, 1, 1, 0, 0, 0)


def _hit(hit: int) -> HitSample:
    return HitSample(BASE, hit)


# --- Consecutive -----------------------------------------------------------


def test_consecutive_below_threshold_does_not_fire() -> None:
    trigger = Consecutive(threshold=3)
    trigger.update(_hit(1))
    trigger.update(_hit(1))

    assert trigger.fired() is False


def test_consecutive_at_threshold_fires() -> None:
    trigger = Consecutive(threshold=3)
    trigger.update(_hit(1))
    trigger.update(_hit(1))
    trigger.update(_hit(1))

    assert trigger.fired() is True


def test_consecutive_threshold_one_fires_on_first_hit() -> None:
    trigger = Consecutive(threshold=1)
    trigger.update(_hit(1))

    assert trigger.fired() is True


def test_consecutive_miss_resets_counter() -> None:
    trigger = Consecutive(threshold=3)
    trigger.update(_hit(1))
    trigger.update(_hit(1))
    trigger.update(_hit(0))  # miss resets the streak
    trigger.update(_hit(1))
    trigger.update(_hit(1))

    assert trigger.fired() is False


def test_consecutive_reset_fully_clears_state() -> None:
    trigger = Consecutive(threshold=2)
    trigger.update(_hit(1))
    trigger.update(_hit(1))
    assert trigger.fired() is True

    trigger.reset()
    assert trigger.fired() is False

    # A single hit after reset must not immediately re-fire a threshold=2 trigger --
    # the counter itself, not just the fired flag, needs to have been cleared.
    trigger.update(_hit(1))
    assert trigger.fired() is False
    trigger.update(_hit(1))
    assert trigger.fired() is True


# --- RateOverWindow ----------------------------------------------------------


def test_rate_over_window_below_threshold_does_not_fire() -> None:
    trigger = RateOverWindow(window_sec=10, threshold=0.75, min_samples=0)
    trigger.update(HitSample(BASE, 0))
    trigger.update(HitSample(BASE + timedelta(seconds=1), 0))
    trigger.update(HitSample(BASE + timedelta(seconds=2), 1))
    # 1/3 hit rate stays below 0.75 throughout.

    assert trigger.fired() is False


def test_rate_over_window_above_threshold_fires() -> None:
    trigger = RateOverWindow(window_sec=10, threshold=0.5, min_samples=0)
    trigger.update(HitSample(BASE, 0))
    trigger.update(HitSample(BASE + timedelta(seconds=1), 1))
    # 1/2 hit rate crosses the 0.5 threshold.

    assert trigger.fired() is True


def test_rate_over_window_min_samples_gates_firing() -> None:
    trigger = RateOverWindow(window_sec=10, threshold=0.5, min_samples=3)
    trigger.update(HitSample(BASE, 1))
    trigger.update(HitSample(BASE + timedelta(seconds=1), 1))
    # Rate is already 1.0, but only 2 samples so far; min_samples=3 gates it.
    assert trigger.fired() is False

    trigger.update(HitSample(BASE + timedelta(seconds=2), 1))
    assert trigger.fired() is True


def test_rate_over_window_evicts_samples_outside_window() -> None:
    trigger = RateOverWindow(window_sec=10, threshold=0.5, min_samples=0)
    # Two early misses keep the rate under threshold ...
    trigger.update(HitSample(BASE, 0))
    trigger.update(HitSample(BASE + timedelta(seconds=1), 0))
    trigger.update(HitSample(BASE + timedelta(seconds=2), 1))
    assert trigger.fired() is False

    # ... until they age out of the 10s window, leaving only hits behind.
    trigger.update(HitSample(BASE + timedelta(seconds=12), 1))
    assert trigger.fired() is True


def test_rate_over_window_threshold_zero_fires_on_first_sample() -> None:
    # threshold=0.0 is a value the config explicitly permits (TriggerRateOverWindow.threshold
    # has ge=0.0). Since rate >= 0.0 is always true once min_samples is satisfied, this fires
    # on the very first sample regardless of hit value. Pinned here as documented behavior.
    trigger = RateOverWindow(window_sec=10, threshold=0.0, min_samples=0)
    trigger.update(HitSample(BASE, 0))

    assert trigger.fired() is True


def test_rate_over_window_reset_clears_state() -> None:
    trigger = RateOverWindow(window_sec=10, threshold=0.5, min_samples=0)
    trigger.update(HitSample(BASE, 1))
    trigger.update(HitSample(BASE + timedelta(seconds=1), 1))
    assert trigger.fired() is True

    trigger.reset()

    assert trigger.fired() is False
    assert len(trigger.buf) == 0
