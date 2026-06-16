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
"""Unit tests for the pure closed-loop saturation estimator.

These are deterministic and require no cluster, queue or event loop config:
async helpers are driven via ``asyncio.run``.
"""

import asyncio
from typing import Awaitable, Callable, List, Tuple

import numpy as np
import pytest

from inference_perf.loadgen.saturation import (
    ProbeResult,
    SaturationConfig,
    batch_means_ci,
    concurrency_ladder,
    find_saturation,
    mser5_truncation,
    throughput_from_counter_series,
)


def _counter_series(rates_per_interval: List[float], dt: float = 0.25) -> List[Tuple[float, int]]:
    """Build a (timestamp, cumulative_completed) series from per-interval rates."""
    samples: List[Tuple[float, int]] = [(0.0, 0)]
    t = 0.0
    c = 0.0
    for r in rates_per_interval:
        t += dt
        c += r * dt
        samples.append((t, int(round(c))))
    return samples


# --------------------------------------------------------------------------- #
# mser5_truncation
# --------------------------------------------------------------------------- #


def test_mser5_discards_warmup_ramp() -> None:
    # 20 low warm-up samples then 80 steady samples: cutoff should land at ~20.
    values = np.array([1.0] * 20 + [50.0] * 80)
    cutoff = mser5_truncation(values)
    assert 15 <= cutoff <= 25


def test_mser5_no_warmup_keeps_most() -> None:
    rng = np.random.default_rng(0)
    values = 50.0 + rng.normal(0, 1.0, size=100)
    cutoff = mser5_truncation(values)
    # Stationary from the start: should not discard a large prefix.
    assert cutoff <= 25


def test_mser5_short_series_is_safe() -> None:
    assert mser5_truncation(np.array([1.0, 2.0, 3.0])) >= 0


# --------------------------------------------------------------------------- #
# batch_means_ci
# --------------------------------------------------------------------------- #


def test_batch_means_constant_has_zero_width() -> None:
    mean, hw, nb = batch_means_ci(np.array([50.0] * 100))
    assert mean == 50.0
    assert hw == 0.0
    assert nb >= 2


def test_batch_means_brackets_true_mean() -> None:
    rng = np.random.default_rng(1)
    rates = 50.0 + rng.normal(0, 3.0, size=400)
    mean, hw, nb = batch_means_ci(rates, min_batches=20)
    assert abs(mean - 50.0) < hw + 1.0
    assert 0.0 < hw < 5.0
    assert nb == 20


def test_batch_means_too_few_samples_infinite_ci() -> None:
    mean, hw, nb = batch_means_ci(np.array([42.0]))
    assert mean == 42.0
    assert hw == float("inf")


# --------------------------------------------------------------------------- #
# throughput_from_counter_series
# --------------------------------------------------------------------------- #


def test_throughput_recovers_steady_rate_after_warmup() -> None:
    rng = np.random.default_rng(2)
    warmup = [10.0] * 12  # slow start (queue filling)
    steady = list(50.0 + rng.normal(0, 2.0, size=120))  # steady 50/s
    samples = _counter_series(warmup + steady)
    tput, hw, nb = throughput_from_counter_series(samples)
    assert abs(tput - 50.0) < 3.0
    assert hw < 5.0
    assert nb >= 2


def test_throughput_two_samples_fallback() -> None:
    tput, hw, nb = throughput_from_counter_series([(0.0, 0), (2.0, 100)])
    assert tput == 50.0
    assert hw == float("inf")


def test_throughput_empty_is_zero() -> None:
    tput, hw, nb = throughput_from_counter_series([])
    assert tput == 0.0


# --------------------------------------------------------------------------- #
# concurrency_ladder
# --------------------------------------------------------------------------- #


def test_ladder_is_geometric_and_includes_ceiling() -> None:
    cfg = SaturationConfig(min_concurrency=1, ladder_growth=2.0, max_concurrency=1024)
    ladder = concurrency_ladder(cfg)
    assert ladder[0] == 1
    assert ladder[-1] == 1024
    assert ladder == sorted(set(ladder))  # strictly increasing, no dupes
    assert [1, 2, 4, 8, 16] == ladder[:5]


def test_ladder_min_equals_max() -> None:
    cfg = SaturationConfig(min_concurrency=4, max_concurrency=4)
    assert concurrency_ladder(cfg) == [4]


# --------------------------------------------------------------------------- #
# find_saturation
# --------------------------------------------------------------------------- #


def _saturating_probe(mu: float, n_half: float) -> Callable[[int], Awaitable[ProbeResult]]:
    """A noise-free probe whose throughput follows X(N) = mu*N/(N+n_half)."""

    async def probe(n: int) -> ProbeResult:
        x = mu * n / (n + n_half)
        return ProbeResult(concurrency=n, throughput=x, ci_half_width=0.0, n_batches=20, converged=True)

    return probe


def test_find_saturation_recovers_mu_and_stops_early() -> None:
    cfg = SaturationConfig(plateau_rel_gain=0.05, max_concurrency=1024)
    result = asyncio.run(find_saturation(_saturating_probe(mu=100.0, n_half=10.0), cfg))
    # mu_max is approached from below; plateau detection lands within ~10%.
    assert 90.0 <= result.max_throughput <= 100.0
    assert not result.harness_limited
    # Should have plateaued well before the 1024 ceiling.
    assert result.probes[-1].concurrency < 1024


def test_find_saturation_flags_harness_limited() -> None:
    # Ceiling far below where this server would plateau: still climbing at max.
    cfg = SaturationConfig(plateau_rel_gain=0.05, max_concurrency=8)
    result = asyncio.run(find_saturation(_saturating_probe(mu=10000.0, n_half=5000.0), cfg))
    assert result.harness_limited
    assert result.probes[-1].concurrency == 8


def test_find_saturation_knee_is_smallest_near_mu() -> None:
    cfg = SaturationConfig(plateau_rel_gain=0.05, max_concurrency=1024)
    result = asyncio.run(find_saturation(_saturating_probe(mu=100.0, n_half=10.0), cfg))
    threshold = result.max_throughput * 0.95
    # Every probe below the knee must be under threshold; the knee meets it.
    knee_probe = next(p for p in result.probes if p.concurrency == result.knee_concurrency)
    assert knee_probe.throughput >= threshold


def _table_probe(table: dict[int, Tuple[float, float, bool]]) -> Callable[[int], Awaitable[ProbeResult]]:
    """Probe returning a scripted ``(throughput, ci_half_width, converged)`` per N."""

    async def probe(n: int) -> ProbeResult:
        tput, ci, conv = table[n]
        return ProbeResult(concurrency=n, throughput=tput, ci_half_width=ci, n_batches=20, converged=conv)

    return probe


def test_find_saturation_does_not_stop_on_a_single_noisy_dip() -> None:
    # N=16 reads LOW (85 < prior best 90) but with a wide CI: the old
    # single-delta rule would have stopped there and reported ~90. The
    # upper-bound rule must keep climbing and recover the true plateau (~112).
    table = {
        1: (20.0, 2.0, True),
        2: (40.0, 2.0, True),
        4: (70.0, 2.0, True),
        8: (90.0, 2.0, True),
        16: (85.0, 15.0, False),  # noisy dip, wide CI
        32: (110.0, 2.0, True),
        64: (112.0, 2.0, True),  # plateau vs best 110
    }
    cfg = SaturationConfig(plateau_rel_gain=0.05, max_concurrency=64)
    result = asyncio.run(find_saturation(_table_probe(table), cfg))
    assert result.max_throughput >= 110.0  # did not get stuck at the dip
    assert {p.concurrency for p in result.probes} >= {32, 64}  # climbed past the dip
    assert not result.harness_limited
    assert result.confident


def test_find_saturation_not_confident_when_deciding_probe_unconverged() -> None:
    # Plateau is reached, but the deciding probe never met its precision target:
    # mu_max still stands, but the result is flagged approximate.
    table = {
        1: (20.0, 1.0, True),
        2: (40.0, 1.0, True),
        4: (70.0, 1.0, True),
        8: (95.0, 1.0, True),
        16: (97.0, 1.0, False),  # within plateau margin of 95, but not converged
    }
    cfg = SaturationConfig(plateau_rel_gain=0.05, max_concurrency=16)
    result = asyncio.run(find_saturation(_table_probe(table), cfg))
    assert not result.harness_limited
    assert not result.confident
    assert 95.0 <= result.max_throughput <= 97.0


def test_saturation_config_validates() -> None:
    for kwargs in (
        {"ladder_growth": 1.0},
        {"min_concurrency": 0},
        {"max_concurrency": 0, "min_concurrency": 1},
        {"relative_precision": 0.0},
        {"plateau_rel_gain": 1.0},
        # Precision must be strictly finer than the plateau threshold (coupling),
        # even when each value is individually in range.
        {"relative_precision": 0.05, "plateau_rel_gain": 0.04},
        {"relative_precision": 0.05, "plateau_rel_gain": 0.05},
    ):
        with pytest.raises(ValueError):
            SaturationConfig(**kwargs)
