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
"""Closed-loop saturation estimation.

This module contains the *pure*, transport-agnostic math used to locate a model
server's maximum sustainable throughput (``mu_max``) by probing it at a ladder
of fixed concurrency levels and finding where the throughput-vs-concurrency
curve flattens.

It deliberately knows nothing about multiprocessing, queues or HTTP. A caller
supplies an async ``probe`` callable that, given a concurrency level ``N``,
runs the server at exactly ``N`` in-flight requests and returns a steady-state
throughput estimate (a :class:`ProbeResult`). Everything here is therefore unit
testable with a synthetic probe and no cluster.

Design notes (why this shape):

* Saturation is a property of a *trend*, not a fixed-length window. We measure
  at fixed concurrency (closed loop) so every probe reaches a stationary steady
  state; the per-probe duration is determined by a precision target, not a
  hardcoded constant.
* ``mu_max`` is the height of the throughput plateau, recovered model-free by
  walking the concurrency ladder until doubling concurrency stops buying
  meaningful throughput. No retrograde/USL peak fit is required (and that fit is
  ill-conditioned for the flat plateaus continuous-batching servers produce).
* The only tunables are statistically meaningful (target precision, confidence,
  plateau gain threshold). Mis-setting them trades cost for precision; they do
  not silently bias the estimate.
"""

from dataclasses import dataclass, field
from typing import Awaitable, Callable, List, Tuple

import numpy as np

# Student-t 0.975 quantiles (two-sided 95% CI) for small degrees of freedom.
# Beyond the table we fall back to the normal approximation (1.96), which is
# within ~3% by df=30. Kept inline to avoid a scipy dependency (numpy only).
_T_0975 = {
    1: 12.706,
    2: 4.303,
    3: 3.182,
    4: 2.776,
    5: 2.571,
    6: 2.447,
    7: 2.365,
    8: 2.306,
    9: 2.262,
    10: 2.228,
    12: 2.179,
    15: 2.131,
    20: 2.086,
    25: 2.060,
    30: 2.042,
}


def _t_critical_95(df: int) -> float:
    """Two-sided 95% Student-t critical value for ``df`` degrees of freedom."""
    if df <= 0:
        return float("inf")
    if df in _T_0975:
        return _T_0975[df]
    # Nearest tabulated df at or below, else normal approximation.
    candidates = [k for k in _T_0975 if k <= df]
    if df >= 30:
        return 1.96
    return _T_0975[max(candidates)] if candidates else 12.706


@dataclass
class ProbeResult:
    """Outcome of running the server at a fixed concurrency level."""

    concurrency: int
    throughput: float  # steady-state completions/sec (request rate)
    ci_half_width: float  # absolute half-width of the CI on ``throughput``
    n_batches: int  # number of independent batches the CI is based on
    converged: bool  # whether the precision target was met before the budget

    @property
    def relative_ci(self) -> float:
        if self.throughput <= 0:
            return float("inf")
        return self.ci_half_width / self.throughput


@dataclass
class SaturationConfig:
    """Internal tunables for the closed-loop search.

    These are not user-facing: the public ``SweepConfig`` exposes only intent
    (how many stages, how long, how far past saturation). The defaults here are
    chosen so the search is robust without configuration; they trade cost for
    precision but never silently bias the estimate.
    """

    # Per-probe steady-state precision target: a probe is "converged" once its
    # CI half-width is within this fraction of the mean throughput.
    relative_precision: float = 0.025
    # Stop climbing the ladder once raising concurrency buys less than this
    # fraction of throughput (the plateau / marginal-gain threshold). Must be
    # strictly coarser than ``relative_precision`` so a plateau is decided
    # outside the measurement noise band, never on noise.
    plateau_rel_gain: float = 0.05
    # Concurrency ladder: start, geometric growth factor, and the hard ceiling
    # imposed by the load generator (num_workers * worker_max_concurrency,
    # capped by the connection pool). The ceiling is a *detectable* limit.
    min_concurrency: int = 1
    ladder_growth: float = 2.0
    max_concurrency: int = 1024
    # Minimum independent batches required before a CI is trusted.
    min_batches: int = 20
    # Per-probe budget: completions per concurrency slot before a probe yields
    # (upper bound on probe work), and a hard wall-clock cap per probe.
    requests_per_slot: int = 40
    max_probe_seconds: float = 120.0

    def __post_init__(self) -> None:
        if self.ladder_growth <= 1.0:
            raise ValueError("ladder_growth must be > 1.0")
        if self.min_concurrency < 1:
            raise ValueError("min_concurrency must be >= 1")
        if self.max_concurrency < self.min_concurrency:
            raise ValueError("max_concurrency must be >= min_concurrency")
        if not (0.0 < self.relative_precision < 1.0):
            raise ValueError("relative_precision must be in (0, 1)")
        if not (0.0 < self.plateau_rel_gain < 1.0):
            raise ValueError("plateau_rel_gain must be in (0, 1)")
        if self.relative_precision >= self.plateau_rel_gain:
            # If the precision floor is as coarse as the plateau threshold, a
            # "negligible gain" is indistinguishable from measurement noise.
            raise ValueError("relative_precision must be strictly less than plateau_rel_gain")


@dataclass
class SaturationResult:
    max_throughput: float  # mu_max estimate (completions/sec)
    knee_concurrency: int  # smallest probed N reaching ~mu_max
    harness_limited: bool  # ladder hit max_concurrency while still climbing
    confident: bool = False  # plateau confirmed by a probe that met its precision target
    probes: List[ProbeResult] = field(default_factory=list)


def mser5_truncation(values: np.ndarray) -> int:
    """Return a warm-up truncation index using the MSER-5 rule.

    MSER-5 batches the series into non-overlapping groups of 5, then picks the
    truncation point that minimises the standard error of the mean of the
    retained batches: ``Var(retained) / len(retained)``. This finds the end of
    the initial transient (queue filling from empty) without assuming a fixed
    warm-up duration. Returns an index into ``values`` (number of leading
    samples to discard).
    """
    n = len(values)
    if n < 10:
        # Too short to batch meaningfully; discard a small fixed fraction.
        return n // 5

    batch = 5
    n_batches = n // batch
    batched = values[: n_batches * batch].reshape(n_batches, batch).mean(axis=1)

    best_d = 0
    best_stat = float("inf")
    # Never truncate more than half the batches away (standard MSER guard).
    for d in range(0, n_batches - 1):
        retained = batched[d:]
        if len(retained) < 2:
            break
        m = retained.mean()
        # Sum of squared deviations / count^2 == Var (pop) / count.
        stat = float(np.sum((retained - m) ** 2)) / (len(retained) ** 2)
        if stat < best_stat:
            best_stat = stat
            best_d = d
    return best_d * batch


def batch_means_ci(rates: np.ndarray, min_batches: int = 20) -> Tuple[float, float, int]:
    """Steady-state mean and 95% CI half-width via the method of batch means.

    ``rates`` are (assumed post-warm-up) per-interval throughput samples. They
    are grouped into contiguous batches whose means are approximately
    independent, and a Student-t CI is formed over the batch means. Returns
    ``(mean, ci_half_width, n_batches)``. With too few samples the CI half-width
    is ``inf`` (never "converged").
    """
    n = len(rates)
    if n == 0:
        return 0.0, float("inf"), 0
    mean = float(np.mean(rates))
    if n < 2:
        return mean, float("inf"), 1

    # Aim for ~min_batches batches; never fewer than 2 samples per batch.
    n_batches = min(min_batches, n // 2)
    n_batches = max(n_batches, 2)
    batch_size = n // n_batches
    usable = batch_size * n_batches
    batch_means = rates[:usable].reshape(n_batches, batch_size).mean(axis=1)

    grand_mean = float(np.mean(batch_means))
    # Sample std of the batch means (ddof=1), std error over n_batches.
    std = float(np.std(batch_means, ddof=1))
    half_width = _t_critical_95(n_batches - 1) * std / np.sqrt(n_batches)
    return grand_mean, float(half_width), n_batches


def throughput_from_counter_series(
    samples: List[Tuple[float, int]],
    min_batches: int = 20,
) -> Tuple[float, float, int]:
    """Estimate steady-state throughput from a cumulative-completion series.

    ``samples`` is a list of ``(timestamp, cumulative_completed)`` readings taken
    at a roughly fixed cadence while the server ran at fixed concurrency. We
    convert to per-interval rates, discard the warm-up transient (MSER-5) and
    form a batch-means CI. Returns ``(throughput, ci_half_width, n_batches)``.

    Falls back to ``total/elapsed`` (with an infinite CI) when there are too few
    samples to analyse.
    """
    if len(samples) < 3:
        if len(samples) == 2:
            (t0, c0), (t1, c1) = samples
            dt = t1 - t0
            return ((c1 - c0) / dt if dt > 0 else 0.0), float("inf"), 1
        return 0.0, float("inf"), 0

    ts = np.array([t for t, _ in samples], dtype=float)
    counts = np.array([c for _, c in samples], dtype=float)
    dts = np.diff(ts)
    dcs = np.diff(counts)
    # Guard against zero/negative intervals (clock coalescing).
    valid = dts > 0
    rates = np.where(valid, dcs / np.where(valid, dts, 1.0), 0.0)
    rates = rates[valid]
    if len(rates) < 2:
        elapsed = ts[-1] - ts[0]
        total = counts[-1] - counts[0]
        return (total / elapsed if elapsed > 0 else 0.0), float("inf"), 1

    cutoff = mser5_truncation(rates)
    steady = rates[cutoff:]
    if len(steady) < 2:
        steady = rates  # warm-up discard ate too much; use everything
    return batch_means_ci(steady, min_batches=min_batches)


def concurrency_ladder(cfg: SaturationConfig) -> List[int]:
    """Geometric ladder of concurrency levels to probe, clamped to the ceiling.

    Always includes ``max_concurrency`` so the ceiling itself is measured (which
    is what lets us tell "plateaued" from "harness-limited").
    """
    ladder: List[int] = []
    n = float(cfg.min_concurrency)
    while int(round(n)) < cfg.max_concurrency:
        level = int(round(n))
        if not ladder or ladder[-1] != level:
            ladder.append(level)
        n *= cfg.ladder_growth
    if not ladder or ladder[-1] != cfg.max_concurrency:
        ladder.append(cfg.max_concurrency)
    return ladder


async def find_saturation(
    probe: Callable[[int], Awaitable[ProbeResult]],
    cfg: SaturationConfig,
) -> SaturationResult:
    """Walk the concurrency ladder until throughput plateaus; return mu_max.

    ``probe(N)`` runs the server at concurrency ``N`` and returns a steady-state
    :class:`ProbeResult`. We climb the geometric ladder and stop once raising
    concurrency no longer buys a *statistically real* gain over the best
    throughput seen so far, or we reach ``max_concurrency``. ``mu_max`` is the
    best throughput observed; the knee is the smallest concurrency that reached
    it.

    The stop rule is deliberately noise-robust. Rather than reacting to a single
    inter-probe delta (which a lone noisy reading can spike or dip), we compare
    each probe's *upper* confidence bound against the running best: a plateau is
    only called when even the optimistic estimate fails to beat the best by
    ``plateau_rel_gain``. A probe with a wide CI therefore cannot trip an early
    stop, and the stop is flagged ``confident`` only if the deciding probe
    actually met its precision target. If throughput was still climbing at the
    ceiling, ``harness_limited`` is set and ``mu_max`` is a lower bound.
    """
    ladder = concurrency_ladder(cfg)
    probes: List[ProbeResult] = []
    best_throughput = 0.0
    plateaued = False
    confident = False

    for level in ladder:
        result = await probe(level)
        probes.append(result)

        # Compare against the best of all *previous* probes. Using the current
        # probe's upper confidence bound makes the decision robust to a single
        # low, noisy reading: only an optimistic estimate that still fails to
        # beat the best by the plateau margin counts as "no real gain left".
        if best_throughput > 0.0:
            upper_bound = result.throughput + result.ci_half_width
            if upper_bound < best_throughput * (1.0 + cfg.plateau_rel_gain):
                best_throughput = max(best_throughput, result.throughput)
                plateaued = True
                # Trust the plateau only if this probe hit its precision target;
                # otherwise mu_max stands but is reported as approximate.
                confident = result.converged
                break

        best_throughput = max(best_throughput, result.throughput)

    # Harness-limited iff we exhausted the ladder (reached the ceiling) without
    # ever detecting a plateau: the curve was still climbing at max concurrency.
    harness_limited = not plateaued and bool(probes) and probes[-1].concurrency >= cfg.max_concurrency

    knee = _knee_concurrency(probes, best_throughput, cfg.plateau_rel_gain)
    return SaturationResult(
        max_throughput=best_throughput,
        knee_concurrency=knee,
        harness_limited=harness_limited,
        confident=confident,
        probes=probes,
    )


def _knee_concurrency(probes: List[ProbeResult], mu_max: float, tol: float) -> int:
    """Smallest probed concurrency whose throughput is within ``tol`` of mu_max."""
    if not probes or mu_max <= 0:
        return probes[-1].concurrency if probes else 0
    threshold = mu_max * (1.0 - tol)
    for p in probes:
        if p.throughput >= threshold:
            return p.concurrency
    return probes[-1].concurrency
