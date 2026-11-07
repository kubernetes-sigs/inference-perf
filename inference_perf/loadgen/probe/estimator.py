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
"""Rung measurement and capacity estimation.

Throughput per rung is measured with batch means: the analysis window is split
into equal sub-windows and the completion rate of each becomes one sample, so
the standard error is estimated without assuming independence of individual
completions. The capacity estimate reads the rung curve X(N) two ways:

- Empirical plateau: isotonic regression (X(N) must be non-decreasing) and
  take the top level. Never extrapolates; valid on any backend topology.
- Saturating-curve fit: X(N) = mu * N / (K + N) via the Hanes linearization
  (N / X regressed on N). Extrapolates to the asymptote, but the form is a
  fitting device, not physics; multi-replica and scheduler-capped backends
  can violate it.

`estimate_constants` prefers the fit only when its asymptote is close to the
measured plateau (i.e. the ladder actually approached saturation) and falls
back to the empirical plateau otherwise. Uncertainty comes from a parametric
bootstrap over the per-rung standard errors.
"""

from typing import Dict, Optional, Sequence, Tuple

import numpy as np
import numpy.typing as npt

from .result import BoundConstant, ConfidenceInterval, RungResult, SaturationSignal
from .stationarity import is_stationary

_FloatArray = npt.NDArray[np.float64]


def batch_means_throughput(
    completion_times: npt.ArrayLike,
    window_start: float,
    window_end: float,
    num_batches: int = 8,
) -> Tuple[float, float, _FloatArray]:
    """Measure throughput over [window_start, window_end) with a batch-means standard error.

    Returns (throughput, standard_error, per_batch_rates). Completion times
    outside the window are ignored; an empty window raises ValueError because
    a rung with zero completions is a measurement failure, not a zero rate.
    """
    if num_batches < 2:
        raise ValueError(f"num_batches must be >= 2, got {num_batches}")
    duration = window_end - window_start
    if duration <= 0:
        raise ValueError(f"window must have positive duration, got {duration}")

    times = np.asarray(completion_times, dtype=np.float64)
    times = times[(times >= window_start) & (times < window_end)]
    if times.size == 0:
        raise ValueError("no completions inside the analysis window")

    throughput = times.size / duration
    edges = np.linspace(window_start, window_end, num_batches + 1)
    counts, _ = np.histogram(times, bins=edges)
    batch_rates = counts.astype(np.float64) / (duration / num_batches)
    standard_error = float(np.std(batch_rates, ddof=1) / np.sqrt(num_batches))
    return float(throughput), standard_error, batch_rates


def make_rung(
    concurrency: int,
    completion_times: npt.ArrayLike,
    latencies: npt.ArrayLike,
    window_start: float,
    window_end: float,
    num_batches: int = 8,
    signal: SaturationSignal = SaturationSignal.NONE,
) -> RungResult:
    """Build a `RungResult` from raw per-request completion times and latencies.

    `latencies[i]` must correspond to `completion_times[i]`. Stationarity is
    judged on the per-batch completion rates, and the Little's law residual
    |N - X * R| / N is computed as the rung's self-consistency check.
    """
    times = np.asarray(completion_times, dtype=np.float64)
    lats = np.asarray(latencies, dtype=np.float64)
    if times.shape != lats.shape:
        raise ValueError(f"completion_times and latencies must align, got {times.shape} vs {lats.shape}")

    throughput, standard_error, batch_rates = batch_means_throughput(times, window_start, window_end, num_batches)
    in_window = (times >= window_start) & (times < window_end)
    latency = float(np.mean(lats[in_window]))
    residual = abs(concurrency - throughput * latency) / concurrency
    return RungResult(
        concurrency=concurrency,
        throughput=throughput,
        throughput_se=standard_error,
        latency=latency,
        littles_law_residual=residual,
        stationary=is_stationary(batch_rates),
        signal=signal,
    )


def isotonic_regression(values: npt.ArrayLike, weights: Optional[npt.ArrayLike] = None) -> _FloatArray:
    """Weighted least-squares fit of a non-decreasing sequence (pool adjacent violators)."""
    y = np.asarray(values, dtype=np.float64)
    if y.ndim != 1:
        raise ValueError(f"values must be one-dimensional, got shape {y.shape}")
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=np.float64)
    if w.shape != y.shape or np.any(w <= 0):
        raise ValueError("weights must be positive and align with values")

    level_means: list[float] = []
    level_weights: list[float] = []
    level_sizes: list[int] = []
    for yi, wi in zip(y, w, strict=True):
        level_means.append(float(yi))
        level_weights.append(float(wi))
        level_sizes.append(1)
        while len(level_means) > 1 and level_means[-2] > level_means[-1]:
            merged_weight = level_weights[-2] + level_weights[-1]
            merged_mean = (level_means[-2] * level_weights[-2] + level_means[-1] * level_weights[-1]) / merged_weight
            merged_size = level_sizes[-2] + level_sizes[-1]
            for level in (level_means, level_weights, level_sizes):
                level.pop()
            level_means[-1] = merged_mean
            level_weights[-1] = merged_weight
            level_sizes[-1] = merged_size
    return np.concatenate([np.full(size, mean) for mean, size in zip(level_means, level_sizes, strict=True)])


def fit_saturating_curve(
    concurrencies: npt.ArrayLike,
    throughputs: npt.ArrayLike,
) -> Optional[Tuple[float, float]]:
    """Fit X(N) = mu * N / (K + N) and return (mu, K), or None if degenerate.

    Uses the Hanes linearization: N / X = N / mu + K / mu is linear in N, so a
    least-squares line gives mu from the slope and K from the intercept. A
    non-positive slope or negative intercept means the data carry no evidence
    of saturation in this functional form.
    """
    n = np.asarray(concurrencies, dtype=np.float64)
    x = np.asarray(throughputs, dtype=np.float64)
    if n.shape != x.shape or n.ndim != 1:
        raise ValueError("concurrencies and throughputs must be aligned one-dimensional arrays")
    if np.unique(n).size < 3:
        return None
    if np.any(n <= 0) or np.any(x <= 0):
        raise ValueError("concurrencies and throughputs must be positive")

    slope, intercept = np.polyfit(n, n / x, 1)
    if slope <= 0 or intercept < 0:
        return None
    mu = 1.0 / float(slope)
    knee_constant = float(intercept) * mu
    return mu, knee_constant


def _point_estimate(
    concurrencies: _FloatArray,
    throughputs: _FloatArray,
    knee_fraction: float,
    fit_acceptance_ratio: float,
) -> Tuple[float, float]:
    """Estimate (r_sat, n_knee) from one realization of the rung curve."""
    isotonic = isotonic_regression(throughputs)
    plateau = float(isotonic[-1])
    fit = fit_saturating_curve(concurrencies, throughputs)
    r_sat = plateau
    if fit is not None and fit[0] <= fit_acceptance_ratio * plateau:
        r_sat = fit[0]
    # Relative tolerance so a rung sitting exactly at the knee fraction is not
    # excluded by floating-point error in the fitted asymptote.
    at_knee = concurrencies[isotonic >= knee_fraction * r_sat * (1.0 - 1e-9)]
    n_knee = float(at_knee[0]) if at_knee.size > 0 else float(concurrencies[-1])
    return r_sat, n_knee


def estimate_constants(
    rungs: Sequence[RungResult],
    rng: Optional[np.random.Generator] = None,
    knee_fraction: float = 0.9,
    fit_acceptance_ratio: float = 1.3,
    num_bootstrap: int = 500,
    confidence: float = 0.95,
) -> Dict[str, BoundConstant]:
    """Estimate the internal symbols {r_sat, n_knee} from a measured ladder.

    Only stationary, non-client-bound rungs participate; retried concurrencies
    keep their latest measurement. The saturating-curve asymptote is used for
    r_sat only when it lies within `fit_acceptance_ratio` of the empirical
    plateau, meaning the ladder actually got close to saturation; otherwise
    the plateau itself is reported, which is an honest lower bound.
    Confidence intervals come from a parametric bootstrap resampling each
    rung's throughput from its batch-means standard error.
    """
    if not 0 < knee_fraction < 1:
        raise ValueError(f"knee_fraction must be in (0, 1), got {knee_fraction}")
    if not 0 < confidence < 1:
        raise ValueError(f"confidence must be in (0, 1), got {confidence}")

    usable = [r for r in rungs if r.stationary and r.signal is not SaturationSignal.CLIENT_BOUND]
    latest_by_concurrency = {r.concurrency: r for r in usable}
    if len(latest_by_concurrency) < 2:
        raise ValueError(f"need at least 2 usable rungs at distinct concurrencies, got {len(latest_by_concurrency)}")
    ordered = [latest_by_concurrency[c] for c in sorted(latest_by_concurrency)]

    concurrencies = np.asarray([r.concurrency for r in ordered], dtype=np.float64)
    throughputs = np.asarray([r.throughput for r in ordered], dtype=np.float64)
    standard_errors = np.asarray([r.throughput_se for r in ordered], dtype=np.float64)

    r_sat_hat, n_knee_hat = _point_estimate(concurrencies, throughputs, knee_fraction, fit_acceptance_ratio)

    generator = rng if rng is not None else np.random.default_rng()
    tiny = np.finfo(np.float64).tiny
    r_sat_draws = np.empty(num_bootstrap, dtype=np.float64)
    n_knee_draws = np.empty(num_bootstrap, dtype=np.float64)
    for i in range(num_bootstrap):
        resampled = np.maximum(throughputs + generator.normal(0.0, 1.0, throughputs.size) * standard_errors, tiny)
        r_sat_draws[i], n_knee_draws[i] = _point_estimate(concurrencies, resampled, knee_fraction, fit_acceptance_ratio)

    alpha = (1.0 - confidence) / 2.0
    r_sat_ci = ConfidenceInterval(
        low=float(np.quantile(r_sat_draws, alpha)),
        high=float(np.quantile(r_sat_draws, 1.0 - alpha)),
    )
    n_knee_ci = ConfidenceInterval(
        low=float(np.quantile(n_knee_draws, alpha)),
        high=float(np.quantile(n_knee_draws, 1.0 - alpha)),
    )
    return {
        "r_sat": BoundConstant(name="r_sat", value=r_sat_hat, ci=r_sat_ci),
        "n_knee": BoundConstant(name="n_knee", value=n_knee_hat, ci=n_knee_ci),
    }
