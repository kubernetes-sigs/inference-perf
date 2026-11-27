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
"""CUSUM drift detection for probe measurement windows.

A rung measurement is only meaningful if the window it averages over was
stationary: no warmup ramp, no cache-fill transient, no capacity drift. CUSUM
(cumulative sum of standardized deviations from the window median) is the
classic sequential test for a sustained mean shift; it accumulates evidence
in both directions and flags the first point where either side exceeds a
threshold. Slack `k` and threshold `h` are in units of the series' robust
standard deviation (MAD-based, falling back to the sample standard deviation
for degenerate series).
"""

from typing import Optional

import numpy as np
import numpy.typing as npt

# Minimum points for the test to have any power; shorter series fail closed
# (treated as non-stationary) so callers re-measure rather than silently trust.
MIN_SERIES_LENGTH = 4

# 1 / Phi^{-1}(0.75): scales the median absolute deviation to estimate the
# standard deviation consistently under normality.
_MAD_TO_SIGMA = 1.4826


def cusum_drift_index(
    series: npt.ArrayLike,
    k: float = 0.5,
    h: float = 5.0,
) -> Optional[int]:
    """Return the index where a sustained mean shift is first detected, or None.

    A perfectly constant series is stationary by definition. Series shorter
    than `MIN_SERIES_LENGTH` raise ValueError; use `is_stationary` for the
    fail-closed boolean form.
    """
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1:
        raise ValueError(f"series must be one-dimensional, got shape {values.shape}")
    if values.size < MIN_SERIES_LENGTH:
        raise ValueError(f"series must have at least {MIN_SERIES_LENGTH} points, got {values.size}")
    if not np.all(np.isfinite(values)):
        raise ValueError("series contains non-finite values")

    center = float(np.median(values))
    scale = float(np.median(np.abs(values - center))) * _MAD_TO_SIGMA
    if scale == 0.0:
        scale = float(np.std(values))
    if scale == 0.0:
        return None

    standardized = (values - center) / scale
    upper = 0.0
    lower = 0.0
    for i, z in enumerate(standardized):
        upper = max(0.0, upper + z - k)
        lower = max(0.0, lower - z - k)
        if upper > h or lower > h:
            return i
    return None


def is_stationary(series: npt.ArrayLike, k: float = 0.5, h: float = 5.0) -> bool:
    """True if no sustained mean shift is detected.

    Fails closed: series too short to test are reported as non-stationary so
    the ladder re-measures instead of trusting an unverifiable window.
    """
    values = np.asarray(series, dtype=np.float64)
    if values.ndim != 1 or values.size < MIN_SERIES_LENGTH:
        return False
    return cusum_drift_index(values, k=k, h=h) is None
