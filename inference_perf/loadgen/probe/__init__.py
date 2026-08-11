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
"""Closed-loop capacity probe for sweep request-rate autoselection.

This package is the computational core of a saturation probe that replaces
transient-based capacity estimation (burst drain-rate percentiles, stepped
open-loop ramps) with a closed-loop concurrency ladder:

1. Hold the number of in-flight requests fixed at N (semaphore dispatch) and
   measure throughput X(N) and mean latency R(N) over a stationary window.
   Little's law (N = X * R) holds for any arrival process and any backend
   topology, so each rung is a self-checking measurement.
2. Grow N geometrically until throughput gains flatten (`ConcurrencyLadder`).
3. Estimate the capacity asymptote and knee from the rung curve
   (`estimate_constants`), preferring a saturating-curve fit when the plateau
   was actually approached and falling back to the empirical plateau when not.

The probe's outputs are named constants with confidence intervals
(`BoundConstant`). The names (`RESERVED_SYMBOLS`) form an internal-only
symbol namespace: sweep stage definitions may reference them, users may never
set them, and the harness binds their values after the probe runs. This
module deliberately has no dependency on the expression grammar or on any
dispatch/wiring code; it is pure measurement and estimation logic.
"""

from .estimator import (
    batch_means_throughput,
    estimate_constants,
    fit_saturating_curve,
    isotonic_regression,
    make_rung,
)
from .ladder import ConcurrencyLadder, LadderConfig
from .result import (
    RESERVED_SYMBOLS,
    BoundConstant,
    ConfidenceInterval,
    ProbeResult,
    RungResult,
    SaturationSignal,
    classify_saturation,
)
from .stationarity import cusum_drift_index, is_stationary

__all__ = [
    "RESERVED_SYMBOLS",
    "BoundConstant",
    "ConcurrencyLadder",
    "ConfidenceInterval",
    "LadderConfig",
    "ProbeResult",
    "RungResult",
    "SaturationSignal",
    "batch_means_throughput",
    "classify_saturation",
    "cusum_drift_index",
    "estimate_constants",
    "fit_saturating_curve",
    "is_stationary",
    "isotonic_regression",
    "make_rung",
]
