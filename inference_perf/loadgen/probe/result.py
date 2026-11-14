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
"""Result types for the capacity probe.

The probe communicates with the rest of the harness exclusively through these
types: per-rung measurements (`RungResult`) and named constants with
confidence intervals (`BoundConstant`). Constant names are drawn from a
reserved, internal-only symbol namespace so that sweep stage definitions can
reference them while user config can never define them.
"""

from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, Mapping, Optional, Tuple

# Internal-only symbol names the probe may bind. Config validation must reject
# any user-supplied definition of these names; stage expressions may reference
# them and the harness binds their values after the probe runs.
#
#   r_sat  - capacity asymptote, requests/sec
#   n_knee - concurrency at which throughput reaches the knee fraction of r_sat
#   r_slo  - highest request rate meeting latency SLOs (reserved; not yet
#            produced by `estimate_constants`, which needs open-loop refinement)
RESERVED_SYMBOLS = frozenset({"r_sat", "n_knee", "r_slo"})


class SaturationSignal(str, Enum):
    """Client-side classification of what bound a rung, if anything.

    On disaggregated or routed backends the bottleneck identity matters as
    much as the knee location: first-token delays indicate queue/prefill/router
    saturation while inter-token stalls indicate decode saturation. A client
    that cannot keep up invalidates the rung entirely.
    """

    NONE = "none"
    PREFILL_BOUND = "prefill_bound"
    DECODE_BOUND = "decode_bound"
    CLIENT_BOUND = "client_bound"


class ProbeStopReason(str, Enum):
    """Why the concurrency ladder stopped growing.

    Only `PLATEAU` means the measured curve flattened on its own. `CLIENT_BOUND`
    and `MAX_CONCURRENCY` end the ladder early, and `MEASUREMENT_FAILED` ends it
    because a rung produced nothing to measure, so in all three the reported
    `r_sat` is a lower bound on capacity rather than a measured asymptote. The
    result carries `censored = True` to keep the two apart in reports.
    """

    PLATEAU = "plateau"
    CLIENT_BOUND = "client_bound"
    MAX_CONCURRENCY = "max_concurrency"
    MEASUREMENT_FAILED = "measurement_failed"


def classify_saturation(
    ttft_inflation: float,
    itl_inflation: float,
    client_lag_inflation: float = 1.0,
    threshold: float = 2.0,
) -> SaturationSignal:
    """Classify a rung's bottleneck from degradation ratios versus an unloaded baseline.

    Each inflation argument is the rung's observed value divided by its
    baseline value (time to first token, inter-token latency, and client
    event-loop lag respectively), so 1.0 means "unchanged". Client saturation
    is checked first because it invalidates the other two signals.
    """
    if ttft_inflation <= 0 or itl_inflation <= 0 or client_lag_inflation <= 0:
        raise ValueError("inflation ratios must be positive")
    if client_lag_inflation >= threshold:
        return SaturationSignal.CLIENT_BOUND
    if ttft_inflation >= threshold and ttft_inflation >= itl_inflation:
        return SaturationSignal.PREFILL_BOUND
    if itl_inflation >= threshold:
        return SaturationSignal.DECODE_BOUND
    return SaturationSignal.NONE


@dataclass(frozen=True)
class ConfidenceInterval:
    low: float
    high: float

    def __post_init__(self) -> None:
        if self.low > self.high:
            raise ValueError(f"invalid confidence interval: low={self.low} > high={self.high}")

    @property
    def width(self) -> float:
        return self.high - self.low


@dataclass(frozen=True)
class BoundConstant:
    """A probe-estimated constant destined for the internal symbol namespace."""

    name: str
    value: float
    ci: ConfidenceInterval

    def __post_init__(self) -> None:
        if self.name not in RESERVED_SYMBOLS:
            raise ValueError(f"'{self.name}' is not a reserved probe symbol; allowed: {sorted(RESERVED_SYMBOLS)}")


@dataclass(frozen=True)
class RungResult:
    """One measured rung of the concurrency ladder.

    `littles_law_residual` is |N - X * R| / N over the analysis window; a large
    value means the window was not stationary or the concurrency was not
    actually held at N, and the rung should not be trusted. `error_rate` is the
    fraction of the rung's requests that failed, because a backend that fails
    fast must not look like a fast backend. `client_lag` is the load
    generator's own event-loop lag during the window, in seconds; it is what
    tells a real server plateau apart from a generator that cannot keep up.
    """

    concurrency: int
    throughput: float
    throughput_se: float = 0.0
    latency: float = 0.0
    littles_law_residual: float = 0.0
    stationary: bool = True
    signal: SaturationSignal = SaturationSignal.NONE
    error_rate: float = 0.0
    client_lag: float = 0.0

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {self.concurrency}")
        if self.throughput <= 0:
            raise ValueError(f"throughput must be positive, got {self.throughput}")
        if self.throughput_se < 0:
            raise ValueError(f"throughput_se must be non-negative, got {self.throughput_se}")
        if not 0.0 <= self.error_rate <= 1.0:
            raise ValueError(f"error_rate must be in [0, 1], got {self.error_rate}")
        if self.client_lag < 0:
            raise ValueError(f"client_lag must be non-negative, got {self.client_lag}")


@dataclass(frozen=True)
class ProbeResult:
    """Everything a probe run produced: the rung curve and the bound constants.

    `stop_reason` records why the ladder stopped, so a capped, client-bound, or
    unmeasurable rung is not read as a measured plateau. `censored` is the same
    fact as a single boolean; it is what report readers and the sweep follow-up
    need in order to treat `r_sat` as a lower bound. `failed_concurrency` names
    the rung that produced no measurement, when that is what ended the ladder,
    because `stop_reason` alone would not say where it broke.
    """

    rungs: Tuple[RungResult, ...]
    constants: Mapping[str, BoundConstant]
    stop_reason: ProbeStopReason = ProbeStopReason.PLATEAU
    # Rungs the estimator was allowed to fit, in ladder order. Exposed so a
    # report can show why a rung did not participate.
    usable_concurrencies: Tuple[int, ...] = ()
    # Set only when stop_reason is MEASUREMENT_FAILED.
    failed_concurrency: Optional[int] = None

    @property
    def censored(self) -> bool:
        """True when the ladder stopped before the curve flattened on its own."""
        return self.stop_reason is not ProbeStopReason.PLATEAU

    def constant(self, name: str) -> BoundConstant:
        if name not in self.constants:
            raise KeyError(f"probe did not bind '{name}'; bound symbols: {sorted(self.constants)}")
        return self.constants[name]

    def to_dict(self) -> Dict[str, Any]:
        """Serialize the probe for a report file.

        This is the reported schema: `stop_reason` and `censored` qualify the
        constants, `failed_concurrency` says which rung ended the ladder when it
        produced no measurement, every rung is listed with the fields the
        estimator gated on, and `usable` says whether that rung entered the fit.
        """
        usable = set(self.usable_concurrencies)
        return {
            "stop_reason": self.stop_reason.value,
            "censored": self.censored,
            "failed_concurrency": self.failed_concurrency,
            "constants": {
                name: {
                    "value": constant.value,
                    "ci": {"low": constant.ci.low, "high": constant.ci.high},
                }
                for name, constant in sorted(self.constants.items())
            },
            "rungs": [
                {
                    "concurrency": rung.concurrency,
                    "throughput": rung.throughput,
                    "throughput_se": rung.throughput_se,
                    "latency": rung.latency,
                    "littles_law_residual": rung.littles_law_residual,
                    "stationary": rung.stationary,
                    "signal": rung.signal.value,
                    "error_rate": rung.error_rate,
                    "client_lag": rung.client_lag,
                    "usable": rung.concurrency in usable,
                }
                for rung in self.rungs
            ],
        }
