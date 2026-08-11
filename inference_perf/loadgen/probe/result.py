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
from typing import Mapping, Tuple

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
    actually held at N, and the rung should not be trusted.
    """

    concurrency: int
    throughput: float
    throughput_se: float = 0.0
    latency: float = 0.0
    littles_law_residual: float = 0.0
    stationary: bool = True
    signal: SaturationSignal = SaturationSignal.NONE

    def __post_init__(self) -> None:
        if self.concurrency < 1:
            raise ValueError(f"concurrency must be >= 1, got {self.concurrency}")
        if self.throughput <= 0:
            raise ValueError(f"throughput must be positive, got {self.throughput}")
        if self.throughput_se < 0:
            raise ValueError(f"throughput_se must be non-negative, got {self.throughput_se}")


@dataclass(frozen=True)
class ProbeResult:
    """Everything a probe run produced: the rung curve and the bound constants."""

    rungs: Tuple[RungResult, ...]
    constants: Mapping[str, BoundConstant]

    def constant(self, name: str) -> BoundConstant:
        if name not in self.constants:
            raise KeyError(f"probe did not bind '{name}'; bound symbols: {sorted(self.constants)}")
        return self.constants[name]
