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
"""Concurrency ladder policy: which N to measure next, and when to stop.

The ladder grows concurrency geometrically and stops when the marginal
throughput gain between the top two rungs is confidently below a threshold.
The confidence direction is deliberate: probing continues until the upper
confidence bound of the gain is small, so noise extends the ladder (bounded
by `max_concurrency`) rather than truncating it into an underestimate of
capacity. A client-bound rung stops the ladder immediately because every
larger rung would measure the client, not the server.

This is pure decision logic: it consumes `RungResult`s and returns the next
concurrency to measure (or None to stop). Dispatch is the caller's job.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence

from .result import RungResult, SaturationSignal


@dataclass(frozen=True)
class LadderConfig:
    start_concurrency: int = 1
    growth_factor: float = 2.0
    max_concurrency: int = 1024
    # Plateau: stop when the relative gain between the top two rungs is
    # confidently below this fraction.
    gain_threshold: float = 0.05
    # One-sided z-score for the gain's upper confidence bound (1.64 ~ 95%).
    z_score: float = 1.64
    # Re-measurements allowed for a non-stationary rung before using it as-is.
    max_retries: int = 1
    # After the plateau is found, measure one geometric-midpoint rung to
    # sharpen the knee estimate.
    refine: bool = True

    def __post_init__(self) -> None:
        if self.start_concurrency < 1:
            raise ValueError(f"start_concurrency must be >= 1, got {self.start_concurrency}")
        if self.growth_factor <= 1.0:
            raise ValueError(f"growth_factor must be > 1, got {self.growth_factor}")
        if self.max_concurrency < self.start_concurrency:
            raise ValueError(f"max_concurrency must be >= start_concurrency, got {self.max_concurrency}")
        if self.gain_threshold <= 0:
            raise ValueError(f"gain_threshold must be positive, got {self.gain_threshold}")
        if self.max_retries < 0:
            raise ValueError(f"max_retries must be non-negative, got {self.max_retries}")


@dataclass
class ConcurrencyLadder:
    """Stateful next-rung policy. Feed it the full measurement history each call."""

    config: LadderConfig = field(default_factory=LadderConfig)
    stop_reason: Optional[str] = None
    _retries: Dict[int, int] = field(default_factory=dict)
    _refined: bool = False

    def next_concurrency(self, history: Sequence[RungResult]) -> Optional[int]:
        """Return the next concurrency to measure, or None when the ladder is done.

        Once None is returned the decision is sticky; `stop_reason` records why
        ("client_bound", "plateau", or "max_concurrency").
        """
        if self.stop_reason is not None:
            return None
        if not history:
            return self.config.start_concurrency

        last = history[-1]
        if last.signal is SaturationSignal.CLIENT_BOUND:
            self.stop_reason = "client_bound"
            return None
        if not last.stationary:
            attempts = self._retries.get(last.concurrency, 0)
            if attempts < self.config.max_retries:
                self._retries[last.concurrency] = attempts + 1
                return last.concurrency

        # Latest measurement wins for retried concurrencies.
        latest = {r.concurrency: r for r in history}
        ordered = sorted(latest)

        if len(ordered) >= 2 and self._plateau_reached(latest[ordered[-2]], latest[ordered[-1]]):
            if self.config.refine and not self._refined:
                self._refined = True
                midpoint = round(math.sqrt(ordered[-2] * ordered[-1]))
                if midpoint not in latest and midpoint >= 1:
                    return midpoint
            self.stop_reason = "plateau"
            return None

        proposed = max(last.concurrency + 1, math.ceil(last.concurrency * self.config.growth_factor))
        if proposed > self.config.max_concurrency:
            self.stop_reason = "max_concurrency"
            return None
        return proposed

    def _plateau_reached(self, lower: RungResult, upper: RungResult) -> bool:
        gain = upper.throughput - lower.throughput
        gain_upper_bound = gain + self.config.z_score * math.hypot(lower.throughput_se, upper.throughput_se)
        return gain_upper_bound < self.config.gain_threshold * lower.throughput
