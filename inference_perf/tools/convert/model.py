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
"""Shared result model for the peer-tool converters (#755).

Every peer flag lands in one of four verdicts:

- ``MAP``: a value-for-value equivalent config path exists; the field is emitted.
- ``MAP_ASSUME``: mappable only under a stated assumption; the field is emitted
  and the assumption is recorded in the emitted comment block.
- ``ANNOTATE``: no equivalent, but the flag cannot change the offered workload
  (reporting, UI, tool plumbing); it is dropped with a ``NO EQUIVALENT`` or
  ``dropped`` line instead of a plausible value.
- ``REFUSE``: no equivalent and the flag, flag value, or flag combination
  changes the offered workload or the measured window; the conversion fails
  with a named reason and no config is written.

The dividing line is the offered workload: request count, prompt lengths,
``max_tokens``, ``stream``/``ignore_eos``, arrival spacing, and in-flight
concurrency. A flag that cannot alter any of those cannot make a comparison
silently wrong, so annotating it is enough; anything that can alter them
refuses rather than guesses.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from inference_perf.config import Config, Distribution, DistributionType


class Verdict(Enum):
    MAP = "map"
    MAP_ASSUME = "map_assume"
    ANNOTATE = "annotate"
    REFUSE = "refuse"


class PeerUsageError(Exception):
    """The peer argv itself is invalid for the mirrored surface (missing value,

    unparsable number, missing required flag). Distinct from a refusal: the
    peer tool would have rejected this argv too.
    """


@dataclass
class Conversion:
    """The outcome of converting one peer argv.

    ``config`` is set only when ``refusals`` is empty: a half-config that
    "looks converted" is the failure mode #755 exists to prevent.
    """

    source_tool: str
    source_version: str
    argv: List[str]
    config: Optional[Config] = None
    assumptions: List[str] = field(default_factory=list)
    no_equivalents: List[str] = field(default_factory=list)
    dropped: List[str] = field(default_factory=list)
    refusals: List[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.refusals and self.config is not None

    def refuse(self, subject: str, reason: str) -> None:
        self.refusals.append(f"{subject}: {reason}")

    def assume(self, text: str) -> None:
        if text not in self.assumptions:
            self.assumptions.append(text)

    def no_equivalent(self, text: str) -> None:
        if text not in self.no_equivalents:
            self.no_equivalents.append(text)

    def drop(self, flag: str) -> None:
        if flag not in self.dropped:
            self.dropped.append(flag)


def fixed_dist(value: int) -> Distribution:
    """A distribution that emits exactly ``value`` for every request.

    ``std_dev`` is left at its (unused) default so the field stays unset in
    the emitted YAML and full-dump comparisons against hand-written fixtures
    do not trip over it.
    """
    return Distribution(type=DistributionType.FIXED, mean=value, min=value, max=value)


def uniform_dist(low: int, high: int) -> Distribution:
    return Distribution(type=DistributionType.UNIFORM, mean=(low + high) / 2, min=low, max=high)


def normal_dist(mean: float, std_dev: float, min_value: int, max_value: int) -> Distribution:
    return Distribution(type=DistributionType.NORMAL, mean=mean, std_dev=std_dev, min=min_value, max=max_value)
