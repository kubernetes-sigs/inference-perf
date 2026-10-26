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

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Callable, FrozenSet, List, Mapping, Optional

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


# argparse's own negative-number rule (``_negative_number_matcher``).
_NEGATIVE_NUMBER = re.compile(r"^-\d+$|^-\d*\.\d+$")


def is_value_token(token: str) -> bool:
    """True when ``token`` reads as a flag's value rather than as a flag.

    A bare negative number counts as a value, the way argparse classifies it
    for ``vllm bench serve``; both frontends use the same rule so an unknown
    flag consumes its one value and refuses once instead of once per token.
    """
    return not token.startswith("-") or _NEGATIVE_NUMBER.match(token) is not None


REDACTED = "<redacted>"


def rewrite_values(argv: List[str], rewriters: Mapping[str, Callable[[str], str]]) -> List[str]:
    """A copy of ``argv`` with the value of every flag in ``rewriters`` passed through its rewriter.

    The emitted comment block records the source argv verbatim, so a credential
    passed on the peer CLI has to be scrubbed there as well as from the config
    body. Both the ``--flag value`` and ``--flag=value`` spellings are covered;
    a trailing flag with no value left is copied as is.
    """
    out: List[str] = []
    i = 0
    while i < len(argv):
        token = argv[i]
        i += 1
        name, eq, inline = token.partition("=")
        rewrite = rewriters.get(name)
        if rewrite is None:
            out.append(token)
        elif eq:
            out.append(f"{name}={rewrite(inline)}")
        else:
            out.append(token)
            if i < len(argv):
                out.append(rewrite(argv[i]))
                i += 1
    return out


def redact_values(argv: List[str], flags: FrozenSet[str]) -> List[str]:
    """A copy of ``argv`` with the whole value of every flag in ``flags`` replaced by ``REDACTED``."""
    return rewrite_values(argv, {flag: lambda _value: REDACTED for flag in flags})


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
