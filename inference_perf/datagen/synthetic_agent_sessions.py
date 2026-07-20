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
"""Synthetic multi-agent session generator.

This module builds synthetic agent-session replay graphs procedurally.
Determinism is a hard requirement: graph generation must be a pure function
of (config, session_index), reproducible byte-for-byte across processes
(e.g. a parent process and its worker processes). To achieve this we avoid
Python's salted `hash()` entirely and derive all randomness from `numpy`
`Generator` instances seeded from stable, path-derived integers.
"""

import hashlib
from typing import Optional

import numpy as np

from inference_perf.config.common import Distribution
from inference_perf.utils.numeric.distribution.utils import sample_from_distribution


def session_seed(base_seed: int, session_index: int) -> int:
    """Derive a stable per-session seed from a base seed and session index.

    Pure function of its inputs -- does NOT use Python's built-in `hash()`,
    which is salted per-process (via PYTHONHASHSEED) and would break
    reproducibility across processes.
    """
    digest = hashlib.blake2b(f"{base_seed}:{session_index}".encode(), digest_size=8).digest()
    return int.from_bytes(digest, "big")


def child_rng(parent_seed: int, *path: int) -> np.random.Generator:
    """Create a numpy random Generator derived from a parent seed and a graph path.

    Folding the path into the seed sequence means different positions in the
    generated graph draw from independent, reproducible random streams.
    """
    return np.random.default_rng([parent_seed, *path])


def sample_int(dist: Optional[Distribution], rng: np.random.Generator, fallback: Distribution) -> int:
    """Resolve `dist` (or `fallback` if None) and draw a single deterministic int.

    Always passes `rng` explicitly to `sample_from_distribution` -- the
    util's default (unseeded) RNG would break determinism.
    """
    d = dist if dist is not None else fallback
    val = sample_from_distribution(d, 1, rng=rng)[0]
    return int(val)
