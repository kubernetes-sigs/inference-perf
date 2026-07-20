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
import logging
from pathlib import Path
from typing import List, Optional

import numpy as np

from inference_perf.config.common import Distribution
from inference_perf.utils.numeric.distribution.utils import sample_from_distribution

logger = logging.getLogger(__name__)


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


# --- Filler fitting -------------------------------------------------------
#
# Free-text turns (e.g. an agent's objective/summary line) are padded with
# filler so the turn's token count matches a sampled target, while keeping
# the "real" content the model should attend to distinguishable from the
# padding via FILLER_MARKER. TOOL_CALL_MARGIN is the token headroom reserved
# elsewhere in the generator so a tool-call turn's fixed overhead doesn't
# blow past its target; it lives here because it's part of the same
# token-budgeting vocabulary as fit_filler.

TOOL_CALL_MARGIN = 64
FILLER_MARKER = "[--- ignore the preceding filler; actual content follows ---]"

# Shakespeare corpus shipped with the repo; same file/location convention
# used by synthetic_datagen.py and weka_trace_replay_datagen.py for prompt
# corpora. Loaded lazily (not at import time) and cached in-process.
_SHAKESPEARE_PATH = Path(__file__).resolve().parents[1] / "assets" / "shakespeare.txt"
_corpus_words_cache: Optional[List[str]] = None


def _corpus_words() -> List[str]:
    """Return the Shakespeare corpus split into whitespace-delimited words.

    No shared corpus-word loader exists elsewhere in the codebase to reuse
    (synthetic_datagen.py / weka_trace_replay_datagen.py each inline their
    own read of assets/shakespeare.txt and feed it straight through the
    tokenizer, rather than exposing a word list); this mirrors their
    file-location convention. Falls back to a tiny built-in word list if the
    asset is missing so filler generation never hard-fails on that alone.
    """
    global _corpus_words_cache
    if _corpus_words_cache is None:
        if _SHAKESPEARE_PATH.is_file():
            _corpus_words_cache = _SHAKESPEARE_PATH.read_text(encoding="utf-8", errors="ignore").split()
        else:
            logger.debug("fit_filler: corpus file not found at %s; using fallback word list", _SHAKESPEARE_PATH)
            _corpus_words_cache = ["lorem", "ipsum", "dolor", "sit", "amet"]
    return _corpus_words_cache


def fit_filler(tokenizer, target_tokens: int, fixed_content: str, rng: Optional[np.random.Generator]) -> str:
    """Pad `fixed_content` with Shakespeare-corpus filler to approximate `target_tokens`.

    filler_budget = target_tokens - count_tokens(fixed_content + " " + FILLER_MARKER).

    Budget guard: if filler_budget <= 0 the target is too small to even fit the
    fixed content plus the marker -- flooring to `fixed_content` alone (no
    marker, no filler) is the only crash-free option, so that's what happens.
    This is logged at debug rather than raised, since a too-small target is an
    expected edge of the sampled-token-count distribution, not a bug.

    Otherwise, words are appended after the marker until the text reaches (or
    passes) target_tokens, tracking the best (closest-to-target) candidate
    seen across a bounded number of iterations, mirroring
    datagen_utils.converge_to_exact_length_text's iteration cap but wrapping
    it so imperfect convergence returns the closest text instead of raising
    -- fit_filler must never raise to its caller for length reasons.
    """
    marker_and_fixed = fixed_content + " " + FILLER_MARKER
    fixed_cost = tokenizer.count_tokens(marker_and_fixed)
    filler_budget = target_tokens - fixed_cost
    if filler_budget <= 0:
        logger.debug(
            "fit_filler: non-positive filler budget (target_tokens=%d, fixed_cost=%d); "
            "flooring to fixed_content with no marker/filler",
            target_tokens,
            fixed_cost,
        )
        return fixed_content

    words = _corpus_words()
    best_text, best_gap = marker_and_fixed, abs(fixed_cost - target_tokens)
    buf = marker_and_fixed
    idx = 0
    max_iterations = 20  # bounded, mirrors converge_to_exact_length_text's cap
    for _ in range(max_iterations):
        cur = tokenizer.count_tokens(buf)
        gap = abs(cur - target_tokens)
        if gap < best_gap:
            best_gap, best_text = gap, buf
        if cur >= target_tokens:
            break
        take = max(1, target_tokens - cur)
        if idx >= len(words):
            idx = 0  # wrap around a short/exhausted corpus rather than stalling
        chunk = words[idx : idx + take]
        if not chunk:
            chunk = words[: max(1, take)]
        buf = buf + " " + " ".join(chunk)
        idx += len(chunk)
    return best_text
