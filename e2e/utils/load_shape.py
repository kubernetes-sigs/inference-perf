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
"""Shared load-shape assertions: was the configured load actually offered (#633).

The golden accuracy helpers in ``utils.accuracy`` check the measurement side
(what the tool reports about each response). These check the stimulus side:
the request rate and the in-flight concurrency the load generator actually
delivered.

Everything here works off the raw ``start_time`` / ``end_time`` pairs in
``per_request_lifecycle_metrics.json``, deliberately not off the numbers
``summarize_requests`` prints, so the reconstruction is independent of
reportgen and can be used to check reportgen's own derivation.

Honest limits of this oracle, stated once so callers do not overclaim:

- The timestamps are recorded by the client around its own send and receive,
  so this is the load generator's view of what it offered. It catches a
  generator that fails to keep up with its own schedule, a semaphore that
  admits too few or too many requests, and a reportgen that derives the rate
  wrongly. It cannot catch a bug in the timestamping itself.
- A server that recorded arrivals independently would be a strictly stronger
  oracle for the same claim. See the Status note on the PR for #633.
"""

import math
from typing import Any, Dict, List, Sequence, Tuple

# Piecewise-constant in-flight count: (segment start, segment end, in-flight).
Segment = Tuple[float, float, int]


def rate_tolerance(n: int, arrival: str) -> float:
    """Relative tolerance on achieved_rate for a stage of ``n`` requests.

    Sized from the arrival process rather than picked to make a run pass.

    ``constant``: ``ConstantLoadTimer`` draws n exponential gaps and rescales
    them so they sum to exactly ``duration``, so the schedule carries no
    cumulative drift. The only stochastic term left in reportgen's
    ``send_duration = max(start) - min(start)`` is the first gap, which n
    points do not span (n points bound n-1 gaps). That gap is Exp(1/rate), so
    the relative error is Exp(1)/n and P(error > k/n) = e^-k. k = 12 puts the
    statistical false-failure rate near 1e-5.

    ``poisson``: ``PoissonLoadTimer`` draws a Poisson(rate) count per second,
    so the wall-clock time to emit n requests is a renewal time with standard
    deviation sqrt(n)/rate against a mean of n/rate. The coefficient of
    variation of the achieved rate is therefore 1/sqrt(n), and 4 sigma is
    4/sqrt(n): tighten it by raising n, never by shrinking the multiplier.

    Both are floored at 5% so ordinary scheduler and socket jitter on a busy
    shared runner cannot fail the gate. At the request counts used by the e2e
    tier the floor binds only for ``constant``.
    """
    if n < 2:
        raise ValueError(f"a rate tolerance is meaningless for n={n} requests")
    floor = 0.05
    if arrival == "poisson":
        return max(floor, 4.0 / math.sqrt(n))
    if arrival == "constant":
        return max(floor, 12.0 / n)
    raise ValueError(f"unknown arrival process: {arrival!r}")


def observed_send_rate(entries: Sequence[Dict[str, Any]]) -> Tuple[float, float]:
    """Recompute (send_duration, achieved_rate) from raw per-request starts.

    Mirrors ``summarize_requests`` exactly (count over the span of send
    times), so the result can be diffed against the reported value to check
    reportgen rather than to restate it.
    """
    starts = sorted(float(e["start_time"]) for e in entries)
    if len(starts) < 2:
        raise ValueError("need at least two requests to observe a send rate")
    send_duration = starts[-1] - starts[0]
    if send_duration <= 0:
        raise ValueError("all requests share one send timestamp; no rate to observe")
    return send_duration, len(starts) / send_duration


def inflight_segments(entries: Sequence[Dict[str, Any]]) -> List[Segment]:
    """Reconstruct the in-flight request count over time as a step function.

    A sweep line over request starts and ends. Ends are applied before starts
    at an identical timestamp, so a tie never credits an extra concurrent
    slot: the reconstruction under-reports rather than over-reports.
    """
    events: List[Tuple[float, int]] = []
    for entry in entries:
        start = float(entry["start_time"])
        end = float(entry["end_time"])
        if end < start:
            raise ValueError(f"request ends before it starts: {start} -> {end}")
        events.append((start, 1))
        events.append((end, -1))
    events.sort(key=lambda ev: (ev[0], ev[1]))

    segments: List[Segment] = []
    inflight = 0
    prev_t = events[0][0]
    for t, delta in events:
        if t > prev_t:
            segments.append((prev_t, t, inflight))
            prev_t = t
        inflight += delta
    return segments


def max_inflight(segments: Sequence[Segment]) -> int:
    return max((n for _, _, n in segments), default=0)


def plateau_window(entries: Sequence[Dict[str, Any]], concurrency: int) -> Tuple[float, float]:
    """Steady-state window of a closed-loop stage, derived not guessed.

    Under a concurrency limit of C, request k cannot start until request k-C
    has finished, so the pipeline is full from the C-th earliest start and
    stays full until the last start, after which only the drain remains.
    ``[C-th earliest start, latest start]`` therefore excludes ramp-up and
    drain by construction, with no hand-tuned margin to tune away a failure.
    """
    if concurrency < 1:
        raise ValueError(f"concurrency must be positive, got {concurrency}")
    starts = sorted(float(e["start_time"]) for e in entries)
    if len(starts) < 2 * concurrency:
        raise ValueError(f"need at least 2*concurrency={2 * concurrency} requests for a plateau, got {len(starts)}")
    return starts[concurrency - 1], starts[-1]


def mean_inflight(segments: Sequence[Segment], window: Tuple[float, float]) -> float:
    """Time-weighted mean in-flight count over ``window``.

    Time weighted, not sample weighted: a run that sits at C for seconds and
    dips to C-1 for microseconds between requests must not read as C-0.5.
    """
    lo, hi = window
    if hi <= lo:
        raise ValueError(f"empty window {window}")
    weighted = 0.0
    covered = 0.0
    for seg_lo, seg_hi, n in segments:
        a, b = max(seg_lo, lo), min(seg_hi, hi)
        if b > a:
            weighted += n * (b - a)
            covered += b - a
    if covered <= 0:
        raise ValueError(f"no in-flight segments overlap window {window}")
    return weighted / covered


def fraction_at_level(segments: Sequence[Segment], window: Tuple[float, float], level: int) -> float:
    """Fraction of ``window`` spent at exactly ``level`` in-flight requests."""
    lo, hi = window
    if hi <= lo:
        raise ValueError(f"empty window {window}")
    at_level = 0.0
    covered = 0.0
    for seg_lo, seg_hi, n in segments:
        a, b = max(seg_lo, lo), min(seg_hi, hi)
        if b > a:
            covered += b - a
            if n == level:
                at_level += b - a
    if covered <= 0:
        raise ValueError(f"no in-flight segments overlap window {window}")
    return at_level / covered


def assert_delivered_concurrency(entries: Sequence[Dict[str, Any]], concurrency: int, *, slack: float = 0.5) -> None:
    """Delivered in-flight concurrency must match the configured level.

    Two claims, both against the configured level as the known-good value:

    - never above it: the semaphore is an upper bound, so this is exact.
    - within ``slack`` of it on time-weighted average across the plateau. The
      default half-slot budget fails an off-by-one distribution bug (C-1 in
      flight) for any C, while absorbing the sub-millisecond gap between one
      request completing and its replacement being dispatched. Measured
      against the sim at C=5 and C=8, that handoff costs about 0.04 of a slot
      (roughly 96% of plateau time sits at exactly C), so the default leaves
      an order of magnitude of headroom over real behaviour and still fails a
      deficit of a whole slot.
    """
    segments = inflight_segments(entries)
    window = plateau_window(entries, concurrency)
    peak = max_inflight(segments)
    assert peak <= concurrency, f"delivered concurrency peaked at {peak}, above the configured limit of {concurrency}"

    mean = mean_inflight(segments, window)
    held = fraction_at_level(segments, window, concurrency)
    assert mean >= concurrency - slack, (
        f"delivered concurrency averaged {mean:.3f} over the plateau, below the configured {concurrency} "
        f"by more than {slack} (only {held:.1%} of plateau time was spent at exactly {concurrency})"
    )
