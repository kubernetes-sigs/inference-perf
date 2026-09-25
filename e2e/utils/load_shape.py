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

The golden accuracy helpers in ``utils.accuracy`` check what the tool reports
about each response. These check the other side: the request rate and the
in-flight concurrency the load generator actually delivered.

Everything here works off the raw ``start_time`` / ``end_time`` pairs in
``per_request_lifecycle_metrics.json``, not off the numbers
``summarize_requests`` prints, so the reconstruction is independent of
reportgen and can be used to check reportgen's own derivation.

Limits, stated once so callers do not overclaim:

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
    """How far achieved_rate may sit from the configured rate, for n requests.

    ``constant`` stages get 12/n, ``poisson`` stages get 4/sqrt(n), and
    neither goes below 5%. More requests means a tighter check; the check is
    never loosened by hand to make a run pass.

    Where the numbers come from:

    ``constant``: ``ConstantLoadTimer`` draws n exponential gaps and rescales
    them to sum to exactly ``duration``, so the schedule has no cumulative
    drift. The only random term left in reportgen's
    ``send_duration = max(start) - min(start)`` is the one gap that n points
    do not span. That gap is Exp(1/rate), so the relative error is Exp(1)/n
    and P(error > k/n) = e^-k. k = 12 puts the false-failure rate near 1e-5.

    ``poisson``: ``PoissonLoadTimer`` draws a Poisson(rate) count per second,
    so the time to emit n requests has standard deviation sqrt(n)/rate around
    a mean of n/rate. The relative spread of the achieved rate is therefore
    1/sqrt(n), and 4 sigma is 4/sqrt(n).

    The 5% floor keeps scheduler and socket jitter on a busy shared runner
    from failing the gate. At the request counts the e2e tier uses, the floor
    binds only for ``constant``.
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

    Same arithmetic as reportgen's ``summarize_requests``: request count over
    the span from first send to last send. Because it is the same arithmetic,
    the reported value can be diffed against it to check reportgen.

    It only sees the first and last send. What happens between them is
    ``assert_arrivals_spread``'s job.
    """
    starts = sorted(float(e["start_time"]) for e in entries)
    if len(starts) < 2:
        raise ValueError("need at least two requests to observe a send rate")
    send_duration = starts[-1] - starts[0]
    if send_duration <= 0:
        raise ValueError("all requests share one send timestamp; no rate to observe")
    return send_duration, len(starts) / send_duration


def arrival_bin_counts(entries: Sequence[Dict[str, Any]], bins: int) -> List[int]:
    """Count request starts in each of ``bins`` equal slices of the send window.

    The window runs from the first start to the last, and the last start lands
    in the final slice. A well-spread stage puts about n/bins starts in every
    slice; a burst puts nearly all of them in one.
    """
    if bins < 2:
        raise ValueError(f"need at least two bins to see a spread, got {bins}")
    starts = sorted(float(e["start_time"]) for e in entries)
    if len(starts) < 2:
        raise ValueError("need at least two requests to bin arrivals")
    lo, hi = starts[0], starts[-1]
    if hi <= lo:
        raise ValueError("all requests share one send timestamp; nothing to bin")
    width = (hi - lo) / bins
    counts = [0] * bins
    for t in starts:
        counts[min(int((t - lo) / width), bins - 1)] += 1
    return counts


def spread_tolerance(n: int, bins: int) -> float:
    """How far one slice's count may sit from n/bins, as a fraction of n/bins.

    4.5/sqrt(n/bins). A slice holds a Poisson-like count with mean m = n/bins
    and standard deviation sqrt(m), so this is 4.5 sigma per slice. It applies
    to both timers: ``constant`` also draws exponential gaps, it only pins
    their total, which makes its slice counts slightly less variable than
    Poisson. With 5 slices, the chance that any slice trips this on a correct
    run is about 3e-5.
    """
    if bins < 2:
        raise ValueError(f"need at least two bins to see a spread, got {bins}")
    m = n / bins
    if m < 1:
        raise ValueError(f"n={n} requests is too few for {bins} bins")
    return 4.5 / math.sqrt(m)


def assert_arrivals_spread(entries: Sequence[Dict[str, Any]], *, bins: int = 5) -> None:
    """Requests must be spread across the send window, not bunched.

    ``observed_send_rate`` only sees the first and last send, so one request
    at t=0 and all the rest at t=duration still reads as the configured rate.
    This closes that hole: each of ``bins`` equal slices of the send window
    must hold n/bins requests within ``spread_tolerance``. It catches a stall
    or burst of roughly half a slice or longer; finer unevenness is the
    timer's normal randomness.
    """
    counts = arrival_bin_counts(entries, bins)
    n = sum(counts)
    expected = n / bins
    tolerance = spread_tolerance(n, bins)
    worst = max(counts, key=lambda c: abs(c - expected))
    error = abs(worst - expected) / expected
    assert error <= tolerance, (
        f"requests are not spread over the send window: per-slice counts {counts}, "
        f"expected about {expected:.0f} each, worst slice is {error:.0%} off (tolerance {tolerance:.0%})"
    )


def inflight_segments(entries: Sequence[Dict[str, Any]]) -> List[Segment]:
    """Turn start/end pairs into a step function of how many requests were in flight.

    A sweep line over request starts and ends. At an identical timestamp the
    end is applied before the start, so a handoff never counts as an extra
    concurrent slot: the reconstruction under-reports rather than over-reports.
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
    """The part of a closed-loop run where the pipeline is full.

    The window is ``[C-th earliest start, latest start]``. Under a concurrency
    limit of C, request k cannot start until request k-C has finished, so the
    pipeline is full from the C-th start and stays full until the last start,
    after which only the drain remains. That excludes ramp-up and drain by
    construction, with no hand-tuned margin to tune away a failure.
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
    """Delivered in-flight concurrency must match the configured level C.

    Two checks: in-flight never goes above C, and the time-weighted average
    over the plateau stays within ``slack`` of C.

    The first is exact because the semaphore is an upper bound. For the
    second, half a slot is the default because it fails an off-by-one
    distribution bug (C-1 in flight) for any C while absorbing the
    sub-millisecond gap between one request completing and its replacement
    being dispatched. Measured against the sim at C=5 and C=8, that handoff
    costs about 0.04 of a slot (roughly 96% of plateau time sits at exactly
    C), so the default has about 10x headroom over real behaviour and still
    fails a deficit of a whole slot.
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
