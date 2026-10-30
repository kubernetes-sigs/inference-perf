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
"""Measured terms: aggregates of finished requests over a window.

A term is written in PromQL's range-vector syntax, ``quantile_over_time(0.99,
ttft[30s])``, or with one of two window functions PromQL has no spelling for:
``<agg>_over_last(series, n)``, the last ``n`` results, and
``<agg>_over_sent(series, a, b)``, requests ``a`` up to ``b`` in dispatch
order. Each finished request is one sample at the stage time it finished, so a
term is computed from the same per-request values the report is, and a
quantile here is the exact quantile rather than a histogram estimate.

:class:`~inference_perf.utils.numeric.expression.Predicate` lifts each term out
of the condition string before SymPy parses the rest, so SymPy sees one symbol
per term. :class:`MeasuredState` feeds finished requests to terms and signals
and reports their values, which ``Predicate.holds`` reads.
"""

from __future__ import annotations

import heapq
import math
import re
from bisect import bisect_left, insort
from collections import deque
from dataclasses import dataclass, field
from typing import AbstractSet, Deque, Dict, List, Mapping, Optional, Sequence, Set, Tuple, Union


@dataclass(frozen=True)
class Series:
    """A per-request value a term can aggregate."""

    name: str
    report_field: Optional[str]
    unit: str
    samples: str


# Series use short names, not the report's field names: a condition reads
# 'quantile_over_time(0.99, ttft[30s])', not '... time_to_first_token[30s]'.
# docs/expressions.md lists the mapping and test_doc_examples.py pins it to this table.
# No inter-token latency: the report re-derives token timestamps from the raw
# chunks at report time (#564), so a live value would differ from the report's.
SERIES: Dict[str, Series] = {
    s.name: s
    for s in (
        Series("latency", "request_latency", "s", "Every successful request: end minus start."),
        Series("ttft", "time_to_first_token", "s", "Successful streamed requests."),
        Series("tpot", "time_per_output_token", "s", "Successful streamed requests with more than one output token."),
        Series("ntpot", "normalized_time_per_output_token", "s", "Successful requests with output tokens."),
        Series("schedule_delay", "schedule_delay", "s", "Every request: how late it was sent against its schedule."),
        Series("error", None, "0 or 1", "Every request: 1 if it failed."),
    )
}

# Read directly, with no window.
SIGNALS: Dict[str, str] = {
    "completed": "Requests finished so far in the stage, successes and failures.",
    "errors": "Failed requests so far in the stage.",
    "in_flight": "Requests sent and not yet finished.",
    "since_last_completion": "Seconds since a request last finished, or since the stage started if none has.",
}

# Each description is its row in docs/expressions.md, and test_doc_examples.py
# checks the doc against these tables in both directions.
AGGREGATES: Dict[str, str] = {
    "avg": "Mean of the samples.",
    "min": "Smallest sample.",
    "max": "Largest sample.",
    "sum": "Sum of the samples.",
    "count": "Number of samples.",
    "stddev": "Population standard deviation, as PromQL computes it.",
    "quantile": "The q-quantile, with q first and between 0 and 1, interpolated linearly as PromQL does.",
}

WINDOWS: Dict[str, str] = {
    "time": "PromQL's window: samples that finished inside a range such as ttft[30s], with optional offset and @.",
    "last": "Ours: the last n samples, written (series, n).",
    "sent": "Ours: requests a up to b in the order they were sent, counting from 0, written (series, a, b).",
}

# Aggregates answered from a sorted copy of the window rather than running sums.
_ORDERED = frozenset({"min", "max", "quantile"})
_SUMMED = frozenset({"sum", "avg", "stddev"})

_TERM_CALL = re.compile(r"\b([A-Za-z0-9]+)_over_([A-Za-z0-9]+)\s*\(")
_PLACEHOLDER = "_term"
_PLACEHOLDER_SPELLING = re.compile(rf"\b{_PLACEHOLDER}\d+\b")
_SELECTOR = re.compile(r"^(?P<series>[A-Za-z_:][A-Za-z0-9_:]*)\s*\[(?P<range>[^\]]*)\](?P<modifiers>.*)$", re.S)
_MODIFIER = re.compile(r"\s*(?:(?P<offset>offset)\s+(?P<duration>[^\s@]+)|@\s*(?P<at>[^\s]+))")
_DURATION_PART = re.compile(r"(\d+)(ms|s|m|h)")
_DURATION_UNITS = {"ms": 0.001, "s": 1.0, "m": 60.0, "h": 3600.0}
_NUMBER = re.compile(r"^(\d+(\.\d*)?|\.\d+)$")
_INTEGER = re.compile(r"^\d+$")
_AT_HINT = (
    "'@' takes the stage second the range ends at, e.g. '@ 30'; PromQL's start(), end() and unix timestamps do not apply."
)


@dataclass(frozen=True)
class Term:
    """One parsed window function, e.g. ``quantile_over_time(0.99, ttft[30s] offset 10s)``.

    ``window`` is ``"time"`` (``range_s``, ``offset_s``, ``at_s``), ``"last"``
    (``last``) or ``"sent"`` (``sent_start`` up to but excluding ``sent_stop``).
    ``source`` is the text as written and does not take part in equality, so
    the same term written twice is evaluated once.
    """

    aggregate: str
    series: str
    window: str
    q: Optional[float] = None
    range_s: Optional[float] = None
    offset_s: float = 0.0
    at_s: Optional[float] = None
    last: Optional[int] = None
    sent_start: Optional[int] = None
    sent_stop: Optional[int] = None
    source: str = field(default="", compare=False)


@dataclass(frozen=True)
class Observation:
    """One finished request, as a measured term sees it.

    ``t`` is the stage time it finished. ``values`` holds the request's value
    for each series it has one for. ``error`` is not passed: it is 1 when
    ``failed`` and 0 otherwise.
    ``sent_index`` is the request's zero-based position in the stage's dispatch
    order, needed only by ``_over_sent`` terms.
    """

    t: float
    failed: bool = False
    values: Mapping[str, float] = field(default_factory=dict)
    sent_index: Optional[int] = None


def extract_terms(raw: str) -> Tuple[str, Dict[str, Term]]:
    """Replace each window function in ``raw`` with a placeholder symbol.

    Returns the rewritten string for SymPy and the term behind each
    placeholder. Raises ``ValueError`` for a malformed term or for PromQL
    syntax left outside one.
    """
    if _PLACEHOLDER_SPELLING.search(raw):
        raise ValueError(f"Predicate {raw!r} uses a name reserved for window functions ({_PLACEHOLDER}<N>).")
    terms: Dict[str, Term] = {}
    out: List[str] = []
    pos = 0
    for match in _TERM_CALL.finditer(raw):
        close = _closing_paren(raw, match.end() - 1)
        if close is None:
            raise ValueError(f"Predicate {raw!r}: {match.group(0)!r} has no closing parenthesis.")
        source = raw[match.start() : close + 1]
        term = parse_term(match.group(1), match.group(2), raw[match.end() : close], source)
        symbol = next((s for s, existing in terms.items() if existing == term), f"{_PLACEHOLDER}{len(terms)}")
        terms.setdefault(symbol, term)
        out.append(raw[pos : match.start()])
        out.append(symbol)
        pos = close + 1
    out.append(raw[pos:])
    rest = "".join(out)
    if "[" in rest or "]" in rest:
        raise ValueError(
            f"Predicate {raw!r} has a range selector outside a window function; "
            f"wrap it in one, e.g. 'avg_over_time(ttft[30s])'."
        )
    if re.search(r"\b(and|or|unless|not|bool)\b", rest):
        raise ValueError(
            f"Predicate {raw!r} uses a PromQL or Python logical keyword; combine comparisons with '&' and '|', "
            f"each parenthesised: '(t >= 600) | (quantile_over_time(0.99, ttft[30s]) > 2)'."
        )
    return rest, terms


def parse_term(aggregate: str, window: str, inner: str, source: str) -> Term:
    """Parse the arguments of ``<aggregate>_over_<window>(...)``."""
    if aggregate not in AGGREGATES:
        hint = " For a median use quantile(0.5, ...)." if aggregate == "median" else ""
        raise ValueError(f"Window function {source!r}: unknown aggregate {aggregate!r}; use one of {list(AGGREGATES)}.{hint}")
    if window not in WINDOWS:
        raise ValueError(f"Window function {source!r}: unknown window {window!r}; use _over_time, _over_last or _over_sent.")
    if re.search(r"@\s*(start|end)\s*\(", inner):
        raise ValueError(f"Window function {source!r}: {_AT_HINT}")
    if "(" in inner or ")" in inner:
        raise ValueError(f"Window function {source!r}: arguments must be constants and a series, not expressions.")
    args = [a.strip() for a in inner.split(",")]
    q: Optional[float] = None
    if aggregate == "quantile":
        if not args or not _NUMBER.match(args[0]):
            raise ValueError(
                f"Window function {source!r}: quantile takes the quantile first, e.g. quantile_over_time(0.99, ...)."
            )
        q = float(args[0])
        if not 0.0 <= q <= 1.0:
            raise ValueError(f"Window function {source!r}: the quantile must be between 0 and 1, got {args[0]}.")
        args = args[1:]
    if window == "time":
        if len(args) != 1:
            raise ValueError(f"Window function {source!r} takes one range selector, e.g. {aggregate}_over_time(ttft[30s]).")
        return _parse_selector(aggregate, q, args[0], source)
    series = _series(args[0] if args else "", source)
    if window == "last":
        if len(args) != 2 or not _INTEGER.match(args[1]) or int(args[1]) < 1:
            raise ValueError(
                f"Window function {source!r} takes a series and a positive count, e.g. {aggregate}_over_last(ttft, 50)."
            )
        return Term(aggregate, series, "last", q=q, last=int(args[1]), source=source)
    if len(args) != 3 or not (_INTEGER.match(args[1]) and _INTEGER.match(args[2])) or int(args[1]) >= int(args[2]):
        raise ValueError(
            f"Window function {source!r} takes a series and two increasing request positions, "
            f"e.g. {aggregate}_over_sent(ttft, 10, 30) for the 11th to 30th request sent."
        )
    return Term(aggregate, series, "sent", q=q, sent_start=int(args[1]), sent_stop=int(args[2]), source=source)


def _parse_selector(aggregate: str, q: Optional[float], selector: str, source: str) -> Term:
    match = _SELECTOR.match(selector)
    if match is None:
        raise ValueError(f"Window function {source!r} needs a range selector, e.g. {aggregate}_over_time(ttft[30s]).")
    series = _series(match.group("series"), source)
    if ":" in match.group("range"):
        raise ValueError(f"Window function {source!r}: subqueries ([range:step]) are not supported.")
    range_s = _duration(match.group("range").strip(), source)
    if range_s <= 0:
        raise ValueError(f"Window function {source!r}: the range must be longer than zero.")
    offset_s: Optional[float] = None
    at_s: Optional[float] = None
    modifiers = match.group("modifiers")
    pos = 0
    while pos < len(modifiers.rstrip()):
        mod = _MODIFIER.match(modifiers, pos)
        if mod is None or mod.end() == pos:
            raise ValueError(f"Window function {source!r}: expected 'offset <duration>' or '@ <seconds>' after the range.")
        if mod.group("offset"):
            if offset_s is not None:
                raise ValueError(f"Window function {source!r} has two offsets.")
            offset_s = _duration(mod.group("duration"), source)
        else:
            if at_s is not None:
                raise ValueError(f"Window function {source!r} has two '@' modifiers.")
            if not _NUMBER.match(mod.group("at")):
                raise ValueError(f"Window function {source!r}: {_AT_HINT}")
            at_s = float(mod.group("at"))
        pos = mod.end()
    if at_s is not None and at_s - (offset_s or 0.0) - range_s < 0:
        raise ValueError(f"Window function {source!r}: the range would start before the stage does, so it never has a value.")
    return Term(aggregate, series, "time", q=q, range_s=range_s, offset_s=offset_s or 0.0, at_s=at_s, source=source)


def _duration(text: str, source: str) -> float:
    """A PromQL duration such as ``30s``, ``1m30s`` or ``500ms``, in seconds."""
    parts = list(_DURATION_PART.finditer(text))
    if not parts or "".join(p.group(0) for p in parts) != text:
        raise ValueError(f"Window function {source!r}: {text!r} is not a duration; write e.g. 30s, 5m, 1m30s or 500ms.")
    return sum(int(p.group(1)) * _DURATION_UNITS[p.group(2)] for p in parts)


def _series(name: str, source: str) -> str:
    if name in SERIES:
        return name
    known = {s.report_field: s.name for s in SERIES.values() if s.report_field}
    hint = f" Series use short names: {name!r} is {known[name]!r}." if name in known else ""
    raise ValueError(f"Window function {source!r}: unknown series {name!r}; use one of {sorted(SERIES)}.{hint}")


def _closing_paren(text: str, open_at: int) -> Optional[int]:
    depth = 0
    for i in range(open_at, len(text)):
        if text[i] == "(":
            depth += 1
        elif text[i] == ")":
            depth -= 1
            if depth == 0:
                return i
    return None


# Every finite float is an integer multiple of 2**-1074, so scaling by 2**1074
# turns running sums into exact integer arithmetic.
_SCALE_BITS = 1074


def _scaled(value: float) -> int:
    num, den = value.as_integer_ratio()
    return num << (_SCALE_BITS - (den.bit_length() - 1))


class _Window:
    """The samples inside one term's window, kept so reading its aggregate never rescans them.

    count, sum, avg and stddev come from running sums held as exact integers,
    so adding and removing samples over a long stage never drifts, and each
    aggregate is rounded once when read: a window of equal samples has a
    stddev of exactly 0. min, max and quantile come from a sorted list.
    """

    def __init__(self, aggregate: str, q: Optional[float]) -> None:
        self.aggregate = aggregate
        self.q = q
        self.count = 0
        self._sum = 0
        self._sumsq = 0
        # the exact sums dominate the per-sample cost, so only keep the ones this aggregate reads
        self._summed = aggregate in _SUMMED
        self._sorted: Optional[List[float]] = [] if aggregate in _ORDERED else None

    def add(self, value: float) -> None:
        self.count += 1
        if self._summed:
            scaled = _scaled(value)
            self._sum += scaled
            self._sumsq += scaled * scaled
        if self._sorted is not None:
            insort(self._sorted, value)

    def remove(self, value: float) -> None:
        self.count -= 1
        if self._summed:
            scaled = _scaled(value)
            self._sum -= scaled
            self._sumsq -= scaled * scaled
        if self._sorted is not None:
            del self._sorted[bisect_left(self._sorted, value)]

    def value(self) -> float:
        n = self.count
        if n == 0:
            return math.nan
        if self.aggregate == "count":
            return float(n)
        # int / int is correctly rounded in Python, however large the operands.
        if self.aggregate == "sum":
            return self._sum / (1 << _SCALE_BITS)
        if self.aggregate == "avg":
            return self._sum / (n << _SCALE_BITS)
        if self.aggregate == "stddev":
            return math.sqrt((n * self._sumsq - self._sum * self._sum) / ((n * n) << (2 * _SCALE_BITS)))
        assert self._sorted is not None
        if self.aggregate == "min":
            return self._sorted[0]
        if self.aggregate == "max":
            return self._sorted[-1]
        # Linear interpolation between closest ranks, as PromQL's quantile_over_time and numpy's default.
        assert self.q is not None
        rank = self.q * (n - 1)
        lo = math.floor(rank)
        hi = min(lo + 1, n - 1)
        return self._sorted[lo] + (self._sorted[hi] - self._sorted[lo]) * (rank - lo)


def _samples(obs: Observation, series: str) -> Sequence[float]:
    if series == "error":
        return (1.0 if obs.failed else 0.0,)
    value = obs.values.get(series)
    # A non-finite value is a broken measurement, not a sample.
    if value is None or not math.isfinite(value):
        return ()
    return (float(value),)


class _TimeEvaluator:
    """``<agg>_over_time(series[range] offset o @ s)``: samples that finished in ``(end - range, end]``.

    ``end`` is ``now - o``, or ``s - o`` with ``@``. The value is unknown until
    the whole range lies inside the stage. Samples wait in ``_pending`` until
    ``end`` passes them and leave ``_active`` once it is ``range`` past them,
    so each sample is added and removed once however many times the term is read.
    """

    def __init__(self, term: Term) -> None:
        assert term.range_s is not None
        self.term = term
        self._range = term.range_s
        self._window = _Window(term.aggregate, term.q)
        self._pending: List[Tuple[float, int, float]] = []
        self._active: List[Tuple[float, int, float]] = []
        self._seq = 0

    def _end(self, now: float) -> float:
        return (self.term.at_s if self.term.at_s is not None else now) - self.term.offset_s

    def observe(self, obs: Observation, now: float) -> None:
        end = self._end(now)
        for value in _samples(obs, self.term.series):
            # A sample past a fixed '@' end can never enter its range.
            if self.term.at_s is not None and obs.t > end:
                continue
            self._seq += 1
            heapq.heappush(self._pending, (obs.t, self._seq, value))
        self._advance(end)

    def _advance(self, end: float) -> None:
        start = end - self._range
        while self._pending and self._pending[0][0] <= end:
            item = heapq.heappop(self._pending)
            heapq.heappush(self._active, item)
            self._window.add(item[2])
        while self._active and self._active[0][0] <= start:
            self._window.remove(heapq.heappop(self._active)[2])

    def value(self, now: float) -> float:
        end = self._end(now)
        if end > now or end - self._range < 0:
            return math.nan
        self._advance(end)
        return self._window.value()


class _LastEvaluator:
    """``<agg>_over_last(series, n)``: the last ``n`` samples to arrive, unknown until there are ``n``."""

    def __init__(self, term: Term) -> None:
        assert term.last is not None
        self.term = term
        self._n = term.last
        self._window = _Window(term.aggregate, term.q)
        self._recent: Deque[float] = deque()

    def observe(self, obs: Observation, now: float) -> None:
        for value in _samples(obs, self.term.series):
            if len(self._recent) == self._n:
                self._window.remove(self._recent.popleft())
            self._recent.append(value)
            self._window.add(value)

    def value(self, now: float) -> float:
        return self._window.value() if len(self._recent) == self._n else math.nan


class _SentEvaluator:
    """``<agg>_over_sent(series, a, b)``: requests ``a`` up to ``b`` in dispatch order.

    Unknown until every one of those requests has finished, then fixed. A
    request with no sample for the series (a failed request has no latency)
    still counts as finished.
    """

    def __init__(self, term: Term) -> None:
        assert term.sent_start is not None and term.sent_stop is not None
        self.term = term
        self._start = term.sent_start
        self._stop = term.sent_stop
        self._window = _Window(term.aggregate, term.q)
        self._finished: Set[int] = set()
        self._value: Optional[float] = None

    def observe(self, obs: Observation, now: float) -> None:
        index = obs.sent_index
        if index is None or not self._start <= index < self._stop or index in self._finished:
            return
        self._finished.add(index)
        for value in _samples(obs, self.term.series):
            self._window.add(value)
        if len(self._finished) == self._stop - self._start:
            self._value = self._window.value()

    def value(self, now: float) -> float:
        return math.nan if self._value is None else self._value


_Evaluator = Union[_TimeEvaluator, _LastEvaluator, _SentEvaluator]


def _evaluator(term: Term) -> _Evaluator:
    if term.window == "time":
        return _TimeEvaluator(term)
    if term.window == "last":
        return _LastEvaluator(term)
    return _SentEvaluator(term)


class MeasuredState:
    """The running value of a set of measured terms and signals over one stage.

    Call :meth:`observe` with each finished request of the stage and
    :meth:`values` for every symbol's value now. ``now`` is stage time and
    must not go backwards. Each sample enters and leaves a term's window once,
    and reading a term never rescans its window, so a large window costs
    memory but not time per result. A predicate reads a state built from its
    own ``terms`` and ``signals``: ``predicate.holds(now, state.values(now))``.
    """

    def __init__(self, terms: Mapping[str, Term], signals: AbstractSet[str] = frozenset()) -> None:
        unknown = set(signals) - set(SIGNALS)
        if unknown:
            raise ValueError(f"Unknown signal(s) {sorted(unknown)}; known: {sorted(SIGNALS)}.")
        self.terms: Dict[str, Term] = dict(terms)
        self.signals = frozenset(signals)
        self.completed = 0
        self.errors = 0
        self._last_finish: Optional[float] = None
        self._now = 0.0
        self._evaluators: Dict[str, _Evaluator] = {symbol: _evaluator(term) for symbol, term in self.terms.items()}

    def _clock(self, now: float) -> float:
        self._now = max(self._now, now)
        return self._now

    def observe(self, obs: Observation, now: float) -> None:
        """Account one finished request, observed at stage time ``now``."""
        now = self._clock(now)
        self.completed += 1
        if obs.failed:
            self.errors += 1
        self._last_finish = obs.t if self._last_finish is None else max(self._last_finish, obs.t)
        for evaluator in self._evaluators.values():
            evaluator.observe(obs, now)

    def values(self, now: float, in_flight: Optional[int] = None) -> Dict[str, float]:
        """Each measured symbol's value at stage time ``now``, NaN while unknown."""
        now = self._clock(now)
        out = {symbol: evaluator.value(now) for symbol, evaluator in self._evaluators.items()}
        signals = self.signals
        if "completed" in signals:
            out["completed"] = float(self.completed)
        if "errors" in signals:
            out["errors"] = float(self.errors)
        if "since_last_completion" in signals:
            out["since_last_completion"] = now - (self._last_finish if self._last_finish is not None else 0.0)
        if "in_flight" in signals:
            out["in_flight"] = math.nan if in_flight is None else float(in_flight)
        return out

    def readings(self, now: float, in_flight: Optional[int] = None) -> Dict[str, float]:
        """:meth:`values` keyed by each term as written, for logs."""
        terms = self.terms
        return {terms[k].source if k in terms else k: v for k, v in self.values(now, in_flight).items()}
