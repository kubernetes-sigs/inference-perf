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
"""Piecewise-linear expressions of stage time ``t``.

A :class:`PiecewiseLinear` is an expression that is a straight line in ``t``
between finitely many breakpoints, e.g. ``"Min(1 + t/6, 64)"`` or
``"Piecewise((8, t < 60), (32, True))"``. It shares the :class:`Expression`
grammar but admits only the nodes that keep that shape, and it is solved at
construction into an explicit list of :class:`Segment` objects. Everything a
caller needs afterwards (the value at any ``t``, the exact bounds, the exact
time a line crosses a value) is then closed-form arithmetic on the segments,
with no numerical search and no sympy at runtime.

It exists for knobs that are set points rather than averages, such as a
concurrency level: a set point has to change at exact, provable times, which a
general expression (``sin``, ``t**2``, ...) cannot promise.
"""

from __future__ import annotations

import bisect
import operator
from dataclasses import dataclass
from typing import Any, Callable, Sequence, Union

import sympy
from sympy.core.relational import Relational
from sympy.logic.boolalg import ITE, And, BooleanFalse, BooleanTrue, Not, Or

from .expression import _T, _is_condition, _is_random_symbol, _parse_raw, _reject_unknown_functions

_INF = float("inf")

# The sympy node types a piecewise-linear expression may contain, besides
# constant subtrees such as sqrt(2), which are always allowed. Each one maps
# piecewise-linear inputs to a piecewise-linear output, except a product of
# two terms in t: products are checked per segment when the expression is
# solved, which accepts t * Heaviside(t - 30) (0, then t) and rejects
# t * Min(t, 5) (t**2 before t = 5).
_PIECEWISE_LINEAR_NODES: tuple[type, ...] = (
    sympy.Symbol,
    sympy.Add,
    sympy.Mul,
    sympy.Min,
    sympy.Max,
    sympy.Abs,
    sympy.Heaviside,
    sympy.Piecewise,
    sympy.functions.elementary.piecewise.ExprCondPair,
    Relational,
    And,
    Or,
    Not,
    # sympy's own spelling of a comparison against a nested Piecewise.
    ITE,
    BooleanTrue,
    BooleanFalse,
)

_PERMITTED = "numbers, t, + - * /, Min, Max, Abs, Heaviside, and Piecewise with conditions such as t < 60"

# How a comparison reads the sign of ``lhs - rhs``.
_COMPARISONS: dict[type, Callable[[float, float], bool]] = {
    sympy.StrictLessThan: operator.lt,
    sympy.LessThan: operator.le,
    sympy.StrictGreaterThan: operator.gt,
    sympy.GreaterThan: operator.ge,
    sympy.Equality: operator.eq,
    sympy.Unequality: operator.ne,
}


@dataclass(frozen=True)
class Segment:
    """The line ``intercept + slope * t`` on ``[start, end)``; ``end`` is ``inf`` for the last one."""

    start: float
    end: float
    slope: float
    intercept: float

    def value(self, t: float) -> float:
        return self.intercept + self.slope * t

    @property
    def probe(self) -> float:
        """A time strictly inside the segment, where no breakpoint can sit."""
        return self.start + 1.0 if self.end == _INF else (self.start + self.end) / 2


class PiecewiseLinear:
    """A deterministic expression that is linear in ``t`` between breakpoints.

    Construction parses ``raw`` with the :class:`Expression` grammar, rejects
    any node that could bend the line (``t**2``, ``1/t``, ``sin``, ``exp``,
    random variables, ...), and solves the expression over ``t >= 0`` into
    :attr:`segments`, node by node: a sum adds lines, ``Min``/``Max`` split
    where their arguments cross, ``Abs``/``Heaviside`` split where their
    argument changes sign, and ``Piecewise`` splits where a condition changes
    truth value. Every split point is the root of a line, so it is exact.

    The value between breakpoints is right-continuous: at a breakpoint the new
    segment applies. What the expression does at the single instant of a jump
    (``Heaviside(0)`` is ``1/2`` in sympy) is deliberately ignored, since no
    caller can act on one instant.

    Raises:
        ValueError: If the expression is unparseable, random, uses a symbol
            other than ``t``, or is not piecewise linear.
    """

    def __init__(self, raw: Union[str, int, float]) -> None:
        self.raw = raw
        expr = _parse_raw("Expression", raw)
        if _is_condition(expr):
            raise ValueError(f"Expression {raw!r} is a condition, not a numeric value.")
        _reject_unknown_functions("Expression", raw, expr)
        free = expr.free_symbols
        if any(_is_random_symbol(s) for s in free):
            raise ValueError(f"Expression {raw!r} contains a random variable; a piecewise-linear value must be deterministic.")
        disallowed = {str(s) for s in free} - {"t"}
        if disallowed:
            raise ValueError(f"Expression {raw!r} uses disallowed symbol(s) {sorted(disallowed)}; permitted: ['t'].")
        _reject_nonlinear_nodes(raw, expr)

        self.segments: list[Segment] = _Solver(raw).solve(expr)
        self._starts = [segment.start for segment in self.segments]

    @property
    def is_constant(self) -> bool:
        """True when the value never changes: one segment with no slope."""
        return len(self.segments) == 1 and self.segments[0].slope == 0

    def segment_at(self, t: float) -> Segment:
        """The segment in effect at ``t >= 0`` (the later one at a breakpoint)."""
        if t < 0:
            raise ValueError(f"t must be >= 0, got {t}.")
        return self.segments[bisect.bisect_right(self._starts, t) - 1]

    def value(self, t: float) -> float:
        """The value at ``t >= 0``."""
        return self.segment_at(t).value(t)

    @property
    def bounds(self) -> tuple[float, float]:
        """Exact ``(lower, upper)`` over ``t >= 0``, including limits at breakpoints.

        A line is extreme at its ends, so the bounds are the segment endpoint
        values. A last segment with a slope is unbounded on that side.
        """
        lows: list[float] = []
        highs: list[float] = []
        for segment in self.segments:
            ends = [segment.value(segment.start)]
            if segment.end == _INF:
                if segment.slope > 0:
                    highs.append(_INF)
                elif segment.slope < 0:
                    lows.append(-_INF)
            else:
                ends.append(segment.value(segment.end))
            lows.append(min(ends))
            highs.append(max(ends))
        return min(lows), max(highs)

    def __repr__(self) -> str:
        return f"PiecewiseLinear({self.raw!r})"


def _is_constant_subtree(node: Any) -> bool:
    return isinstance(node, sympy.Expr) and not node.free_symbols and bool(node.is_number)


def _reject_nonlinear_nodes(raw: Any, expr: Any) -> None:
    """Reject any node outside ``_PIECEWISE_LINEAR_NODES`` that depends on ``t``, naming it."""
    pending: list[Any] = [expr]
    while pending:
        node: Any = pending.pop()
        # Read before the isinstance check below, which narrows node to object.
        children = node.args
        if _is_constant_subtree(node):
            continue
        if isinstance(node, sympy.Pow):
            raise ValueError(
                f"Expression {raw!r} uses {node}, which is not piecewise linear; t may not appear in a power or a denominator."
            )
        if not isinstance(node, _PIECEWISE_LINEAR_NODES):
            raise ValueError(
                f"Expression {raw!r} uses {node.func.__name__!r}, which is not piecewise linear; permitted: {_PERMITTED}."
            )
        pending.extend(children)


def _root(segment: Segment) -> list[float]:
    """Where the segment's line crosses zero, if strictly inside the segment."""
    if segment.slope == 0:
        return []
    root = -segment.intercept / segment.slope
    return [root] if segment.start < root < segment.end else []


def _align(functions: Sequence[list[Segment]], extra: Sequence[float] = ()) -> list[tuple[float, float, list[Segment]]]:
    """Cut ``[0, inf)`` at every breakpoint of every function (and at ``extra``).

    Returns ``(start, end, segments)`` with one segment per function, each the
    one in effect on ``[start, end)``.
    """
    edges = sorted({s.start for f in functions for s in f} | {p for p in extra if p > 0})
    edges.append(_INF)
    starts = [[s.start for s in f] for f in functions]
    out = []
    for start, end in zip(edges, edges[1:], strict=False):
        active = [f[bisect.bisect_right(st, start) - 1] for f, st in zip(functions, starts, strict=True)]
        out.append((start, end, active))
    return out


def _merge(pieces: list[Segment]) -> list[Segment]:
    """Join neighbours that are the same line."""
    merged: list[Segment] = []
    for piece in pieces:
        last = merged[-1] if merged else None
        if last is not None and last.slope == piece.slope and last.intercept == piece.intercept:
            merged[-1] = Segment(last.start, piece.end, last.slope, last.intercept)
        else:
            merged.append(piece)
    return merged


def _constant(value: float) -> list[Segment]:
    return [Segment(0.0, _INF, 0.0, value)]


class _Solver:
    """Turns an allowlisted sympy tree into segments; booleans are 0/1 constants."""

    def __init__(self, raw: Any) -> None:
        self.raw = raw

    def solve(self, node: Any) -> list[Segment]:
        if _is_constant_subtree(node):
            return _constant(float(node))
        if node == _T:
            return [Segment(0.0, _INF, 1.0, 0.0)]
        if isinstance(node, BooleanTrue):
            return _constant(1.0)
        if isinstance(node, BooleanFalse):
            return _constant(0.0)
        if isinstance(node, sympy.Add):
            return self._combine(node.args, lambda lines: (sum(s for s, _ in lines), sum(c for _, c in lines)))
        if isinstance(node, sympy.Mul):
            return self._combine(node.args, self._product(node))
        if isinstance(node, (sympy.Min, sympy.Max, And, Or)):
            return self._extreme(node.args, lowest=isinstance(node, (sympy.Min, And)))
        if isinstance(node, sympy.Abs):
            return self._by_sign(
                node.args[0], lambda seg, v: (-seg.slope, -seg.intercept) if v < 0 else (seg.slope, seg.intercept)
            )
        if isinstance(node, sympy.Heaviside):
            return self._by_sign(
                node.args[0], lambda seg, v: (0.0, 1.0 if v > 0 else 0.0 if v < 0 else float(node.func(0, *node.args[1:])))
            )
        if isinstance(node, Relational):
            compare = _COMPARISONS[type(node)]
            return self._by_sign(node.lhs - node.rhs, lambda seg, v: (0.0, 1.0 if compare(v, 0.0) else 0.0))
        if isinstance(node, Not):
            return self._combine(node.args, lambda lines: (0.0, 1.0 - lines[0][1]))
        if isinstance(node, ITE):
            return self._combine(node.args, lambda lines: (0.0, lines[1][1] if lines[0][1] > 0.5 else lines[2][1]))
        if isinstance(node, sympy.Piecewise):
            return self._piecewise(node)
        raise ValueError(f"Expression {self.raw!r} uses {node}, which is not piecewise linear; permitted: {_PERMITTED}.")

    def _combine(self, args: Sequence[Any], line: Callable[[list[tuple[float, float]]], tuple[float, float]]) -> list[Segment]:
        pieces = []
        for start, end, active in _align([self.solve(a) for a in args]):
            slope, intercept = line([(s.slope, s.intercept) for s in active])
            pieces.append(Segment(start, end, slope, intercept))
        return _merge(pieces)

    def _product(self, node: Any) -> Callable[[list[tuple[float, float]]], tuple[float, float]]:
        def line(lines: list[tuple[float, float]]) -> tuple[float, float]:
            slope, intercept = 0.0, 1.0
            for s, c in lines:
                if slope != 0 and s != 0:
                    raise ValueError(
                        f"Expression {self.raw!r} multiplies terms in t in {node}, which is not linear in t there; "
                        f"t may only be scaled by a constant."
                    )
                slope, intercept = slope * c + s * intercept, intercept * c
            return slope, intercept

        return line

    def _extreme(self, args: Sequence[Any], lowest: bool) -> list[Segment]:
        functions = [self.solve(a) for a in args]
        pieces = []
        for start, end, active in _align(functions):
            # Lines can cross inside the segment; cut there so one line wins each part.
            cuts = {start, end}
            for i, a in enumerate(active):
                for b in active[i + 1 :]:
                    cuts.update(_root(Segment(start, end, a.slope - b.slope, a.intercept - b.intercept)))
            edges = sorted(cuts)
            for lo, hi in zip(edges, edges[1:], strict=False):
                probe = Segment(lo, hi, 0.0, 0.0).probe
                values = [s.value(probe) for s in active]
                best = active[values.index(min(values) if lowest else max(values))]
                pieces.append(Segment(lo, hi, best.slope, best.intercept))
        return _merge(pieces)

    def _by_sign(self, arg: Any, line: Callable[[Segment, float], tuple[float, float]]) -> list[Segment]:
        """Cut where ``arg`` crosses zero, then build each part from the sign of ``arg`` there."""
        inner = self.solve(arg)
        pieces = []
        for start, end, (segment,) in _align([inner], extra=[r for s in inner for r in _root(s)]):
            part = Segment(start, end, segment.slope, segment.intercept)
            slope, intercept = line(part, part.value(part.probe))
            pieces.append(Segment(start, end, slope, intercept))
        return _merge(pieces)

    def _piecewise(self, node: Any) -> list[Segment]:
        branches = [(self.solve(value), self.solve(condition)) for value, condition in node.args]
        functions = [f for pair in branches for f in pair]
        pieces = []
        for start, end, active in _align(functions):
            probe = Segment(start, end, 0.0, 0.0).probe
            for value, condition in zip(active[0::2], active[1::2], strict=True):
                if condition.value(probe) > 0.5:
                    pieces.append(Segment(start, end, value.slope, value.intercept))
                    break
            else:
                raise ValueError(
                    f"Expression {self.raw!r} has no value at t={probe:g}; give the last Piecewise branch the condition True."
                )
        return _merge(pieces)
