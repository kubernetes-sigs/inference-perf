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
"""First-class numeric expressions for config values.

An :class:`Expression` turns a numeric config knob (a constant, a function of
stage time ``t``, and/or a draw from a random variable) into a
single object that is parsed and validated once at construction and sampled
many times. It is the string-grammar counterpart to
:mod:`inference_perf.utils.numeric.distribution`, which remains the fast,
vectorised path for structured ``Distribution`` configs.

Grammar: constants, the single time variable ``t``, standard math, and the
distribution constructors that have a row in :data:`.distributions.DISTRIBUTIONS`
(``Normal(512, 200)``, ``Uniform(10, 50)``, ``Poisson(10)``, ...). Any other
``sympy.stats`` distribution is rejected by name. Each call site decides whether ``t``
and random variables are permitted, and may constrain the value range. Bounds
are assertions by default: a value that can provably escape them is rejected at
construction, and an escape only detectable at sample time raises there. A call
site that would rather truncate random draws, e.g. so a request-rate expression
can never go negative, opts in with ``clip=True``.

A :class:`Predicate` is the boolean counterpart: a condition over ``t`` such as
``"t >= 60"``, for a field that waits for something to become true. It shares
the grammar and parser but is validated differently: it must be false at
``t = 0`` and then hold from some time onward, which is proved statically so
the condition compiles to an exact boundary instead of being polled. A predicate may also read measured terms
(``quantile_over_time(0.99, ttft[30s])``, see :mod:`.measured`), which are the
system under test and cannot be solved: they are evaluated as results arrive,
and the proof is made with their comparisons all true and all false,
bracketing when the condition can and must hold.
"""

from __future__ import annotations

import inspect
import logging
import math
import re
from tokenize import TokenError
from typing import Callable, Iterator, Mapping, NoReturn, Optional, TypeGuard, Union

import numpy as np
import sympy
import sympy.stats
from numpy.typing import ArrayLike, NDArray
from sympy import AccumBounds, Interval, Symbol, oo
from sympy.calculus.util import function_range
from sympy.core.function import AppliedUndef
from sympy.core.relational import GreaterThan, LessThan, Relational, StrictGreaterThan, StrictLessThan
from sympy.logic.boolalg import And, Boolean, Or
from sympy.parsing.sympy_parser import parse_expr

from .distributions import DISTRIBUTIONS, Distribution, DistributionUse, lookup, render_condition, unsupported_message
from .measured import SERIES, SIGNALS, Term, extract_terms
from .sympy_types import SympyExpr, SympyNode, SympyRandomSymbol, SympySet

logger = logging.getLogger(__name__)

# The single time variable the grammar ever permits.
_T = Symbol("t")


def _distribution_constructors() -> dict[str, Callable[..., SympyRandomSymbol]]:
    """Collect every ``sympy.stats`` distribution constructor (name-first)."""
    ctors: dict[str, Callable[..., SympyRandomSymbol]] = {}
    for name in dir(sympy.stats):
        if name.startswith("_") or not name[0].isupper():
            continue
        obj = getattr(sympy.stats, name)
        try:
            params = list(inspect.signature(obj).parameters)
        except (TypeError, ValueError):
            continue
        if params and params[0] == "name":
            ctors[name] = obj
    return ctors


# Built once at import; the set of sympy.stats distributions is static. Only
# the ones with a row in DISTRIBUTIONS are callable from the grammar: the rest
# are kept so that using one is rejected by name.
_DISTRIBUTION_CTORS = _distribution_constructors()


def _unsupported(name: str) -> Callable[..., NoReturn]:
    def reject(*args: object) -> NoReturn:
        raise ValueError(unsupported_message(name))

    return reject


def _parse_namespace(counter: list[int]) -> dict[str, Callable[..., SympyRandomSymbol]]:
    """A fresh distribution namespace whose constructors take no name argument.

    Config strings read ``Normal(512, 200)`` rather than ``Normal("x", ...)``,
    so each constructor binds a name for us. Crucially the name is *unique per
    occurrence* (``rv_1``, ``rv_2``, ...) via the shared ``counter``, so that an
    expression with two draws of the same distribution (e.g. a skew-normal built
    from two independent normals) yields two independent random variables rather
    than one aliased symbol.

    A constructor is callable only when its distribution has a row in
    ``DISTRIBUTIONS``, and then only with constant arguments that meet the
    row's conditions. Every other ``sympy.stats`` name raises.
    """
    namespace: dict[str, Callable[..., SympyRandomSymbol]] = {name: _unsupported(name) for name in _DISTRIBUTION_CTORS}
    for name, row in DISTRIBUTIONS.items():

        def bound(
            *args: object,
            _row: Distribution = row,
            _ctor: Callable[..., SympyRandomSymbol] = _DISTRIBUTION_CTORS[name],
            _counter: list[int] = counter,
        ) -> SympyRandomSymbol:
            use = _row.use(args)
            use.require_constant()
            _counter[0] += 1
            return _ctor(f"rv_{_counter[0]}", *use.constructor_args)

        namespace[name] = bound
    return namespace


def _use_of(rv: SympyRandomSymbol) -> DistributionUse:
    """The table row and arguments behind a random variable in a parsed tree.

    Raises ``ValueError`` for a distribution with no row. The parse namespace
    already refuses those, so this is the check that still holds if a tree is
    ever built some other way.
    """
    distribution = rv.pspace.distribution
    row = lookup(distribution)
    if row is None:
        raise ValueError(unsupported_message(type(distribution).__name__.removesuffix("Distribution")))
    return row.use_of_sympy(distribution.args)


def _is_random_symbol(sym: object) -> TypeGuard[SympyRandomSymbol]:
    return isinstance(sym, sympy.stats.rv.RandomSymbol)


def _is_node(value: object) -> TypeGuard[SympyNode]:
    return isinstance(value, sympy.Basic)


def _is_expr(value: object) -> TypeGuard[SympyExpr]:
    return isinstance(value, sympy.Expr)


def _parse_raw(kind: str, raw: Union[str, int, float], shown: Optional[str] = None) -> object:
    """Parse a config value into a sympy object, shared by both grammars.

    ``shown`` is the text to name in a parse error when ``raw`` has been
    rewritten from it, so the author sees their own spelling. The result is
    whatever ``parse_expr`` built, which is a Python bool, tuple or list for
    input like ``True``, ``1, 2`` or ``[1]``, so callers check it.
    """
    if isinstance(raw, bool):
        raise TypeError(f"{kind} does not accept bool input.")
    if isinstance(raw, (int, float)):
        return sympy.sympify(raw)
    if isinstance(raw, str):
        # parse_expr ultimately eval()s the transformed string, so an
        # expression is only as trustworthy as the config it came from.
        # These strings are config-author input, never end-user input.
        try:
            return parse_expr(raw, local_dict=_parse_namespace([0]))
        except (SyntaxError, TypeError, AttributeError, sympy.SympifyError) as e:
            raise ValueError(f"Could not parse {kind.lower()} {shown if shown is not None else raw!r}: {e}") from e
    raise TypeError(f"{kind} accepts str or number, got {type(raw).__name__}.")


def _interval(expr: SympyNode, duration: Optional[float]) -> Optional[tuple[SympyExpr, SympyExpr]]:
    """Sound ``(lower, upper)`` sympy bounds for ``expr``, or ``None`` when undecidable."""
    if _is_expr(expr) and (expr.is_Number or expr.is_NumberSymbol):
        return expr, expr
    if _is_random_symbol(expr):
        use = _use_of(expr)
        return use.lower, use.upper
    if expr == _T:
        return sympy.S.Zero, (sympy.Float(duration) if duration is not None else oo)
    found = [_interval(arg, duration) for arg in expr.args]
    parts = [part for part in found if part is not None]
    if len(parts) != len(found):
        return None
    if isinstance(expr, sympy.Abs):
        lo, hi = parts[0]
        if lo >= 0:
            return lo, hi
        if hi <= 0:
            return -hi, -lo
        return sympy.S.Zero, sympy.Max(-lo, hi)
    if isinstance(expr, (sympy.Min, sympy.Max)):
        # Min/Max are monotone in every argument, and AccumBounds does not
        # reduce them, so combine the endpoints directly.
        return expr.func(*(lo for lo, _hi in parts)), expr.func(*(hi for _lo, hi in parts))
    if isinstance(expr, (sympy.floor, sympy.ceiling)):
        # monotone, so the endpoints map straight through; AccumBounds has no rule for them
        lo, hi = parts[0]
        rounding = sympy.floor if isinstance(expr, sympy.floor) else sympy.ceiling
        return rounding(lo), rounding(hi)
    try:
        result = expr.func(*(lo if lo == hi else AccumBounds(lo, hi) for lo, hi in parts))
    except (TypeError, ValueError, NotImplementedError):
        return None
    if isinstance(result, AccumBounds):
        return result.min, result.max
    if _is_expr(result) and result.is_number and result.is_extended_real:
        return result, result
    return None


def _support(use: DistributionUse) -> SympySet:
    """The values a draw can take: the whole numbers or the reals from the row's lowest to its highest."""
    if use.row.takes_list:
        listed: SympySet = sympy.FiniteSet(*use.args)
        return listed
    lower, upper = use.lower, use.upper
    if use.row.integer:
        whole: SympySet = sympy.Range(sympy.ceiling(lower), upper if upper == oo else sympy.Add(sympy.floor(upper), 1))
        return whole
    reals: SympySet = Interval(lower, upper)
    return reals


def _is_condition(expr: object) -> bool:
    """True for a truth value or a relational/logical combination, false for a number.

    A bare ``isinstance(expr, Boolean)`` is not enough: sympy's ``Symbol``
    subclasses ``Boolean`` (so it can stand in ``And(x, y)``), which would
    make the numeric expression ``"t"`` look like a condition.
    """
    return isinstance(expr, (bool, Boolean)) and not isinstance(expr, sympy.Expr)


def _reject_unknown_functions(kind: str, raw: object, expr: SympyNode) -> None:
    """Reject unknown functions, e.g. a misspelled distribution InvalidDist(10)."""
    undefined = {f.func.__name__ for f in expr.atoms(AppliedUndef)}
    if undefined:
        raise ValueError(f"{kind} {raw!r} uses unknown function(s): {sorted(undefined)}.")


class Expression:
    """A numeric config value: constant, time-varying (``t``), and/or random.

    The expression is parsed and validated when constructed, then sampled via
    :meth:`sample`. Construction raises :class:`ValueError` for unparseable
    input, unknown symbols or functions, a disallowed time variable or random
    variable, or a value range that is provably outside ``[minimum, maximum]``.

    Args:
        raw: The expression as a string (``"10 + 5*sin(2*pi*t/60)"``) or a bare
            number.
        allow_time: Whether the time variable ``t`` may appear. ``t`` is the
            only variable the grammar permits; when ``False`` the expression
            must be constant (and/or random).
        allow_random: Whether random variables may appear.
        minimum: Lower bound for the value, or ``None`` for unbounded. Bounds
            are assertions: a value that can provably escape them is rejected
            at construction, and an escape only detectable at sample time
            raises there. See ``clip`` for the truncating alternative.
        maximum: Upper bound for the value, or ``None`` for unbounded.
        clip: Truncate instead of assert for *random* draws: samples are
            clamped into ``[minimum, maximum]``. This piles the out-of-range
            probability mass onto the bounds, which is why it is opt-in.
            Requires at least one bound, and rejects at construction an
            expression whose value provably never intersects the bounds
            (every draw would collapse onto a single bound). Deterministic
            values are always asserted, never clamped, even with ``clip``.
        duration: The upper end of the ``t`` domain (stage duration, seconds).
            Required to statically bound a time-varying expression; without it
            a bounded time-varying expression is checked at sample time instead.
    """

    def __init__(
        self,
        raw: Union[str, int, float],
        *,
        allow_time: bool = True,
        allow_random: bool = True,
        minimum: Optional[float] = None,
        maximum: Optional[float] = None,
        clip: bool = False,
        duration: Optional[float] = None,
    ) -> None:
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"minimum ({minimum}) cannot be greater than maximum ({maximum}).")
        if clip and minimum is None and maximum is None:
            raise ValueError("clip=True requires minimum and/or maximum to clip to.")

        self.raw = raw
        self.minimum = minimum
        self.maximum = maximum
        self.clip = clip
        self._duration = duration
        self._expr = self._parse(raw)
        _reject_unknown_functions("Expression", raw, self._expr)

        free = self._expr.free_symbols
        self._random_symbols = sorted((s for s in free if _is_random_symbol(s)), key=str)
        # every random variable must have a row: this raises for one that does not
        self._uses = {rv: _use_of(rv) for rv in self._random_symbols}
        self._is_random = bool(self._random_symbols)
        self._free_symbols = {str(s) for s in free if not _is_random_symbol(s)}

        if self._is_random and not allow_random:
            raise ValueError(f"Expression {raw!r} contains a random variable, which is not permitted here.")

        allowed = {"t"} if allow_time else set()
        disallowed = self._free_symbols - allowed
        if disallowed:
            permitted = sorted(allowed) if allowed else "constants only"
            raise ValueError(f"Expression {raw!r} uses disallowed symbol(s) {sorted(disallowed)}; permitted: {permitted}.")

        # Resolve the range policy once, raising now for provable violations.
        self._clip_random = clip and self._is_random
        self._runtime_check = False
        self._validate_range(duration)
        self._compile_random_sampler()
        self._evaluator: Optional[Callable[[NDArray[np.float64]], ArrayLike]] = None

    def _compile_random_sampler(self) -> None:
        """Precompute a numpy-vectorised sampler for the random fast path.

        Splits the parsed tree into random *leaves* (the ``RandomSymbol`` atoms)
        and a deterministic *skeleton* (the arithmetic/time combination of them).
        Each leaf is drawn through its row's numpy sampler, and the skeleton is
        compiled once with ``sympy.lambdify``. At sample time we draw the
        leaves with numpy and feed them through the lambdified closure, so
        sympy's own sampling is never involved. Every row has a sampler, so
        this covers every expression the grammar accepts, including
        ``Piecewise`` mixtures, which lambdify to a numpy select.
        """
        self._random_leaves: list[tuple[SympyNode, DistributionUse]] = []
        self._lambdified: Optional[Callable[..., ArrayLike]] = None
        if not self._is_random:
            return

        placeholders: dict[SympyNode, SympyNode] = {}
        for i, rv in enumerate(self._random_symbols):
            placeholder = Symbol(f"_leaf_{i}")
            placeholders[rv] = placeholder
            self._random_leaves.append((placeholder, self._uses[rv]))

        skeleton = self._expr.xreplace(placeholders)
        order = [ph for ph, _use in self._random_leaves] + [_T]
        self._lambdified = sympy.lambdify(order, skeleton, "numpy")

    def _parse(self, raw: Union[str, int, float]) -> SympyExpr:
        expr = _parse_raw("Expression", raw)
        if _is_condition(expr):
            raise ValueError(f"Expression {raw!r} is a condition, not a numeric value; use Predicate for conditions.")
        if not _is_expr(expr):
            raise ValueError(
                f"Expression {raw!r} is not a numeric value; write a number or an expression such as '512 + 2*t'."
            )
        return expr

    def _validate_range(self, duration: Optional[float]) -> None:
        if self.minimum is None and self.maximum is None:
            return

        bounds = Interval(
            sympy.Float(self.minimum) if self.minimum is not None else -oo,
            sympy.Float(self.maximum) if self.maximum is not None else oo,
        )
        static_range = self._static_range(duration)

        if static_range is None:
            # Undecided: defer to sample time. With ``clip`` random draws are
            # clamped; otherwise any out-of-range value raises when sampled.
            self._runtime_check = not self._clip_random
            return

        if self._clip_random:
            # Truncation is opted into, so a support wider than the bounds is
            # fine; only a support entirely outside them is rejected, because
            # every draw would collapse onto a single bound.
            if static_range.intersect(bounds) == sympy.S.EmptySet:
                raise ValueError(
                    f"Expression {self.raw!r} evaluates within {static_range}, which never "
                    f"intersects the permitted range {bounds}; clipping would collapse every draw to a bound."
                )
            return

        contained = static_range.is_subset(bounds)
        if contained is True:
            return
        if contained is False:
            # The hint is for someone editing a config, who has no clip
            # switch: keep the value in range inside the expression.
            hint = (
                " Keep it in range with Min(...) or Max(...), or pick a distribution bounded to the range."
                if self._is_random
                else ""
            )
            raise ValueError(
                f"Expression {self.raw!r} can evaluate to {static_range}, which is outside the permitted range {bounds}.{hint}"
            )
        # is_subset returned None (couldn't decide): treat as undecided.
        self._runtime_check = True

    def _static_range(self, duration: Optional[float]) -> Optional[SympySet]:
        """Best-effort provable value range, or ``None`` when undecidable."""
        if self._is_random:
            # Only a bare random variable has a readily available support set.
            # Transformed/mixed random expressions are left for the sample-time
            # clamp.
            if _is_random_symbol(self._expr):
                return _support(self._uses[self._expr])
            return None

        if not self._free_symbols:
            value = float(self._expr.evalf())
            point: SympySet = Interval(value, value)
            return point

        # Time-varying: the range is only knowable over a bounded t domain.
        if duration is None:
            return None
        try:
            values: SympySet = function_range(self._expr, _T, Interval(0, sympy.Float(duration)))
        except Exception:
            return None
        return values

    @property
    def bounds(self) -> Optional[tuple[float, float]]:
        """Provable ``(lower, upper)`` bounds on every value the expression can take.

        Worked out by interval arithmetic over the tree: each random variable
        contributes its support, ``t`` contributes ``[0, duration]`` (or
        ``[0, inf)`` without a duration), and ``Min``/``Max`` combine their
        arguments' bounds. The result is sound but can be wider than the true
        range, e.g. when one variable appears twice. An unbounded side is
        ``inf``/``-inf``. ``None`` means some part of the tree has no interval
        rule, so nothing can be proven.

        This is what a caller needs when it must budget for the worst case,
        e.g. ``Min(LogNormal(6, 0.5), 4096)`` is at most 4096.
        """
        found = _interval(self._expr, self._duration)
        if found is None:
            return None
        return float(found[0]), float(found[1])

    def is_within(self, minimum: Optional[float] = None, maximum: Optional[float] = None) -> bool:
        """True when :attr:`bounds` proves every value lies in ``[minimum, maximum]``.

        A side left as ``None`` is open. An expression with no provable bounds
        is never within, so a field that must stay in range rejects it rather
        than finding out while sampling.
        """
        found = self.bounds
        if found is None:
            return False
        lower, upper = found
        return (minimum is None or lower >= minimum) and (maximum is None or upper <= maximum)

    @property
    def is_integer_valued(self) -> bool:
        """True when every value the expression can take is provably an integer.

        sympy does not know a discrete random variable is integer-valued, so each
        one is stood in for by an integer symbol when its row says its draws are
        whole numbers, and by a real symbol otherwise. ``Poisson(2) + 1`` and
        ``floor(Normal(512, 128))`` prove, ``Normal(512, 128)`` and
        ``Poisson(2) / 2`` do not.
        """
        if _is_expr(self._expr) and self._expr.is_number:
            return float(self._expr).is_integer()
        stand_ins: dict[SympyNode, SympyNode] = {
            rv: sympy.Dummy(integer=True) if use.integer else sympy.Dummy(real=True) for rv, use in self._uses.items()
        }
        return self._expr.xreplace(stand_ins).is_integer is True

    def mean(self, t: Optional[float] = None) -> Optional[float]:
        """The exact mean of the value, or ``None`` when it cannot be derived.

        Each random variable's mean comes from its row. The mean of the whole
        expression follows only where the expression is linear in its random
        variables, e.g. ``128 + 2*Poisson(64)``. Through anything else
        (``Min``, ``floor``, a product of two draws, a ``Piecewise`` mixture)
        the mean is not a function of the means, so the answer is ``None``.
        ``inf`` or ``-inf`` means a draw with an infinite mean decides it,
        e.g. ``Pareto(1, 1)``.

        Args:
            t: Stage time in seconds, substituted for ``t`` when present.
        """
        expr = self._expr
        if t is not None:
            expr = expr.subs(_T, sympy.Float(t))
        remaining = {str(s) for s in expr.free_symbols if not _is_random_symbol(s)}
        if remaining:
            raise ValueError(f"Expression {self.raw!r} requires {sorted(remaining)} but none was provided to mean().")
        present = [rv for rv in self._random_symbols if expr.has(rv)]
        if present:
            try:
                degree = sympy.Poly(expr, *present).total_degree()
            except sympy.PolynomialError:
                return None
            if degree > 1:
                return None
        means: dict[SympyNode, SympyNode] = {rv: self._uses[rv].mean for rv in present}
        value = float(expr.xreplace(means))
        return None if math.isnan(value) else value

    @property
    def is_constant(self) -> bool:
        """True when the value never varies: no time variable and no randomness."""
        return not self._is_random and not self._free_symbols

    @property
    def is_random(self) -> bool:
        """True when the expression contains a random variable."""
        return self._is_random

    @property
    def free_symbols(self) -> set[str]:
        """Names of non-random free symbols (a subset of ``{"t"}``)."""
        return set(self._free_symbols)

    def sample(
        self,
        t: Optional[float] = None,
        *,
        rng: Optional[np.random.Generator] = None,
        size: int = 1,
    ) -> Union[float, NDArray[np.float64]]:
        """Draw value(s) from the expression.

        Args:
            t: Stage time in seconds, substituted for ``t`` when present.
            rng: numpy Generator for deterministic random draws. Ignored for
                non-random expressions; defaults to a fresh Generator otherwise.
            size: Number of values to draw. ``1`` returns a float; ``>1`` returns
                a numpy array.

        Returns:
            A float when ``size == 1``, otherwise a numpy array of floats.
        """
        if size < 1:
            raise ValueError(f"size must be a positive integer, got {size}.")

        expr = self._expr
        if t is not None:
            expr = expr.subs(_T, sympy.Float(t))

        remaining = {str(s) for s in expr.free_symbols if not _is_random_symbol(s)}
        if remaining:
            raise ValueError(f"Expression {self.raw!r} requires {sorted(remaining)} but none was provided to sample().")

        if not self._is_random:
            value = float(expr.evalf())
            if self._runtime_check and not self._within_bounds(value):
                raise ValueError(
                    f"Expression {self.raw!r} evaluated to {value}"
                    f"{f' at t={t}' if t is not None else ''}, outside the permitted range "
                    f"[{self.minimum}, {self.maximum}]."
                )
            return value if size == 1 else np.full(size, value, dtype=np.float64)

        if rng is None:
            rng = np.random.default_rng()

        # Draw each leaf through its row's numpy sampler (independent draws,
        # since the shared rng advances between leaves), then evaluate the
        # lambdified deterministic skeleton over the drawn arrays.
        assert self._lambdified is not None
        leaf_draws = [use.sample(rng, size) for _ph, use in self._random_leaves]
        tval = 0.0 if t is None else float(t)
        samples = np.asarray(self._lambdified(*leaf_draws, tval), dtype=np.float64)
        if samples.shape == ():
            samples = np.full(size, float(samples), dtype=np.float64)

        if self._clip_random:
            samples = np.clip(
                samples,
                self.minimum if self.minimum is not None else -np.inf,
                self.maximum if self.maximum is not None else np.inf,
            )
        elif self._runtime_check:
            lo = self.minimum if self.minimum is not None else -np.inf
            hi = self.maximum if self.maximum is not None else np.inf
            out_of_range = (samples < lo) | (samples > hi)
            if bool(np.any(out_of_range)):
                offender = float(samples.reshape(-1)[np.argmax(out_of_range.reshape(-1))])
                raise ValueError(
                    f"Expression {self.raw!r} drew {offender}, outside the permitted range "
                    f"[{self.minimum}, {self.maximum}]. Keep it in range with Min(...) or Max(...), "
                    "or pick a distribution bounded to the range."
                )
        return float(samples.reshape(-1)[0]) if size == 1 else samples

    def evaluate(self, t: Union[float, NDArray[np.float64]]) -> NDArray[np.float64]:
        """Evaluate a deterministic expression at many times at once.

        :meth:`sample` substitutes ``t`` symbolically on every call, which is
        fine for one value and far too slow for a grid (a time-varying rate is
        integrated over thousands of points). This compiles the expression
        once with ``sympy.lambdify`` and evaluates it over a numpy array.

        Args:
            t: Stage time(s) in seconds; a scalar or an array.

        Returns:
            A float array with the same shape as ``t``.

        Raises:
            ValueError: If the expression is random, or if a bound that could
                not be proven at construction is violated at some ``t``.
        """
        if self._is_random:
            raise ValueError(f"Expression {self.raw!r} is random; evaluate() is only for deterministic expressions.")
        if self._evaluator is None:
            self._evaluator = sympy.lambdify([_T], self._expr, "numpy")
        times = np.asarray(t, dtype=np.float64)
        # A constant lambdifies to a scalar; broadcast it to the input's shape.
        values = np.broadcast_to(np.asarray(self._evaluator(times), dtype=np.float64), times.shape).copy()
        if self._runtime_check:
            lo = self.minimum if self.minimum is not None else -np.inf
            hi = self.maximum if self.maximum is not None else np.inf
            out_of_range = (values < lo) | (values > hi)
            if bool(np.any(out_of_range)):
                index = int(np.argmax(out_of_range.reshape(-1)))
                raise ValueError(
                    f"Expression {self.raw!r} evaluated to {float(values.reshape(-1)[index])} "
                    f"at t={float(np.broadcast_to(times, values.shape).reshape(-1)[index])}, outside the permitted range "
                    f"[{self.minimum}, {self.maximum}]."
                )
        return values

    def _within_bounds(self, value: float) -> bool:
        if self.minimum is not None and value < self.minimum:
            return False
        if self.maximum is not None and value > self.maximum:
            return False
        return True

    def __repr__(self) -> str:
        return f"Expression({self.raw!r})"


# The ``t`` domain a predicate is proved over: a stage runs from t = 0 onward.
_TIME_DOMAIN: SympySet = Interval(0, oo)
_NO_TIME: SympySet = sympy.S.EmptySet


class Predicate:
    """A boolean condition over stage time ``t`` and measured terms, e.g. ``"t >= 60"``.

    The string-grammar counterpart of :class:`Expression` for a trigger: a
    field that waits for the condition to become true and checks it at
    discrete instants. It is parsed and validated once at construction and
    evaluated with :meth:`holds`. Construction raises :class:`ValueError`
    unless the condition is false at ``t = 0`` and then holds for every later
    ``t`` from some time onward. That shape is what makes a trigger meaningful
    when it is checked at discrete instants: a condition that holds only at a
    single time (``Eq(t, 60)``) or over a window (``(t >= 60) & (t < 120)``)
    can be skipped over by the check, and one that already holds at ``t = 0``
    (``t < 60``, ``t >= 0``) would fire before anything has run.

    The shape is proved statically by solving the condition over ``t >= 0``,
    which is why :attr:`boundary` is always known: the condition compiles to
    the exact time it starts holding, so a caller can schedule against it
    instead of polling. The proof is only as good as sympy's inequality
    solver, so a condition may use only the nodes the solver answers
    correctly (see ``_PREDICATE_NODES``): ``+``, ``*``, ``/``, powers,
    ``sqrt``, ``exp``, ``log``, ``Min``/``Max``, the inequalities ``>``,
    ``>=``, ``<``, ``<=`` and their ``&``/``|`` combinations. Anything else,
    e.g. ``sin`` or ``Abs``, is rejected by name rather than evaluated on an
    unproven shape. Random variables are never permitted: a condition must be
    the same on every run.

    Besides ``t`` a condition may read what the stage measures: window
    functions over finished requests (``quantile_over_time(0.99, ttft[30s])``,
    ``avg_over_last(latency, 100)``, ``avg_over_sent(ttft, 10, 30)``, see
    :mod:`.measured`) and the signals ``completed``, ``errors``, ``in_flight``
    and ``since_last_completion``. A count can be read by status and in the
    past: ``completed{status="success"} offset 60s``. Those are the system under test and cannot
    be solved, so a comparison on one is evaluated as results arrive. The time
    part is still proved, twice. With every measured comparison false the
    condition must hold from some time onward on its own. That time is
    :attr:`boundary`, the latest the condition can start holding. With every
    measured comparison true it must again hold from some time onward. That
    time is :attr:`earliest`, before which no result can make it hold. So a
    measured term can only make the condition hold sooner, never keep it from
    holding, and ``quantile_over_time(0.99, ttft[30s]) > 2`` alone is rejected.

    Equality in any spelling (``t = 60``, ``t == 60``, ``Eq(t, 60)``) and its
    negation are rejected up front: equality holds at a single instant, which
    a check at discrete instants can step over.

    Note the grammar quirk: ``&`` and ``|`` bind tighter than a comparison, so
    each comparison must be parenthesised, ``(t >= 60) & (t < 120)``.

    An error names the piece at fault rather than repeating the condition: it
    quotes at most ``_QUOTE_LIMIT`` characters of it, then the comparison or
    name that broke the rule and the outermost ``|`` clause that holds it. So
    the message stays the same size whether the condition has one clause or
    fifty.

    Args:
        raw: The condition as a string.
        name: What errors call the condition, e.g. the config field it came from.
    """

    def __init__(self, raw: str, name: str = "Predicate") -> None:
        if not isinstance(raw, str):
            raise TypeError(f"{name} accepts str, got {type(raw).__name__}.")
        self.raw = raw
        self._subject = subject = f"{name} {_quote(raw)}"
        # the '=' inside a label selector, completed{status="failure"}, is not an equality.
        # blanked rather than cut, so a match keeps its offset in raw
        equality = _EQUALITY_SPELLING.search(re.sub(r"\{[^{}]*\}", lambda m: " " * len(m.group()), raw))
        if equality:
            at = equality.start()
            raise ValueError(_equality_message(subject, _where(raw, lambda start, text: start <= at < start + len(text))))
        rewritten, self._terms = extract_terms(raw)
        try:
            expr = _parse_raw("Predicate", rewritten, shown=raw)
        except ValueError as e:
            if not isinstance(e.__cause__, (SyntaxError, TypeError, AttributeError, sympy.SympifyError)):
                # not a parse failure: a distribution the table refuses already says what is wrong
                raise
            where = _where(raw, lambda _start, text: _parse_clause(text) is None)
            found = f" Found{where}." if where else ""
            if "&" in raw or "|" in raw:
                # a TypeError here is Python applying '&' or '|' to a bare number, which says nothing to a config author
                detail = (
                    "'&' or '|' joins a comparison that has no parentheses"
                    if isinstance(e.__cause__, TypeError)
                    else e.__cause__
                )
                raise ValueError(
                    f"Could not parse {subject}: {detail}. Hint: '&' and '|' bind tighter than a comparison, "
                    f"so parenthesise each one: '(t >= 60) & (t < 120)'.{found}"
                ) from e
            raise ValueError(f"Could not parse {subject}: {e.__cause__}.") from e

        if isinstance(expr, bool) or expr in (sympy.true, sympy.false):
            raise ValueError(f"{subject} is constant ({expr!r}); a condition must vary with t, e.g. 't >= 60'.")
        if not (_is_node(expr) and _is_condition(expr)):
            raise ValueError(f"{subject} is not a condition; use a comparison such as 't >= 60'.")
        undefined = sorted({f.func.__name__ for f in expr.atoms(AppliedUndef)})
        if undefined:
            raise ValueError(f"{subject} uses unknown function(s): {undefined}.{_found(raw, undefined[0])}")

        free = expr.free_symbols
        if any(_is_random_symbol(s) for s in free):
            raise ValueError(f"{subject} contains a random variable; a condition must be the same on every run.")
        names = {str(s) for s in free}
        disallowed = names - {"t"} - set(SIGNALS) - set(self._terms)
        if disallowed:
            raise ValueError(_disallowed_message(subject, raw, disallowed))
        # sympy can fold a term away ('t + 0*avg_over_last(ttft, 5)'); a term
        # the compiled condition never reads must not be evaluated either.
        self._terms = {symbol: term for symbol, term in self._terms.items() if symbol in names}
        self._measured = frozenset(names - {"t"})
        if "t" not in names:
            if self._measured:
                raise ValueError(
                    f"{subject} can only hold on measured values; it must also hold on time alone, so that it holds "
                    f"even if they never do. Add a time bound as one more clause: '| (t >= 600)'."
                )
            raise ValueError(f"{subject} does not depend on t; a condition must vary with t.")
        _reject_unprovable_nodes(subject, raw, expr)

        self._expr = expr
        self._symbols = [_T] + [Symbol(name) for name in sorted(self._measured)]
        self._holds = sympy.lambdify(self._symbols, expr, "numpy")

        # A comparison that reads a measured value is unknown until the run.
        # With them all false the condition must still hold from some time on:
        # that is the boundary. With them all true it holds from the earliest
        # time a result could make it hold. Both are plain conditions over t.
        measured_atoms = [atom for atom in expr.atoms(Relational) if {str(s) for s in atom.free_symbols} & self._measured]
        on_time_alone = expr.xreplace({atom: sympy.false for atom in measured_atoms})
        self._boundary = self._prove_boundary(on_time_alone)
        if measured_atoms:
            self._earliest = self._prove_earliest(expr.xreplace({atom: sympy.true for atom in measured_atoms}))
        else:
            self._earliest = self._boundary

    def _prove_boundary(self, on_time_alone: SympyNode) -> float:
        """Solve the condition with its measured comparisons false and require the shape ``[b, oo)`` with ``b > 0``."""
        holding = _holding_set(on_time_alone)
        if holding is None:
            raise ValueError(f"{self._subject} could not be proved to hold from some time onward.")
        if holding == sympy.S.EmptySet:
            if self._measured:
                clauses = len(_clauses(self.raw))
                which = f", and none of its {clauses} clauses does" if clauses > 1 else ""
                raise ValueError(
                    f"{self._subject} never holds on time alone: it must hold with every measured comparison false{which}. "
                    f"Add a time bound as one more clause: '| (t >= 600)'."
                )
            raise ValueError(f"{self._subject} never holds for t >= 0.")
        if not isinstance(holding, Interval) or holding.sup != oo:
            raise ValueError(
                f"{self._subject} holds only on {_format_time_set(holding)}; a condition must keep holding once it "
                f"starts to, as 't >= 60' does.{_cause(self.raw, _lapsing(on_time_alone))}"
            )
        if holding.inf == 0:
            raise ValueError(
                f"{self._subject} already holds at (or immediately after) t=0, before anything has run; "
                f"a condition must be false at t=0, e.g. 't >= 60'.{_cause(self.raw, _holding_at_start(on_time_alone))}"
            )
        boundary = float(holding.inf)
        if not math.isfinite(boundary):
            raise ValueError(f"{self._subject} holds only from t = {holding.inf}, which is too large for a float.")
        return boundary

    def _prove_earliest(self, at_the_earliest: SympyNode) -> float:
        """Solve the condition with its measured comparisons true and require the shape ``[b, oo)``, ``b >= 0``."""
        holding = _holding_set(at_the_earliest)
        if holding is None:
            raise ValueError(f"{self._subject} could not be proved to hold from some time onward.")
        if not isinstance(holding, Interval) or holding.sup != oo:
            raise ValueError(
                f"{self._subject} holds only on {_format_time_set(holding)} once its measured values hold; a condition "
                f"must keep holding once it starts to, as 't >= 60' does.{_cause(self.raw, _lapsing(at_the_earliest))}"
            )
        return float(holding.inf)

    @property
    def boundary(self) -> float:
        """The time from which the condition holds on time alone, whatever the measured values read."""
        return self._boundary

    @property
    def earliest(self) -> float:
        """The time before which no result can make the condition hold; equal to :attr:`boundary` without measured values."""
        return self._earliest

    @property
    def terms(self) -> Mapping[str, Term]:
        """The window functions and selected or offset signals the condition reads, keyed by their symbol in :attr:`measured_symbols`."""
        return self._terms

    @property
    def signals(self) -> frozenset[str]:
        """The signals (``completed``, ``in_flight``, ...) the condition reads."""
        return self._measured & frozenset(SIGNALS)

    @property
    def measured_symbols(self) -> frozenset[str]:
        """Every symbol besides ``t``: one per term, plus the signals. Empty for a condition over ``t`` alone."""
        return self._measured

    @property
    def free_symbols(self) -> set[str]:
        """Names of free symbols: ``t`` and :attr:`measured_symbols`."""
        return {str(s) for s in self._expr.free_symbols}

    def holds(self, t: float, values: Optional[Mapping[str, float]] = None) -> bool:
        """Evaluate the condition at stage time ``t`` (seconds) with each measured symbol's current value.

        A measured symbol missing from ``values``, or NaN, has no value yet:
        every comparison that reads it is false, so the rest of the condition
        decides. :meth:`~.measured.MeasuredState.values` supplies ``values``.
        """
        args = [np.float64(t)] + [np.float64((values or {}).get(name, math.nan)) for name in sorted(self._measured)]
        with np.errstate(all="ignore"):
            return bool(self._holds(*args))

    def __repr__(self) -> str:
        return f"Predicate({self.raw!r})"


# Python's '=' and '==' (structural comparison in sympy) and '!=' are caught
# on the raw string, before parsing turns them into a syntax error or a
# constant; '>=' and '<=' are not matched.
_EQUALITY_SPELLING = re.compile(r"(?<![<>=!])=(?!=)|==|!=")

# The only sympy node types a predicate may contain. An allowlist rather than a
# denylist: sympy's inequality solver can return a confident but incomplete
# answer (sin(t) > 0 solves to (0, pi) alone) or a wrong one (nested Abs), so a
# node is allowed only once test_predicate.py pins the boundary the solver
# gives for it. Powers are further restricted in _reject_unprovable_nodes.
_PREDICATE_NODES: tuple[type, ...] = (
    And,
    Or,
    GreaterThan,
    StrictGreaterThan,
    LessThan,
    StrictLessThan,
    sympy.Symbol,
    sympy.Number,
    sympy.NumberSymbol,
    sympy.Add,
    sympy.Mul,
    sympy.Pow,
    sympy.exp,
    sympy.log,
    sympy.Min,
    sympy.Max,
)


def _format_time_set(s: SympySet) -> str:
    """Render a solved set of times in interval notation, e.g. ``[0, 60)`` or ``[0, 30) or [60, infinity)``.

    sympy's own repr (``Interval.Ropen(0, 60)``) is exact but reads as code in
    a config error.
    """
    if isinstance(s, sympy.Union):
        return " or ".join(_format_time_set(part) for part in s.args)
    if isinstance(s, Interval):

        def end(x: SympyExpr) -> str:
            if x == oo:
                return "infinity"
            return str(x) if x.is_Integer else format(float(x), "g")

        return f"{'(' if s.left_open else '['}{end(s.inf)}, {end(s.sup)}{')' if s.right_open else ']'}"
    return str(s)


def _disallowed_message(subject: str, raw: str, disallowed: set[str]) -> str:
    names = sorted(disallowed)
    message = (
        f"{subject} uses disallowed symbol(s) {names}; permitted: t, the signals {sorted(SIGNALS)}, "
        f"and window functions such as 'quantile_over_time(0.99, ttft[30s])' or 'avg_over_last(latency, 100)'."
        f"{_found(raw, names[0])}"
    )
    series = [name for name in names if name in SERIES]
    if series:
        message += f" {series[0]!r} is a series: read it through a window function, e.g. 'avg_over_last({series[0]}, 100)'."
    report_names = {s.report_field: s.name for s in SERIES.values() if s.report_field}
    renamed = [name for name in names if name in report_names]
    if renamed:
        message += f" Series use short names: {renamed[0]!r} is {report_names[renamed[0]]!r}."
    return message


def _equality_message(subject: str, where: str) -> str:
    return (
        f"{subject} uses equality{where}; a condition must be an inequality such as 't >= 60'. "
        f"Equality holds at a single instant, which a check at discrete instants can miss."
    )


def _reject_unprovable_nodes(subject: str, raw: str, expr: SympyNode) -> None:
    """Reject any node outside ``_PREDICATE_NODES``, naming it."""
    for node in sympy.preorder_traversal(expr):
        if isinstance(node, (sympy.Eq, sympy.Ne)):
            # sympy calls these Equality and Unequality; the config spells them Eq and Ne
            raise ValueError(_equality_message(subject, _where_named(raw, "Eq" if isinstance(node, sympy.Eq) else "Ne")))
        if isinstance(node, sympy.Pow):
            base, exponent = node.args
            # t**2, sqrt(t), 1/t (constant exponent) and 2**t (positive constant base) only.
            if exponent.free_symbols and not (base.is_number and base.is_positive):
                raise ValueError(f"{subject} uses {node}; a power needs a constant exponent or a positive constant base.")
            continue
        if not isinstance(node, _PREDICATE_NODES):
            name = node.func.__name__
            raise ValueError(
                f"{subject} uses {name!r}, which is not allowed in a condition; "
                f"permitted: + - * / **, sqrt, exp, log, Min, Max, and > >= < <= combined with & and |. "
                f"These are the ones a condition can be proved to keep holding with.{_found(raw, name)}"
            )


# The most of a condition an error quotes. Past it the message relies on the
# comparison and clause it names, so its size does not grow with the condition.
_QUOTE_LIMIT = 80


def _quote(raw: str) -> str:
    """``raw`` in quotes, cut at ``_QUOTE_LIMIT`` characters with its full length noted."""
    if len(raw) <= _QUOTE_LIMIT:
        return repr(raw)
    return f"{raw[:_QUOTE_LIMIT]!r}... ({len(raw)} characters)"


def _clauses(raw: str) -> list[tuple[int, str]]:
    """The pieces of ``raw`` between its outermost ``|``, each with the offset it starts at."""
    pieces: list[tuple[int, str]] = []
    depth = start = 0
    for i, ch in enumerate(raw):
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        elif ch == "|" and depth == 0:
            pieces.append((start, raw[start:i]))
            start = i + 1
    pieces.append((start, raw[start:]))
    return pieces


def _where(raw: str, at_fault: Callable[[int, str], bool]) -> str:
    """Name the first outermost ``|`` clause ``at_fault`` picks: `` in clause 37 of 50, '(t >= 60) & (t < 120)'``.

    ``at_fault`` gets each clause's offset and text. Empty when the condition
    is a single clause, where the quoted condition already is the clause, or
    when no clause is picked.
    """
    pieces = _clauses(raw)
    if len(pieces) < 2:
        return ""
    hits = [(n, text.strip()) for n, (start, text) in enumerate(pieces, 1) if at_fault(start, text)]
    if not hits:
        return ""
    n, text = hits[0]
    more = f" and {len(hits) - 1} more" if len(hits) > 1 else ""
    return f" in clause {n} of {len(pieces)}, {_quote(text)}{more}"


def _mentions(text: str, name: str) -> bool:
    """Whether ``name`` appears in ``text`` as a whole word."""
    return re.search(rf"(?<![A-Za-z0-9_]){re.escape(name)}(?![A-Za-z0-9_])", text) is not None


def _where_named(raw: str, name: str) -> str:
    """Name the first clause that mentions ``name``."""
    return _where(raw, lambda _start, text: _mentions(text, name))


def _found(raw: str, name: str) -> str:
    """A sentence placing ``name`` in its clause, or nothing when there is one clause."""
    where = _where_named(raw, name)
    return f" Found{where}." if where else ""


def _parse_clause(text: str) -> object:
    """One clause parsed on its own, or ``None`` when it does not parse."""
    try:
        return _parse_raw("Predicate", extract_terms(text)[0])
    except (ValueError, TypeError):
        return None


def _where_compared(raw: str, comparison: SympyNode) -> str:
    """Name the first clause that contains ``comparison``."""

    def contains(_start: int, text: str) -> bool:
        parsed = _parse_clause(text)
        return _is_node(parsed) and comparison in parsed.atoms(Relational)

    return _where(raw, contains)


def _cause(raw: str, comparisons: list[SympyNode]) -> str:
    """A sentence naming up to three comparisons at fault and the clause each sits in; empty when there are none."""
    if not comparisons:
        return ""
    shown = "; ".join(f"'{c}'{_where_compared(raw, c)}" for c in comparisons[:3])
    more = f"; and {len(comparisons) - 3} more" if len(comparisons) > 3 else ""
    return f" Cause: {shown}{more}."


def _time_comparisons(expr: SympyNode) -> list[tuple[SympyNode, SympySet]]:
    """Each comparison in a condition over ``t`` alone with the times it holds on, skipping any that never holds."""
    solved = [(c, _holding_set(c)) for c in sorted(expr.atoms(Relational), key=str)]
    return [(c, s) for c, s in solved if s is not None and s != sympy.S.EmptySet]


def _lapsing(expr: SympyNode) -> list[SympyNode]:
    """The comparisons that stop holding as ``t`` grows, e.g. ``t < 120``.

    ``&`` and ``|`` of comparisons that each keep holding keep holding, so a
    condition that lapses always has one of these in it.
    """
    return [c for c, s in _time_comparisons(expr) if not (isinstance(s, Interval) and s.sup == oo)]


def _holding_at_start(expr: SympyNode) -> list[SympyNode]:
    """The comparisons that already hold at ``t = 0``, e.g. ``t >= 0``."""
    return [c for c, s in _time_comparisons(expr) if s.inf == sympy.S.Zero]


def _holding_set(expr: SympyNode) -> Optional[SympySet]:
    """The set of ``t >= 0`` where ``expr`` holds, or ``None`` when not provable.

    Leaves are solved by sympy, and ``&``/``|`` are folded as set
    intersection/union of their solved leaves. A leaf the solver cannot
    answer (a ``ConditionSet``) makes the whole condition unprovable. The
    constants arise when a measured comparison is replaced by true or false.
    """
    if expr is sympy.true:
        return _TIME_DOMAIN
    if expr is sympy.false:
        return _NO_TIME
    if isinstance(expr, And):
        sets = [_holding_set(arg) for arg in expr.args]
        if any(s is None for s in sets):
            return None
        both: SympySet = sympy.Intersection(*sets)
        return both
    if isinstance(expr, Or):
        sets = [_holding_set(arg) for arg in expr.args]
        if any(s is None for s in sets):
            return None
        either: SympySet = sympy.Union(*sets)
        return either
    if isinstance(expr, Relational):
        try:
            solved: SympySet = sympy.solveset(expr, _T, _TIME_DOMAIN)
        except Exception:
            return None
        return None if solved.has(sympy.ConditionSet) else solved
    return None


class ParameterConditionError(ValueError):
    """A distribution parameter that varies with ``t`` breaks, or cannot be shown to meet, one of its conditions.

    Attributes:
        distribution: The use as written, e.g. ``Exponential(t)``.
        parameters: The parameters the condition reads, e.g. ``("rate",)``.
        condition: The condition, e.g. ``rate > 0``. ``rate is a finite real``
            is the one condition every parameter has that no row lists.
        t: A stage time in the window at which the condition is false, or
            ``None`` when no such time was found and the condition still could
            not be proved.
    """

    def __init__(
        self, use: DistributionUse, parameters: tuple[str, ...], condition: str, duration: float, t: Optional[float]
    ) -> None:
        self.distribution = str(use)
        self.parameters = parameters
        self.condition = condition
        self.t = t
        outcome = f"fails at t = {t:g}" if t is not None else "could not be proved"
        super().__init__(
            f"{self.distribution}: {use.row.signature} requires '{condition}', which {outcome} over [0, {duration:g}]."
        )


def parse_distribution_use(raw: str) -> DistributionUse:
    """Parse one distribution call whose arguments may be expressions of ``t``, e.g. ``"Exponential(5 + t/10)"``.

    This is the input to :func:`prove_parameter_conditions`. :class:`Expression`
    itself takes constant arguments only.
    """
    namespace: dict[str, Callable[..., object]] = {name: _unsupported(name) for name in _DISTRIBUTION_CTORS}
    for name, row in DISTRIBUTIONS.items():
        namespace[name] = lambda *args, _row=row: _row.use(args)
    try:
        use = parse_expr(raw, local_dict=namespace)
    except (SyntaxError, TypeError, AttributeError, TokenError, sympy.SympifyError) as e:
        raise ValueError(f"Could not parse distribution {raw!r}: {e}") from e
    if not isinstance(use, DistributionUse):
        raise ValueError(f"{raw!r} is not a single distribution call such as 'Exponential(5 + t/10)'.")
    for arg in use.args:
        _reject_unknown_functions("Distribution", raw, arg)
        disallowed = sorted(str(symbol) for symbol in arg.free_symbols if symbol != _T)
        if disallowed:
            raise ValueError(f"Distribution {raw!r} uses disallowed symbol(s) {disallowed}; permitted: ['t'].")
    return use


# where 'lhs - rhs' must lie for each kind of condition to hold
_HOLDING_GAPS: Mapping[str, SympySet] = {
    ">": Interval.open(0, oo),
    ">=": Interval(0, oo),
    "<": Interval.open(-oo, 0),
    "<=": Interval(-oo, 0),
    "==": sympy.FiniteSet(0),
}
_WITNESS_GRID = 64


def prove_parameter_conditions(use: DistributionUse, duration: float) -> None:
    """Prove every parameter condition of ``use`` for all ``t`` in ``[0, duration]``.

    A condition such as ``rate > 0`` becomes a statement about the range of an
    expression of ``t`` once the arguments are substituted. It is proved from
    the interval bounds of that expression, or failing that from its exact
    range over the window. Every parameter must also stay a finite real, which
    rules out ``Exponential(1/t)``. Anything short of a proof is a rejection: the
    error names a ``t`` at which the condition is false when one is found
    among the window's ends, the points where the condition's two sides meet
    and an even grid, and carries ``t = None`` otherwise.

    Raises:
        ParameterConditionError: for the first condition that is not proved.
    """
    if not (math.isfinite(duration) and duration >= 0):
        raise ValueError(f"duration must be a finite non-negative number, got {duration}.")
    if use.row.takes_list:
        # a list of values has no condition to prove over time: the values are constants or it is refused
        use.require_constant()
        return
    for parameter, arg in zip(use.row.parameters, use.args, strict=True):
        # evaluated as well as proved: the range machinery assumes a real value, which sqrt(t - 5) is not at t = 0
        failing = _time_outside(arg, duration, sympy.S.Reals)
        if failing is not None or not any(_is_bounded(values) for values in _ranges(arg, duration)):
            raise ParameterConditionError(use, (parameter,), f"{parameter} is a finite real", duration, failing)
    for condition in use.row.conditions:
        gap = use.bind(sympy.Add(condition.lhs, -condition.rhs))
        holding = _HOLDING_GAPS[condition.rel_op]
        if not any(values.is_subset(holding) is True for values in _ranges(gap, duration)):
            parameters = tuple(
                name for name, symbol in zip(use.row.parameters, use.row.symbols, strict=True) if condition.has(symbol)
            )
            failing = _time_outside(gap, duration, holding)
            raise ParameterConditionError(use, parameters, render_condition(condition), duration, failing)


def _ranges(expr: SympyExpr, duration: float) -> Iterator[SympySet]:
    """Sets proved to contain every value of ``expr`` over ``[0, duration]``, the cheap loose one first."""
    found = _interval(expr, duration)
    if found is not None:
        hull: SympySet = Interval(found[0], found[1])
        yield hull
    # interval bounds are sound but loose, e.g. for (t - 1)**2 + 1, so ask for the exact range
    try:
        values: SympySet = function_range(expr, _T, Interval(0, sympy.Float(duration)))
    except Exception:
        return
    yield values


def _is_bounded(values: SympySet) -> bool:
    try:
        return math.isfinite(float(values.inf)) and math.isfinite(float(values.sup))
    except (TypeError, ValueError, NotImplementedError):
        return False


def _time_outside(expr: SympyExpr, duration: float, holding: SympySet) -> Optional[float]:
    """The earliest candidate ``t`` in ``[0, duration]`` at which ``expr`` evaluates outside ``holding``."""
    candidates: list[SympyExpr] = [sympy.Float(duration * i / _WITNESS_GRID) for i in range(_WITNESS_GRID + 1)]
    try:
        # where the two sides of a condition meet, which a strict condition fails exactly
        meeting = sympy.solveset(expr, _T, Interval(0, sympy.Float(duration)))
    except Exception:
        meeting = sympy.S.EmptySet
    if isinstance(meeting, sympy.FiniteSet):
        candidates.extend(meeting)
    for at in sorted(candidates, key=float):
        value = expr.subs(_T, at)
        # a value that is not real, e.g. 1/t at 0, is in no holding set
        if holding.contains(value) != sympy.true:
            return float(at)
    return None
