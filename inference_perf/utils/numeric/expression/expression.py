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
stage time ``t``, and/or a draw from a ``sympy.stats`` random variable) into a
single object that is parsed and validated once at construction and sampled
many times. It is the string-grammar counterpart to
:mod:`inference_perf.utils.numeric.distribution`, which remains the fast,
vectorised path for structured ``Distribution`` configs.

Grammar: constants, the single time variable ``t``, standard math, and
``sympy.stats`` distribution constructors (``Normal(512, 200)``,
``Uniform(10, 50)``, ``Poisson(10)``, ...). Each call site decides whether ``t``
and random variables are permitted, and may constrain the value range so that,
for example, a request-rate expression can never evaluate negative.
"""

from __future__ import annotations

import inspect
import logging
from typing import Any, Optional, Union

import numpy as np
import sympy
import sympy.stats
from numpy.typing import NDArray
from sympy import Interval, Symbol, oo
from sympy.calculus.util import function_range
from sympy.core.function import AppliedUndef
from sympy.parsing.sympy_parser import parse_expr

logger = logging.getLogger(__name__)

# sympy maintains its own per-distribution numpy samplers in a singledispatch
# registry; the implementation registered to ``object`` is a stub that raises.
# Resolving this dispatch ourselves (rather than calling the ``sympy.stats.sample``
# wrapper, which re-does all of its symbolic bookkeeping on every call) lets us
# sample any distribution sympy supports at near-numpy speed, with no hardcoded
# table to keep in sync. The import is guarded: if a future sympy moves this
# internal module we degrade to the slow but always-correct sampling path.
try:
    from sympy.stats.sampling.sample_numpy import do_sample_numpy

    _NO_NUMPY_SAMPLER = do_sample_numpy.dispatch(object)
except Exception:  # pragma: no cover - defensive: sympy internal layout changed
    do_sample_numpy = None  # type: ignore[assignment]
    _NO_NUMPY_SAMPLER = None

# The single time variable the grammar ever permits.
_T = Symbol("t")


def _distribution_constructors() -> dict[str, Any]:
    """Collect every ``sympy.stats`` distribution constructor (name-first)."""
    ctors: dict[str, Any] = {}
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


# Built once at import; the set of sympy.stats distributions is static.
_DISTRIBUTION_CTORS = _distribution_constructors()


def _parse_namespace(counter: list[int]) -> dict[str, Any]:
    """A fresh distribution namespace whose constructors take no name argument.

    Config strings read ``Normal(512, 200)`` rather than ``Normal("x", ...)``,
    so each constructor binds a name for us. Crucially the name is *unique per
    occurrence* (``rv_1``, ``rv_2``, ...) via the shared ``counter``, so that an
    expression with two draws of the same distribution (e.g. a skew-normal built
    from two independent normals) yields two independent random variables rather
    than one aliased symbol.
    """
    namespace: dict[str, Any] = {}
    for name, ctor in _DISTRIBUTION_CTORS.items():

        def bound(*args: Any, _ctor: Any = ctor, _counter: list[int] = counter) -> Any:
            _counter[0] += 1
            return _ctor(f"rv_{_counter[0]}", *args)

        namespace[name] = bound
    return namespace


def _is_random_symbol(sym: Any) -> bool:
    return isinstance(sym, sympy.stats.rv.RandomSymbol)


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
        allow_random: Whether ``sympy.stats`` random variables may appear.
        minimum: Lower bound for sampled values, or ``None`` for unbounded.
        maximum: Upper bound for sampled values, or ``None`` for unbounded.
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
        duration: Optional[float] = None,
    ) -> None:
        if minimum is not None and maximum is not None and minimum > maximum:
            raise ValueError(f"minimum ({minimum}) cannot be greater than maximum ({maximum}).")

        self.raw = raw
        self.minimum = minimum
        self.maximum = maximum
        self._expr = self._parse(raw)

        # Reject unknown functions, e.g. a misspelled distribution InvalidDist(10).
        undefined = {f.func.__name__ for f in self._expr.atoms(AppliedUndef)}
        if undefined:
            raise ValueError(f"Expression {raw!r} uses unknown function(s): {sorted(undefined)}.")

        free = self._expr.free_symbols
        self._random_symbols = sorted((s for s in free if _is_random_symbol(s)), key=str)
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
        self._clip_random = self._is_random and (minimum is not None or maximum is not None)
        self._runtime_check = False
        self._validate_range(duration)
        self._compile_random_sampler()

    def _compile_random_sampler(self) -> None:
        """Precompute a numpy-vectorised sampler for the random fast path.

        Splits the parsed tree into random *leaves* (the ``RandomSymbol`` atoms)
        and a deterministic *skeleton* (the arithmetic/time combination of them).
        Each leaf is drawn through sympy's own per-distribution numpy sampler
        (``do_sample_numpy``), resolved once here because the expression is fixed
        at construction; the skeleton is compiled once with ``sympy.lambdify``. At
        sample time we draw the leaves with numpy and feed them through the
        lambdified closure, never touching the slow ``sympy.stats.sample`` wrapper.
        If any leaf distribution has no numpy sampler (or the sympy internal is
        unavailable) we set ``_fallback`` and keep the sympy path.
        """
        self._random_leaves: list[tuple[Any, Any]] = []
        self._lambdified: Optional[Any] = None
        self._fallback = False
        if not self._is_random:
            return

        if do_sample_numpy is None:
            self._fallback = True
            return

        placeholders: dict[Any, Any] = {}
        for i, rv in enumerate(self._random_symbols):
            dist = rv.pspace.distribution
            if do_sample_numpy.dispatch(type(dist)) is _NO_NUMPY_SAMPLER:
                self._fallback = True
                self._random_leaves = []
                return
            placeholder = Symbol(f"_leaf_{i}")
            placeholders[rv] = placeholder
            self._random_leaves.append((placeholder, dist))

        skeleton = self._expr.xreplace(placeholders)
        order = [ph for ph, _dist in self._random_leaves] + [_T]
        self._lambdified = sympy.lambdify(order, skeleton, "numpy")

    def _parse(self, raw: Union[str, int, float]) -> Any:
        if isinstance(raw, bool):
            raise TypeError("Expression does not accept bool input.")
        if isinstance(raw, (int, float)):
            return sympy.sympify(raw)
        if isinstance(raw, str):
            try:
                return parse_expr(raw, local_dict=_parse_namespace([0]))
            except (SyntaxError, TypeError, AttributeError, sympy.SympifyError) as e:
                raise ValueError(f"Could not parse expression {raw!r}: {e}") from e
        raise TypeError(f"Expression accepts str or number, got {type(raw).__name__}.")

    def _validate_range(self, duration: Optional[float]) -> None:
        if self.minimum is None and self.maximum is None:
            return

        bounds = Interval(
            sympy.Float(self.minimum) if self.minimum is not None else -oo,
            sympy.Float(self.maximum) if self.maximum is not None else oo,
        )
        static_range = self._static_range(duration)

        if static_range is None:
            # Undecided: defer to sample time. Random draws are clamped; a
            # deterministic value out of range is a config error and raises.
            if not self._is_random:
                self._runtime_check = True
            return

        contained = static_range.is_subset(bounds)
        if contained is True:
            return
        if contained is False:
            raise ValueError(
                f"Expression {self.raw!r} can evaluate to {static_range}, which is outside the permitted range {bounds}."
            )
        # is_subset returned None (couldn't decide): treat as undecided.
        if not self._is_random:
            self._runtime_check = True

    def _static_range(self, duration: Optional[float]) -> Any:
        """Best-effort provable value range, or ``None`` when undecidable."""
        if self._is_random:
            # Only a bare random variable has a readily available support set.
            # Transformed/mixed random expressions are left for the sample-time
            # clamp.
            if _is_random_symbol(self._expr):
                try:
                    return self._expr.pspace.distribution.set
                except Exception:
                    return None
            return None

        if not self._free_symbols:
            value = float(self._expr.evalf())
            return Interval(value, value)

        # Time-varying: the range is only knowable over a bounded t domain.
        if duration is None:
            return None
        try:
            return function_range(self._expr, _T, Interval(0, sympy.Float(duration)))
        except Exception:
            return None

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

        if self._fallback:
            # Some leaf distribution has no direct numpy sampler; fall back to
            # the (slower) symbolic sampler for correctness.
            draw = sympy.stats.sample(
                expr,
                size=(size,) if size > 1 else (),
                library="numpy",
                seed=rng,
            )
            samples = np.asarray(draw, dtype=np.float64)
        else:
            # Fast path: draw each leaf via sympy's per-distribution numpy
            # sampler (independent draws, since the shared rng advances between
            # leaves), then evaluate the lambdified deterministic skeleton over
            # the drawn arrays.
            leaf_draws = [np.asarray(do_sample_numpy(dist, (size,), rng), dtype=np.float64) for _ph, dist in self._random_leaves]
            tval = 0.0 if t is None else float(t)
            out = self._lambdified(*leaf_draws, tval)  # type: ignore[misc]
            samples = np.asarray(out, dtype=np.float64)
            if samples.shape == ():
                samples = np.full(size, float(samples), dtype=np.float64)

        if self._clip_random:
            samples = np.clip(
                samples,
                self.minimum if self.minimum is not None else -np.inf,
                self.maximum if self.maximum is not None else np.inf,
            )
        # The fallback yields a 0-d scalar for size==1 while the numpy fast path
        # yields a shape-(1,) array; flatten before scalarising so both work.
        return float(samples.reshape(-1)[0]) if size == 1 else samples

    def _within_bounds(self, value: float) -> bool:
        if self.minimum is not None and value < self.minimum:
            return False
        if self.maximum is not None and value > self.maximum:
            return False
        return True

    def __repr__(self) -> str:
        return f"Expression({self.raw!r})"
