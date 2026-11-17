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
"""The table of random distributions the expression grammar supports.

A distribution may appear in an expression only if it has a row in
:data:`DISTRIBUTIONS`. The row carries everything the grammar needs to know
about it: its parameters in the order a config writes them, the conditions
they must meet, the support, the mean and a numpy sampler. Nothing about a
distribution is asked of sympy after parsing, so a distribution that sympy
can build but not sample, or whose mean never exists, cannot get in.

Parameters, conditions, support and mean follow sympy's definition of each
distribution, and ``test_distributions.py`` checks every row against sympy.
The samplers reparameterise numpy where its convention differs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable, Mapping, Optional, Sequence

import numpy as np
import sympy
from numpy.typing import NDArray
from sympy import Eq, Ge, Gt, Le, Lt, oo

from .sympy_types import SympyExpr, SympyNode, SympyRelational

# draws ``size`` values given the parameters as floats, in the row's order
Sampler = Callable[[np.random.Generator, Sequence[float], int], NDArray[np.float64]]


# the base draw in a row's standard form
BASE: SympyExpr = sympy.Symbol("z", real=True)


@dataclass(frozen=True)
class Standard:
    """A row's draw written as a function of a base draw that its time-varying parameters do not touch.

    ``Exponential(rate)`` is ``z / rate`` for ``z`` drawn from ``Exponential(1)``,
    and ``Normal(mean, std)`` is ``mean + std * z`` for a standard normal ``z``.
    Holding ``z`` fixed while the parameters move is what lets one draw be
    followed along a curve: the same draw gives a shorter gap as the rate
    climbs. A parameter the base draw itself reads, such as a gamma's shape,
    is listed in ``fixed`` and cannot vary with ``t``.
    """

    # draws the base values. reads only the ``fixed`` parameters, the rest may be nan
    base: Sampler
    # the draw, over BASE and the row's parameter symbols
    value: SympyExpr
    fixed: tuple[str, ...] = ()


@dataclass(frozen=True)
class Distribution:
    """One supported distribution. Every sympy field is written over :attr:`symbols`."""

    name: str
    # as written in a config, in sympy's argument order
    parameters: tuple[str, ...]
    # relationals over the parameter symbols, all of which must hold
    conditions: tuple[SympyRelational, ...]
    # the lowest and highest value a draw can take, either may be infinite
    lower: SympyExpr
    upper: SympyExpr
    # whether every draw is a whole number
    integer: bool
    mean: SympyExpr
    # further relationals the mean needs to be finite, empty when it always is
    mean_finite_when: tuple[SympyRelational, ...]
    sampler: Sampler
    # the one parameter is a list of values to pick from, e.g. DiscreteUniform([128, 256, 512]).
    # lower, upper and mean are then display forms only, DistributionUse works them out from the values
    takes_list: bool = False
    # None for a row whose parameters cannot vary with t
    standard: Optional[Standard] = None

    @property
    def symbols(self) -> tuple[SympyExpr, ...]:
        return _symbols(*self.parameters)

    @property
    def signature(self) -> str:
        """The call as a config writes it, e.g. ``Normal(mean, std)``."""
        if self.takes_list:
            return f"{self.name}([{self.parameters[0]}])"
        return f"{self.name}({', '.join(self.parameters)})"

    def use(self, args: Sequence[object]) -> DistributionUse:
        """Bind arguments to the parameters. Raises ``ValueError`` on a wrong count or a non-numeric argument."""
        if self.takes_list:
            return self._use_list(args)
        if len(args) != len(self.parameters):
            raise ValueError(f"{self.signature} takes {len(self.parameters)} argument(s), got {len(args)}.")
        bound: list[SympyExpr] = []
        for parameter, arg in zip(self.parameters, args, strict=True):
            if isinstance(arg, bool) or not isinstance(arg, (int, float, sympy.Expr)):
                raise ValueError(f"{self.signature}: {parameter} must be a number, got {arg!r}.")
            value: SympyExpr = sympy.sympify(arg)
            bound.append(value)
        return DistributionUse(self, tuple(bound))

    def _use_list(self, args: Sequence[object]) -> DistributionUse:
        if len(args) != 1 or not isinstance(args[0], (list, tuple)):
            raise ValueError(f"{self.signature} takes one list of numbers, e.g. {self.name}([128, 256, 512]).")
        items: list[SympyExpr] = []
        for item in args[0]:
            if isinstance(item, bool) or not isinstance(item, (int, float, sympy.Expr)):
                raise ValueError(f"{self.signature}: every value must be a number, got {item!r}.")
            value: SympyExpr = sympy.sympify(item)
            items.append(value)
        if not items:
            raise ValueError(f"{self.signature} needs at least one value.")
        # a repeat would have to mean a weight, which this distribution does not have
        repeated = sorted({str(item) for item in items if items.count(item) > 1})
        if repeated:
            raise ValueError(f"{self.signature}: each value may appear once, got {', '.join(repeated)} more than once.")
        return DistributionUse(self, tuple(items))

    def use_of_sympy(self, sympy_args: Sequence[object]) -> DistributionUse:
        """The use behind a sympy distribution object's ``args``, e.g. ``NormalDistribution(512, 128).args``."""
        if self.takes_list:
            return self.use([list(sympy_args)])
        # sympy appends arguments of its own to some distributions, e.g. Binomial's succ and fail
        return self.use(sympy_args[: len(self.parameters)])


@dataclass(frozen=True)
class DistributionUse:
    """A distribution with arguments bound to its parameters, e.g. ``Normal(512, 128)``.

    The arguments are numbers wherever the grammar builds one. They may be
    expressions of ``t`` when a caller is proving the conditions over a window.
    For a row that takes a list, the arguments are the values of the list.
    """

    row: Distribution
    args: tuple[SympyExpr, ...]

    def __str__(self) -> str:
        written = ", ".join(str(arg) for arg in self.args)
        return f"{self.row.name}([{written}])" if self.row.takes_list else f"{self.row.name}({written})"

    @property
    def constructor_args(self) -> tuple[object, ...]:
        """The arguments as sympy's constructor takes them after the name."""
        return (list(self.args),) if self.row.takes_list else self.args

    @property
    def integer(self) -> bool:
        """Whether every draw is a whole number."""
        if self.row.takes_list:
            return all(sympy.floor(arg) == arg for arg in self.args)
        return self.row.integer

    @property
    def _binding(self) -> Mapping[SympyNode, SympyNode]:
        return dict(zip(self.row.symbols, self.args, strict=True))

    def bind(self, expr: SympyExpr) -> SympyExpr:
        """Substitute the arguments into an expression of the parameters, e.g. the row's mean."""
        return expr.xreplace(self._binding)

    def substitute(self, rule: Mapping[SympyNode, SympyNode]) -> DistributionUse:
        """The same use with ``rule`` applied to every argument, e.g. ``{t: 3.0}`` for the use as it stands at 3s."""
        return DistributionUse(self.row, tuple(arg.xreplace(rule) for arg in self.args))

    @property
    def over_time(self) -> tuple[str, ...]:
        """The parameters whose argument is not a constant, e.g. ``("rate",)`` for ``Exponential(5 + t/10)``."""
        if self.row.takes_list:
            return ()
        return tuple(name for name, arg in zip(self.row.parameters, self.args, strict=True) if arg.free_symbols)

    def require_standard(self) -> None:
        """Raise ``ValueError`` when an argument varies but the row cannot follow it along a curve."""
        varying = self.over_time
        if not varying:
            return
        standard = self.row.standard
        if standard is None:
            raise ValueError(f"{self}: the parameters of {self.row.signature} cannot vary with t.")
        stuck = [name for name in varying if name in standard.fixed]
        if stuck:
            free = [name for name in self.row.parameters if name not in standard.fixed]
            raise ValueError(
                f"{self}: {', '.join(stuck)} of {self.row.signature} cannot vary with t. Only {', '.join(free)} can."
            )

    def coupled(self) -> tuple[Callable[[np.random.Generator, int], NDArray[np.float64]], SympyExpr]:
        """A base sampler and the draw as an expression of :data:`BASE`, for following one draw as ``t`` moves.

        With constant arguments the base draw is the draw itself. Otherwise it
        is the row's standard form with the arguments substituted, e.g.
        ``z / (5 + t/10)`` over ``z`` from ``Exponential(1)``.
        """
        if not self.over_time:
            return self.sample, BASE
        self.require_standard()
        standard = self.row.standard
        assert standard is not None
        parameters = [float(arg) if not arg.free_symbols else math.nan for arg in self.args]

        def base(rng: np.random.Generator, size: int) -> NDArray[np.float64]:
            return standard.base(rng, parameters, size)

        return base, self.bind(standard.value)

    def holds(self, condition: SympyRelational) -> bool:
        """True when the condition is decided and true for these arguments."""
        return bool(condition.xreplace(self._binding) == sympy.true)

    def require_constant(self) -> None:
        """Raise ``ValueError`` unless every argument is a finite number and every condition holds."""
        names = self.row.parameters
        if self.row.takes_list:
            names = tuple(f"{self.row.parameters[0]}[{index}]" for index in range(len(self.args)))
        for parameter, arg in zip(names, self.args, strict=True):
            constant = arg.is_number and arg.is_extended_real is True
            if not constant or not math.isfinite(float(arg)):
                raise ValueError(f"{self}: {parameter} must be a finite constant, got {arg}.")
        for condition in self.row.conditions:
            if not self.holds(condition):
                named = sorted(str(symbol) for symbol in condition.free_symbols)
                got = ", ".join(f"{name} = {self.args[self.row.parameters.index(name)]}" for name in named)
                raise ValueError(f"{self}: {self.row.signature} requires {render_condition(condition)}, got {got}.")

    @property
    def lower(self) -> SympyExpr:
        if self.row.takes_list:
            lowest: SympyExpr = sympy.Min(*self.args)
            return lowest
        return self.bind(self.row.lower)

    @property
    def upper(self) -> SympyExpr:
        if self.row.takes_list:
            highest: SympyExpr = sympy.Max(*self.args)
            return highest
        return self.bind(self.row.upper)

    @property
    def mean(self) -> SympyExpr:
        """The mean for these arguments, ``oo`` where the row says it is not finite."""
        if self.row.takes_list:
            average: SympyExpr = sympy.Add(*self.args) / len(self.args)
            return average
        if not all(self.holds(condition) for condition in self.row.mean_finite_when):
            infinite: SympyExpr = oo
            return infinite
        return self.bind(self.row.mean)

    def sample(self, rng: np.random.Generator, size: int) -> NDArray[np.float64]:
        return self.row.sampler(rng, [float(arg) for arg in self.args], size)


def _symbols(*names: str) -> tuple[SympyExpr, ...]:
    # real, so a condition such as 'std > 0' is a well-formed relational
    return tuple(sympy.Symbol(name, real=True) for name in names)


def render_condition(condition: SympyRelational) -> str:
    """A condition as a config author reads it, e.g. ``std > 0`` or ``floor(n) == n``."""
    return f"{condition.lhs} {condition.rel_op} {condition.rhs}"


def _floats(draws: object) -> NDArray[np.float64]:
    return np.asarray(draws, dtype=np.float64)


def _sample_normal(rng: np.random.Generator, p: Sequence[float], size: int) -> NDArray[np.float64]:
    return _floats(rng.normal(loc=p[0], scale=p[1], size=size))


def _sample_lognormal(rng: np.random.Generator, p: Sequence[float], size: int) -> NDArray[np.float64]:
    return _floats(rng.lognormal(mean=p[0], sigma=p[1], size=size))


def _sample_exponential(rng: np.random.Generator, p: Sequence[float], size: int) -> NDArray[np.float64]:
    # sympy takes a rate, numpy a scale
    return _floats(rng.exponential(1 / p[0], size=size))


def _sample_gamma(rng: np.random.Generator, p: Sequence[float], size: int) -> NDArray[np.float64]:
    return _floats(rng.gamma(shape=p[0], scale=p[1], size=size))


def _sample_uniform(rng: np.random.Generator, p: Sequence[float], size: int) -> NDArray[np.float64]:
    return _floats(rng.uniform(low=p[0], high=p[1], size=size))


def _sample_beta(rng: np.random.Generator, p: Sequence[float], size: int) -> NDArray[np.float64]:
    return _floats(rng.beta(a=p[0], b=p[1], size=size))


def _sample_weibull(rng: np.random.Generator, p: Sequence[float], size: int) -> NDArray[np.float64]:
    # sympy takes (scale, shape), numpy draws the unit-scale Weibull from the shape alone
    return _floats(p[0] * rng.weibull(a=p[1], size=size))


def _sample_pareto(rng: np.random.Generator, p: Sequence[float], size: int) -> NDArray[np.float64]:
    # numpy draws the Lomax form, which starts at 0: shift by 1 and scale by the minimum
    return _floats((rng.pareto(a=p[1], size=size) + 1) * p[0])


def _sample_poisson(rng: np.random.Generator, p: Sequence[float], size: int) -> NDArray[np.float64]:
    return _floats(rng.poisson(lam=p[0], size=size))


def _sample_binomial(rng: np.random.Generator, p: Sequence[float], size: int) -> NDArray[np.float64]:
    return _floats(rng.binomial(n=int(p[0]), p=p[1], size=size))


def _sample_geometric(rng: np.random.Generator, p: Sequence[float], size: int) -> NDArray[np.float64]:
    # numpy counts trials up to and including the first success, so draws start at 1 as sympy's do
    return _floats(rng.geometric(p=p[0], size=size))


def _sample_discrete_uniform(rng: np.random.Generator, p: Sequence[float], size: int) -> NDArray[np.float64]:
    return _floats(rng.choice(np.asarray(p, dtype=np.float64), size=size))


def _base_normal(rng: np.random.Generator, p: Sequence[float], size: int) -> NDArray[np.float64]:
    return _floats(rng.standard_normal(size))


def _base_exponential(rng: np.random.Generator, p: Sequence[float], size: int) -> NDArray[np.float64]:
    return _floats(rng.standard_exponential(size))


def _base_uniform(rng: np.random.Generator, p: Sequence[float], size: int) -> NDArray[np.float64]:
    return _floats(rng.random(size))


def _base_gamma(rng: np.random.Generator, p: Sequence[float], size: int) -> NDArray[np.float64]:
    return _floats(rng.standard_gamma(p[0], size))


def _normal() -> Distribution:
    mean, std = sympy.symbols("mean std", real=True)
    standard = Standard(_base_normal, mean + std * BASE)
    return Distribution("Normal", ("mean", "std"), (Gt(std, 0),), -oo, oo, False, mean, (), _sample_normal, standard=standard)


def _lognormal() -> Distribution:
    # mean and std are those of the underlying normal, as in sympy
    mean, std = sympy.symbols("mean std", real=True)
    return Distribution(
        "LogNormal",
        ("mean", "std"),
        (Gt(std, 0),),
        sympy.S.Zero,
        oo,
        False,
        sympy.exp(mean + std**2 / 2),
        (),
        _sample_lognormal,
        standard=Standard(_base_normal, sympy.exp(mean + std * BASE)),
    )


def _exponential() -> Distribution:
    rate = sympy.Symbol("rate", real=True)
    standard = Standard(_base_exponential, BASE / rate)
    return Distribution(
        "Exponential", ("rate",), (Gt(rate, 0),), sympy.S.Zero, oo, False, 1 / rate, (), _sample_exponential, standard=standard
    )


def _gamma() -> Distribution:
    shape, scale = sympy.symbols("shape scale", real=True)
    # the base draw is Gamma(shape, 1), so the shape is part of the draw and only the scale can move
    standard = Standard(_base_gamma, scale * BASE, fixed=("shape",))
    return Distribution(
        "Gamma",
        ("shape", "scale"),
        (Gt(shape, 0), Gt(scale, 0)),
        sympy.S.Zero,
        oo,
        False,
        shape * scale,
        (),
        _sample_gamma,
        standard=standard,
    )


def _uniform() -> Distribution:
    left, right = sympy.symbols("left right", real=True)
    standard = Standard(_base_uniform, left + (right - left) * BASE)
    return Distribution(
        "Uniform",
        ("left", "right"),
        (Lt(left, right),),
        left,
        right,
        False,
        (left + right) / 2,
        (),
        _sample_uniform,
        standard=standard,
    )


def _beta() -> Distribution:
    alpha, beta = sympy.symbols("alpha beta", real=True)
    return Distribution(
        "Beta",
        ("alpha", "beta"),
        (Gt(alpha, 0), Gt(beta, 0)),
        sympy.S.Zero,
        sympy.S.One,
        False,
        alpha / (alpha + beta),
        (),
        _sample_beta,
    )


def _weibull() -> Distribution:
    scale, shape = sympy.symbols("scale shape", real=True)
    return Distribution(
        "Weibull",
        ("scale", "shape"),
        (Gt(scale, 0), Gt(shape, 0)),
        sympy.S.Zero,
        oo,
        False,
        scale * sympy.gamma(1 + 1 / shape),
        (),
        _sample_weibull,
        # a Weibull is a power of an exponential
        standard=Standard(_base_exponential, scale * BASE ** (1 / shape)),
    )


def _pareto() -> Distribution:
    minimum, alpha = sympy.symbols("minimum alpha", real=True)
    return Distribution(
        "Pareto",
        ("minimum", "alpha"),
        (Gt(minimum, 0), Gt(alpha, 0)),
        minimum,
        oo,
        False,
        alpha * minimum / (alpha - 1),
        (Gt(alpha, 1),),
        _sample_pareto,
        # the log of a Pareto over its minimum is an exponential
        standard=Standard(_base_exponential, minimum * sympy.exp(BASE / alpha)),
    )


def _poisson() -> Distribution:
    mean = sympy.Symbol("mean", real=True)
    return Distribution("Poisson", ("mean",), (Gt(mean, 0),), sympy.S.Zero, oo, True, mean, (), _sample_poisson)


def _binomial() -> Distribution:
    n, p = sympy.symbols("n p", real=True)
    return Distribution(
        "Binomial",
        ("n", "p"),
        # floor(n) == n says n is a whole number in a form the range machinery can evaluate
        (Ge(n, 0), Eq(sympy.floor(n), n), Ge(p, 0), Le(p, 1)),
        sympy.S.Zero,
        n,
        True,
        n * p,
        (),
        _sample_binomial,
    )


def _geometric() -> Distribution:
    p = sympy.Symbol("p", real=True)
    return Distribution("Geometric", ("p",), (Gt(p, 0), Le(p, 1)), sympy.S.One, oo, True, 1 / p, (), _sample_geometric)


def _discrete_uniform() -> Distribution:
    values = sympy.Symbol("values", real=True)
    # shown in the docs table, a use reads the list itself
    lowest, highest, average = (sympy.Function(name)(values) for name in ("min", "max", "mean"))
    return Distribution(
        "DiscreteUniform", ("values",), (), lowest, highest, False, average, (), _sample_discrete_uniform, takes_list=True
    )


DISTRIBUTIONS: Mapping[str, Distribution] = {
    row.name: row
    for row in (
        _normal(),
        _lognormal(),
        _exponential(),
        _gamma(),
        _uniform(),
        _beta(),
        _weibull(),
        _pareto(),
        _poisson(),
        _binomial(),
        _geometric(),
        _discrete_uniform(),
    )
}


def unsupported_message(name: str) -> str:
    return f"Distribution {name!r} is not supported. Supported distributions: {', '.join(DISTRIBUTIONS)}."


def lookup(sympy_distribution: object) -> Optional[Distribution]:
    """The row for a sympy distribution object such as ``NormalDistribution(512, 128)``, or ``None``."""
    return DISTRIBUTIONS.get(type(sympy_distribution).__name__.removesuffix("Distribution"))


def _render_bound(bound: SympyExpr) -> str:
    return str(bound).replace("oo", "inf")


DOCS_HEADER = ("Distribution", "Requires", "Lowest", "Highest", "Whole numbers", "Mean")


def docs_rows() -> list[tuple[str, ...]]:
    """The supported-distributions table in ``docs/expressions.md``, one tuple of cells per row.

    The docs test compares the doc against this, so the doc cannot drift from the table.
    """
    rows: list[tuple[str, ...]] = []
    for row in DISTRIBUTIONS.values():
        mean = f"`{row.mean}`"
        if row.mean_finite_when:
            mean += f" when {' and '.join(f'`{render_condition(c)}`' for c in row.mean_finite_when)}, otherwise infinite"
        rows.append(
            (
                f"`{row.signature}`",
                "at least one value, none repeated"
                if row.takes_list
                else ", ".join(f"`{render_condition(c)}`" for c in row.conditions),
                f"`{_render_bound(row.lower)}`",
                f"`{_render_bound(row.upper)}`",
                "when every value is" if row.takes_list else "yes" if row.integer else "no",
                mean,
            )
        )
    return rows
