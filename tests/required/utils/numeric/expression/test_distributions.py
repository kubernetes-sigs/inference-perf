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
"""The distribution table: every row is complete, samples what it states, and agrees with sympy's definition."""

import math
from typing import Callable, Sequence

import numpy as np
import pytest
import sympy
import sympy.stats
from numpy.typing import NDArray

from inference_perf.utils.numeric.expression.distributions import DISTRIBUTIONS, Distribution, DistributionUse, lookup
from inference_perf.utils.numeric.expression.sympy_types import SympyExpr, SympyRandomSymbol

_HALF = sympy.Rational(1, 2)
_QUARTER = sympy.Rational(1, 4)

# One valid argument list per row. The values are lopsided on purpose: swapping two parameters, or reading a
# rate as a scale, moves the mean or the support.
GOOD: dict[str, tuple[object, ...]] = {
    "Normal": (10, 2),
    "LogNormal": (1, _HALF),
    "Exponential": (5,),
    "Gamma": (2, 3),
    "Uniform": (2, 6),
    "Beta": (2, 5),
    "Weibull": (3, 2),
    "Pareto": (2, 3),
    "Poisson": (4,),
    "Binomial": (10, _QUARTER),
    "Geometric": (_QUARTER,),
    "DiscreteUniform": ([128, 256, 1024],),
}

# Argument lists that break exactly one condition each, in the order the row lists its conditions.
BAD: dict[str, list[tuple[object, ...]]] = {
    "Normal": [(10, 0)],
    "LogNormal": [(1, -1)],
    "Exponential": [(0,)],
    "Gamma": [(0, 3), (2, -1)],
    "Uniform": [(6, 6)],
    "Beta": [(0, 5), (2, 0)],
    "Weibull": [(0, 2), (3, 0)],
    "Pareto": [(0, 3), (2, 0)],
    "Poisson": [(0,)],
    "Binomial": [(-1, _QUARTER), (_HALF * 5, _QUARTER), (10, -_QUARTER), (10, _HALF * 3)],
    "Geometric": [(0,), (_HALF * 3,)],
    # a list has no condition over a parameter symbol, its checks are in test_discrete_uniform_rejects_a_bad_list
    "DiscreteUniform": [],
}

# What sympy itself cannot compute in this version, so the row is not compared against it there.
SYMPY_CANNOT_COMPUTE_MEAN = {"LogNormal": "sympy.stats.E raises NotImplementedError for the log-normal"}

# Rows whose sympy numpy sampler takes the caller's seed. Weibull has no sympy sampler, and sympy's Pareto
# sampler draws from numpy's global state.
# DiscreteUniform has no sympy numpy sampler either.
SYMPY_SEEDED_SAMPLER = sorted(set(GOOD) - {"Weibull", "Pareto", "DiscreteUniform"})

# Rows that take one list of values in place of named parameters.
LIST_ROWS = {"DiscreteUniform"}

_DRAWS = 200_000
_ROWS = sorted(DISTRIBUTIONS)


# The row named 'Normal' bound to its GOOD arguments (10, 2), checked as constants.
def _use(name: str, args: Sequence[object] | None = None) -> DistributionUse:
    use = DISTRIBUTIONS[name].use(GOOD[name] if args is None else args)
    use.require_constant()
    return use


# 200000 draws of a row's sampler at its GOOD arguments from numpy's default_rng(seed).
def _draws(name: str, seed: int = 0) -> NDArray[np.float64]:
    return _use(name).sample(np.random.default_rng(seed), _DRAWS)


# sympy's own random variable for a row, e.g. sympy.stats.Normal('x', 10, 2).
def _sympy_rv(name: str, args: Sequence[object] | None = None) -> SympyRandomSymbol:
    rv: SympyRandomSymbol = getattr(sympy.stats, name)("x", *(GOOD[name] if args is None else args))
    return rv


# sympy's P(X <= x) as a float. A finite distribution's cdf is a dict of value to cumulative probability.
def _sympy_cdf(name: str, x: float) -> float:
    cdf = sympy.stats.cdf(_sympy_rv(name))
    if isinstance(cdf, dict):
        return float(sum(p for value, p in sympy.stats.density(_sympy_rv(name)).dict.items() if value <= x))
    return float(cdf(sympy.Float(x)))


# The table holds exactly these twelve rows, keyed by name. Adding or dropping one must change this list.
def test_table_lists_exactly_the_supported_distributions() -> None:
    assert list(DISTRIBUTIONS) == [
        "Normal",
        "LogNormal",
        "Exponential",
        "Gamma",
        "Uniform",
        "Beta",
        "Weibull",
        "Pareto",
        "Poisson",
        "Binomial",
        "Geometric",
        "DiscreteUniform",
    ]
    assert all(name == row.name for name, row in DISTRIBUTIONS.items())
    assert set(GOOD) == set(BAD) == set(DISTRIBUTIONS)


# Every row has all five parts: a name with parameters, at least one condition and one on every parameter, a
# support, a mean and a sampler. Each sympy part reads only the row's own parameter symbols.
@pytest.mark.parametrize("name", sorted(set(_ROWS) - LIST_ROWS))
def test_row_has_every_part(name: str) -> None:
    row = DISTRIBUTIONS[name]
    symbols = set(row.symbols)
    assert row.parameters and len(set(row.parameters)) == len(row.parameters)
    assert row.signature == f"{name}({', '.join(row.parameters)})"
    assert row.conditions
    for part in (*row.conditions, *row.mean_finite_when, row.lower, row.upper, row.mean):
        assert part.free_symbols <= symbols, f"{name}: {part} reads something that is not a parameter"
    # Normal's mean is the one parameter in the table that may be any real number
    unconditioned = symbols - set().union(*(condition.free_symbols for condition in row.conditions))
    assert {str(symbol) for symbol in unconditioned} == ({"mean"} if name in ("Normal", "LogNormal") else set())
    assert isinstance(row.integer, bool)
    assert row.mean.free_symbols, f"{name}: the mean must be an expression of the parameters"
    assert callable(row.sampler)


# 200000 seeded draws per row at its GOOD arguments, e.g. Exponential(5). Expected: every draw lies in the stated
# support, is a whole number when the row says so, and the sample mean is within 5 standard errors of the row's mean.
@pytest.mark.parametrize("name", _ROWS)
def test_sampler_matches_the_row(name: str) -> None:
    use = _use(name)
    draws = _draws(name)
    assert draws.shape == (_DRAWS,) and draws.dtype == np.float64
    assert float(use.lower) <= draws.min() and draws.max() <= float(use.upper)
    assert bool(np.all(draws == np.floor(draws))) is use.integer
    standard_error = float(draws.std(ddof=1)) / math.sqrt(_DRAWS)
    assert abs(float(draws.mean()) - float(use.mean)) <= 5 * standard_error


# The sampler draws sympy's distribution, not numpy's default one. At the 10th, 50th and 90th percentile of
# 200000 draws, the fraction of draws at or below that value is within 5 standard errors of sympy's cdf there.
@pytest.mark.parametrize("name", _ROWS)
def test_sampler_follows_sympy_cdf(name: str) -> None:
    draws = _draws(name)
    for x in np.quantile(draws, [0.1, 0.5, 0.9]):
        expected = _sympy_cdf(name, float(x))
        observed = float(np.mean(draws <= x))
        tolerance = 5 * math.sqrt(expected * (1 - expected) / _DRAWS)
        assert abs(observed - expected) <= tolerance, f"{name}: P(X <= {x}) drew {observed}, sympy says {expected}"


# Same seed, same numbers as sympy's own numpy sampler: 1000 draws from default_rng(42) through the row equal
# 1000 through sympy.stats' do_sample_numpy. This is what keeps seeded runs unchanged by the table.
@pytest.mark.parametrize("name", SYMPY_SEEDED_SAMPLER)
def test_seeded_draws_equal_sympy_numpy_sampler(name: str) -> None:
    sample_numpy = pytest.importorskip("sympy.stats.sampling.sample_numpy")
    ours = _use(name).sample(np.random.default_rng(42), 1000)
    theirs = sample_numpy.do_sample_numpy(_sympy_rv(name).pspace.distribution, (1000,), np.random.default_rng(42))
    np.testing.assert_array_equal(ours, np.asarray(theirs, dtype=np.float64))


# Pareto(2, 3) is seeded now: two runs from default_rng(42) draw the same 1000 numbers. sympy's sampler ignored
# the seed.
def test_pareto_is_reproducible_with_a_seed() -> None:
    first = _use("Pareto").sample(np.random.default_rng(42), 1000)
    second = _use("Pareto").sample(np.random.default_rng(42), 1000)
    np.testing.assert_array_equal(first, second)


# sympy builds '<Name>Distribution' for each row with the arguments in the row's parameter order, and lookup()
# maps that object back to the row. Normal('x', 10, 2) -> NormalDistribution with args (10, 2) -> the Normal row.
@pytest.mark.parametrize("name", _ROWS)
def test_row_maps_to_sympy_distribution(name: str) -> None:
    row = DISTRIBUTIONS[name]
    distribution = _sympy_rv(name).pspace.distribution
    assert lookup(distribution) is row
    assert row.use_of_sympy(distribution.args).args == _use(name).args
    if name not in LIST_ROWS:
        assert distribution.args[: len(row.parameters)] == _use(name).args


# The row's support equals sympy's at the GOOD arguments: Uniform(2, 6) -> [2, 6], Geometric(1/4) -> 1 upward,
# Binomial(10, 1/4) -> 0 to 10. Whole-number rows are exactly those whose sympy support is a set of integers.
@pytest.mark.parametrize("name", _ROWS)
def test_support_agrees_with_sympy(name: str) -> None:
    use = _use(name)
    support = _sympy_rv(name).pspace.distribution.set
    if isinstance(support, (set, frozenset)):
        # a finite distribution gives a Python set of its values
        lowest, highest = min(support), max(support)
        integer = all(sympy.sympify(value).is_integer for value in support)
    else:
        lowest, highest = support.inf, support.sup
        integer = bool(support.is_subset(sympy.S.Integers))
    assert (use.lower, use.upper) == (lowest, highest)
    assert use.integer is integer


# The row's mean equals sympy's E at the GOOD arguments, e.g. Weibull(3, 2) -> 3*sqrt(pi)/2. The only skip is
# LogNormal, where sympy's E raises. That is asserted, so the skip ends when sympy learns to compute it.
@pytest.mark.parametrize("name", _ROWS)
def test_mean_agrees_with_sympy(name: str) -> None:
    if name in SYMPY_CANNOT_COMPUTE_MEAN:
        with pytest.raises(NotImplementedError):
            sympy.stats.E(_sympy_rv(name))
        pytest.skip(SYMPY_CANNOT_COMPUTE_MEAN[name])
    expected: SympyExpr = sympy.stats.E(_sympy_rv(name))
    assert sympy.simplify(sympy.Add(_use(name).mean, -expected)) == 0


# LogNormal(1, 1/2), which sympy cannot take the mean of: the row says exp(1 + (1/2)**2/2) = exp(9/8).
def test_lognormal_mean_is_of_the_result_not_the_underlying_normal() -> None:
    assert _use("LogNormal").mean == sympy.exp(sympy.Rational(9, 8))


# Pareto's mean is finite only for alpha > 1. Pareto(2, 3) -> 3, Pareto(2, 1) and Pareto(2, 1/2) -> oo,
# and sympy's E agrees on all three.
@pytest.mark.parametrize(("alpha", "expected"), [(3, 3), (1, sympy.oo), (_HALF, sympy.oo)])
def test_pareto_mean_is_infinite_at_or_below_alpha_one(alpha: object, expected: object) -> None:
    assert _use("Pareto", (2, alpha)).mean == expected
    assert sympy.stats.E(_sympy_rv("Pareto", (2, alpha))) == expected


# The GOOD arguments pass the row's conditions and sympy's own check, e.g. Normal(10, 2).
@pytest.mark.parametrize("name", _ROWS)
def test_good_arguments_pass_like_sympy(name: str) -> None:
    _use(name)
    _sympy_rv(name)


# Each BAD argument list breaks one condition, e.g. Normal(10, 0) breaks std > 0. Expected: the row rejects it
# naming that condition, sympy's check rejects it too, and every condition of the row is broken by some list.
@pytest.mark.parametrize("name", _ROWS)
def test_bad_arguments_fail_like_sympy(name: str) -> None:
    row = DISTRIBUTIONS[name]
    broken = set()
    for args in BAD[name]:
        use = row.use(args)
        failing = [condition for condition in row.conditions if not use.holds(condition)]
        assert len(failing) == 1, f"{use}: expected one broken condition, got {failing}"
        broken.add(failing[0])
        with pytest.raises(ValueError, match="requires"):
            use.require_constant()
        with pytest.raises(ValueError):
            _sympy_rv(name, args)
    assert broken == set(row.conditions)


# DiscreteUniform([128, 256, 1024]) reads its support and mean from the list: lowest 128, highest 1024, mean
# 1408/3, whole numbers. DiscreteUniform([0.5, 2]) is not whole-number. It prints the way a config writes it.
def test_discrete_uniform_reads_the_list() -> None:
    use = _use("DiscreteUniform")
    assert (use.lower, use.upper, use.mean) == (sympy.Integer(128), sympy.Integer(1024), sympy.Rational(1408, 3))
    assert use.integer
    assert str(use) == "DiscreteUniform([128, 256, 1024])"
    assert DISTRIBUTIONS["DiscreteUniform"].signature == "DiscreteUniform([values])"
    assert _use("DiscreteUniform", ([0.5, 2],)).integer is False
    assert set(_draws("DiscreteUniform").tolist()) == {128.0, 256.0, 1024.0}


# A list that cannot be picked from is refused with the reason: no list, an empty list, a repeated value, a
# value that is not a number, and a value that is not a finite constant.
@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((1, 5), "takes one list of numbers"),
        (([1, 2], [3]), "takes one list of numbers"),
        (([],), "needs at least one value"),
        (([1, 2, 1],), "each value may appear once, got 1 more than once"),
        ((["a", 2],), "every value must be a number"),
        (([True, 2],), "every value must be a number"),
        (([sympy.Symbol("t"), 2],), r"values\[0\] must be a finite constant"),
        (([1, sympy.oo],), r"values\[1\] must be a finite constant"),
    ],
)
def test_discrete_uniform_rejects_a_bad_list(args: tuple[object, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DISTRIBUTIONS["DiscreteUniform"].use(args).require_constant()


# DiscreteUniform compares values numerically, not by sympy type: [128.0, 256] is whole-numbered.
def test_discrete_uniform_float_whole_numbers_are_integer() -> None:
    assert _use("DiscreteUniform", ([128.0, 256],)).integer is True


# DiscreteUniform rejects numerically equal repeats even across types: [128, 128.0] and [1/2, 0.5] raise.
@pytest.mark.parametrize("values", [[128, 128.0], [_HALF, 0.5]], ids=["int-float", "rational-float"])
def test_discrete_uniform_rejects_numeric_duplicates(values: list[object]) -> None:
    with pytest.raises(ValueError, match="each value may appear once"):
        DISTRIBUTIONS["DiscreteUniform"].use((values,))


# Edge values sympy accepts are accepted: Binomial(0, 1/4), Binomial(10, 0), Binomial(10, 1), Geometric(1)
# and a whole number written as a float, Binomial(10.0, 1/4).
@pytest.mark.parametrize(
    ("name", "args"),
    [
        ("Binomial", (0, _QUARTER)),
        ("Binomial", (10, 0)),
        ("Binomial", (10, 1)),
        ("Geometric", (1,)),
        ("Binomial", (10.0, _QUARTER)),
    ],
)
def test_boundary_arguments_pass_like_sympy(name: str, args: tuple[object, ...]) -> None:
    _use(name, args)
    _sympy_rv(name, args)


# A wrong argument count or a non-number is refused before any condition is read: Normal(1) -> 'takes 2',
# Normal([1], 2) and Normal(True, 2) -> 'must be a number'.
@pytest.mark.parametrize(
    ("args", "message"),
    [
        ((1,), "takes 2 argument"),
        ((1, 2, 3), "takes 2 argument"),
        (([1], 2), "mean must be a number"),
        ((True, 2), "mean must be a number"),
    ],
)
def test_use_rejects_wrong_arguments(args: tuple[object, ...], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        DISTRIBUTIONS["Normal"].use(args)


# A constant use needs finite numbers: Normal(oo, 1) and Normal(t, 1) -> 'must be a finite constant'.
@pytest.mark.parametrize("mean", [sympy.oo, sympy.Symbol("t"), sympy.nan], ids=["oo", "t", "nan"])
def test_require_constant_rejects_non_finite_or_symbolic(mean: object) -> None:
    with pytest.raises(ValueError, match="mean must be a finite constant"):
        DISTRIBUTIONS["Normal"].use((mean, 1)).require_constant()


# Each sampler reparameterises numpy to sympy's convention. With one draw per row from default_rng(0):
# Exponential(5) is numpy's exponential with scale 1/5, Weibull(3, 2) is 3 times numpy's weibull(2),
# Pareto(2, 3) is 2 times (numpy's pareto(3) + 1), Geometric(1/4) is numpy's geometric(1/4), which starts at 1.
@pytest.mark.parametrize(
    ("name", "expected"),
    [
        ("Exponential", lambda rng: rng.exponential(1 / 5, 4)),
        ("Weibull", lambda rng: 3 * rng.weibull(2, 4)),
        ("Pareto", lambda rng: 2 * (rng.pareto(3, 4) + 1)),
        ("Geometric", lambda rng: rng.geometric(0.25, 4)),
    ],
)
def test_sampler_parameterisation(name: str, expected: Callable[[np.random.Generator], NDArray[np.float64]]) -> None:
    np.testing.assert_array_equal(_use(name).sample(np.random.default_rng(0), 4), expected(np.random.default_rng(0)))


# lookup() of a sympy distribution with no row, Rayleigh(2), is None. The row type is exported for callers.
def test_lookup_of_unlisted_distribution() -> None:
    assert lookup(sympy.stats.Rayleigh("x", 2).pspace.distribution) is None
    assert isinstance(DISTRIBUTIONS["Normal"], Distribution)
