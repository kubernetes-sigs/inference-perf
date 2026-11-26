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
"""Expression reads distributions through the table: the allowlist, sampling, support, mean and the time-varying hook."""

import math
from typing import Optional

import numpy as np
import pytest
import sympy
import sympy.stats

from inference_perf.utils.numeric.expression import (
    DISTRIBUTIONS,
    Expression,
    ParameterConditionError,
    Predicate,
    parse_distribution_use,
    prove_parameter_conditions,
)
from inference_perf.utils.numeric.expression import expression as expression_module

_SUPPORTED = "Supported distributions: Normal, LogNormal, Exponential, Gamma, Uniform, Beta, Weibull, Pareto, Poisson, Binomial, Geometric, DiscreteUniform."


# A sympy.stats distribution with no row is rejected when parsed, alone or inside arithmetic. The message names it
# and lists the supported ones: "Distribution 'Rayleigh' is not supported. Supported distributions: Normal, ...".
@pytest.mark.parametrize(
    ("raw", "name"),
    [
        ("Rayleigh(2)", "Rayleigh"),
        ("Bernoulli(0.5)", "Bernoulli"),
        ("Cauchy(0, 1)", "Cauchy"),
        ("Triangular(0, 2, 1)", "Triangular"),
        ("Die(6)", "Die"),
        ("floor(Min(Laplace(0, 1), 10)) + Normal(0, 1)", "Laplace"),
    ],
)
def test_unlisted_distribution_is_rejected_by_name(raw: str, name: str) -> None:
    with pytest.raises(ValueError) as error:
        Expression(raw)
    assert str(error.value) == f"Distribution {name!r} is not supported. {_SUPPORTED}"


# The same rejection reaches a condition: 'Rayleigh(2) >= t' names Rayleigh instead of calling it a random variable.
def test_unlisted_distribution_is_rejected_in_a_predicate() -> None:
    with pytest.raises(ValueError, match="Distribution 'Rayleigh' is not supported"):
        Predicate("Rayleigh(2) >= t")


# Every sympy.stats constructor without a row is refused by the parse namespace, about 60 names. None is left
# callable by accident, and every row's name is callable.
def test_every_sympy_constructor_is_either_a_row_or_refused() -> None:
    namespace = expression_module._parse_namespace([0])
    unlisted = set(expression_module._DISTRIBUTION_CTORS) - set(DISTRIBUTIONS)
    assert len(unlisted) > 50
    for name in unlisted:
        with pytest.raises(ValueError, match=f"Distribution '{name}' is not supported"):
            namespace[name](1, 2)
    assert set(DISTRIBUTIONS) <= set(namespace)


# A random variable that reached a tree without going through the parse namespace is still refused: looking up
# sympy's own Rayleigh('x', 2) raises the same message.
def test_random_symbol_without_a_row_is_refused_after_parsing() -> None:
    with pytest.raises(ValueError, match="Distribution 'Rayleigh' is not supported"):
        expression_module._use_of(sympy.stats.Rayleigh("x", 2))


# Arguments that break a row's condition are rejected when parsed, with the condition and the value:
# Normal(512, 0) -> 'requires std > 0, got std = 0', Uniform(5, 3) -> 'requires left < right, got left = 5, right = 3'.
@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("Normal(512, 0)", r"Normal\(512, 0\): Normal\(mean, std\) requires std > 0, got std = 0\."),
        ("Uniform(5, 3)", r"Uniform\(left, right\) requires left < right, got left = 5, right = 3\."),
        ("Binomial(5/2, 1/2)", r"Binomial\(n, p\) requires floor\(n\) == n, got n = 5/2\."),
        ("Geometric(0)", r"Geometric\(p\) requires p > 0, got p = 0\."),
        ("Normal(512)", r"Normal\(mean, std\) takes 2 argument\(s\), got 1\."),
    ],
)
def test_arguments_are_checked_against_the_row(raw: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Expression(raw)


# A distribution argument must be a constant here: Normal(t, 1), Normal(Uniform(0, 1), 1) and Normal(oo, 1) are
# rejected when parsed. The first two used to construct and then fail on the first draw.
@pytest.mark.parametrize("raw", ["Normal(t, 1)", "Normal(Uniform(0, 1), 1)", "Normal(oo, 1)"])
def test_arguments_must_be_constant(raw: str) -> None:
    with pytest.raises(ValueError, match="mean must be a finite constant"):
        Expression(raw)


# A constant argument may be any constant expression: Exponential(1/300), Normal(sqrt(2), pi), Uniform(0, 2*pi).
@pytest.mark.parametrize("raw", ["Exponential(1/300)", "Normal(sqrt(2), pi)", "Uniform(0, 2*pi)", "LogNormal(6, 0.5)"])
def test_constant_expressions_are_accepted_as_arguments(raw: str) -> None:
    assert Expression(raw).is_random


# One bare draw per row, with the row's name and the arguments as the floats its sampler receives.
_BARE_DRAWS: list[tuple[str, str, list[float]]] = [
    ("Normal(10, 2)", "Normal", [10.0, 2.0]),
    ("LogNormal(1, 1/2)", "LogNormal", [1.0, 0.5]),
    ("Exponential(5)", "Exponential", [5.0]),
    ("Gamma(2, 3)", "Gamma", [2.0, 3.0]),
    ("Uniform(2, 6)", "Uniform", [2.0, 6.0]),
    ("Beta(2, 5)", "Beta", [2.0, 5.0]),
    ("Weibull(3, 2)", "Weibull", [3.0, 2.0]),
    ("Pareto(2, 3)", "Pareto", [2.0, 3.0]),
    ("Poisson(4)", "Poisson", [4.0]),
    ("Binomial(10, 1/4)", "Binomial", [10.0, 0.25]),
    ("Geometric(1/4)", "Geometric", [0.25]),
    ("DiscreteUniform([128, 256, 512])", "DiscreteUniform", [128.0, 256.0, 512.0]),
]


# Expression.sample of a bare row draws exactly what the row's sampler draws from the same seed, for every row.
# Exponential(5) from default_rng(3) with size 50 equals the Exponential row's sampler on [5.0].
@pytest.mark.parametrize(("raw", "name", "values"), _BARE_DRAWS)
def test_sample_draws_through_the_row_sampler(raw: str, name: str, values: list[float]) -> None:
    drawn = Expression(raw).sample(rng=np.random.default_rng(3), size=50)
    np.testing.assert_array_equal(drawn, DISTRIBUTIONS[name].sampler(np.random.default_rng(3), values, 50))


# The list above has one case for every row in the table.
def test_sample_cases_cover_every_row() -> None:
    assert sorted(name for _raw, name, _values in _BARE_DRAWS) == sorted(DISTRIBUTIONS)


# Arithmetic around a draw is applied to the sampler's output: floor(Min(LogNormal(6, 0.5), 4096)) from
# default_rng(1) equals floor(minimum(rng.lognormal(6, 0.5), 4096)) from the same seed, as whole numbers up to 4096.
def test_sample_applies_arithmetic_around_a_draw() -> None:
    drawn = np.asarray(Expression("floor(Min(LogNormal(6, 0.5), 4096))").sample(rng=np.random.default_rng(1), size=5000))
    expected = np.floor(np.minimum(np.random.default_rng(1).lognormal(6, 0.5, 5000), 4096))
    np.testing.assert_array_equal(drawn, expected)
    assert drawn.max() <= 4096


# A sum draws its terms in order from one generator: Normal(512, 128) + Poisson(4) from default_rng(2) equals
# rng.normal(512, 128, n) + rng.poisson(4, n) with the normal drawn first.
def test_sample_draws_each_term_of_a_sum_in_order() -> None:
    drawn = Expression("Normal(512, 128) + Poisson(4)").sample(rng=np.random.default_rng(2), size=100)
    rng = np.random.default_rng(2)
    np.testing.assert_array_equal(drawn, rng.normal(512, 128, 100) + rng.poisson(4, 100))


# A Piecewise mixture: 75% Uniform(100, 200), else Uniform(1000, 2000). From default_rng(0) with 20000 draws,
# every draw is in one of the two ranges, 73% to 77% are in the low one, and the draws equal the same three
# uniforms drawn by hand and selected on the middle one.
def test_sample_piecewise_mixture() -> None:
    raw = "Piecewise((Uniform(100, 200), Uniform(0, 1) < 0.75), (Uniform(1000, 2000), True))"
    drawn = np.asarray(Expression(raw).sample(rng=np.random.default_rng(0), size=20000))
    low = (drawn >= 100) & (drawn <= 200)
    assert bool(np.all(low | ((drawn >= 1000) & (drawn <= 2000))))
    assert 0.73 <= float(low.mean()) <= 0.77
    rng = np.random.default_rng(0)
    first, chooser, second = rng.uniform(100, 200, 20000), rng.uniform(0, 1, 20000), rng.uniform(1000, 2000, 20000)
    np.testing.assert_array_equal(drawn, np.where(chooser < 0.75, first, second))


# Weibull(3, 2) samples now: 20000 draws from default_rng(0) are non-negative with a mean within 2% of
# 3*gamma(3/2) = 2.6587. It used to parse and then raise NotImplementedError on the first draw.
def test_weibull_samples() -> None:
    drawn = np.asarray(Expression("Weibull(3, 2)").sample(rng=np.random.default_rng(0), size=20000))
    assert drawn.min() >= 0
    assert math.isclose(float(drawn.mean()), 3 * math.gamma(1.5), rel_tol=0.02)


# A size-1 draw is a float and equals the first of a seeded size-1 sampler call: Exponential(5) from default_rng(7).
def test_single_draw_is_a_float() -> None:
    value = Expression("Exponential(5)").sample(rng=np.random.default_rng(7))
    assert isinstance(value, float)
    assert value == float(np.random.default_rng(7).exponential(1 / 5, 1)[0])


# bounds come from the row's support with the arguments substituted: Pareto(100, 3) -> (100, inf),
# Geometric(1/4) -> (1, inf), Weibull(3, 2) -> (0, inf), Binomial(10, 1/4) + Uniform(2, 6) -> (2, 16).
@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("Pareto(100, 3)", (100.0, math.inf)),
        ("Geometric(1/4)", (1.0, math.inf)),
        ("Weibull(3, 2)", (0.0, math.inf)),
        ("Binomial(10, 1/4) + Uniform(2, 6)", (2.0, 16.0)),
    ],
)
def test_bounds_read_the_row_support(raw: str, expected: tuple[float, float]) -> None:
    assert Expression(raw).bounds == expected


# is_integer_valued reads the row's whole-number flag: Geometric(1/4) and Binomial(10, 1/4) + Poisson(4) are
# whole, Weibull(3, 2) and Pareto(100, 3) are not.
@pytest.mark.parametrize(
    ("raw", "expected"),
    [("Geometric(1/4)", True), ("Binomial(10, 1/4) + Poisson(4)", True), ("Weibull(3, 2)", False), ("Pareto(100, 3)", False)],
)
def test_integrality_reads_the_row(raw: str, expected: bool) -> None:
    assert Expression(raw).is_integer_valued is expected


# A range on a bare whole-number draw is decided from the row's support. Binomial(10, 1/2) in [0, 10] constructs
# (it used to raise AttributeError), Binomial(10, 1/2) in [0, 9] and Poisson(3) in [0, 100] are rejected, and
# Geometric(1/4) with a minimum of 1 constructs because its draws start at 1.
@pytest.mark.parametrize(
    ("raw", "minimum", "maximum", "accepted"),
    [
        ("Binomial(10, 1/2)", 0, 10, True),
        ("Binomial(10, 1/2)", 0, 9, False),
        ("Poisson(3)", 0, 100, False),
        ("Geometric(1/4)", 1, None, True),
        ("Poisson(3)", 1, None, False),
    ],
)
def test_range_validation_reads_the_row_support(raw: str, minimum: float, maximum: Optional[float], accepted: bool) -> None:
    if accepted:
        Expression(raw, minimum=minimum, maximum=maximum)
        return
    with pytest.raises(ValueError, match="outside the permitted range"):
        Expression(raw, minimum=minimum, maximum=maximum)


# clip rejects a support that never meets the range: Poisson(3) clipped to [-5, -1], Uniform(2, 6) clipped to
# [7, 9] and Poisson(3) clipped to [0.2, 0.8], which holds no whole number, raise. Poisson(3) clipped to [0, 2] constructs.
def test_clip_reads_the_row_support() -> None:
    for raw, minimum, maximum in [("Poisson(3)", -5, -1), ("Uniform(2, 6)", 7, 9), ("Poisson(3)", 0.2, 0.8)]:
        with pytest.raises(ValueError, match="never intersects"):
            Expression(raw, minimum=minimum, maximum=maximum, clip=True)
    Expression("Poisson(3)", minimum=0, maximum=2, clip=True)


# mean() is exact where the expression is linear in its draws, from the rows' means: a constant, a bare draw,
# a shifted and scaled draw, a sum of draws, and a coefficient of t. LogNormal(6, 0.5) -> exp(6.125).
@pytest.mark.parametrize(
    ("raw", "t", "expected"),
    [
        ("512", None, 512.0),
        ("Exponential(1/300)", None, 300.0),
        ("LogNormal(6, 0.5)", None, math.exp(6.125)),
        ("128 + 2*Poisson(64)", None, 256.0),
        ("Normal(512, 128) + Uniform(2, 6) - Gamma(2, 3)", None, 510.0),
        ("t*Normal(2, 1) + 5", 3.0, 11.0),
        ("5 + t/2", 10.0, 10.0),
    ],
)
def test_mean_of_a_linear_expression(raw: str, t: Optional[float], expected: float) -> None:
    assert Expression(raw).mean(t) == pytest.approx(expected)


# mean() is None where the mean is not a function of the rows' means: a cap, a rounding, a product of two
# draws, a square and a Piecewise mixture.
@pytest.mark.parametrize(
    "raw",
    [
        "Min(LogNormal(6, 0.5), 4096)",
        "floor(Normal(512, 128))",
        "Normal(1, 2)*Normal(3, 1)",
        "Normal(0, 1)**2",
        "Piecewise((Uniform(100, 200), Uniform(0, 1) < 0.75), (Uniform(1000, 2000), True))",
    ],
)
def test_mean_is_none_when_not_linear(raw: str) -> None:
    assert Expression(raw).mean() is None


# An infinite mean is reported as inf, not as a number: Pareto(1, 1) -> inf, 5 - Pareto(1, 1/2) -> -inf.
# The difference of two infinite means is undefined -> None. Pareto(100, 3) -> 150.
def test_mean_reports_an_infinite_mean() -> None:
    assert Expression("Pareto(1, 1)").mean() == math.inf
    assert Expression("5 - Pareto(1, 1/2)").mean() == -math.inf
    assert Expression("Pareto(1, 1) - Pareto(1, 1)").mean() is None
    assert Expression("Pareto(100, 3)").mean() == pytest.approx(150.0)


# mean() of a time-varying expression needs t, as sample() does: '5 + t/2' with no t raises.
def test_mean_requires_t_when_the_expression_uses_it() -> None:
    with pytest.raises(ValueError, match="requires \\['t'\\]"):
        Expression("5 + t/2").mean()


# Conditions that hold over the whole window prove, so nothing is raised: a rate of 5 + t/10 over [0, 60] stays in
# [5, 11], Uniform(t, 10) over [0, 5] keeps left < right, a constant use proves, and t**2 - 2*t + 2 >= 1 is
# proved from its exact range where interval bounds alone are too loose.
@pytest.mark.parametrize(
    ("raw", "duration"),
    [
        ("Exponential(5 + t/10)", 60.0),
        ("Uniform(t, 10)", 5.0),
        ("Normal(t, 1 + t)", 60.0),
        ("Exponential(5)", 60.0),
        ("Exponential(t**2 - 2*t + 2)", 10.0),
        ("Binomial(5, t/10)", 10.0),
        ("Geometric(exp(-t))", 10.0),
        ("Exponential(t + 1)", 0.0),
    ],
)
def test_parameter_conditions_prove_over_the_window(raw: str, duration: float) -> None:
    prove_parameter_conditions(parse_distribution_use(raw), duration)


# A condition that breaks inside the window is rejected with the parameter, the condition and a t where it is false:
#   Exponential(t) over [0, 10]       -> rate, 'rate > 0', t = 0
#   Normal(10, 2 - t) over [0, 5]     -> std, 'std > 0', t = 2 (where std reaches 0)
#   Uniform(t, 10) over [0, 10]       -> left and right, 'left < right', t = 10
#   Binomial(5, t/10) over [0, 20]    -> p, 'p <= 1', some t after 10
#   Binomial(5 + t, 1/2) over [0, 3]  -> n, 'floor(n) == n', some t where 5 + t is a fraction
#   Exponential(1/t) over [0, 10]     -> rate, 'rate is a finite real', t = 0
#   Gamma(2, t - 1) over [0, 10]      -> scale (the second condition), t = 0
#   Exponential(sqrt(t - 5) + 1)      -> rate, 'rate is a finite real', t = 0 (not a real number before t = 5)
@pytest.mark.parametrize(
    ("raw", "duration", "parameters", "condition", "t"),
    [
        ("Exponential(t)", 10.0, ("rate",), "rate > 0", 0.0),
        ("Normal(10, 2 - t)", 5.0, ("std",), "std > 0", 2.0),
        ("Uniform(t, 10)", 10.0, ("left", "right"), "left < right", 10.0),
        ("Binomial(5, t/10)", 20.0, ("p",), "p <= 1", None),
        ("Binomial(5 + t, 1/2)", 3.0, ("n",), "floor(n) == n", None),
        ("Exponential(1/t)", 10.0, ("rate",), "rate is a finite real", 0.0),
        ("Gamma(2, t - 1)", 10.0, ("scale",), "scale > 0", 0.0),
        ("Exponential(sqrt(t - 5) + 1)", 10.0, ("rate",), "rate is a finite real", 0.0),
    ],
)
def test_parameter_condition_fails_inside_the_window(
    raw: str, duration: float, parameters: tuple[str, ...], condition: str, t: Optional[float]
) -> None:
    use = parse_distribution_use(raw)
    with pytest.raises(ParameterConditionError) as error:
        prove_parameter_conditions(use, duration)
    assert (error.value.parameters, error.value.condition) == (parameters, condition)
    assert error.value.t is not None and 0 <= error.value.t <= duration
    if t is not None:
        assert error.value.t == t
    assert f"fails at t = {error.value.t:g} over [0, {duration:g}]" in str(error.value)
    # the reported t is a real counterexample: the arguments at that t do not make a valid constant use
    with pytest.raises((ValueError, TypeError)):
        use.row.use([arg.subs(sympy.Symbol("t"), sympy.Float(error.value.t)) for arg in use.args]).require_constant()


# "Cannot be proved" is a rejection, with t = None. 2 + floor(t) - t stays in (1, 2] but neither the interval
# bounds nor the exact range can show it, a Piecewise rate has no interval rule, and 1/(t - 1/3)**2 blows up
# between grid points. None of the three passes.
@pytest.mark.parametrize(
    ("raw", "condition"),
    [
        ("Exponential(2 + floor(t) - t)", "rate > 0"),
        ("Exponential(Piecewise((1, t < 3), (2, True)))", "rate is a finite real"),
        ("Exponential(1/(t - 1/3)**2)", "rate is a finite real"),
    ],
)
def test_unprovable_parameter_condition_is_rejected(raw: str, condition: str) -> None:
    with pytest.raises(ParameterConditionError) as error:
        prove_parameter_conditions(parse_distribution_use(raw), 10.0)
    assert (error.value.t, error.value.condition) == (None, condition)
    assert "could not be proved over [0, 10]" in str(error.value)


# The window matters: Normal(10, 2 - t) proves over [0, 1.5] and fails over [0, 5].
def test_parameter_condition_depends_on_the_window() -> None:
    use = parse_distribution_use("Normal(10, 2 - t)")
    prove_parameter_conditions(use, 1.5)
    with pytest.raises(ParameterConditionError):
        prove_parameter_conditions(use, 5.0)


# The window must be a finite length from 0: a duration of -1, inf or nan raises ValueError before any proof.
@pytest.mark.parametrize("duration", [-1.0, math.inf, math.nan])
def test_prove_rejects_a_bad_window(duration: float) -> None:
    with pytest.raises(ValueError, match="duration must be a finite non-negative number"):
        prove_parameter_conditions(parse_distribution_use("Exponential(5)"), duration)


# parse_distribution_use takes exactly one supported distribution call over t. An unlisted distribution, a call
# inside arithmetic, a plain number, an unknown symbol or function, a wrong argument count and bad syntax all raise.
@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("Rayleigh(t)", "Distribution 'Rayleigh' is not supported"),
        ("Normal(1, 2) + 1", "Could not parse distribution"),
        ("5 + t", "is not a single distribution call"),
        ("Exponential(x)", r"disallowed symbol\(s\) \['x'\]"),
        ("Exponential(foo(t))", r"unknown function\(s\): \['foo'\]"),
        ("Exponential(t, 2)", r"takes 1 argument\(s\), got 2"),
        ("Exponential(", "Could not parse distribution"),
    ],
)
def test_parse_distribution_use_rejects(raw: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        parse_distribution_use(raw)


# parse_distribution_use('Normal(10, 2 - t)') binds the row and the arguments in order: Normal, (10, 2 - t).
def test_parse_distribution_use_binds_arguments() -> None:
    use = parse_distribution_use("Normal(10, 2 - t)")
    assert use.row is DISTRIBUTIONS["Normal"]
    assert use.args == (sympy.Integer(10), 2 - sympy.Symbol("t"))
    assert str(use) == "Normal(10, 2 - t)"


# DiscreteUniform takes a list. "DiscreteUniform([128, 256, 512])" draws only those three values, is bounded
# 128 to 512 with mean 896/3 and is whole-number. Arithmetic carries through: 2*DiscreteUniform([1, 3]) + 1 -> 3 or 7.
def test_discrete_uniform_in_an_expression() -> None:
    picks = Expression("DiscreteUniform([128, 256, 512])")
    draws = np.asarray(picks.sample(rng=np.random.default_rng(0), size=2000))
    assert set(draws.tolist()) == {128.0, 256.0, 512.0}
    assert picks.bounds == (128.0, 512.0)
    assert picks.mean() == pytest.approx(896 / 3)
    assert picks.is_integer_valued
    scaled = Expression("2*DiscreteUniform([1, 3]) + 1")
    assert set(np.asarray(scaled.sample(rng=np.random.default_rng(0), size=2000)).tolist()) == {3.0, 7.0}
    assert not Expression("DiscreteUniform([0.5, 2])").is_integer_valued


# A range check reads the listed values, not the span between them: [128, 256, 512] with minimum=200 is rejected
# naming the values, and a list is refused when a value depends on t, both in an expression and in the proof.
def test_discrete_uniform_range_and_time() -> None:
    with pytest.raises(ValueError, match=r"can evaluate to \{128, 256, 512\}"):
        Expression("DiscreteUniform([128, 256, 512])", minimum=200)
    Expression("DiscreteUniform([128, 256, 512])", minimum=128, maximum=512)
    with pytest.raises(ValueError, match=r"values\[0\] must be a finite constant, got t"):
        Expression("DiscreteUniform([t, 2])")
    prove_parameter_conditions(parse_distribution_use("DiscreteUniform([1, 2.5])"), 10)
    with pytest.raises(ValueError, match=r"values\[0\] must be a finite constant, got t"):
        prove_parameter_conditions(parse_distribution_use("DiscreteUniform([t, 2])"), 10)


# DiscreteUniform([128.0, 256]) is integer-valued even with a float literal, while
# DiscreteUniform([128, 128.0]) is rejected as a repeat rather than double-weighting 128.
def test_discrete_uniform_float_equivalence_in_expression() -> None:
    assert Expression("DiscreteUniform([128.0, 256])").is_integer_valued is True
    with pytest.raises(ValueError, match="each value may appear once"):
        Expression("DiscreteUniform([128, 128.0])")
