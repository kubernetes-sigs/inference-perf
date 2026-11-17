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
import numpy as np
import pytest

from inference_perf.utils.numeric.expression import Expression


class TestConstant:
    def test_int(self) -> None:
        expr = Expression(10)
        assert expr.is_constant
        assert not expr.is_random
        assert expr.sample() == 10.0

    def test_float_string(self) -> None:
        assert Expression("3.5").sample() == 3.5

    def test_t_does_not_affect_constant(self) -> None:
        assert Expression("10").sample(t=100) == 10.0

    def test_size_broadcasts(self) -> None:
        result = Expression("7").sample(size=4)
        assert isinstance(result, np.ndarray)
        assert result.tolist() == [7.0, 7.0, 7.0, 7.0]


class TestTimeVarying:
    def test_linear(self) -> None:
        expr = Expression("10 + t")
        assert expr.free_symbols == {"t"}
        assert expr.sample(t=0) == 10.0
        assert expr.sample(t=5) == 15.0

    def test_division(self) -> None:
        assert Expression("t / 2").sample(t=10) == 5.0

    def test_oscillating(self) -> None:
        expr = Expression("10 + 5*sin(2*pi*t/60)")
        assert expr.sample(t=0) == pytest.approx(10.0)
        assert expr.sample(t=15) == pytest.approx(15.0)

    def test_requires_t_when_missing(self) -> None:
        with pytest.raises(ValueError):
            Expression("10 + t").sample()


class TestVariablePermissions:
    def test_t_disallowed(self) -> None:
        with pytest.raises(ValueError):
            Expression("10 + t", allow_time=False)

    def test_t_allowed_by_default(self) -> None:
        assert Expression("10 + t").sample(t=1) == 11.0

    def test_only_t_is_a_valid_variable(self) -> None:
        with pytest.raises(ValueError):
            Expression("10 + x")

    def test_invalid_bareword_is_disallowed_symbol(self) -> None:
        with pytest.raises(ValueError):
            Expression("invalid_expr")

    def test_unknown_function_rejected(self) -> None:
        with pytest.raises(ValueError):
            Expression("InvalidDist(10)")

    def test_unparseable(self) -> None:
        with pytest.raises((ValueError, TypeError)):
            Expression("10 +* 2")

    # '1, 2' parses to a Python tuple and '[1]' to a list. Both are rejected as not a numeric value,
    # where they used to fail with AttributeError: 'tuple' object has no attribute 'atoms'.
    @pytest.mark.parametrize("raw", ["1, 2", "[1]"])
    def test_tuple_or_list_rejected(self, raw: str) -> None:
        with pytest.raises(ValueError, match="is not a numeric value"):
            Expression(raw)


class TestRandom:
    def test_normal_returns_float(self) -> None:
        expr = Expression("Normal(512, 200)")
        assert expr.is_random
        assert isinstance(expr.sample(), float)

    def test_uniform_in_support(self) -> None:
        val = Expression("Uniform(10, 50)").sample(rng=np.random.default_rng(0))
        assert 10 <= val <= 50

    def test_poisson(self) -> None:
        assert isinstance(Expression("Poisson(10)").sample(), float)

    def test_random_disallowed(self) -> None:
        with pytest.raises(ValueError):
            Expression("Normal(10, 2)", allow_random=False)

    def test_size_returns_array(self) -> None:
        result = Expression("Uniform(10, 50)").sample(size=100, rng=np.random.default_rng(1))
        assert isinstance(result, np.ndarray)
        assert result.shape == (100,)
        assert result.min() >= 10 and result.max() <= 50

    def test_reproducible_with_seed(self) -> None:
        a = Expression("Normal(10, 2)").sample(size=50, rng=np.random.default_rng(42))
        b = Expression("Normal(10, 2)").sample(size=50, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(a, b)

    def test_mixed_time_and_random(self) -> None:
        # 10 + Normal(0, 1) shifted by t; just needs to produce a float.
        expr = Expression("10 + t + Normal(0, 1)")
        assert expr.is_random
        assert isinstance(expr.sample(t=5, rng=np.random.default_rng(0)), float)

    def test_distribution_beyond_legacy_fast_path(self) -> None:
        # Rayleigh was never in the hand-written sampler table; it is served by
        # sympy's own numpy dispatcher on the fast path (not the slow fallback)
        # and its draws stay within the [0, oo) support.
        expr = Expression("Rayleigh(2)")
        assert expr.is_random
        assert not expr._fallback
        result = expr.sample(size=200, rng=np.random.default_rng(7))
        assert isinstance(result, np.ndarray)
        assert result.min() >= 0


class TestRangeValidation:
    def test_constant_out_of_range_rejected(self) -> None:
        with pytest.raises(ValueError):
            Expression("-5", minimum=0)

    def test_constant_within_range_ok(self) -> None:
        assert Expression("5", minimum=0, maximum=10).sample() == 5.0

    def test_time_varying_provably_out_of_range_rejected(self) -> None:
        # 10 + 5*sin(...) dips to 5 over the domain; min=8 is violated.
        with pytest.raises(ValueError):
            Expression("10 + 5*sin(2*pi*t/60)", minimum=8, duration=120)

    def test_time_varying_provably_in_range_ok(self) -> None:
        # Range over [0, 120] is [5, 15]; bounds [0, 20] contain it.
        expr = Expression("10 + 5*sin(2*pi*t/60)", minimum=0, maximum=20, duration=120)
        assert expr.sample(t=0) == pytest.approx(10.0)

    def test_unbounded_support_random_rejected(self) -> None:
        # Normal has support (-oo, oo); a non-negative rate is impossible to guarantee.
        with pytest.raises(ValueError):
            Expression("Normal(10, 2)", minimum=0)

    def test_bounded_support_random_ok(self) -> None:
        # Uniform(10, 50) support is within [0, 100].
        expr = Expression("Uniform(10, 50)", minimum=0, maximum=100)
        assert 10 <= expr.sample(rng=np.random.default_rng(0)) <= 50

    def test_deterministic_undecided_raises_at_sample_time(self) -> None:
        # No duration given, so the t range can't be proven up front; the
        # negative value surfaces when sampled.
        expr = Expression("10 - t", minimum=0)
        assert expr.sample(t=5) == 5.0
        with pytest.raises(ValueError):
            expr.sample(t=20)

    def test_min_greater_than_max_rejected(self) -> None:
        with pytest.raises(ValueError):
            Expression("5", minimum=10, maximum=0)


class TestClipPolicy:
    """Bounds assert by default; ``clip=True`` opts random draws into truncation."""

    def test_clip_clamps_bare_rv_with_unbounded_support(self) -> None:
        # Without clip this exact expression is rejected at construction
        # (see test_unbounded_support_random_rejected).
        expr = Expression("Normal(512, 200)", minimum=0, clip=True)
        result = expr.sample(size=500, rng=np.random.default_rng(0))
        assert isinstance(result, np.ndarray)
        assert result.min() >= 0

    def test_clip_clamps_transformed_rv(self) -> None:
        expr = Expression("100 + 50*Normal(0, 1)", minimum=0, maximum=120, clip=True)
        result = expr.sample(size=500, rng=np.random.default_rng(3))
        assert isinstance(result, np.ndarray)
        assert result.min() >= 0
        assert result.max() <= 120

    def test_random_undecided_without_clip_raises_on_out_of_range_draw(self) -> None:
        # Transformed randomness is statically undecidable, so construction
        # succeeds; every draw lands in [10, 11] and violates maximum=5.
        expr = Expression("Uniform(0, 1) + 10", maximum=5)
        with pytest.raises(ValueError, match="outside the permitted range .* Keep it in range"):
            expr.sample(size=10, rng=np.random.default_rng(0))

    def test_random_undecided_without_clip_in_range_ok(self) -> None:
        expr = Expression("Uniform(0, 1) + 10", minimum=0, maximum=20)
        val = expr.sample(rng=np.random.default_rng(0))
        assert 10 <= val <= 11

    def test_clip_rejects_provably_disjoint_support(self) -> None:
        # Uniform(10, 20) never intersects (-oo, 5]; clipping would collapse
        # every draw to the bound, which is a config error even with clip.
        with pytest.raises(ValueError):
            Expression("Uniform(10, 20)", maximum=5, clip=True)

    def test_clip_requires_bounds(self) -> None:
        with pytest.raises(ValueError):
            Expression("Normal(0, 1)", clip=True)

    def test_clip_never_clamps_deterministic_values(self) -> None:
        # A deterministic out-of-range value is a config error regardless of
        # clip: provable violations raise at construction, time-varying ones
        # at sample time.
        with pytest.raises(ValueError):
            Expression("-5", minimum=0, clip=True)
        expr = Expression("10 - t", minimum=0, clip=True)
        assert expr.sample(t=5) == 5.0
        with pytest.raises(ValueError):
            expr.sample(t=20)


class TestSympyInternalsFence:
    """Fail loudly if a sympy upgrade breaks the internals the fast path leans on.

    Both fences guard silent degradation: if sympy moves ``do_sample_numpy``
    or reshapes its distribution constructors, Expression would still work but
    every random draw would take the slow symbolic path.
    """

    def test_numpy_fast_path_compiled(self) -> None:
        from inference_perf.utils.numeric.expression import expression as mod

        # getattr rather than attribute access: do_sample_numpy is an import,
        # not a definition, so mypy --strict flags cross-module access to it.
        assert getattr(mod, "do_sample_numpy", None) is not None
        expr = Expression("Normal(0, 1) + Uniform(0, 1) + t")
        assert not expr._fallback
        assert expr._lambdified is not None

    def test_distribution_constructors_discovered(self) -> None:
        from inference_perf.utils.numeric.expression import expression as mod

        for name in ("Normal", "Uniform", "Poisson", "Exponential", "LogNormal", "Gamma"):
            assert name in mod._DISTRIBUTION_CTORS, f"sympy.stats constructor discovery lost {name}"


# evaluate(): vectorized evaluation of a deterministic expression over an array
# of times. Inputs are a constant, a function of t, and a random expression;
# expected: values match sample() at each t, constants broadcast to the input
# shape, random expressions are refused, and an unprovable bound still raises.
class TestEvaluate:
    def test_matches_sample_at_each_time(self) -> None:
        expr = Expression("10 + 5*sin(t)")
        times = np.linspace(0.0, 60.0, 7)
        expected = np.array([float(expr.sample(t=float(x))) for x in times])
        assert np.allclose(expr.evaluate(times), expected)

    def test_constant_broadcasts_to_input_shape(self) -> None:
        values = Expression("4").evaluate(np.zeros(5))
        assert values.shape == (5,)
        assert np.all(values == 4.0)

    def test_scalar_time_returns_zero_dim_array(self) -> None:
        assert float(Expression("2*t").evaluate(3.0)) == 6.0

    def test_random_expression_rejected(self) -> None:
        with pytest.raises(ValueError, match="random"):
            Expression("Normal(0, 1)").evaluate(np.zeros(3))

    def test_unprovable_bound_checked_over_the_array(self) -> None:
        # No duration, so the range of 10 - t can't be proven at construction
        # and the minimum of 0 is checked per value: t=20 gives -10.
        expr = Expression("10 - t", minimum=0)
        with pytest.raises(ValueError, match="t=20"):
            expr.evaluate(np.array([0.0, 5.0, 20.0]))


# A bare 't' (and '1*t', which sympy folds back to 't') is a numeric value, not a
# condition, even though sympy's Symbol subclasses Boolean. Expected: both construct,
# and evaluate at t=3 gives 3.
@pytest.mark.parametrize("raw", ["t", "1*t"])
def test_bare_time_symbol_is_numeric(raw: str) -> None:
    assert float(Expression(raw).evaluate(3.0)) == 3.0


# bounds: provable (lower, upper) over every value. Inputs and expected bounds:
#   Min(LogNormal(6, 0.5), 4096) -> (0, 4096)     a capped long tail
#   Min(2*Uniform(10, 20), 30)   -> (20, 30)      arithmetic inside Min
#   Uniform(10, 20) + Uniform(0, 5) -> (10, 25)   sums of supports
#   128 + Poisson(64)            -> (128, inf)    unbounded above
#   Abs(Normal(0, 1))            -> (0, inf)      the skew_normal building block
#   Beta(2, 5)                   -> (0, 1)
#   5 + t/2 with duration 60     -> (5, 35);  without a duration -> (5, inf)
#   512                          -> (512, 512)
@pytest.mark.parametrize(
    ("raw", "duration", "expected"),
    [
        ("Min(LogNormal(6, 0.5), 4096)", None, (0.0, 4096.0)),
        ("Min(2*Uniform(10, 20), 30)", None, (20.0, 30.0)),
        ("Uniform(10, 20) + Uniform(0, 5)", None, (10.0, 25.0)),
        ("128 + Poisson(64)", None, (128.0, float("inf"))),
        ("Abs(Normal(0, 1))", None, (0.0, float("inf"))),
        ("Beta(2, 5)", None, (0.0, 1.0)),
        ("5 + t/2", 60.0, (5.0, 35.0)),
        ("5 + t/2", None, (5.0, float("inf"))),
        ("512", None, (512.0, 512.0)),
    ],
)
def test_bounds(raw: str, duration: float | None, expected: tuple[float, float]) -> None:
    assert Expression(raw, duration=duration).bounds == expected


# bounds is None when some node has no interval rule: a Piecewise over t. Expected: None, not a guess.
def test_bounds_undecidable() -> None:
    assert Expression("Piecewise((2, t < 30), (20, True))", duration=60).bounds is None


# bounds is sound: over 20000 draws of each random expression, every sample lies inside the proven bounds.
@pytest.mark.parametrize("raw", ["Min(LogNormal(6, 0.5), 4096)", "Min(2*Uniform(10, 20), 30)", "Abs(Normal(0, 1)) - 3"])
def test_bounds_contain_samples(raw: str) -> None:
    expr = Expression(raw)
    assert expr.bounds is not None
    lo, hi = expr.bounds
    samples = np.asarray(expr.sample(rng=np.random.default_rng(0), size=20000))
    assert lo <= samples.min() and samples.max() <= hi


# t inside a distribution's parameters counts as t: "Exponential(5 + t/10)" varies with time and lists t
# among its symbols, "Exponential(5)" does not, and a field that does not allow t rejects it there too.
def test_time_in_a_distribution_parameter_is_seen() -> None:
    assert Expression("Exponential(5 + t/10)").varies_with_time
    assert Expression("Exponential(5 + t/10)").free_symbols == {"t"}
    assert not Expression("Exponential(5)").varies_with_time
    assert Expression("t + Exponential(5)").varies_with_time
    with pytest.raises(ValueError, match="disallowed symbol"):
        Expression("Normal(t, 1)", allow_time=False)


# A draw uses the parameters as they stand at the given t: 2000 seeded draws of "Exponential(10 + t)"
# average 0.1 at t=0 and 0.01 at t=90 (within 10%). Drawing without a t is an error.
def test_draw_uses_the_parameters_at_t() -> None:
    expression = Expression("Exponential(10 + t)")
    for t, mean in ((0.0, 0.1), (90.0, 0.01)):
        draws = np.asarray(expression.sample(t, rng=np.random.default_rng(0), size=2000))
        assert float(draws.mean()) == pytest.approx(mean, rel=0.1)
    with pytest.raises(ValueError, match="requires \\['t'\\]"):
        expression.sample(rng=np.random.default_rng(0))


# A support that moves with t is bounded over the window: "Uniform(t, t + 1)" over 60s lies in [0, 61],
# and "0.05 + Exponential(20 + t)" is at least 0.05.
def test_bounds_of_a_support_that_moves_with_time() -> None:
    assert Expression("Uniform(t, t + 1)", duration=60).bounds == (0.0, 61.0)
    assert Expression("0.05 + Exponential(20 + t)", duration=60).bounds == (0.05, float("inf"))


# Random expressions over t that can be drawn from at every t in a 60s window pass the check, and so
# do expressions the check does not apply to. A LogNormal with no t passes although sympy cannot
# compute its mean, and a gap with t and no distribution passes although its bounds are undecidable.
@pytest.mark.parametrize(
    "raw",
    [
        "Exponential(5 + t/10)",
        "Exponential(0.1 + t)",
        "Gamma(2, 0.1 + t)",
        "Uniform(t, t + 1)",
        "0.05 + Exponential(20 + t)",
        "Max(0, Normal(1 + t, 0.1))",
        "Exponential(10)",
        "LogNormal(-2.55, 0.7)",
        "1/t",
        "Piecewise((oo, t < 5), (0.1, True))",
    ],
)
def test_valid_draws_over_time_are_accepted(raw: str) -> None:
    Expression(raw, minimum=0, duration=60).require_valid_draws()


# Over a 60s window each of these is rejected, naming the parameter and a t: a rate of 0 at t=0, a rate
# that goes negative just after t=10, a rate with a pole at t=5 that cannot be shown finite, a
# distribution with no mean, and a gap that cannot be shown to stay at or above 0.
@pytest.mark.parametrize(
    ("raw", "message"),
    [
        ("Exponential(t)", "Exponential is not valid at t=0, where rate = 0: Rate must be positive"),
        ("Exponential(10 - t)", "Exponential is not valid at t=10.02, where rate = -0.02: Rate must be positive"),
        (
            "Exponential(1/(t - 5))",
            "the rate of Exponential, 1/\\(t - 5\\), cannot be shown to be finite for every t from 0 to 60",
        ),
        ("Exponential(1/t)", "the rate of Exponential, 1/t, is not finite at t=0"),
        ("Abs(Cauchy(1 + t, 1))", "the mean of Cauchy cannot be computed at t=0"),
        ("Pareto(1 + t, 1)", "the mean of Pareto"),
        ("Exponential(5 + t) - 0.01", "cannot be shown to stay inside the permitted range"),
    ],
)
def test_invalid_draws_over_time_are_rejected(raw: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        Expression(raw, minimum=0, duration=60).require_valid_draws()


# The check needs a window: a random expression over t built with no duration is refused.
def test_valid_draws_need_a_duration() -> None:
    with pytest.raises(ValueError, match="needs a duration"):
        Expression("Exponential(5 + t)", minimum=0).require_valid_draws()
