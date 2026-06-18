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

    def test_random_undecided_support_is_clamped(self) -> None:
        # Mixed random support is undecidable statically, so values are clamped.
        expr = Expression("100 + 50*Normal(0, 1)", minimum=0, maximum=120)
        result = expr.sample(size=500, rng=np.random.default_rng(3))
        assert isinstance(result, np.ndarray)
        assert result.min() >= 0
        assert result.max() <= 120

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
