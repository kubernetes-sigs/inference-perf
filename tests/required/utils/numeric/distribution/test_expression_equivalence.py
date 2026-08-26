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
"""Every Distribution config has a corresponding Expression.

``distribution_to_expression`` is the compiler the Distribution samplers now
run on. These tests prove the mapping is total (every DistributionType member
compiles to a string Expression accepts) and that each compiled expression
draws from the distribution the legacy numpy sampler drew from, checked
against closed-form moments. Same-seed draws match the legacy sampler
statistically, not bit-for-bit: the draw streams changed when sampling moved
to Expression.
"""

from math import pi, sqrt

import numpy as np
import pytest

from inference_perf.config import Distribution, DistributionType
from inference_perf.utils.numeric.distribution import distribution_to_expression, sample_from_distribution
from inference_perf.utils.numeric.expression import Expression


# One representative config per DistributionType, parameterised so each type's
# distinguishing parameter matters (skew for skew_normal, mean-as-lambda for
# poisson, ...). Bounds are wide so moment checks below see the distribution,
# not the clip tail.
_CONFIGS = {
    DistributionType.NORMAL: Distribution(type=DistributionType.NORMAL, mean=500.0, min=0, max=10000, std_dev=100.0),
    DistributionType.SKEW_NORMAL: Distribution(
        type=DistributionType.SKEW_NORMAL, mean=500.0, min=-10000, max=10000, std_dev=100.0, skew=4.0
    ),
    DistributionType.LOGNORMAL: Distribution(type=DistributionType.LOGNORMAL, mean=150.0, min=0, max=100000, std_dev=60.0),
    DistributionType.UNIFORM: Distribution(type=DistributionType.UNIFORM, mean=0.0, min=10, max=100),
    DistributionType.POISSON: Distribution(type=DistributionType.POISSON, mean=10.0, min=0, max=1000),
    DistributionType.FIXED: Distribution(type=DistributionType.FIXED, mean=99.9, min=0, max=1000),
}


class TestEveryDistributionTypeCompiles:
    # The compiler must cover the whole enum: adding a DistributionType member
    # without a mapping fails here, not at first use.
    def test_config_table_covers_every_type(self) -> None:
        assert set(_CONFIGS) == set(DistributionType)

    # Each type's config compiles to a string that Expression parses with
    # randomness allowed and time disallowed (a length can never depend on t).
    @pytest.mark.parametrize("dist_type", list(DistributionType), ids=lambda d: d.value)
    def test_compiles_to_valid_expression(self, dist_type: DistributionType) -> None:
        raw = distribution_to_expression(_CONFIGS[dist_type])
        expression = Expression(raw, allow_time=False)
        assert expression.free_symbols == set()

    # Sampling through the public API works for every type: 100 draws from
    # each config come back as 100 integers.
    @pytest.mark.parametrize("dist_type", list(DistributionType), ids=lambda d: d.value)
    def test_samples_through_public_api(self, dist_type: DistributionType) -> None:
        result = sample_from_distribution(_CONFIGS[dist_type], 100, rng=np.random.default_rng(0))
        assert len(result) == 100
        assert np.issubdtype(result.dtype, np.integer)

    # Same seed, same array, for every type: the compiled expressions stay
    # reproducible under a seeded Generator (the old functional prototype lost
    # this for skew_normal).
    @pytest.mark.parametrize("dist_type", list(DistributionType), ids=lambda d: d.value)
    def test_reproducible_with_same_seed(self, dist_type: DistributionType) -> None:
        a = sample_from_distribution(_CONFIGS[dist_type], 200, rng=np.random.default_rng(11))
        b = sample_from_distribution(_CONFIGS[dist_type], 200, rng=np.random.default_rng(11))
        np.testing.assert_array_equal(a, b)


class TestCompiledMomentsMatchLegacySampler:
    # Normal(500, 100): 20000 draws should have mean ~500 and std ~100.
    def test_normal(self) -> None:
        result = sample_from_distribution(_CONFIGS[DistributionType.NORMAL], 20000, rng=np.random.default_rng(1))
        assert abs(float(result.mean()) - 500.0) < 3.0
        assert abs(float(result.std()) - 100.0) < 3.0

    # Azzalini skew-normal with location 500, scale 100, shape a=4:
    # delta = a/sqrt(1+a^2), E[X] = loc + scale*delta*sqrt(2/pi) ~ 577.4 and
    # Var[X] = scale^2*(1 - 2*delta^2/pi) ~ 100^2*0.4 — matching the legacy
    # two-normal construction, whose "mean" was the location parameter.
    def test_skew_normal(self) -> None:
        delta = 4.0 / sqrt(1.0 + 16.0)
        expected_mean = 500.0 + 100.0 * delta * sqrt(2.0 / pi)
        expected_std = 100.0 * sqrt(1.0 - 2.0 * delta**2 / pi)
        result = sample_from_distribution(_CONFIGS[DistributionType.SKEW_NORMAL], 20000, rng=np.random.default_rng(2))
        assert abs(float(result.mean()) - expected_mean) < 3.0
        assert abs(float(result.std()) - expected_std) < 3.0
        # Positive skew puts the median below the mean.
        assert float(np.median(result)) < float(result.mean())

    # Moment-matched lognormal: mean/std_dev describe the lognormal itself,
    # so 20000 draws from LogNormal(mean=150, std_dev=60) should have sample
    # mean ~150 and sample std ~60.
    def test_lognormal(self) -> None:
        result = sample_from_distribution(_CONFIGS[DistributionType.LOGNORMAL], 20000, rng=np.random.default_rng(3))
        assert abs(float(result.mean()) - 150.0) < 3.0
        assert abs(float(result.std()) - 60.0) < 3.0

    # Uniform over [10, 100]: draws come from [min, max+1) then clip, so both
    # endpoints appear and the mean sits near (10 + 101)/2 = 55.5.
    def test_uniform(self) -> None:
        result = sample_from_distribution(_CONFIGS[DistributionType.UNIFORM], 20000, rng=np.random.default_rng(4))
        values = set(result.tolist())
        assert 10 in values
        assert 100 in values
        assert abs(float(result.mean()) - 55.5) < 1.0

    # Poisson(10): sample mean and variance both ~10.
    def test_poisson(self) -> None:
        result = sample_from_distribution(_CONFIGS[DistributionType.POISSON], 20000, rng=np.random.default_rng(5))
        assert abs(float(result.mean()) - 10.0) < 0.3
        assert abs(float(result.var()) - 10.0) < 0.5

    # Fixed with mean=99.9 returns the truncated constant 99 for every draw.
    def test_fixed(self) -> None:
        result = sample_from_distribution(_CONFIGS[DistributionType.FIXED], 100, rng=np.random.default_rng(6))
        assert all(v == 99 for v in result)


class TestRepeatedVariablesAreIndependent:
    # The skew-normal compilation only works if its two Normal(0, 1) draws are
    # independent. Var(X + Y) = 2 for independent standard normals; identical
    # (perfectly correlated) streams — the old functional prototype's seeding
    # bug — would give Var(X + X) = 4.
    def test_sum_of_two_standard_normals_has_variance_two(self) -> None:
        expression = Expression("Normal(0, 1) + Normal(0, 1)", allow_time=False)
        draws = np.asarray(expression.sample(rng=np.random.default_rng(7), size=20000))
        assert abs(float(draws.var()) - 2.0) < 0.1


class TestDegenerateConfigs:
    # std_dev=0 normal collapses to the constant mean: every draw is
    # round(clip(250.4)) = 250.
    def test_normal_zero_std_dev_is_constant(self) -> None:
        config = Distribution(type=DistributionType.NORMAL, mean=250.4, min=0, max=1000, std_dev=0.0)
        result = sample_from_distribution(config, 50, rng=np.random.default_rng(8))
        assert all(v == 250 for v in result)

    # std_dev=0 skew_normal also collapses to the constant mean; the skew
    # parameter has nothing to act on.
    def test_skew_normal_zero_std_dev_is_constant(self) -> None:
        config = Distribution(type=DistributionType.SKEW_NORMAL, mean=250.4, min=0, max=1000, std_dev=0.0, skew=5.0)
        result = sample_from_distribution(config, 50, rng=np.random.default_rng(9))
        assert all(v == 250 for v in result)

    # skew=0 skew_normal degenerates to a plain normal: mean ~500, std ~100
    # over 20000 draws.
    def test_skew_normal_zero_skew_is_plain_normal(self) -> None:
        config = Distribution(type=DistributionType.SKEW_NORMAL, mean=500.0, min=0, max=10000, std_dev=100.0, skew=0.0)
        result = sample_from_distribution(config, 20000, rng=np.random.default_rng(10))
        assert abs(float(result.mean()) - 500.0) < 3.0
        assert abs(float(result.std()) - 100.0) < 3.0
