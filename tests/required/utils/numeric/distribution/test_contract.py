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
"""Characterization tests for the Distribution sampling API.

``utils.numeric.expression`` is intended to subsume this API. These tests pin
the current behavioral contract, especially where it differs from Expression
semantics (silent clipping vs assert-by-default bounds, integer rounding,
parameters that are reinterpreted or ignored per distribution type), so the
migration changes each behavior knowingly rather than accidentally.
"""

import numpy as np
import pytest

from inference_perf.config import Distribution, DistributionType
from inference_perf.utils.numeric.distribution import generate_distribution, sample_from_distribution


class TestSampleFromDistributionContract:
    def test_out_of_range_mass_silently_clips_to_bounds(self) -> None:
        # Draws beyond [min, max] are clipped, piling probability mass onto the
        # exact bound values. Expression makes this an explicit clip=True opt-in.
        config = Distribution(type=DistributionType.NORMAL, mean=500.0, min=400, max=600, std_dev=300.0)
        result = sample_from_distribution(config, 2000, rng=np.random.default_rng(0))
        assert 400 in result
        assert 600 in result

    def test_mean_outside_bounds_accepted_and_collapses_to_bound(self) -> None:
        # No mean-within-bounds validation: every draw clips to the near bound.
        # (Contrast generate_distribution, which rejects this for "normal".)
        config = Distribution(type=DistributionType.NORMAL, mean=1000.0, min=0, max=100, std_dev=10.0)
        result = sample_from_distribution(config, 100, rng=np.random.default_rng(0))
        assert all(v == 100 for v in result)

    def test_always_returns_rounded_integers(self) -> None:
        # Output is always rounded to an integer dtype; Expression returns floats.
        config = Distribution(type=DistributionType.NORMAL, mean=100.5, min=0, max=200, std_dev=10.0)
        result = sample_from_distribution(config, 100, rng=np.random.default_rng(0))
        assert np.issubdtype(result.dtype, np.integer)

    def test_fixed_truncates_mean_instead_of_rounding(self) -> None:
        config = Distribution(type=DistributionType.FIXED, mean=99.9, min=0, max=200)
        result = sample_from_distribution(config, 10, rng=np.random.default_rng(0))
        assert all(v == 99 for v in result)

    def test_uniform_ignores_mean_and_std_dev(self) -> None:
        a = Distribution(type=DistributionType.UNIFORM, mean=20.0, min=1, max=10, std_dev=5.0)
        b = Distribution(type=DistributionType.UNIFORM, mean=80.0, min=1, max=10, std_dev=50.0)
        result_a = sample_from_distribution(a, 100, rng=np.random.default_rng(7))
        result_b = sample_from_distribution(b, 100, rng=np.random.default_rng(7))
        np.testing.assert_array_equal(result_a, result_b)

    def test_uniform_reaches_both_endpoints(self) -> None:
        # Draws come from [min, max + 1) then clip, so max is fully reachable
        # rather than a half-weight rounding edge.
        config = Distribution(type=DistributionType.UNIFORM, mean=5.0, min=1, max=10)
        values = set(sample_from_distribution(config, 5000, rng=np.random.default_rng(1)).tolist())
        assert 1 in values
        assert 10 in values

    def test_poisson_nonpositive_mean_falls_back_to_lambda_one(self) -> None:
        config = Distribution(type=DistributionType.POISSON, mean=0.0, min=0, max=50, std_dev=0.0)
        result = sample_from_distribution(config, 5000, rng=np.random.default_rng(2))
        assert abs(float(result.mean()) - 1.0) < 0.15

    def test_poisson_ignores_std_dev(self) -> None:
        a = Distribution(type=DistributionType.POISSON, mean=10.0, min=0, max=100, std_dev=1.0)
        b = Distribution(type=DistributionType.POISSON, mean=10.0, min=0, max=100, std_dev=99.0)
        result_a = sample_from_distribution(a, 100, rng=np.random.default_rng(3))
        result_b = sample_from_distribution(b, 100, rng=np.random.default_rng(3))
        np.testing.assert_array_equal(result_a, result_b)

    def test_skew_normal_mean_param_is_location_not_mean(self) -> None:
        # The Azzalini construction treats "mean" as the location parameter, so
        # positive skew shifts the actual sample mean well above it.
        config = Distribution(type=DistributionType.SKEW_NORMAL, mean=100.0, min=-1000, max=1000, std_dev=50.0, skew=5.0)
        result = sample_from_distribution(config, 10000, rng=np.random.default_rng(4))
        assert float(result.mean()) > 110.0

    def test_lognormal_is_moment_matched(self) -> None:
        # mean/std_dev describe the lognormal itself, not the underlying normal.
        config = Distribution(type=DistributionType.LOGNORMAL, mean=150.0, min=1, max=100000, std_dev=60.0)
        result = sample_from_distribution(config, 20000, rng=np.random.default_rng(5))
        assert abs(float(result.mean()) - 150.0) < 5.0

    def test_unseeded_calls_are_not_reproducible(self) -> None:
        # rng=None means a fresh default generator per call; reproducible runs
        # must thread a seeded Generator. Expression keeps this contract.
        config = Distribution(type=DistributionType.NORMAL, mean=500.0, min=0, max=1000, std_dev=100.0)
        result_a = sample_from_distribution(config, 100)
        result_b = sample_from_distribution(config, 100)
        assert not np.array_equal(result_a, result_b)


class TestGenerateDistributionContract:
    def test_normal_mean_outside_bounds_rejected(self) -> None:
        # The legacy entry point validates mean against the bounds for "normal";
        # sample_from_distribution does not (see the collapse test above).
        with pytest.raises(ValueError, match="[Mm]ean"):
            generate_distribution(min=0, max=100, mean=1000, std_dev=10, total_count=10)

    def test_unknown_dist_type_rejected(self) -> None:
        with pytest.raises(ValueError, match="dist_type"):
            generate_distribution(min=0, max=100, mean=50, std_dev=10, total_count=10, dist_type="zipf")

    def test_lognormal_is_shifted_by_min(self) -> None:
        # The legacy lognormal moment-matches the distribution of (value - min),
        # then shifts by min; the floor is min, not zero.
        result = generate_distribution(
            min=100, max=100000, mean=150, std_dev=30, total_count=5000, rng=np.random.default_rng(6)
        )
        assert result.min() >= 100
        assert abs(float(result.mean()) - 150.0) < 5.0
