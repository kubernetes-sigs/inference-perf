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

from inference_perf.config import Distribution, DistributionType
from inference_perf.utils.distribution import generate_distribution, sample_from_distribution


class TestSampleFromDistribution:
    def test_normal_within_bounds(self) -> None:
        config = Distribution(type=DistributionType.NORMAL, mean=500.0, min=100, max=900, std_dev=100.0)
        result = sample_from_distribution(config, 1000, rng=np.random.default_rng(42))
        assert result.min() >= 100
        assert result.max() <= 900
        assert len(result) == 1000

    def test_normal_approximate_mean(self) -> None:
        config = Distribution(type=DistributionType.NORMAL, mean=200.0, min=10, max=2000, std_dev=50.0)
        result = sample_from_distribution(config, 10000, rng=np.random.default_rng(42))
        assert abs(result.mean() - 200.0) < 5.0

    def test_skew_normal_within_bounds(self) -> None:
        config = Distribution(
            type=DistributionType.SKEW_NORMAL, mean=200.0, min=10, max=2000, std_dev=80.0, skew=2.5
        )
        result = sample_from_distribution(config, 1000, rng=np.random.default_rng(42))
        assert result.min() >= 10
        assert result.max() <= 2000

    def test_skew_normal_positive_skew_has_right_tail(self) -> None:
        config = Distribution(
            type=DistributionType.SKEW_NORMAL, mean=100.0, min=0, max=1000, std_dev=50.0, skew=5.0
        )
        result = sample_from_distribution(config, 10000, rng=np.random.default_rng(42))
        # With positive skew, the median should be less than the mean of the samples
        median = float(np.median(result))
        mean = float(result.mean())
        assert median <= mean

    def test_skew_normal_zero_skew_approximates_normal(self) -> None:
        config_skew = Distribution(
            type=DistributionType.SKEW_NORMAL, mean=500.0, min=0, max=1000, std_dev=100.0, skew=0.0
        )
        config_normal = Distribution(type=DistributionType.NORMAL, mean=500.0, min=0, max=1000, std_dev=100.0)
        rng1 = np.random.default_rng(42)
        rng2 = np.random.default_rng(42)
        result_skew = sample_from_distribution(config_skew, 10000, rng=rng1)
        result_normal = sample_from_distribution(config_normal, 10000, rng=rng2)
        # Means should be close (both target 500)
        assert abs(float(result_skew.mean()) - float(result_normal.mean())) < 20.0

    def test_lognormal_within_bounds(self) -> None:
        config = Distribution(type=DistributionType.LOGNORMAL, mean=150.0, min=1, max=4096, std_dev=60.0)
        result = sample_from_distribution(config, 1000, rng=np.random.default_rng(42))
        assert result.min() >= 1
        assert result.max() <= 4096

    def test_lognormal_zero_std_dev_constant(self) -> None:
        config = Distribution(type=DistributionType.LOGNORMAL, mean=100.0, min=1, max=500, std_dev=0.0)
        result = sample_from_distribution(config, 100, rng=np.random.default_rng(42))
        assert all(v == 100 for v in result)

    def test_lognormal_negative_mean_raises(self) -> None:
        config = Distribution(type=DistributionType.LOGNORMAL, mean=-5.0, min=-10, max=100, std_dev=10.0)
        with pytest.raises(ValueError, match="mean > 0"):
            sample_from_distribution(config, 100)

    def test_uniform_within_bounds(self) -> None:
        config = Distribution(type=DistributionType.UNIFORM, mean=50.0, min=10, max=100)
        result = sample_from_distribution(config, 1000, rng=np.random.default_rng(42))
        assert result.min() >= 10
        assert result.max() <= 100

    def test_uniform_covers_range(self) -> None:
        config = Distribution(type=DistributionType.UNIFORM, mean=50.0, min=1, max=10)
        result = sample_from_distribution(config, 10000, rng=np.random.default_rng(42))
        unique_vals = set(result.tolist())
        # Should cover most of the [1, 10] range
        assert len(unique_vals) >= 9

    def test_poisson_within_bounds(self) -> None:
        config = Distribution(type=DistributionType.POISSON, mean=4.2, min=0, max=50)
        result = sample_from_distribution(config, 1000, rng=np.random.default_rng(42))
        assert result.min() >= 0
        assert result.max() <= 50

    def test_poisson_approximate_mean(self) -> None:
        config = Distribution(type=DistributionType.POISSON, mean=10.0, min=0, max=100)
        result = sample_from_distribution(config, 10000, rng=np.random.default_rng(42))
        assert abs(float(result.mean()) - 10.0) < 1.0

    def test_deterministic_seeding(self) -> None:
        config = Distribution(type=DistributionType.NORMAL, mean=500.0, min=100, max=900, std_dev=100.0)
        result1 = sample_from_distribution(config, 100, rng=np.random.default_rng(42))
        result2 = sample_from_distribution(config, 100, rng=np.random.default_rng(42))
        np.testing.assert_array_equal(result1, result2)

    def test_different_seeds_differ(self) -> None:
        config = Distribution(type=DistributionType.NORMAL, mean=500.0, min=100, max=900, std_dev=100.0)
        result1 = sample_from_distribution(config, 100, rng=np.random.default_rng(42))
        result2 = sample_from_distribution(config, 100, rng=np.random.default_rng(99))
        assert not np.array_equal(result1, result2)

    def test_fixed_value_std_dev_zero(self) -> None:
        config = Distribution(type=DistributionType.NORMAL, mean=50.0, min=50, max=50, std_dev=0.0)
        result = sample_from_distribution(config, 100, rng=np.random.default_rng(42))
        assert all(v == 50 for v in result)

    def test_invalid_count_zero(self) -> None:
        config = Distribution(type=DistributionType.NORMAL, mean=50.0, min=10, max=100, std_dev=10.0)
        with pytest.raises(ValueError, match="positive"):
            sample_from_distribution(config, 0)

    def test_invalid_min_greater_than_max(self) -> None:
        # Pydantic validator catches this at construction time
        with pytest.raises(Exception, match="min.*max"):
            Distribution(type=DistributionType.NORMAL, mean=50.0, min=100, max=10, std_dev=10.0)


class TestLegacyGenerateDistribution:
    """Ensure the original generate_distribution function still works unchanged."""

    def test_output_shape_and_bounds(self) -> None:
        result = generate_distribution(min=10, max=100, mean=50, std_dev=20, total_count=500)
        assert len(result) == 500
        assert result.min() >= 10
        assert result.max() <= 100

    def test_fixed_value(self) -> None:
        result = generate_distribution(min=42, max=42, mean=42, std_dev=0, total_count=100)
        assert all(v == 42 for v in result)
