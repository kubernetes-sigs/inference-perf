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
from inference_perf.datagen.multimodal_sampling import sample_insertion_point
from inference_perf.utils.numeric.distribution import sample_from_distribution


# insertion_point 'Uniform(0, 1)' sampled 100 times under seed 42 is not rounded: the draws keep their
# fractional parts and spread over (0, 1) rather than collapsing onto 0 and 1.
def test_uniform_insertion_preserves_continuous_range() -> None:
    rng = np.random.default_rng(42)
    actual = np.array([sample_insertion_point("Uniform(0, 1)", rng) for _ in range(100)])
    assert np.all((actual >= 0) & (actual <= 1))
    assert np.any(actual != np.round(actual))
    assert len(set(actual.tolist())) > 90


# An insertion_point expression that can leave [0, 1] (a wide normal, Uniform(-10, 10), a Poisson count,
# a lognormal, a skew-normal build) is clamped per draw: 100 samples each all land within [0, 1].
@pytest.mark.parametrize(
    "expression",
    [
        "Normal(2, 3)",
        "Uniform(-10, 10)",
        "Poisson(2)",
        "LogNormal(0.5, 0.8)",
        "2 + 3*(0.9*Abs(Normal(0, 1)) + 0.4*Normal(0, 1))",
    ],
)
def test_insertion_distribution_is_clamped_to_prompt(expression: str) -> None:
    rng = np.random.default_rng(42)
    assert all(0 <= sample_insertion_point(expression, rng) <= 1 for _ in range(100))


@pytest.mark.parametrize("kind", [DistributionType.FIXED, DistributionType.NORMAL, DistributionType.UNIFORM])
def test_count_sampling_keeps_integer_semantics(kind: DistributionType) -> None:
    config = Distribution(type=kind, min=0, max=4, mean=1.7, std_dev=0.5)
    actual = sample_from_distribution(config, 100, np.random.default_rng(42))
    rng = np.random.default_rng(42)
    if kind == DistributionType.FIXED:
        expected = np.full(100, 1, dtype=int)
    elif kind == DistributionType.NORMAL:
        expected = np.round(np.clip(rng.normal(1.7, 0.5, 100), 0, 4)).astype(int)
    else:
        expected = np.round(np.clip(rng.uniform(0, 5, 100), 0, 4)).astype(int)
    assert np.issubdtype(actual.dtype, np.integer)
    np.testing.assert_array_equal(actual, expected)


# An expression insertion_point is sampled as a fraction. Input: 'Beta(2, 5)' over 200 draws. Expected: every
# draw in [0, 1], and the mean near Beta(2, 5)'s 2/7 (within 0.05).
def test_expression_insertion_point_is_fractional() -> None:
    rng = np.random.default_rng(0)
    draws = [sample_insertion_point("Beta(2, 5)", rng) for _ in range(200)]
    assert all(0.0 <= d <= 1.0 for d in draws)
    assert abs(float(np.mean(draws)) - 2 / 7) < 0.05
