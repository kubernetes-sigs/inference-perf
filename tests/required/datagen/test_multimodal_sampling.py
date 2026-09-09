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


def test_uniform_insertion_preserves_continuous_range() -> None:
    config = Distribution(type=DistributionType.UNIFORM, min=0, max=1)
    rng = np.random.default_rng(42)
    actual = [sample_insertion_point(config, rng) for _ in range(100)]
    expected = np.random.default_rng(42).uniform(0, 1, 100)
    np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize("kind", list(DistributionType))
def test_insertion_distribution_is_clamped_to_prompt(kind: DistributionType) -> None:
    config = Distribution(type=kind, min=-10, max=10, mean=2, std_dev=3)
    rng = np.random.default_rng(42)
    assert all(0 <= sample_insertion_point(config, rng) <= 1 for _ in range(100))


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
