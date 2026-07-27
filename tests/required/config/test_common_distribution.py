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
"""Validity rules for ``inference_perf.config.common.Distribution``."""

import math

import pytest

from inference_perf.config import Distribution, DistributionType


def test_defaults() -> None:
    d = Distribution()
    assert d.type == DistributionType.NORMAL
    assert d.min == 10
    assert d.max == 1024
    assert d.mean == 512
    assert d.std_dev == 200
    assert d.variance is None
    assert d.skew == 0.0


def test_variance_conversion_to_std_dev() -> None:
    d = Distribution(type=DistributionType.NORMAL, mean=100.0, variance=6400.0, std_dev=0.0)
    assert math.isclose(d.std_dev, 80.0, abs_tol=1e-6)


def test_both_variance_and_std_dev_is_error() -> None:
    with pytest.raises(ValueError, match="Specify either"):
        Distribution(type=DistributionType.NORMAL, mean=100.0, std_dev=10.0, variance=100.0)


def test_negative_variance_is_error() -> None:
    with pytest.raises(ValueError, match="Variance cannot be negative"):
        Distribution(mean=100.0, std_dev=0.0, variance=-1.0)


def test_negative_std_dev_is_error() -> None:
    with pytest.raises(ValueError, match="std_dev cannot be negative"):
        Distribution(mean=100.0, std_dev=-1.0)


def test_min_greater_than_max_is_error() -> None:
    with pytest.raises(ValueError, match=r"min \(2000\) cannot be greater than max \(10\)"):
        Distribution(min=2000, max=10)


def test_min_equal_to_max_is_allowed() -> None:
    d = Distribution(min=5, max=5)
    assert d.min == d.max == 5
