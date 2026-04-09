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

import pytest
from inference_perf.utils.expressions import evaluate_rate, sample_distribution


def test_evaluate_rate_constant() -> None:
    assert evaluate_rate("10", 0) == 10.0
    assert evaluate_rate("10", 100) == 10.0


def test_evaluate_rate_time_dependent() -> None:
    assert evaluate_rate("10 + t", 0) == 10.0
    assert evaluate_rate("10 + t", 5) == 15.0
    assert evaluate_rate("t / 2", 10) == 5.0


def test_sample_distribution() -> None:
    # We can't easily assert the exact value due to randomness,
    # but we can check if it returns a float and doesn't crash.
    val = sample_distribution("Normal(512, 200)")
    assert isinstance(val, float)

    val = sample_distribution("Poisson(10)")
    assert isinstance(val, float)

    val = sample_distribution("Uniform(10, 50)")
    assert isinstance(val, float)
    assert 10 <= val <= 50


def test_invalid_expression() -> None:
    with pytest.raises((ValueError, TypeError)):
        evaluate_rate("invalid_expr", 0)

    with pytest.raises((ValueError, TypeError)):
        sample_distribution("InvalidDist(10)")


def test_has_random_variables() -> None:
    from inference_perf.utils.expressions import has_random_variables

    assert not has_random_variables("10")
    assert not has_random_variables("10 + t/60")
    assert has_random_variables("Normal(10, 2)")
    assert has_random_variables("10 + Normal(0, 1)")
    assert not has_random_variables("invalid_expr")
