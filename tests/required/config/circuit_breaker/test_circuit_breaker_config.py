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
"""Validity rules for ``inference_perf.config.circuit_breaker``.

Covers only constraints the pydantic models declare explicitly (Field
ge/gt/le/min_length); no invented validation rules.
"""

import pytest
from pydantic import ValidationError

from inference_perf.config.circuit_breaker import (
    CircuitBreakerConfig,
    MetricsSpec,
    TriggerConsecutive,
    TriggerRateOverWindow,
)

# --- TriggerConsecutive -----------------------------------------------------


def test_trigger_consecutive_rejects_threshold_below_one() -> None:
    with pytest.raises(ValidationError):
        TriggerConsecutive(type="consecutive", threshold=0)


def test_trigger_consecutive_threshold_one_is_valid() -> None:
    trigger = TriggerConsecutive(type="consecutive", threshold=1)

    assert trigger.threshold == 1


# --- TriggerRateOverWindow ---------------------------------------------------


def test_trigger_rate_over_window_rejects_non_positive_window_sec() -> None:
    with pytest.raises(ValidationError):
        TriggerRateOverWindow(type="rate_over_window", window_sec=0.0, threshold=0.5)


def test_trigger_rate_over_window_rejects_threshold_above_one() -> None:
    with pytest.raises(ValidationError):
        TriggerRateOverWindow(type="rate_over_window", window_sec=10.0, threshold=1.1)


def test_trigger_rate_over_window_rejects_negative_threshold() -> None:
    with pytest.raises(ValidationError):
        TriggerRateOverWindow(type="rate_over_window", window_sec=10.0, threshold=-0.1)


def test_trigger_rate_over_window_threshold_boundaries_are_valid() -> None:
    low = TriggerRateOverWindow(type="rate_over_window", window_sec=10.0, threshold=0.0)
    high = TriggerRateOverWindow(type="rate_over_window", window_sec=10.0, threshold=1.0)

    assert low.threshold == 0.0
    assert high.threshold == 1.0


def test_trigger_rate_over_window_rejects_negative_min_samples() -> None:
    with pytest.raises(ValidationError):
        TriggerRateOverWindow(type="rate_over_window", window_sec=10.0, threshold=0.5, min_samples=-1)


def test_trigger_rate_over_window_min_samples_defaults_to_zero() -> None:
    trigger = TriggerRateOverWindow(type="rate_over_window", window_sec=10.0, threshold=0.5)

    assert trigger.min_samples == 0


# --- MetricsSpec --------------------------------------------------------------


def test_metrics_spec_requires_at_least_one_match() -> None:
    with pytest.raises(ValidationError):
        MetricsSpec(matches=[])


def test_metrics_spec_rules_default_to_empty() -> None:
    spec = MetricsSpec(matches=["latency > `0`"])

    assert spec.rules == []


# --- CircuitBreakerConfig ------------------------------------------------------


def test_circuit_breaker_config_accepts_mixed_trigger_types() -> None:
    config = CircuitBreakerConfig(
        name="breaker",
        metrics=MetricsSpec(matches=["latency > `0`"], rules=["error != `null`"]),
        triggers=[
            TriggerConsecutive(type="consecutive", threshold=3),
            TriggerRateOverWindow(type="rate_over_window", window_sec=30.0, threshold=0.5, min_samples=5),
        ],
    )

    assert config.name == "breaker"
    assert isinstance(config.triggers[0], TriggerConsecutive)
    assert isinstance(config.triggers[1], TriggerRateOverWindow)
