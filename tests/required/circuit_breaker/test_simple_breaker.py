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
"""Unit tests for SimpleCircuitBreaker.

SimpleCircuitBreaker only ever calls `metric.model_dump(...)` on whatever is
fed to it, so a small purpose-built BaseModel stands in for the metric here;
its exact type is irrelevant to the behavior under test.
"""

from typing import List, Optional

import pytest
from pydantic import BaseModel

from inference_perf.circuit_breaker.simple_breaker import SimpleCircuitBreaker
from inference_perf.config.circuit_breaker import (
    CircuitBreakerConfig,
    MetricsSpec,
    TriggerConsecutive,
)


class _Metric(BaseModel):
    latency: float
    error: Optional[str] = None


def _config(*, matches: List[str], rules: Optional[List[str]] = None, threshold: int = 1) -> CircuitBreakerConfig:
    return CircuitBreakerConfig(
        name="test-breaker",
        metrics=MetricsSpec(matches=matches, rules=rules if rules is not None else []),
        triggers=[TriggerConsecutive(type="consecutive", threshold=threshold)],
    )


# --- matches: OR semantics --------------------------------------------------


def test_matches_or_semantics_first_expression() -> None:
    config = _config(matches=["error == 'timeout'", "latency > `100`"])
    breaker = SimpleCircuitBreaker(config)

    breaker.feed(_Metric(latency=1.0, error="timeout"))

    assert breaker.is_open() is True


def test_matches_or_semantics_second_expression() -> None:
    config = _config(matches=["error == 'timeout'", "latency > `100`"])
    breaker = SimpleCircuitBreaker(config)

    breaker.feed(_Metric(latency=200.0))

    assert breaker.is_open() is True


def test_no_match_leaves_breaker_closed() -> None:
    config = _config(matches=["error == 'timeout'"])
    breaker = SimpleCircuitBreaker(config)

    breaker.feed(_Metric(latency=1.0))

    assert breaker.is_open() is False


# --- rules: OR semantics, empty rules ---------------------------------------


def test_rules_or_semantics() -> None:
    config = _config(matches=["latency > `0`"], rules=["error == 'timeout'", "latency > `100`"])
    breaker = SimpleCircuitBreaker(config)

    breaker.feed(_Metric(latency=200.0))  # matches, and 2nd rule makes it a hit

    assert breaker.is_open() is True


def test_rules_miss_does_not_open_breaker() -> None:
    config = _config(matches=["latency > `0`"], rules=["error == 'timeout'"])
    breaker = SimpleCircuitBreaker(config)

    breaker.feed(_Metric(latency=1.0))  # matches, but the rule is False -> miss

    assert breaker.is_open() is False


def test_empty_rules_means_a_match_is_a_hit() -> None:
    config = _config(matches=["latency > `0`"], rules=[])
    breaker = SimpleCircuitBreaker(config)

    breaker.feed(_Metric(latency=1.0))

    assert breaker.is_open() is True


# --- _search swallowing exceptions: characterization, not endorsement ------


def test_search_swallows_runtime_exception(monkeypatch: pytest.MonkeyPatch) -> None:
    """CHARACTERIZATION test, not an endorsement (see kubernetes-sigs/inference-perf#732).

    `_search` catches whatever exception a compiled expression's search() raises and
    treats it as "no match" -- feed() must not propagate it, and the breaker must stay
    closed. Pinned here deterministically by making search() itself raise, rather than
    relying on any particular JMESPath expression or JMESPath version raising. This is
    not an assertion that swallowing the exception is correct; production behavior is
    deliberately left unchanged here.
    """
    config = _config(matches=["latency > `0`"], rules=["latency > `0`"])
    breaker = SimpleCircuitBreaker(config)

    def _raise(value: object) -> bool:
        raise RuntimeError("boom")

    monkeypatch.setattr(breaker._rules[0], "search", _raise)

    breaker.feed(_Metric(latency=1.0))  # feed() must not propagate the exception

    assert breaker.is_open() is False


# --- trigger integration -----------------------------------------------------


def test_breaker_opens_when_a_real_trigger_fires() -> None:
    config = _config(matches=["latency > `0`"], threshold=2)
    breaker = SimpleCircuitBreaker(config)

    breaker.feed(_Metric(latency=1.0))
    assert breaker.is_open() is False

    breaker.feed(_Metric(latency=1.0))
    assert breaker.is_open() is True


def test_multiple_triggers_or_semantics() -> None:
    config = CircuitBreakerConfig(
        name="multi-trigger",
        metrics=MetricsSpec(matches=["latency > `0`"]),
        triggers=[
            TriggerConsecutive(type="consecutive", threshold=100),
            TriggerConsecutive(type="consecutive", threshold=1),
        ],
    )
    breaker = SimpleCircuitBreaker(config)

    # The first trigger needs 100 consecutive hits; the second needs only 1.
    breaker.feed(_Metric(latency=1.0))

    assert breaker.is_open() is True


# --- reset -------------------------------------------------------------------


def test_reset_clears_open_flag() -> None:
    config = _config(matches=["latency > `0`"], threshold=1)
    breaker = SimpleCircuitBreaker(config)
    breaker.feed(_Metric(latency=1.0))
    assert breaker.is_open() is True

    breaker.reset()

    assert breaker.is_open() is False


def test_reset_regression_trip_reset_partial_refeed_full_refeed() -> None:
    """Regression for the SimpleCircuitBreaker.reset() fix (kubernetes-sigs/inference-perf#732).

    Before the fix, reset() only cleared `_open` and left trigger state
    (Consecutive._fired / .c) latched, so a single matching metric fed after
    reset would re-open the breaker immediately. reset() must also reset()
    every trigger so a fresh, complete threshold is required to reopen.
    """
    config = _config(matches=["latency > `0`"], threshold=3)
    breaker = SimpleCircuitBreaker(config)

    # Trip the breaker.
    for _ in range(3):
        breaker.feed(_Metric(latency=1.0))
    assert breaker.is_open() is True

    # Reset: breaker is closed again.
    breaker.reset()
    assert breaker.is_open() is False

    # Fewer than the full threshold after reset must NOT immediately reopen it.
    for _ in range(2):
        breaker.feed(_Metric(latency=1.0))
    assert breaker.is_open() is False

    # A fresh, complete threshold does reopen it.
    breaker.feed(_Metric(latency=1.0))
    assert breaker.is_open() is True
