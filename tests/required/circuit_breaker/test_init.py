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
"""Unit tests for the circuit-breaker registry: init/get/feed_breakers.

`_initialized_circuit_breakers` is process-global state. Every test here gets
an isolated, fresh dict via monkeypatch so registration in one test can never
leak into another, and the original module state is restored automatically
once the test ends.
"""

import pytest
from pydantic import BaseModel

import inference_perf.circuit_breaker as cb_module
from inference_perf.circuit_breaker import feed_breakers, get_circuit_breaker, init_circuit_breakers
from inference_perf.config.circuit_breaker import CircuitBreakerConfig, MetricsSpec, TriggerConsecutive


class _Metric(BaseModel):
    latency: float


@pytest.fixture(autouse=True)
def _isolated_registry(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cb_module, "_initialized_circuit_breakers", {})


def _config(name: str) -> CircuitBreakerConfig:
    return CircuitBreakerConfig(
        name=name,
        metrics=MetricsSpec(matches=["latency > `0`"]),
        triggers=[TriggerConsecutive(type="consecutive", threshold=1)],
    )


def test_init_circuit_breakers_registers_by_name() -> None:
    init_circuit_breakers([_config("a"), _config("b")])

    assert get_circuit_breaker("a").name == "a"
    assert get_circuit_breaker("b").name == "b"


def test_init_circuit_breakers_double_init_raises() -> None:
    init_circuit_breakers([_config("a")])

    with pytest.raises(RuntimeError, match="already initialized"):
        init_circuit_breakers([_config("b")])


def test_get_circuit_breaker_unknown_name_raises() -> None:
    init_circuit_breakers([_config("a")])

    with pytest.raises(ValueError, match="Unknown circuit breaker: missing"):
        get_circuit_breaker("missing")


def test_feed_breakers_fans_out_to_all_registered() -> None:
    init_circuit_breakers([_config("a"), _config("b")])

    feed_breakers(_Metric(latency=1.0))

    assert get_circuit_breaker("a").is_open() is True
    assert get_circuit_breaker("b").is_open() is True


def test_feed_breakers_is_a_noop_when_none_registered() -> None:
    feed_breakers(_Metric(latency=1.0))  # must not raise
