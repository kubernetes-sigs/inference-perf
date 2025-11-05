# Copyright 2025 The Kubernetes Authors.
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
from typing import Dict, List

from pydantic import BaseModel

from .base import CircuitBreaker
from .config import CircuitBreakerConfig
from .simple_breaker import SimpleCircuitBreaker

_initialized_circuit_breakers: Dict[str, CircuitBreaker] = {}


def init_circuit_breakers(configs: List[CircuitBreakerConfig]) -> None:
    if _initialized_circuit_breakers:
        raise RuntimeError("Circuit breakers already initialized")
    for config in configs:
        _initialized_circuit_breakers[config.name] = SimpleCircuitBreaker(config)


def feed_breakers(data: BaseModel) -> None:
    for breaker in _initialized_circuit_breakers.values():
        breaker.feed(data)


def get_circuit_breaker(name: str) -> CircuitBreaker:
    if name not in _initialized_circuit_breakers:
        raise ValueError(f"Unknown circuit breaker: {name}")
    return _initialized_circuit_breakers[name]


__all__ = [
    "feed_breakers",
    "get_circuit_breaker",
    "init_circuit_breakers",
    "CircuitBreaker",
    "CircuitBreakerConfig",
]
