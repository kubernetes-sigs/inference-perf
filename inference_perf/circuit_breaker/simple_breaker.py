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
import logging
import time
import jmespath

from typing import Any, Dict, List, Set
from pydantic import BaseModel
from inference_perf.apis import RequestLifecycleMetric
from inference_perf.config.circuit_breaker import CircuitBreakerConfig

from .base import CircuitBreaker
from .triggers import HitSample

logger = logging.getLogger(__name__)


class SimpleCircuitBreaker(CircuitBreaker):
    """
    Simple Expression-driven breaker.
    - rules: JMESPath boolean expressions (OR semantics).
    - triggers: OR semantics; any fired -> open.
    """

    def __init__(self, config: CircuitBreakerConfig) -> None:
        super().__init__(config)
        self._matches = [jmespath.compile(expr) for expr in config.metrics.matches]
        self._rules = [jmespath.compile(expr) for expr in config.metrics.rules]
        self._failed_exprs: Set[str] = set()

    def _search(self, exprs: List[jmespath.parser.ParsedResult], data: Dict[str, Any]) -> bool:
        for expr in exprs:
            try:
                if bool(expr.search(data)):
                    return True
            except Exception as e:
                # Data-dependent (e.g. comparing a string field to a number). Never let it
                # reach the collector's drain task, but say once per expression that it is
                # not being evaluated, since the breaker will otherwise just never trip on it.
                if expr.expression not in self._failed_exprs:
                    self._failed_exprs.add(expr.expression)
                    logger.warning(
                        f"Circuit breaker '{self.name}': expression {expr.expression!r} "
                        f"failed to evaluate and is treated as not matching ({type(e).__name__}: {e}). "
                        f"Further failures of this expression are not logged."
                    )
        return False

    def feed(self, metric: BaseModel) -> None:
        data = metric.model_dump(mode="json", exclude_unset=True, exclude_none=True)
        if self._search(self._matches, data):
            hit = 1 if not self._rules or self._search(self._rules, data) else 0
            sample_time = metric.end_time if isinstance(metric, RequestLifecycleMetric) else time.perf_counter()
            hit_sample = HitSample(sample_time, hit)
            for t in self._triggers:
                t.update(hit_sample)
                if t.fired():
                    self._open = True

    def is_open(self) -> bool:
        return self._open

    def reset(self) -> None:
        self._open = False
        for t in self._triggers:
            t.reset()
