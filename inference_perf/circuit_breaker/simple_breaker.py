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
import jmespath

from datetime import datetime
from typing import Any, Dict, List
from pydantic import BaseModel
from inference_perf.config.circuit_breaker import CircuitBreakerConfig

from .base import CircuitBreaker
from .triggers import HitSample


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

    def _search(self, exprs: List[jmespath.parser.ParsedResult], data: Dict[str, Any]) -> bool:
        for expr in exprs:
            if bool(expr.search(data)):
                return True
        return False

    def feed(self, metric: BaseModel) -> None:
        data = metric.model_dump(mode="json", exclude_unset=True, exclude_none=True)
        if self._search(self._matches, data):
            hit = 1 if not self._rules or self._search(self._rules, data) else 0

            # Use metric.end_time if available to get request time
            if hasattr(metric, "end_time") and metric.end_time is not None:
                ts = datetime.fromtimestamp(metric.end_time)
            elif hasattr(metric, "end_timestamp_ns") and metric.end_timestamp_ns is not None:
                ts = datetime.fromtimestamp(metric.end_timestamp_ns / 1e9)
            elif hasattr(metric, "timestamp") and metric.timestamp is not None:
                if isinstance(metric.timestamp, datetime):
                    ts = metric.timestamp
                else:
                    ts = datetime.fromtimestamp(metric.timestamp)
            else:
                ts = datetime.now()

            hit_sample = HitSample(ts, hit)
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
