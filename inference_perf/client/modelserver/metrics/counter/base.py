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
from typing import List
from pydantic import BaseModel

from ..base import Metric


class CounterResult(BaseModel):
    """Result of a counter query: the windowed total (increase), the average per-second rate
    over the window, and the overall per-second rate."""

    total: float = 0.0
    avg: float = 0.0
    per_second: float = 0.0


class CounterMetric(Metric[CounterResult]):
    """avg is the average per-second rate over the window (avg_over_time of the rate), matching the
    pre-refactor counter "mean" semantics rather than the window total.

    A counter's stored series name depends on the exporter: prometheus_client appends `_total`
    to every counter sample, while older exporters (and OTel re-exports) keep the bare family
    name. Queries therefore match both exact forms joined with `or` rather than a
    `{__name__=~"name(_total)?"}` selector: Google Managed Prometheus rejects regex matchers
    on `__name__` with HTTP 400, which silently zeroed every counter backed by one (#567).
    """

    def __init__(self, metric_name: str) -> None:
        # `{__name__=~...}` selector names are not supported: GMP rejects regex matchers on
        # `__name__`, so both name forms are queried exactly (see get_queries).
        if metric_name.startswith("{"):
            raise ValueError(f"CounterMetric does not support `{{__name__=~...}}` selector metric names: {metric_name}")
        self.metric_name = metric_name

    def _spanning(self, fn: str, duration: float, filters: str) -> str:
        # `fn` applied to the `_total`-suffixed and bare forms of the name, whichever exists;
        # `or` unions the two so mixed fleets (old and new exporters) still sum correctly.
        # Histogram series names (`_count`/`_sum`/`_bucket`) can never carry `_total`, so
        # counters over them keep a single exact leg.
        base = self.metric_name.removesuffix("_total")
        d = f"{duration:.0f}s"
        if base.endswith(("_count", "_sum", "_bucket")):
            return f"{fn}({base}{{{filters}}}[{d}])"
        return f"{fn}({base}_total{{{filters}}}[{d}]) or {fn}({base}{{{filters}}}[{d}])"

    def get_queries(self, duration: float, filters: str) -> List[str]:
        return [
            f"sum({self._spanning('increase', duration, filters)})",
            f"avg_over_time(({self._spanning('rate', duration, filters)})[{duration:.0f}s:{duration:.0f}s])",
            f"sum({self._spanning('rate', duration, filters)})",
        ]

    def parse(self, results: List[float]) -> CounterResult:
        return CounterResult(total=results[0], avg=results[1], per_second=results[2])
