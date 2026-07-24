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
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, Generic, Iterator, List, Optional, Tuple, TypeVar
from pydantic import BaseModel

R = TypeVar("R", bound=BaseModel)


class Metric(ABC, Generic[R]):
    metric_name: str

    @abstractmethod
    def get_queries(self, duration: float, filters: str) -> List[str]:
        """Returns the ordered list of PromQL queries this metric requires.

        filters is the comma-joined label selector (e.g. "model_name='m'"), supplied by the
        owning container so it is not repeated on every metric.
        """
        ...

    @abstractmethod
    def parse(self, results: List[float]) -> R:
        """Convert the ordered query results into a typed result object."""
        ...

    def collect(self, execute: Callable[[str], float], duration: float, filters: str) -> R:
        """Run this metric's queries via execute and parse them into its typed result.

        Keeps query execution and parsing together on the metric so callers never
        need to know the query/result shape of a particular metric type.
        """
        return self.parse([execute(query) for query in self.get_queries(duration, filters)])


class BaseMetrics:
    """A collection of metrics keyed by the ModelServerMetrics field they populate.

    The field name lives in the container (the custom_metrics dict key, or a named
    field's attribute name) rather than on the metric, so a metric only carries its
    PromQL name and never repeats the target field. The label filters are uniform across
    a model server, so they live here too and are applied at query-build time.
    """

    def __init__(self, filters: Optional[List[str]] = None, custom_metrics: Optional[Dict[str, Metric[Any]]] = None) -> None:
        self.filters = ",".join(filters or [])
        self.custom_metrics = custom_metrics or {}

    def _iter_metrics(self) -> Iterator[Tuple[str, Metric[Any]]]:
        """Yield (target_field, metric) for every metric this container holds.

        Subclasses override this to yield their named fields first (and then defer to
        super() for the custom metrics) so iteration is explicit rather than a
        reflection walk over __dict__.
        """
        yield from self.custom_metrics.items()

    def __iter__(self) -> Iterator[Tuple[str, Metric[Any]]]:
        seen: set[str] = set()
        for field, metric in self._iter_metrics():
            if field not in seen:
                seen.add(field)
                yield field, metric
