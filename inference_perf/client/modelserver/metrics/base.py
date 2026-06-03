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
from typing import Any, Callable, Generic, Iterator, List, Optional, TypeVar
from pydantic import BaseModel

R = TypeVar("R", bound=BaseModel)


class Metric(ABC, Generic[R]):
    metric_name: str
    target_field: str

    @abstractmethod
    def get_queries(self, duration: float) -> List[str]:
        """Returns the ordered list of PromQL queries this metric requires."""
        ...

    @abstractmethod
    def parse(self, results: List[float]) -> R:
        """Convert the ordered query results into a typed result object."""
        ...

    def collect(self, execute: Callable[[str], float], duration: float) -> R:
        """Run this metric's queries via execute and parse them into its typed result.

        Keeps query execution and parsing together on the metric so callers never
        need to know the query/result shape of a particular metric type.
        """
        return self.parse([execute(query) for query in self.get_queries(duration)])


class BaseMetrics:
    def __init__(self, custom_metrics: Optional[List[Metric[Any]]] = None) -> None:
        self.custom_metrics = custom_metrics or []

    def _iter_metrics(self) -> Iterator[Metric[Any]]:
        """Yield every metric this container holds.

        Subclasses override this to yield their named fields (and then defer to
        super() for the custom metrics) so iteration is explicit rather than a
        reflection walk over __dict__.
        """
        yield from self.custom_metrics

    def __iter__(self) -> Iterator[Metric[Any]]:
        seen: set[int] = set()
        for metric in self._iter_metrics():
            if id(metric) not in seen:
                seen.add(id(metric))
                yield metric
