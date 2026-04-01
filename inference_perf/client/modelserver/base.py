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
from typing import Any, Generic, List, Optional, Tuple, TypeVar
from pydantic import BaseModel
from inference_perf.config import APIConfig, APIType
from inference_perf.apis import InferenceAPIData


class GaugeResult(BaseModel):
    avg: float = 0.0
    median: float = 0.0
    p90: float = 0.0
    p99: float = 0.0


class HistogramResult(GaugeResult):
    per_second: float = 0.0


class RequestsResult(BaseModel):
    total: float = 0.0
    per_second: float = 0.0


class CounterResult(BaseModel):
    value: float = 0.0


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


class GaugeMetric(Metric[GaugeResult]):
    def __init__(self, target_field: str, metric_name: str, filters: List[str]) -> None:
        self.target_field = target_field
        self.metric_name = metric_name
        self.filters = ",".join(filters)

    def get_queries(self, duration: float) -> List[str]:
        f, m = self.filters, self.metric_name
        return [
            f"avg_over_time({m}{{{f}}}[{duration:.0f}s])",
            f"quantile_over_time(0.5, {m}{{{f}}}[{duration:.0f}s])",
            f"quantile_over_time(0.9, {m}{{{f}}}[{duration:.0f}s])",
            f"quantile_over_time(0.99, {m}{{{f}}}[{duration:.0f}s])",
        ]

    def parse(self, results: List[float]) -> GaugeResult:
        return GaugeResult(avg=results[0], median=results[1], p90=results[2], p99=results[3])


class CounterMetric(Metric[HistogramResult]):
    """Returns HistogramResult (not CounterResult) because target fields like prompt_tokens may
    be populated by either CounterMetric (vllm/sglang) or HistogramMetric (tgi). Counter fills
    only avg+per_second; percentile fields default to 0."""

    def __init__(self, target_field: str, metric_name: str, filters: List[str]) -> None:
        self.target_field = target_field
        self.metric_name = metric_name
        self.filters = ",".join(filters)

    def get_queries(self, duration: float) -> List[str]:
        f, m = self.filters, self.metric_name
        return [
            f"sum(increase({m}{{{f}}}[{duration:.0f}s]))",
            f"sum(rate({m}{{{f}}}[{duration:.0f}s]))",
        ]

    def parse(self, results: List[float]) -> HistogramResult:
        return HistogramResult(avg=results[0], per_second=results[1])


class RequestsMetric(Metric[RequestsResult]):
    def __init__(self, metric_name: str, filters: List[str], target_field: str = "requests") -> None:
        self.target_field = target_field
        self.metric_name = metric_name
        self.filters = ",".join(filters)

    def _selector(self) -> str:
        m = self.metric_name
        if m.startswith("{") and m.endswith("}"):
            return f"{{{m[1:-1]},{self.filters}}}" if self.filters else m
        return f"{m}{{{self.filters}}}"

    def get_queries(self, duration: float) -> List[str]:
        selector = self._selector()
        return [
            f"sum(increase({selector}[{duration:.0f}s]))",
            f"sum(rate({selector}[{duration:.0f}s]))",
        ]

    def parse(self, results: List[float]) -> RequestsResult:
        return RequestsResult(total=results[0], per_second=results[1])


class HistogramMetric(Metric[HistogramResult]):
    def __init__(self, target_field: str, metric_name: str, filters: List[str]) -> None:
        self.target_field = target_field
        self.metric_name = metric_name
        self.filters = ",".join(filters)

    def get_queries(self, duration: float) -> List[str]:
        f, m = self.filters, self.metric_name
        return [
            f"sum(rate({m}_sum{{{f}}}[{duration:.0f}s])) / (sum(rate({m}_count{{{f}}}[{duration:.0f}s])) > 0)",
            f"histogram_quantile(0.5, sum(rate({m}_bucket{{{f}}}[{duration:.0f}s])) by (le))",
            f"histogram_quantile(0.9, sum(rate({m}_bucket{{{f}}}[{duration:.0f}s])) by (le))",
            f"histogram_quantile(0.99, sum(rate({m}_bucket{{{f}}}[{duration:.0f}s])) by (le))",
            f"sum(rate({m}_sum{{{f}}}[{duration:.0f}s]))",
        ]

    def parse(self, results: List[float]) -> HistogramResult:
        return HistogramResult(avg=results[0], median=results[1], p90=results[2], p99=results[3], per_second=results[4])


class CustomMetric(Metric[CounterResult]):
    def __init__(self, target_field: str, metric_name: str, op: str, type: str, filters: List[str]) -> None:
        self.target_field = target_field
        self.metric_name = metric_name
        self.op = op
        self.type = type
        self.filters = ",".join(filters)

    def _selector(self) -> str:
        # If the metric name is wrapped in `{...}` (e.g. `{__name__=~"foo(_total)?"}`),
        # merge the filters into the selector instead of appending a second `{...}` group.
        m = self.metric_name
        if m.startswith("{") and m.endswith("}"):
            return f"{{{m[1:-1]},{self.filters}}}" if self.filters else m
        return f"{m}{{{self.filters}}}"

    def get_queries(self, duration: float) -> List[str]:
        if self.type != "counter":
            return []
        selector = self._selector()
        if self.op == "rate":
            return [f"sum(rate({selector}[{duration:.0f}s]))"]
        if self.op == "increase":
            return [f"sum(increase({selector}[{duration:.0f}s]))"]
        return []

    def parse(self, results: List[float]) -> CounterResult:
        return CounterResult(value=results[0]) if results else CounterResult()


class BaseMetrics:
    def __init__(self, custom_metrics: Optional[List[Metric[Any]]] = None) -> None:
        self.custom_metrics = custom_metrics or []

    def get_all_metrics(self) -> List[Metric[Any]]:
        return list(self.custom_metrics)


class ModelServerClient(ABC):
    @abstractmethod
    def __init__(self, api_config: APIConfig, timeout: Optional[float] = None, *args: Tuple[int, ...]) -> None:
        if api_config.type not in self.get_supported_apis():
            raise Exception(f"Unsupported API type {api_config}")

        self.api_config = api_config
        self.timeout = timeout

    def new_session(self) -> "ModelServerClientSession":
        return ModelServerClientSession(self)

    @abstractmethod
    def get_supported_apis(self) -> List[APIType]:
        raise NotImplementedError

    @abstractmethod
    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        raise NotImplementedError

    @abstractmethod
    def get_prometheus_metric_metadata(self) -> BaseMetrics:
        # assumption: all metrics clients have metrics exported in Prometheus format
        raise NotImplementedError


class ModelServerClientSession:
    def __init__(self, client: ModelServerClient):
        self.client = client

    async def process_request(
        self, data: InferenceAPIData, stage_id: int, scheduled_time: float, lora_adapter: Optional[str] = None
    ) -> None:
        await self.client.process_request(data, stage_id, scheduled_time, lora_adapter)

    async def close(self) -> None:  # noqa - subclasses optionally override this
        pass
