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
from typing import List, Optional, Tuple
from inference_perf.config import APIConfig, APIType
from inference_perf.apis import InferenceAPIData


class Metric(ABC):
    metric_name: str

    @abstractmethod
    def get_queries(self, duration: float) -> List[Tuple[str, str]]:
        """Returns a list of tuples containing (target_attr, query)."""
        pass


class GaugeMetric(Metric):
    def __init__(self, base_name: str, metric_name: str, filters: List[str]) -> None:
        self.base_name = base_name
        self.metric_name = metric_name
        self.filters = ",".join(filters)

    def get_queries(self, duration: float) -> List[Tuple[str, str]]:
        f = self.filters
        m = self.metric_name
        return [
            (f"avg_{self.base_name}", f"avg_over_time({m}{{{f}}}[{duration:.0f}s])"),
            (f"median_{self.base_name}", f"quantile_over_time(0.5, {m}{{{f}}}[{duration:.0f}s])"),
            (f"p90_{self.base_name}", f"quantile_over_time(0.9, {m}{{{f}}}[{duration:.0f}s])"),
            (f"p99_{self.base_name}", f"quantile_over_time(0.99, {m}{{{f}}}[{duration:.0f}s])"),
        ]


class CounterMetric(Metric):
    def __init__(self, base_name: str, metric_name: str, filters: List[str]) -> None:
        self.base_name = base_name
        self.metric_name = metric_name
        self.filters = ",".join(filters)

    def get_queries(self, duration: float) -> List[Tuple[str, str]]:
        f = self.filters
        m = self.metric_name
        return [
            (f"avg_{self.base_name}", f"sum(increase({m}{{{f}}}[{duration:.0f}s]))"),
            (f"{self.base_name}_per_second", f"sum(rate({m}{{{f}}}[{duration:.0f}s]))"),
        ]


class RequestsMetric(Metric):
    def __init__(self, metric_name: str, filters: List[str]) -> None:
        self.metric_name = metric_name
        self.filters = ",".join(filters)

    def get_queries(self, duration: float) -> List[Tuple[str, str]]:
        m = self.metric_name
        if m.startswith("{") and m.endswith("}"):
            selector = f"{{{m[1:-1]},{self.filters}}}" if self.filters else m
        else:
            selector = f"{m}{{{self.filters}}}"
        return [
            ("total_requests", f"sum(increase({selector}[{duration:.0f}s]))"),
            ("requests_per_second", f"sum(rate({selector}[{duration:.0f}s]))"),
        ]


class HistogramMetric(Metric):
    def __init__(self, base_name: str, metric_name: str, filters: List[str]) -> None:
        self.base_name = base_name
        self.metric_name = metric_name
        self.filters = ",".join(filters)

    def get_queries(self, duration: float) -> List[Tuple[str, str]]:
        f = self.filters
        m = self.metric_name
        return [
            (
                f"avg_{self.base_name}",
                f"sum(rate({m}_sum{{{f}}}[{duration:.0f}s])) / (sum(rate({m}_count{{{f}}}[{duration:.0f}s])) > 0)",
            ),
            (f"median_{self.base_name}", f"histogram_quantile(0.5, sum(rate({m}_bucket{{{f}}}[{duration:.0f}s])) by (le))"),
            (f"p90_{self.base_name}", f"histogram_quantile(0.9, sum(rate({m}_bucket{{{f}}}[{duration:.0f}s])) by (le))"),
            (f"p99_{self.base_name}", f"histogram_quantile(0.99, sum(rate({m}_bucket{{{f}}}[{duration:.0f}s])) by (le))"),
            (f"{self.base_name}_per_second", f"sum(rate({m}_sum{{{f}}}[{duration:.0f}s]))"),
        ]


class CustomMetric(Metric):
    def __init__(self, target_field: str, metric_name: str, op: str, type: str, filters: List[str]) -> None:
        self.target_field = target_field
        self.metric_name = metric_name
        self.op = op
        self.type = type
        self.filters = ",".join(filters)

    def get_queries(self, duration: float) -> List[Tuple[str, str]]:
        # If the metric name is wrapped in `{...}` (e.g. `{__name__=~"foo(_total)?"}`),
        # merge the filters into the selector instead of appending a second `{...}` group.
        m = self.metric_name
        if m.startswith("{") and m.endswith("}"):
            selector = f"{{{m[1:-1]},{self.filters}}}" if self.filters else m
        else:
            selector = f"{m}{{{self.filters}}}"

        query = ""
        if self.type == "counter":
            if self.op == "rate":
                query = f"sum(rate({selector}[{duration:.0f}s]))"
            elif self.op == "increase":
                query = f"sum(increase({selector}[{duration:.0f}s]))"

        if query:
            return [(self.target_field, query)]
        return []


class BaseMetrics:
    def __init__(self, custom_metrics: Optional[List[Metric]] = None) -> None:
        self.custom_metrics = custom_metrics or []


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
