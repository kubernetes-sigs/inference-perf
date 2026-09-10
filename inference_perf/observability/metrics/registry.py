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

"""Config-driven registry of inference-perf's own Prometheus metrics.

Each exported metric is declared as a :class:`MetricSpec` whose ``enabled``
predicate inspects the run :class:`~inference_perf.config.Config`.
:func:`build_metrics` instantiates only the enabled specs into a fresh
``CollectorRegistry`` and returns a :class:`MetricsHub` that fans run
lifecycle events out to them. Disabled metrics are never instantiated, so
they are absent from the exposition output rather than present at zero.

The metric definitions themselves live under ``sets/``: ``sets/core.py``
holds the specs exported on every run, and config-conditional sets get
sibling modules aggregated in ``sets/__init__.py``. The exposition server in
``prometheus.py`` just serves the registry built here; it does not know which
metrics exist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any, Callable, Generic, List, Optional, Sequence, Tuple, TypeVar, Union

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram

from inference_perf.apis.base import RequestLifecycleMetric
from inference_perf.config import Config

logger = logging.getLogger(__name__)

PrometheusMetric = Union[Counter, Gauge, Histogram]
MetricT = TypeVar("MetricT", bound=PrometheusMetric)


def always(config: Config) -> bool:
    """Predicate for core metrics, exported regardless of config."""
    return True


@dataclass(frozen=True)
class MetricSpec(Generic[MetricT]):
    """Declares one exported metric and how it reacts to run lifecycle events.

    ``enabled`` decides from the run config whether the metric is exported at
    all. The lifecycle hooks are the only population path; each hook receives
    the live ``metric_type`` instance, typed via ``MetricT``, so a spec whose
    hook expects a different metric type than it declares fails type checking.
    Anything a hook needs to know about a request must travel on the
    ``RequestLifecycleMetric`` it is passed.
    """

    name: str
    documentation: str
    metric_type: type[MetricT]
    labelnames: Tuple[str, ...] = ()
    buckets: Optional[Tuple[float, ...]] = None  # histograms only
    enabled: Callable[[Config], bool] = always
    on_run_start: Optional[Callable[[MetricT], None]] = None
    on_request: Optional[Callable[[MetricT, RequestLifecycleMetric], None]] = None

    def __post_init__(self) -> None:
        if self.buckets is not None and self.metric_type is not Histogram:
            raise ValueError(f"metric {self.name!r}: buckets are only valid for Histogram metrics")


class MetricsHub:
    """Holds the metrics instantiated for one run and fans events out to them.

    Wire :meth:`observe_request` as an observer on the request metric
    collector and call :meth:`on_run_start` when the benchmark run begins.
    :attr:`registry` is what a ``PrometheusMetricsServer`` should serve.

    Hook exceptions are logged and swallowed, never propagated: these methods
    run inside the benchmark's request-recording path, and a buggy metric must
    not fail the run it is observing.
    """

    def __init__(self, registry: CollectorRegistry, bindings: Sequence[Tuple[MetricSpec[Any], PrometheusMetric]]) -> None:
        self.registry = registry
        self._bindings = tuple(bindings)
        self._failed_request_hooks: set[str] = set()

    def on_run_start(self) -> None:
        for spec, prom_metric in self._bindings:
            if spec.on_run_start is None:
                continue
            try:
                spec.on_run_start(prom_metric)
            except Exception:
                logger.exception("on_run_start hook for metric %r failed", spec.name)

    def observe_request(self, metric: RequestLifecycleMetric) -> None:
        for spec, prom_metric in self._bindings:
            if spec.on_request is None:
                continue
            try:
                spec.on_request(prom_metric, metric)
            except Exception:
                # Log once per spec: this runs per request, and a hook that
                # fails once will usually fail on every request.
                if spec.name not in self._failed_request_hooks:
                    self._failed_request_hooks.add(spec.name)
                    logger.exception("on_request hook for metric %r failed; suppressing further logs for this hook", spec.name)


def build_metrics(config: Config, specs: Optional[Sequence[MetricSpec[Any]]] = None) -> MetricsHub:
    """Instantiate the specs enabled under ``config`` into a fresh registry.

    Uses a fresh ``CollectorRegistry`` (never the global default one) so
    multiple runs in the same process do not collide.
    """
    if specs is None:
        # Imported lazily: the sets modules import MetricSpec from here.
        from inference_perf.observability.metrics.sets import ALL_SPECS

        specs = ALL_SPECS

    names = [spec.name for spec in specs]
    duplicates = sorted({name for name in names if names.count(name) > 1})
    if duplicates:
        raise ValueError(f"duplicate metric names in specs: {duplicates}")

    registry = CollectorRegistry()
    bindings: List[Tuple[MetricSpec[Any], PrometheusMetric]] = []
    for spec in specs:
        if spec.enabled(config):
            bindings.append((spec, _instantiate(spec, registry)))
    return MetricsHub(registry, bindings)


def _instantiate(spec: MetricSpec[Any], registry: CollectorRegistry) -> PrometheusMetric:
    if spec.metric_type is Histogram and spec.buckets is not None:
        return Histogram(spec.name, spec.documentation, labelnames=spec.labelnames, registry=registry, buckets=spec.buckets)
    prom_metric: PrometheusMetric = spec.metric_type(
        spec.name, spec.documentation, labelnames=spec.labelnames, registry=registry
    )
    return prom_metric
