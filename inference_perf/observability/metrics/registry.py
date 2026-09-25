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

Every spec carries a :class:`MetricStability` level, and the level is
prepended to the metric's HELP text so it reaches whoever is reading the
exposition output rather than living only in this repo's docs. New specs are
``ALPHA`` unless they say otherwise.

The metric definitions themselves live under ``sets/``: ``sets/core.py``
holds the specs exported on every run, and config-conditional sets get
sibling modules aggregated in ``sets/__init__.py``. The exposition server in
``prometheus.py`` just serves the registry built here; it does not know which
metrics exist.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import StrEnum
from typing import Any, Callable, Generic, List, Optional, Sequence, Tuple, TypeVar, Union

from prometheus_client import CollectorRegistry, Counter, Gauge, Histogram  # noqa: TID251 (instantiates the declared specs)

from inference_perf.apis.base import RequestLifecycleMetric
from inference_perf.config import Config
from inference_perf.observability.context import RunContext, StageContext

logger = logging.getLogger(__name__)

PrometheusMetric = Union[Counter, Gauge, Histogram]
MetricT = TypeVar("MetricT", bound=PrometheusMetric)


class MetricStability(StrEnum):
    """How much a metric's shape is promised to callers, per metric.

    The level travels into the HELP text (see :func:`_instantiate`) and into
    the generated ``docs/runtime_metrics.md``. It is deliberately per metric, not
    per set, so a metric can graduate on its own without moving modules.

    Nothing here is ``STABLE`` before v1.0.0; ``ALPHA`` is the default so a
    new metric cannot become a promise by omission.
    """

    ALPHA = "ALPHA"
    BETA = "BETA"
    STABLE = "STABLE"

    @property
    def promise(self) -> str:
        """What this level guarantees to whoever is scraping the metric.

        Defined once here so the generated ``docs/runtime_metrics.md`` and any
        reviewer reading the specs get the same wording.
        """
        return _PROMISES[self]


_PROMISES = {
    MetricStability.ALPHA: "May be renamed, relabeled or removed in any release, with no notice.",
    MetricStability.BETA: ("Labels may still change, but no rename or removal without a release that deprecates it first."),
    MetricStability.STABLE: "Name, type and label names are fixed for the current major version.",
}


def always(config: Config) -> bool:
    """Predicate for core metrics, exported regardless of config."""
    return True


def exposition_name(spec: "MetricSpec[Any]") -> str:
    """The sample name a metric appears under in the exposition format.

    ``prometheus_client`` appends ``_total`` to Counter sample names, so a
    Counter declared as ``inference_perf_requests`` is scraped, and read back
    via ``CollectorRegistry.get_sample_value``, as
    ``inference_perf_requests_total``. Anything resolving a spec to a sample
    must go through here rather than using ``spec.name`` directly.
    """
    if spec.metric_type is Counter:
        return f"{spec.name}_total"
    return spec.name


def help_text(spec: "MetricSpec[Any]") -> str:
    """The HELP string a metric is exported with: its stability, then its docs.

    Kubernetes components mark stability the same way, so a scraped
    ``inference_perf_*`` metric reads like the rest of the ecosystem and a
    consumer learns what is promised from the exposition output alone, without
    finding this repo. The level is only here, never in the metric name and
    never in a label, so promoting a metric does not break the queries and
    dashboards written against it.
    """
    return f"[{spec.stability.value}] {spec.documentation}"


@dataclass(frozen=True)
class MetricSpec(Generic[MetricT]):
    """Declares one exported metric and how it reacts to run lifecycle events.

    ``enabled`` decides from the run config whether the metric is exported at
    all. The lifecycle hooks are the only population path; each hook receives
    the live ``metric_type`` instance, typed via ``MetricT``, so a spec whose
    hook expects a different metric type than it declares fails type checking.
    Anything a hook needs to know about a request must travel on the
    ``RequestLifecycleMetric`` it is passed; anything it needs about the run
    travels on the :class:`RunContext`.

    ``stability`` is the compatibility promise for this one metric; leaving it
    at the default keeps the metric changeable.
    """

    name: str
    documentation: str
    metric_type: type[MetricT]
    labelnames: Tuple[str, ...] = ()
    buckets: Optional[Tuple[float, ...]] = None  # histograms only
    stability: MetricStability = MetricStability.ALPHA
    enabled: Callable[[Config], bool] = always
    on_run_start: Optional[Callable[[MetricT, RunContext], None]] = None
    on_stage_start: Optional[Callable[[MetricT, StageContext], None]] = None
    on_stage_end: Optional[Callable[[MetricT, StageContext], None]] = None
    on_request: Optional[Callable[[MetricT, RequestLifecycleMetric], None]] = None

    def __post_init__(self) -> None:
        if self.buckets is not None and self.metric_type is not Histogram:
            raise ValueError(f"metric {self.name!r}: buckets are only valid for Histogram metrics")


class MetricsHub:
    """Holds the metrics instantiated for one run and fans events out to them.

    Wire :meth:`observe_request` as an observer on the request metric
    collector, hand the hub to the load generator as its stage observer, and
    call :meth:`on_run_start` when the benchmark run begins.
    :attr:`registry` is what a ``PrometheusMetricsServer`` should serve.

    Hook exceptions are logged and swallowed, never propagated: these methods
    run inside the benchmark's request-recording path, and a buggy metric must
    not fail the run it is observing.
    """

    def __init__(
        self,
        registry: CollectorRegistry,
        bindings: Sequence[Tuple[MetricSpec[Any], PrometheusMetric]],
        config: Optional[Config] = None,
    ) -> None:
        self.registry = registry
        self._bindings = tuple(bindings)
        self._config = config
        self._failed_request_hooks: set[str] = set()

    def on_run_start(self, context: Optional[RunContext] = None) -> None:
        if context is None:
            context = RunContext(config=self._config if self._config is not None else Config())
        self._dispatch("on_run_start", context)

    def on_stage_start(self, context: StageContext) -> None:
        self._dispatch("on_stage_start", context)

    def on_stage_end(self, context: StageContext) -> None:
        self._dispatch("on_stage_end", context)

    def _dispatch(self, event: str, *args: Any) -> None:
        for spec, prom_metric in self._bindings:
            hook = getattr(spec, event)
            if hook is None:
                continue
            try:
                hook(prom_metric, *args)
            except Exception:
                logger.exception("%s hook for metric %r failed", event, spec.name)

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
    return MetricsHub(registry, bindings, config=config)


def _instantiate(spec: MetricSpec[Any], registry: CollectorRegistry) -> PrometheusMetric:
    documentation = help_text(spec)
    if spec.metric_type is Histogram and spec.buckets is not None:
        return Histogram(spec.name, documentation, labelnames=spec.labelnames, registry=registry, buckets=spec.buckets)
    prom_metric: PrometheusMetric = spec.metric_type(spec.name, documentation, labelnames=spec.labelnames, registry=registry)
    return prom_metric
