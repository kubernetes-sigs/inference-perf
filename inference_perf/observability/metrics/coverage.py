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

"""Proof that every metric the endpoint exposes is a documented one.

``docs/runtime_metrics.md`` is generated from ``ALL_SPECS``, so on its own it only
promises that the declared metrics are written down. What makes it a complete
reference is the pair of checks here, run by
``pdm run check:runtime-metrics`` inside the merge-blocking ``validate``:

- :func:`unaggregated_spec_names` catches a metric declared in a ``sets/``
  module that ``sets/__init__.py`` forgot to fold into ``ALL_SPECS``, which
  would otherwise be silently absent from both the endpoint and the doc.
- :func:`emitted_metric_names` reads the family names back out of a rendered
  exposition rather than off the specs, so the comparison is against what a
  scraper actually sees.

The third leg is a ruff ``TID251`` ban on the ``prometheus_client``
constructors (see ``pyproject.toml``), which stops a metric from reaching a
registry without passing through a spec in the first place.
"""

from __future__ import annotations

import importlib
import pkgutil
import re
from dataclasses import replace
from typing import Any, List, Sequence, Set, Tuple

from prometheus_client.exposition import generate_latest

from inference_perf.config import Config
from inference_perf.observability.metrics.registry import MetricSpec, always, build_metrics, exposition_name
from inference_perf.observability.metrics.sets import ALL_SPECS

SETS_PACKAGE = "inference_perf.observability.metrics.sets"

# `# TYPE <name> <type>` is the exposition's own declaration of a metric
# family, and counters appear there already suffixed with _total.
TYPE_LINE = re.compile(r"^# TYPE (\S+) (\S+)$", re.MULTILINE)


def emitted_metric_names(specs: Sequence[MetricSpec[Any]] = ALL_SPECS) -> Set[str]:
    """Every metric name a scrape of this build can expose.

    Instantiates the specs with every ``enabled`` predicate forced on, so the
    result is the union over all configs rather than the subset that one
    config happens to turn on.
    """
    ungated = tuple(replace(spec, enabled=always) for spec in specs)
    body = generate_latest(build_metrics(Config(), specs=ungated).registry).decode()
    return _families(TYPE_LINE.findall(body))


def _families(declarations: Sequence[Tuple[str, str]]) -> Set[str]:
    """The metric names from `# TYPE` lines, minus prometheus_client's own.

    A counter or histogram with no labels has a child from the start, so
    prometheus_client emits a `<name>_created` gauge for it with a TYPE line
    of its own. That series belongs to the metric it is derived from and gets
    no row of its own in the doc, so counting it would report a phantom
    undocumented metric the moment anyone declares an unlabelled counter.
    """
    names = {name for name, _ in declarations}
    derived = set()
    for name, metric_type in declarations:
        if metric_type != "gauge" or not name.endswith("_created"):
            continue
        base = name[: -len("_created")]
        if f"{base}_total" in names or base in names:
            derived.add(name)
    return names - derived


def documented_metric_names(specs: Sequence[MetricSpec[Any]] = ALL_SPECS) -> Set[str]:
    """The metric names ``docs/runtime_metrics.md`` gets a row for."""
    return {exposition_name(spec) for spec in specs}


def specs_declared_under_sets() -> List[MetricSpec[Any]]:
    """Every MetricSpec reachable at module level under ``sets/``.

    ALL_SPECS is hand-assembled in ``sets/__init__.py``, so a new set module
    that nobody remembered to add there would be undocumented and unexported
    with nothing to say so. Importing the package's modules and reading their
    module-level specs is what makes that omission visible.
    """
    declared: List[MetricSpec[Any]] = []
    package = importlib.import_module(SETS_PACKAGE)
    for module_info in pkgutil.iter_modules(list(package.__path__)):
        module = importlib.import_module(f"{SETS_PACKAGE}.{module_info.name}")
        for value in vars(module).values():
            if isinstance(value, MetricSpec):
                declared.append(value)
            elif isinstance(value, tuple):
                declared.extend(item for item in value if isinstance(item, MetricSpec))
    return declared


def unaggregated_spec_names(specs: Sequence[MetricSpec[Any]] = ALL_SPECS) -> List[str]:
    """Names of specs declared under ``sets/`` that ``specs`` leaves out."""
    aggregated = {id(spec) for spec in specs}
    return sorted({spec.name for spec in specs_declared_under_sets() if id(spec) not in aggregated})


def coverage_problems(specs: Sequence[MetricSpec[Any]] = ALL_SPECS) -> List[str]:
    """Every way the doc and the exposition currently disagree, as prose.

    Empty means the table in ``docs/runtime_metrics.md`` names exactly the metrics
    a scrape can show. Returns problems rather than raising so the caller
    decides how to report them; ``scripts/sync_runtime_metrics_doc.py``
    prints them and exits non-zero.
    """
    orphans = unaggregated_spec_names(specs)
    if orphans:
        # Reported alone: these specs are not in the registry either, so the
        # emitted-vs-documented comparison below would say nothing about them.
        return [
            f"Metrics declared under sets/ but missing from ALL_SPECS: {', '.join(orphans)}. "
            "Every metric must be exported and documented through ALL_SPECS. Add the set's specs to "
            "ALL_SPECS in sets/__init__.py, then run `pdm run update:runtime-metrics`."
        ]

    documented = documented_metric_names(specs)
    emitted = emitted_metric_names(specs)
    problems: List[str] = []

    undocumented = sorted(emitted - documented)
    if undocumented:
        problems.append(
            f"Metrics the endpoint can expose with no row in the doc: {', '.join(undocumented)}. "
            "Declare each one as a MetricSpec in a sets/ module reachable from ALL_SPECS; the doc is "
            "generated from there."
        )

    stale = sorted(documented - emitted)
    if stale:
        problems.append(
            f"Metrics documented but not present in the exposition: {', '.join(stale)}. "
            "The exposition name of a spec is what a scraper sees; fix the spec or remove it."
        )
    return problems
