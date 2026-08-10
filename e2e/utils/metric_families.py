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
"""Prometheus metric-family bookkeeping shared by the metric-name checks.

Three representations of "which metrics exist" meet in these tests, and this
module converts between them:

- declared: the (base name, type) pairs the vLLM client will query, read from
  ``get_prometheus_metric_metadata()``. A name is either bare
  (``vllm:num_requests_waiting``) or a version-spanning PromQL selector
  (``{__name__=~"vllm:request_success(_total)?"}``), so matching is by base
  name plus the type's naming convention, never string equality.
- exposed: what a live server's ``/metrics`` text actually contains.
- golden: the committed per-release snapshot of the exposed ``vllm:*``
  families (``e2e/testdata/vllm_metric_families/<release-tag>.txt``), which
  stands in for a live server in the serverless check.

Golden files hold one ``<family> <type>`` pair per line, sorted;  ``#``
comments and blank lines are ignored. ``*_created`` gauges are dropped at
capture time: they are prometheus_client per-series creation timestamps,
mechanical companions of their family, and vLLM can toggle them via
``PROMETHEUS_DISABLE_CREATED_SERIES`` without any semantic change.
"""

import re
from pathlib import Path
from typing import Dict, Set

from inference_perf.client.modelserver.metrics import CounterMetric, GaugeMetric, HistogramMetric
from inference_perf.client.modelserver.openai_client import OpenAIMetrics

GOLDEN_DIR = Path(__file__).resolve().parents[1] / "testdata" / "vllm_metric_families"

_BASE_NAME = re.compile(r"vllm:[A-Za-z0-9_]+")


def declared_metrics(metadata: OpenAIMetrics) -> Dict[str, str]:
    """Metric base names -> prometheus type, as declared by a client's metadata."""
    types = ((CounterMetric, "counter"), (GaugeMetric, "gauge"), (HistogramMetric, "histogram"))
    declared: Dict[str, str] = {}
    for _field, metric in metadata:
        metric_type = next(t for cls, t in types if isinstance(metric, cls))
        for name in _BASE_NAME.findall(metric.metric_name):
            declared[name] = metric_type
    return declared


def exposed_names(metrics_text: str) -> Set[str]:
    """All family and sample names present in a /metrics exposition."""
    names = set()
    for line in metrics_text.splitlines():
        if line.startswith("# TYPE ") or line.startswith("# HELP "):
            names.add(line.split(" ")[2])
        elif line and not line.startswith("#"):
            names.add(line.split("{")[0].split(" ")[0])
    return names


def exposed_vllm_families(metrics_text: str) -> Dict[str, str]:
    """The exposition's ``vllm:*`` family -> type map, ``*_created`` dropped."""
    families: Dict[str, str] = {}
    for line in metrics_text.splitlines():
        if not line.startswith("# TYPE vllm:"):
            continue
        _, _, name, metric_type = line.split(" ")
        if not name.endswith("_created"):
            families[name] = metric_type
    return families


def is_exposed(name: str, metric_type: str, names: Set[str]) -> bool:
    """Whether a declared (name, type) is present in a live exposition's names.

    Presence is type-aware, mirroring what a Prometheus scrape stores: gauges
    by bare name, counters by bare or ``_total``-suffixed name, histograms by
    their ``_bucket``/``_count``/``_sum`` series.
    """
    if metric_type == "histogram":
        return all(f"{name}{suffix}" in names for suffix in ("_bucket", "_count", "_sum"))
    if metric_type == "counter":
        return name in names or f"{name}_total" in names
    return name in names


def in_golden(name: str, metric_type: str, golden: Dict[str, str]) -> bool:
    """Whether a declared (name, type) resolves against a golden family map."""
    if metric_type == "counter":
        return golden.get(name) == "counter" or golden.get(f"{name}_total") == "counter"
    return golden.get(name) == metric_type


def golden_path(release_tag: str) -> Path:
    return GOLDEN_DIR / f"{release_tag}.txt"


def load_golden(path: Path) -> Dict[str, str]:
    families: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        name, metric_type = line.split(" ")
        families[name] = metric_type
    return families


def format_golden(families: Dict[str, str], release_tag: str) -> str:
    header = (
        f"# vllm:* metric families exposed by vllm-openai-cpu:{release_tag} under the\n"
        "# e2e launch config (e2e/vllm_cpu_server.sh), captured after one warmup\n"
        "# request. Regenerate against a freshly started server of this release:\n"
        "#   e2e/vllm_cpu_server.sh start <release-tag>\n"
        "#   E2E_VLLM_BASE_URL=http://127.0.0.1:8000 E2E_VLLM_VERSION=<release-tag> \\\n"
        "#     E2E_UPDATE_METRIC_GOLDENS=1 pdm run test:e2e:live -k metric_names\n"
    )
    body = "".join(f"{name} {metric_type}\n" for name, metric_type in sorted(families.items()))
    return f"{header}{body}"
