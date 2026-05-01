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
import asyncio
import logging
from typing import TYPE_CHECKING, Any, Dict, List

from ..base import CollectedStackObservability, StackObservabilityCollector
from ._client import detect_namespace, in_cluster, load_clients, selector_string
from .component_inspector import ComponentInspector
from .per_pod_prom_scraper import PerPodPromScraper, PodScrapeTarget
from .pod_lifecycle_watcher import PodLifecycleWatcher
from .replica_status_sampler import ReplicaStatusSampler

if TYPE_CHECKING:
    from inference_perf.config import BRV02Config

logger = logging.getLogger(__name__)

# Default vLLM /metrics port; override via the `prometheus.io/port` pod annotation
# when present.
_DEFAULT_METRICS_PORT = 8000


class KubernetesStackCollector(StackObservabilityCollector):
    """In-cluster BR0.2 collector. Composes:

    - ComponentInspector       -> scenario.stack[]
    - PodLifecycleWatcher      -> observability.pod_startup_times
    - ReplicaStatusSampler     -> observability.replica_status
    - PerPodPromScraper        -> observability vllm_*/epp_* fields

    The `kubernetes` package is imported lazily by each sub-collector so the
    module is importable even without the `k8s` extra.
    """

    def __init__(self, config: "BRV02Config") -> None:
        self.config = config
        self._stack: List[Dict[str, Any]] = []
        k8s = config.kubernetes
        namespace = (k8s.namespace if k8s else None) or detect_namespace()
        label_selectors = (k8s.label_selectors if k8s else None) or {}
        interval = k8s.scrape_interval_seconds if k8s else 15

        self._inspector = ComponentInspector(namespace=namespace, label_selectors=label_selectors)
        self._pod_watcher = PodLifecycleWatcher(namespace=namespace, label_selectors=label_selectors)
        self._replica_sampler = ReplicaStatusSampler(
            namespace=namespace, label_selectors=label_selectors, interval_seconds=interval
        )
        self._prom_scraper = PerPodPromScraper(interval_seconds=interval)
        self._namespace = namespace
        self._label_selectors = label_selectors

    async def start(self) -> None:
        if not in_cluster():
            logger.warning("KubernetesStackCollector: not in-cluster; skipping K8s discovery")
            return
        try:
            self._stack = await asyncio.to_thread(self._inspector.inspect)
        except Exception as e:
            logger.warning("KubernetesStackCollector: inspector failed: %s", e)
            self._stack = []
        try:
            scrape_targets = await asyncio.to_thread(self._discover_scrape_targets)
            self._prom_scraper.set_targets(scrape_targets)
        except Exception as e:
            logger.warning("KubernetesStackCollector: scrape target discovery failed: %s", e)
        await asyncio.gather(
            self._pod_watcher.start(),
            self._replica_sampler.start(),
            self._prom_scraper.start(),
        )

    async def stop(self) -> None:
        await asyncio.gather(
            self._pod_watcher.stop(),
            self._replica_sampler.stop(),
            self._prom_scraper.stop(),
            return_exceptions=True,
        )

    def collect(self) -> CollectedStackObservability:
        observability: Dict[str, Any] = {}
        pod_records = self._pod_watcher.snapshot()
        if pod_records:
            observability["pod_startup_times"] = {"pods": pod_records}
        replica_snapshots = self._replica_sampler.snapshots()
        latest = self._replica_sampler.latest()
        if replica_snapshots:
            observability["replica_status"] = {
                "namespace": self._namespace,
                "controllers": latest,
                "time_series": replica_snapshots,
            }
        observability.update(self._prom_scraper.collect())

        return CollectedStackObservability(
            stack=self._stack,
            observability=observability or None,
        )

    def _discover_scrape_targets(self) -> List[PodScrapeTarget]:
        if not self._namespace:
            return []
        try:
            core_v1, _ = load_clients()
        except Exception as e:
            logger.warning("KubernetesStackCollector: cannot load K8s client: %s", e)
            return []
        selector = selector_string(self._label_selectors) if self._label_selectors else ""
        pods = core_v1.list_namespaced_pod(self._namespace, label_selector=selector).items
        targets: List[PodScrapeTarget] = []
        for pod in pods:
            ip = pod.status.pod_ip if pod.status else None
            if not ip:
                continue
            port = _resolve_metrics_port(pod)
            url = f"http://{ip}:{port}/metrics"
            labels = pod.metadata.labels or {}
            component_label = (
                labels.get("app.kubernetes.io/instance") or labels.get("app.kubernetes.io/name") or pod.metadata.name
            )
            role = labels.get("llm-d.ai/role") or labels.get("app.kubernetes.io/component") or "replica"
            targets.append(PodScrapeTarget(pod_name=pod.metadata.name, component_label=component_label, role=role, url=url))
        return targets


def _resolve_metrics_port(pod: Any) -> int:
    annotations = pod.metadata.annotations or {}
    raw = annotations.get("prometheus.io/port")
    if raw:
        try:
            return int(raw)
        except (TypeError, ValueError):
            pass
    return _DEFAULT_METRICS_PORT
