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
import datetime
import logging
from typing import Any, Dict, List, Optional

from ._client import detect_namespace, load_clients, selector_string

logger = logging.getLogger(__name__)


class ReplicaStatusSampler:
    """Polls Deployments + StatefulSets in a namespace at a fixed interval and
    snapshots desired/available/ready/updated counts. Output feeds
    observability.replica_status.
    """

    def __init__(
        self,
        namespace: Optional[str] = None,
        label_selectors: Optional[Dict[str, str]] = None,
        interval_seconds: float = 15.0,
    ) -> None:
        self.namespace = namespace or detect_namespace()
        self.label_selectors = label_selectors or {}
        self.interval_seconds = max(1.0, interval_seconds)
        self._task: Optional[asyncio.Task[None]] = None
        self._snapshots: List[Dict[str, Any]] = []
        self._latest: List[Dict[str, Any]] = []

    async def start(self) -> None:
        if self.namespace is None or self._task is not None:
            return
        self._task = asyncio.create_task(self._loop())

    async def stop(self) -> None:
        if self._task is None:
            return
        self._task.cancel()
        try:
            await self._task
        except asyncio.CancelledError:
            pass
        self._task = None

    def snapshots(self) -> List[Dict[str, Any]]:
        return list(self._snapshots)

    def latest(self) -> List[Dict[str, Any]]:
        return list(self._latest)

    async def _loop(self) -> None:
        try:
            _, apps_v1 = await asyncio.to_thread(load_clients)
        except Exception as e:
            logger.warning("ReplicaStatusSampler: cannot load K8s client: %s", e)
            return
        selector = selector_string(self.label_selectors) if self.label_selectors else ""
        while True:
            try:
                snapshot = await asyncio.to_thread(self._sample, apps_v1, selector)
                self._snapshots.append(snapshot)
                self._latest = snapshot["controllers"]
            except asyncio.CancelledError:
                raise
            except Exception as e:
                logger.warning("ReplicaStatusSampler: sample failed: %s", e)
            await asyncio.sleep(self.interval_seconds)

    def _sample(self, apps_v1: Any, selector: str) -> Dict[str, Any]:
        controllers: List[Dict[str, Any]] = []
        for d in apps_v1.list_namespaced_deployment(self.namespace, label_selector=selector).items:
            controllers.append(_controller_snapshot("Deployment", d))
        for s in apps_v1.list_namespaced_stateful_set(self.namespace, label_selector=selector).items:
            controllers.append(_controller_snapshot("StatefulSet", s))
        return {
            "timestamp": datetime.datetime.now(datetime.timezone.utc).isoformat(),
            "namespace": self.namespace,
            "controllers": controllers,
        }


def _controller_snapshot(kind: str, obj: Any) -> Dict[str, Any]:
    status = obj.status
    spec = obj.spec
    labels = obj.metadata.labels or {}
    return {
        "kind": kind,
        "name": obj.metadata.name,
        "model": labels.get("llm-d.ai/model") or labels.get("inference-perf.x-k8s.io/model"),
        "role": labels.get("llm-d.ai/role") or labels.get("app.kubernetes.io/component"),
        "desired_replicas": getattr(spec, "replicas", None) or 0,
        "available_replicas": getattr(status, "available_replicas", None) or 0,
        "ready_replicas": getattr(status, "ready_replicas", None) or 0,
        "updated_replicas": getattr(status, "updated_replicas", None) or 0,
    }
