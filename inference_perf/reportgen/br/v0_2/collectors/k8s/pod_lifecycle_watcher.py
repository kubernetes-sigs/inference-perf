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
import threading
from typing import Any, Dict, List, Optional

from ._client import detect_namespace, load_clients, selector_string

logger = logging.getLogger(__name__)


class PodLifecycleWatcher:
    """Watches pods in a namespace and records creationTimestamp + the moment
    the Ready condition transitions to True. Output feeds
    observability.pod_startup_times.

    Uses the kubernetes Watch API on a background thread (the python client is
    sync only); start()/stop() bridge it to asyncio.
    """

    def __init__(
        self,
        namespace: Optional[str] = None,
        label_selectors: Optional[Dict[str, str]] = None,
    ) -> None:
        self.namespace = namespace or detect_namespace()
        self.label_selectors = label_selectors or {}
        self._thread: Optional[threading.Thread] = None
        self._stop_event = threading.Event()
        self._watch: Any = None
        self._lock = threading.Lock()
        # Per-pod state: name -> {creation_timestamp, ready_timestamp, role, model, node}
        self._pods: Dict[str, Dict[str, Any]] = {}

    async def start(self) -> None:
        if self.namespace is None or self._thread is not None:
            return
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._watch_loop, name="br-v0-2-pod-watcher", daemon=True)
        self._thread.start()

    async def stop(self) -> None:
        if self._thread is None:
            return
        self._stop_event.set()
        watch = self._watch
        if watch is not None:
            try:
                watch.stop()
            except Exception:
                pass
        # Don't block on join more than briefly; the thread is daemon.
        await asyncio.to_thread(self._thread.join, 5.0)
        self._thread = None

    def snapshot(self) -> List[Dict[str, Any]]:
        with self._lock:
            return [_pod_record(name, state) for name, state in self._pods.items()]

    def _watch_loop(self) -> None:
        try:
            from kubernetes import watch

            core_v1, _ = load_clients()
        except Exception as e:
            logger.warning("PodLifecycleWatcher: cannot start K8s watch: %s", e)
            return

        selector = selector_string(self.label_selectors) if self.label_selectors else ""
        self._watch = watch.Watch()
        try:
            for event in self._watch.stream(
                core_v1.list_namespaced_pod,
                self.namespace,
                label_selector=selector,
                timeout_seconds=0,
            ):
                if self._stop_event.is_set():
                    break
                pod = event.get("object")
                if pod is None:
                    continue
                self._record(pod)
        except Exception as e:
            if not self._stop_event.is_set():
                logger.warning("PodLifecycleWatcher: stream ended: %s", e)
        finally:
            self._watch = None

    def _record(self, pod: Any) -> None:
        labels = pod.metadata.labels or {}
        name = pod.metadata.name
        creation = pod.metadata.creation_timestamp
        ready_ts = _ready_transition_time(pod)
        with self._lock:
            state = self._pods.setdefault(
                name,
                {
                    "creation_timestamp": creation,
                    "ready_timestamp": None,
                    "role": labels.get("llm-d.ai/role") or labels.get("app.kubernetes.io/component"),
                    "model": labels.get("llm-d.ai/model") or labels.get("inference-perf.x-k8s.io/model"),
                    "node": pod.spec.node_name,
                },
            )
            if state["ready_timestamp"] is None and ready_ts is not None:
                state["ready_timestamp"] = ready_ts


def _ready_transition_time(pod: Any) -> Optional[datetime.datetime]:
    if not pod.status or not pod.status.conditions:
        return None
    for cond in pod.status.conditions:
        if cond.type == "Ready" and cond.status == "True":
            ts = cond.last_transition_time
            return ts if isinstance(ts, datetime.datetime) else None
    return None


def _pod_record(name: str, state: Dict[str, Any]) -> Dict[str, Any]:
    creation = state.get("creation_timestamp")
    ready = state.get("ready_timestamp")
    startup_seconds = None
    if creation is not None and ready is not None:
        startup_seconds = max(0.0, (ready - creation).total_seconds())
    return {
        "name": name,
        "model": state.get("model"),
        "role": state.get("role"),
        "node": state.get("node"),
        "creation_timestamp": creation.isoformat() if creation else None,
        "ready_timestamp": ready.isoformat() if ready else None,
        "startup_seconds": startup_seconds,
    }
