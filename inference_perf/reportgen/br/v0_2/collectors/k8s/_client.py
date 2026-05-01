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
import os
from typing import Any, Optional, Tuple


_NAMESPACE_FILE = "/var/run/secrets/kubernetes.io/serviceaccount/namespace"


def in_cluster() -> bool:
    return bool(os.environ.get("KUBERNETES_SERVICE_HOST"))


def detect_namespace(fallback: Optional[str] = None) -> Optional[str]:
    try:
        with open(_NAMESPACE_FILE, "r", encoding="utf-8") as f:
            return f.read().strip() or fallback
    except OSError:
        return fallback


class KubernetesUnavailableError(RuntimeError):
    """Raised when the `kubernetes` extra is not installed."""


def load_clients() -> Tuple[Any, Any]:
    """Load the K8s client. In-cluster config first, then ~/.kube/config.

    Returns (CoreV1Api, AppsV1Api). Raises KubernetesUnavailableError if the
    `kubernetes` package isn't installed; raises ConfigException on auth failure.
    """
    try:
        from kubernetes import client, config
    except ImportError as e:
        raise KubernetesUnavailableError(
            "BR0.2 KubernetesStackCollector requires the 'k8s' extra: pip install 'inference-perf[k8s]'"
        ) from e

    if in_cluster():
        config.load_incluster_config()
    else:
        config.load_kube_config()
    return client.CoreV1Api(), client.AppsV1Api()


def selector_string(label_selectors: dict[str, str]) -> str:
    return ",".join(f"{k}={v}" for k, v in label_selectors.items())
