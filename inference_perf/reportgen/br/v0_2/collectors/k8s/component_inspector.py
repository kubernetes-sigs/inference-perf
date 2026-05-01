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
import hashlib
import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from ._client import detect_namespace, load_clients, selector_string

logger = logging.getLogger(__name__)


_ROLE_LABELS = (
    "llm-d.ai/role",
    "inference.networking.x-k8s.io/role",
    "app.kubernetes.io/component",
)
_VALID_ROLES = {"prefill", "decode", "replica"}

_TP_ARGS = ("--tensor-parallel-size", "-tp")
_DP_ARGS = ("--data-parallel-size", "-dp")
_DP_LOCAL_ARGS = ("--data-parallel-size-local",)
_PP_ARGS = ("--pipeline-parallel-size", "-pp")
_EP_ARGS = ("--enable-expert-parallel",)
_MODEL_ARGS = ("--model",)


class ComponentInspector:
    """Discovers stack components via the K8s API.

    For each matching pod: image -> tool/tool_version, env + args -> native,
    GPU resource requests + node labels -> accelerator, role label -> role.
    Pods sharing the same component label are aggregated into one entry with
    a replicas count.
    """

    def __init__(
        self,
        namespace: Optional[str] = None,
        label_selectors: Optional[Dict[str, str]] = None,
    ) -> None:
        self.namespace = namespace or detect_namespace()
        self.label_selectors = label_selectors or {}

    def inspect(self) -> List[Dict[str, Any]]:
        if not self.namespace:
            logger.warning("ComponentInspector: no namespace; returning empty stack")
            return []

        core_v1, _ = load_clients()
        selector = selector_string(self.label_selectors) if self.label_selectors else ""
        pods = core_v1.list_namespaced_pod(self.namespace, label_selector=selector).items

        components_by_label: Dict[str, Dict[str, Any]] = {}
        node_labels_cache: Dict[str, Dict[str, str]] = {}
        for pod in pods:
            label = self._component_label(pod)
            if label in components_by_label:
                components_by_label[label]["_replicas"] += 1
                continue
            node_name = pod.spec.node_name
            node_labels = node_labels_cache.get(node_name) if node_name else None
            if node_name and node_labels is None:
                node_labels = self._read_node_labels(core_v1, node_name)
                node_labels_cache[node_name] = node_labels or {}
            component = self._component_from_pod(pod, label, node_labels or {})
            component["_replicas"] = 1
            components_by_label[label] = component

        finalized: List[Dict[str, Any]] = []
        for component in components_by_label.values():
            component["standardized"]["replicas"] = component.pop("_replicas")
            component["metadata"]["cfg_id"] = self._cfg_id(component["native"])
            finalized.append(component)
        return finalized

    @staticmethod
    def _read_node_labels(core_v1: Any, node_name: str) -> Dict[str, str]:
        try:
            node = core_v1.read_node(node_name)
            return dict(node.metadata.labels or {})
        except Exception as e:
            logger.warning("ComponentInspector: failed to read node %s: %s", node_name, e)
            return {}

    @classmethod
    def _component_from_pod(cls, pod: Any, label: str, node_labels: Dict[str, str]) -> Dict[str, Any]:
        container = pod.spec.containers[0]
        image = container.image or ""
        args, envars = _extract_args_envars(container)
        accelerator = _extract_accelerator(container, node_labels, args)
        role = _extract_role(pod)
        model_name = _extract_model_name(args)

        return {
            "metadata": {
                "schema_version": "0.0.1",
                "label": label,
            },
            "standardized": {
                "kind": "inference_engine",
                "tool": _image_repo(image),
                "tool_version": image,
                "role": role,
                "model": {"name": model_name} if model_name else None,
                "accelerator": accelerator,
            },
            "native": {
                "args": args,
                "envars": envars,
            },
        }

    @staticmethod
    def _component_label(pod: Any) -> str:
        labels = pod.metadata.labels or {}
        for key in ("app.kubernetes.io/instance", "app.kubernetes.io/name", "app"):
            if labels.get(key):
                return str(labels[key])
        return str(pod.metadata.name)

    @staticmethod
    def _cfg_id(native: Dict[str, Any]) -> str:
        payload = json.dumps(native, sort_keys=True, default=str).encode("utf-8")
        return hashlib.md5(payload).hexdigest()


def _image_repo(image: str) -> str:
    no_tag = image.split("@")[0].split(":")[0]
    return no_tag.rsplit("/", 1)[-1] or no_tag or "unknown"


def _extract_args_envars(container: Any) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    args: Dict[str, Any] = {}
    cli = list(container.command or []) + list(container.args or [])
    i = 0
    while i < len(cli):
        token = cli[i]
        if not isinstance(token, str) or not token.startswith("-"):
            i += 1
            continue
        if "=" in token:
            flag, value = token.split("=", 1)
            args[flag] = value
            i += 1
            continue
        if i + 1 < len(cli) and not str(cli[i + 1]).startswith("-"):
            args[token] = cli[i + 1]
            i += 2
        else:
            args[token] = None
            i += 1

    envars: Dict[str, Any] = {}
    for env in container.env or []:
        if getattr(env, "value", None) is not None:
            envars[env.name] = env.value
        elif getattr(env, "value_from", None) is not None:
            envars[env.name] = str(env.value_from)
    return args, envars


def _extract_accelerator(
    container: Any,
    node_labels: Dict[str, str],
    args: Dict[str, Any],
) -> Optional[Dict[str, Any]]:
    requests = (container.resources.requests if container.resources else None) or {}
    limits = (container.resources.limits if container.resources else None) or {}
    gpu_count = _parse_int(requests.get("nvidia.com/gpu") or limits.get("nvidia.com/gpu"))
    if not gpu_count:
        return None
    return {
        "model": node_labels.get("nvidia.com/gpu.product") or node_labels.get("gpu.nvidia.com/class") or "unknown",
        "count": gpu_count,
        "parallelism": _extract_parallelism(args),
    }


def _extract_parallelism(args: Dict[str, Any]) -> Dict[str, int]:
    return {
        "tp": _arg_int(args, _TP_ARGS, default=1),
        "dp": _arg_int(args, _DP_ARGS, default=1),
        "dp_local": _arg_int(args, _DP_LOCAL_ARGS, default=1),
        "pp": _arg_int(args, _PP_ARGS, default=1),
        "workers": 1,
        "ep": 1 if any(a in args for a in _EP_ARGS) else 1,
    }


def _extract_role(pod: Any) -> str:
    labels = pod.metadata.labels or {}
    for key in _ROLE_LABELS:
        value = labels.get(key)
        if value and str(value).lower() in _VALID_ROLES:
            return str(value).lower()
    return "replica"


def _extract_model_name(args: Dict[str, Any]) -> Optional[str]:
    for k in _MODEL_ARGS:
        value = args.get(k)
        if value:
            return str(value)
    return None


def _parse_int(value: Any) -> int:
    if value is None:
        return 0
    if isinstance(value, int):
        return value
    m = re.match(r"^\d+", str(value))
    return int(m.group(0)) if m else 0


def _arg_int(args: Dict[str, Any], keys: Tuple[str, ...], default: int) -> int:
    for k in keys:
        if k in args:
            return _parse_int(args[k]) or default
    return default
