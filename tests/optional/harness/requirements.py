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
"""Infer a test's hardware requirement from its manifest and match it to clusters.

The manifest is the single source of truth: a case's hardware requirement is just
the ``nodeSelector`` on its pod-bearing objects. Matching a cluster is then the
same question the scheduler asks, restricted to label selection: does this cluster
have at least one node whose labels are a superset of that nodeSelector.

v1 scope (deliberately narrow): nodeSelector only (no affinity, no resource
accounting), and matching is live-nodes-only (a node pool scaled to zero is
invisible and reads as "no hardware").
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

import yaml

# Kinds that carry a pod template (or are a pod). Everything else (Service,
# ConfigMap, ...) has no scheduling constraints and is ignored.
_POD_BEARING_KINDS = {"Pod", "Deployment", "StatefulSet", "Job", "DaemonSet", "ReplicaSet"}

NodeSelector = dict[str, str]


def _pod_spec(doc: dict[str, object]) -> dict[str, object] | None:
    kind = doc.get("kind")
    if kind == "Pod":
        spec = doc.get("spec")
        return spec if isinstance(spec, dict) else None
    if kind in _POD_BEARING_KINDS:
        spec = doc.get("spec")
        template = spec.get("template") if isinstance(spec, dict) else None
        pod = template.get("spec") if isinstance(template, dict) else None
        return pod if isinstance(pod, dict) else None
    return None


def infer_node_selector(manifest_path: str | Path) -> NodeSelector:
    """Return the merged nodeSelector across all pod-bearing docs in a manifest.

    Raises ValueError if two pods in the same manifest disagree on a label, since
    that would make the case's hardware requirement ambiguous.
    """
    selector: NodeSelector = {}
    text = Path(manifest_path).read_text()
    for doc in yaml.safe_load_all(text):
        if not isinstance(doc, dict):
            continue
        pod = _pod_spec(doc)
        if pod is None:
            continue
        node_selector = pod.get("nodeSelector") or {}
        if not isinstance(node_selector, dict):
            continue
        for key, value in node_selector.items():
            existing = selector.get(key)
            if existing is not None and existing != value:
                raise ValueError(
                    f"{manifest_path}: conflicting nodeSelector for {key!r}: "
                    f"{existing!r} vs {value!r}"
                )
            selector[str(key)] = str(value)
    return selector


def node_fits(node_selector: NodeSelector, node_labels: dict[str, str]) -> bool:
    """A node fits if its labels satisfy every nodeSelector key (subset match)."""
    return all(node_labels.get(key) == value for key, value in node_selector.items())


def _list_node_labels(kubeconfig: str | None) -> list[dict[str, str]]:
    cmd = ["kubectl", "get", "nodes", "-o", "json"]
    if kubeconfig:
        cmd[1:1] = ["--kubeconfig", kubeconfig]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    return [item.get("metadata", {}).get("labels", {}) or {} for item in data.get("items", [])]


def matching_node_count(node_selector: NodeSelector, kubeconfig: str | None) -> int:
    """Number of live nodes in the cluster that satisfy the nodeSelector.

    Doubles as the contention-class capacity: 0 means skip, N>0 means up to N of
    these tests can run concurrently before they have to queue.
    """
    return sum(1 for labels in _list_node_labels(kubeconfig) if node_fits(node_selector, labels))
