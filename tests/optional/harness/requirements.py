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

The manifest is the single source of truth: a case's hardware requirement is the
required ``nodeAffinity`` on its pod-bearing objects. Affinity (not a plain
nodeSelector) is used so one portable manifest can name a GPU SKU under whatever
label each provider applies: its ``nodeSelectorTerms`` are OR'd, so a term for GFD's
``nvidia.com/gpu.product`` and a term for GKE's ``cloud.google.com/gke-accelerator``
let the same pod schedule on either, with the scheduler enforcing the SKU natively.

Matching a cluster is then the same question the scheduler asks, restricted to label
selection: does this cluster have at least one live node that satisfies the affinity
(any term, all of that term's expressions). The harness applies the manifest as-is
(no rewriting); it only reads the affinity to skip cleanly when no cluster has the
hardware and to size the per-class slot semaphore.

v1 scope (deliberately narrow): required nodeAffinity matchExpressions with the
In/NotIn/Exists/DoesNotExist operators (no matchFields, no preferred affinity, no
resource accounting), and matching is live-nodes-only (a node pool scaled to zero is
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

_SUPPORTED_OPERATORS = {"In", "NotIn", "Exists", "DoesNotExist"}

# A required nodeAffinity, parsed and normalized for matching. ``Requirement`` is a
# list of OR'd terms; each term is an AND of (key, operator, values) expressions. An
# empty Requirement means the case sets no node constraint (matches any node).
MatchExpr = tuple[str, str, tuple[str, ...]]
Term = tuple[MatchExpr, ...]
Requirement = list[Term]


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


def _parse_terms(pod: dict[str, object]) -> Requirement:
    """Pull the required nodeAffinity nodeSelectorTerms out of a pod spec."""
    affinity = pod.get("affinity")
    node_affinity = affinity.get("nodeAffinity") if isinstance(affinity, dict) else None
    required = node_affinity.get("requiredDuringSchedulingIgnoredDuringExecution") if isinstance(node_affinity, dict) else None
    raw_terms = required.get("nodeSelectorTerms") if isinstance(required, dict) else None
    if not isinstance(raw_terms, list):
        return []
    terms: Requirement = []
    for raw_term in raw_terms:
        if not isinstance(raw_term, dict):
            continue
        raw_exprs = raw_term.get("matchExpressions")
        if not isinstance(raw_exprs, list):
            continue
        exprs: list[MatchExpr] = []
        for raw in raw_exprs:
            if not isinstance(raw, dict):
                continue
            key = raw.get("key")
            operator = raw.get("operator")
            if not isinstance(key, str) or not isinstance(operator, str):
                continue
            if operator not in _SUPPORTED_OPERATORS:
                raise ValueError(f"unsupported nodeAffinity operator: {operator!r}")
            raw_values = raw.get("values")
            values = tuple(str(v) for v in raw_values) if isinstance(raw_values, list) else ()
            exprs.append((key, operator, values))
        if exprs:
            terms.append(tuple(exprs))
    return terms


def infer_node_affinity(manifest_path: str | Path) -> Requirement:
    """Return the case's required nodeAffinity, or [] if it sets no constraint.

    Read from the pod-bearing docs. Raises ValueError if two pods carry different
    affinities, since that makes the case's hardware requirement ambiguous; a
    GPU server alongside a CPU-only sidecar (no affinity) is fine.
    """
    found: Requirement | None = None
    text = Path(manifest_path).read_text()
    for doc in yaml.safe_load_all(text):
        if not isinstance(doc, dict):
            continue
        pod = _pod_spec(doc)
        if pod is None:
            continue
        terms = _parse_terms(pod)
        if not terms:
            continue
        if found is not None and found != terms:
            raise ValueError(f"{manifest_path}: conflicting nodeAffinity: {found!r} vs {terms!r}")
        found = terms
    return found or []


def deployment_names(manifest_path: str | Path) -> list[str]:
    """Return the metadata.name of every Deployment in a manifest, in document order.

    The runner waits on these rollouts before driving load, so the server's name
    is read from the manifest rather than hardcoded. This is what lets a new suite
    bring its own Deployment (named anything) without touching the harness.
    """
    names: list[str] = []
    text = Path(manifest_path).read_text()
    for doc in yaml.safe_load_all(text):
        if not isinstance(doc, dict) or doc.get("kind") != "Deployment":
            continue
        metadata = doc.get("metadata")
        name = metadata.get("name") if isinstance(metadata, dict) else None
        if isinstance(name, str) and name:
            names.append(name)
    return names


def _expr_satisfied(node_labels: dict[str, str], expr: MatchExpr) -> bool:
    key, operator, values = expr
    present = key in node_labels
    value = node_labels.get(key)
    if operator == "In":
        return present and value in values
    if operator == "NotIn":
        return not present or value not in values
    if operator == "Exists":
        return present
    if operator == "DoesNotExist":
        return not present
    raise ValueError(f"unsupported nodeAffinity operator: {operator!r}")


def node_matches(node_labels: dict[str, str], requirement: Requirement) -> bool:
    """A node matches if it satisfies any term (terms are OR'd; expressions AND'd).

    An empty requirement matches every node (the case set no node constraint).
    """
    if not requirement:
        return True
    return any(all(_expr_satisfied(node_labels, expr) for expr in term) for term in requirement)


def describe(requirement: Requirement) -> str:
    """A short human description of a requirement, for skip messages."""
    if not requirement:
        return "any node"
    parts = []
    for term in requirement:
        anded = " and ".join(f"{key} {op} {list(values)}" for key, op, values in term)
        parts.append(f"({anded})")
    return " or ".join(parts)


def _list_node_labels(kubeconfig: str | None) -> list[dict[str, str]]:
    cmd = ["kubectl", "get", "nodes", "-o", "json"]
    if kubeconfig:
        cmd[1:1] = ["--kubeconfig", kubeconfig]
    out = subprocess.run(cmd, capture_output=True, text=True, check=True).stdout
    data = json.loads(out)
    return [item.get("metadata", {}).get("labels", {}) or {} for item in data.get("items", [])]


def matching_node_count(requirement: Requirement, kubeconfig: str | None) -> int:
    """Number of live nodes in the cluster that satisfy the requirement.

    Doubles as the contention-class capacity: 0 means skip, N>0 means up to N of
    these tests can run concurrently before they have to queue.
    """
    return sum(1 for labels in _list_node_labels(kubeconfig) if node_matches(labels, requirement))
