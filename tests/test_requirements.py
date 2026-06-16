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
"""Pure unit tests for requirement inference and node matching.

Intentionally NOT marked ``live``: these run in default CI with no cluster, which
is the whole point of splitting inference (testable anywhere) from matching
(needs a cluster).
"""

from __future__ import annotations

import sys
import textwrap
from pathlib import Path

import pytest

# The live-tier harness package lives under tests/optional/ (it is e2e
# infrastructure). Its requirement-inference logic is pure and unit-tested here,
# so put that dir on the path to import `harness` by absolute name. Done in-file
# rather than via a tests/conftest.py to avoid a second `conftest` module, which
# `mypy --strict ./tests` rejects as a duplicate.
sys.path.insert(0, str(Path(__file__).parent / "optional"))

from harness import requirements  # noqa: E402
from harness.requirements import Requirement  # noqa: E402

# A two-term requirement (GFD OR GKE) like the real case manifests use.
_GPU_REQ: Requirement = [
    (("nvidia.com/gpu.product", "In", ("NVIDIA-H100-80GB-HBM3",)),),
    (("cloud.google.com/gke-accelerator", "In", ("nvidia-h100-80gb", "nvidia-h100-mega-80gb")),),
]


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "vllm.yaml"
    path.write_text(textwrap.dedent(body))
    return path


_AFFINITY_YAML = """
apiVersion: apps/v1
kind: Deployment
spec:
  template:
    spec:
      affinity:
        nodeAffinity:
          requiredDuringSchedulingIgnoredDuringExecution:
            nodeSelectorTerms:
            - matchExpressions:
              - key: nvidia.com/gpu.product
                operator: In
                values: [NVIDIA-H100-80GB-HBM3]
            - matchExpressions:
              - key: cloud.google.com/gke-accelerator
                operator: In
                values: [nvidia-h100-80gb, nvidia-h100-mega-80gb]
---
apiVersion: v1
kind: Service
spec:
  ports: [{port: 8000}]
"""


def test_infers_node_affinity_from_deployment(tmp_path: Path) -> None:
    assert requirements.infer_node_affinity(_write(tmp_path, _AFFINITY_YAML)) == _GPU_REQ


def test_no_affinity_is_empty(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path,
        """
        apiVersion: v1
        kind: Pod
        spec:
          containers: [{name: c, image: busybox}]
        """,
    )
    assert requirements.infer_node_affinity(manifest) == []


def test_conflicting_affinity_raises(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path,
        """
        apiVersion: v1
        kind: Pod
        spec:
          affinity:
            nodeAffinity:
              requiredDuringSchedulingIgnoredDuringExecution:
                nodeSelectorTerms:
                - matchExpressions: [{key: nvidia.com/gpu.product, operator: In, values: [NVIDIA-L4]}]
        ---
        apiVersion: apps/v1
        kind: Deployment
        spec:
          template:
            spec:
              affinity:
                nodeAffinity:
                  requiredDuringSchedulingIgnoredDuringExecution:
                    nodeSelectorTerms:
                    - matchExpressions: [{key: nvidia.com/gpu.product, operator: In, values: [NVIDIA-A100-SXM4-80GB]}]
        """,
    )
    with pytest.raises(ValueError):
        requirements.infer_node_affinity(manifest)


def test_unsupported_operator_raises(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path,
        """
        apiVersion: v1
        kind: Pod
        spec:
          affinity:
            nodeAffinity:
              requiredDuringSchedulingIgnoredDuringExecution:
                nodeSelectorTerms:
                - matchExpressions: [{key: gpu-count, operator: Gt, values: ["0"]}]
        """,
    )
    with pytest.raises(ValueError):
        requirements.infer_node_affinity(manifest)


@pytest.mark.parametrize(
    "labels, matches",
    [
        ({"cloud.google.com/gke-accelerator": "nvidia-h100-80gb", "zone": "us"}, True),  # GKE term
        ({"cloud.google.com/gke-accelerator": "nvidia-h100-mega-80gb"}, True),  # GKE alt value
        ({"nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3"}, True),  # GFD term
        ({"nvidia.com/gpu.product": "NVIDIA-L4"}, False),  # wrong SKU
        ({"cloud.google.com/gke-accelerator": "nvidia-l4"}, False),  # wrong SKU
        ({"zone": "us"}, False),  # not a GPU node
    ],
)
def test_node_matches_ord_terms(labels: dict[str, str], matches: bool) -> None:
    assert requirements.node_matches(labels, _GPU_REQ) is matches


def test_empty_requirement_matches_any_node() -> None:
    assert requirements.node_matches({"anything": "x"}, []) is True


def test_expressions_within_a_term_are_anded() -> None:
    # One term, two expressions: a node must satisfy both.
    req: Requirement = [
        (
            ("nvidia.com/gpu.product", "In", ("NVIDIA-H100-80GB-HBM3",)),
            ("topology.kubernetes.io/zone", "In", ("us-east1",)),
        )
    ]
    assert requirements.node_matches(
        {"nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3", "topology.kubernetes.io/zone": "us-east1"}, req
    )
    # GPU matches but zone does not -> term fails.
    assert not requirements.node_matches(
        {"nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3", "topology.kubernetes.io/zone": "eu-west1"}, req
    )


@pytest.mark.parametrize(
    "operator, values, labels, matches",
    [
        ("NotIn", ("NVIDIA-L4",), {"nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3"}, True),
        ("NotIn", ("NVIDIA-L4",), {"nvidia.com/gpu.product": "NVIDIA-L4"}, False),
        ("NotIn", ("NVIDIA-L4",), {"zone": "us"}, True),  # absent key satisfies NotIn
        ("Exists", (), {"nvidia.com/gpu.product": "anything"}, True),
        ("Exists", (), {"zone": "us"}, False),
        ("DoesNotExist", (), {"zone": "us"}, True),
        ("DoesNotExist", (), {"nvidia.com/gpu.product": "x"}, False),
    ],
)
def test_match_operators(operator: str, values: tuple[str, ...], labels: dict[str, str], matches: bool) -> None:
    req: Requirement = [(("nvidia.com/gpu.product", operator, values),)]
    assert requirements.node_matches(labels, req) is matches


def test_matching_node_count(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_nodes = [
        {"cloud.google.com/gke-accelerator": "nvidia-h100-80gb"},  # match (GKE)
        {"nvidia.com/gpu.product": "NVIDIA-H100-80GB-HBM3"},  # match (GFD)
        {"nvidia.com/gpu.product": "NVIDIA-L4"},  # wrong SKU
        {"zone": "us"},  # not a GPU node
    ]
    monkeypatch.setattr(requirements, "_list_node_labels", lambda kubeconfig: fake_nodes)
    assert requirements.matching_node_count(_GPU_REQ, None) == 2
    # A requirement no node satisfies -> 0 -> the fixture skips the case.
    a100: Requirement = [(("nvidia.com/gpu.product", "In", ("NVIDIA-A100-SXM4-80GB",)),)]
    assert requirements.matching_node_count(a100, None) == 0


def test_deployment_names_read_from_manifest_in_order(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path,
        """
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: qwen3-text-vllm-deployment
        ---
        apiVersion: v1
        kind: Service
        metadata:
          name: qwen3-text-vllm-service
        ---
        apiVersion: apps/v1
        kind: Deployment
        metadata:
          name: sidecar-deployment
        """,
    )
    # Only Deployments, in document order; the Service is ignored. This is what
    # lets a suite name its server anything without the harness hardcoding it.
    assert requirements.deployment_names(manifest) == ["qwen3-text-vllm-deployment", "sidecar-deployment"]


def test_deployment_names_empty_when_none(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path,
        """
        apiVersion: v1
        kind: Pod
        metadata:
          name: standalone
        spec:
          containers: [{name: c, image: busybox}]
        """,
    )
    assert requirements.deployment_names(manifest) == []
