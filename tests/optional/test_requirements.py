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

import textwrap
from pathlib import Path

import pytest

from harness import requirements


def _write(tmp_path: Path, body: str) -> Path:
    path = tmp_path / "vllm.yaml"
    path.write_text(textwrap.dedent(body))
    return path


def test_infers_node_selector_from_deployment(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path,
        """
        apiVersion: apps/v1
        kind: Deployment
        spec:
          template:
            spec:
              nodeSelector:
                cloud.google.com/gke-accelerator: nvidia-h100-80gb
        ---
        apiVersion: v1
        kind: Service
        spec:
          ports: [{port: 8000}]
        """,
    )
    assert requirements.infer_node_selector(manifest) == {
        "cloud.google.com/gke-accelerator": "nvidia-h100-80gb"
    }


def test_no_constraint_matches_everything(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path,
        """
        apiVersion: v1
        kind: Pod
        spec:
          containers: [{name: c, image: busybox}]
        """,
    )
    assert requirements.infer_node_selector(manifest) == {}


def test_conflicting_selectors_raise(tmp_path: Path) -> None:
    manifest = _write(
        tmp_path,
        """
        apiVersion: v1
        kind: Pod
        spec:
          nodeSelector: {gpu: h100}
        ---
        apiVersion: apps/v1
        kind: Deployment
        spec:
          template:
            spec:
              nodeSelector: {gpu: a100}
        """,
    )
    with pytest.raises(ValueError):
        requirements.infer_node_selector(manifest)


@pytest.mark.parametrize(
    "selector, labels, fits",
    [
        ({"gpu": "h100"}, {"gpu": "h100", "zone": "us"}, True),  # subset match
        ({"gpu": "h100"}, {"gpu": "a100"}, False),  # wrong value
        ({"gpu": "h100"}, {"zone": "us"}, False),  # missing key
        ({}, {"anything": "x"}, True),  # empty selector matches any node
    ],
)
def test_node_fits(selector: dict[str, str], labels: dict[str, str], fits: bool) -> None:
    assert requirements.node_fits(selector, labels) is fits
