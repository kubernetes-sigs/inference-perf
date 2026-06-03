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
"""Pytest plumbing for the optional (live-backend) test tier.

Flow per live test:
  infer nodeSelector from the case manifest
    -> find supplied clusters with >=1 matching node
    -> skip loudly if none
    -> acquire a slot in that (cluster, nodeSelector) class (capacity = node count)
    -> hand the test a Cluster handle (kubeconfig + fresh namespace)
    -> on teardown, delete the namespace (frees the GPU) and release the slot.
"""
from __future__ import annotations

import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

# conftest is loaded as a top-level module, so make the local harness package
# importable by absolute name for both this file and the test modules.
sys.path.insert(0, str(Path(__file__).parent))

from harness import requirements, slots  # noqa: E402
from harness.runner import Cluster  # noqa: E402


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--kubeconfigs",
        default="",
        help="Comma-separated kubeconfig paths; each is one candidate cluster. "
        "Empty means use the ambient kubeconfig as the single cluster.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live: requires a real model-server backend on a matching cluster "
        "(excluded from default CI via -m 'not live').",
    )


@pytest.fixture(scope="session")
def kubeconfigs(request: pytest.FixtureRequest) -> list[str | None]:
    raw = str(request.config.getoption("--kubeconfigs")).strip()
    return [p.strip() for p in raw.split(",") if p.strip()] or [None]


@pytest.fixture
def cluster_for_case(
    request: pytest.FixtureRequest, kubeconfigs: list[str | None]
) -> Iterator[Cluster]:
    """Match the case's manifest to a cluster, serialize on scarce hardware, clean up.

    The case's manifest path is supplied via indirect parametrize, e.g.:

        @pytest.mark.live
        @pytest.mark.parametrize(
            "cluster_for_case", sorted(CASES_DIR.glob("*/vllm.yaml")), indirect=True
        )
        def test_case(cluster_for_case): ...
    """
    manifest_path = request.param
    node_selector = requirements.infer_node_selector(manifest_path)

    # Live-nodes-only matching. capacity is the contention-class size.
    candidates = [
        (kc, requirements.matching_node_count(node_selector, kc)) for kc in kubeconfigs
    ]
    usable = [(kc, n) for kc, n in candidates if n > 0]
    if not usable:
        pytest.skip(
            f"no supplied cluster has a live node matching nodeSelector "
            f"{node_selector or '{}'} (scaled-to-zero pools are not detected)"
        )

    kubeconfig, capacity = usable[0]
    key = slots.class_key(kubeconfig, node_selector)
    namespace = f"inference-perf-e2e-{uuid.uuid4().hex[:8]}"

    with slots.acquire_slot(key, capacity):
        _kubectl(kubeconfig, "create", "namespace", namespace)
        try:
            yield Cluster(
                kubeconfig=kubeconfig, namespace=namespace, manifest_path=Path(manifest_path)
            )
        finally:
            # Release == delete the namespace so the next queued test in this
            # class can schedule onto the freed node. Non-negotiable for reuse.
            _kubectl(kubeconfig, "delete", "namespace", namespace, "--wait=true", check=False)


def _kubectl(kubeconfig: str | None, *args: str, check: bool = True) -> None:
    import subprocess

    cmd = ["kubectl", *args]
    if kubeconfig:
        cmd[1:1] = ["--kubeconfig", kubeconfig]
    subprocess.run(cmd, check=check)
