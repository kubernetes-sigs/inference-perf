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
  infer the required nodeAffinity from the case manifest
    -> find supplied clusters with >=1 live node satisfying it (any OR'd term)
    -> skip loudly if none
    -> acquire a slot in that (cluster, requirement) class (capacity = node count)
    -> hand the test a Cluster handle (kubeconfig + fresh namespace)
    -> on teardown, delete the namespace (frees the GPU) and release the slot.
"""

from __future__ import annotations

import json
import os
import subprocess
import sys
import uuid
from collections.abc import Iterator
from pathlib import Path

import pytest

# conftest is loaded as a top-level module, so make the local harness package
# importable by absolute name for both this file and the test modules.
sys.path.insert(0, str(Path(__file__).parent))

from harness import requirements, slots  # noqa: E402
from harness.runner import DEFAULT_IMAGE, Cluster  # noqa: E402

# Namespace naming, shared by per-case creation and the orphan sweep.
NAMESPACE_PREFIX = "inference-perf-e2e"
_IMAGE_ENV = "INFERENCE_PERF_IMAGE"


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--kubeconfigs",
        default="",
        help="Comma-separated kubeconfig paths; each is one candidate cluster. "
        "Empty means use the ambient kubeconfig as the single cluster.",
    )
    parser.addoption(
        "--image",
        default="",
        help=f"inference-perf image for the job (default: ${_IMAGE_ENV} or {DEFAULT_IMAGE}).",
    )
    parser.addoption(
        "--sweep-orphan-namespaces",
        action="store_true",
        default=False,
        help=f"Before running, delete leftover {NAMESPACE_PREFIX}-* namespaces from "
        "killed prior runs. Off by default; unsafe alongside concurrent runs.",
    )


def pytest_configure(config: pytest.Config) -> None:
    config.addinivalue_line(
        "markers",
        "live: requires a real model-server backend on a matching cluster "
        "(auto-skipped unless --kubeconfigs is passed; also excludable via -m 'not live').",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """Auto-skip the live tier unless the run explicitly points at clusters.

    A live test needs a real backend, so without --kubeconfigs it can only error
    (no kubectl, no cluster). Skipping rather than erroring keeps the default
    `pytest tests` / `pdm run test` / coverage runs green without every caller
    having to remember `-m "not live"`. Passing --kubeconfigs opts the tier in.
    """
    if str(config.getoption("--kubeconfigs")).strip():
        return
    skip_live = pytest.mark.skip(reason="live tier: pass --kubeconfigs=<path,...> to run live-backend tests")
    for item in items:
        if "live" in item.keywords:
            item.add_marker(skip_live)


@pytest.fixture(scope="session")
def kubeconfigs(request: pytest.FixtureRequest) -> list[str | None]:
    raw = str(request.config.getoption("--kubeconfigs")).strip()
    return [p.strip() for p in raw.split(",") if p.strip()] or [None]


@pytest.fixture(scope="session")
def image(request: pytest.FixtureRequest) -> str:
    return str(request.config.getoption("--image")) or os.environ.get(_IMAGE_ENV, "") or DEFAULT_IMAGE


@pytest.fixture(scope="session", autouse=True)
def _sweep_orphan_namespaces(request: pytest.FixtureRequest, kubeconfigs: list[str | None]) -> None:
    """Reclaim namespaces leaked by killed prior runs (opt-in via flag).

    Per-case teardown already deletes namespaces on a clean exit; this covers
    hard kills (SIGKILL, OOM, CI timeout) where teardown never ran.
    """
    if not request.config.getoption("--sweep-orphan-namespaces"):
        return
    for kubeconfig in kubeconfigs:
        raw = _kubectl(kubeconfig, "get", "namespaces", "-o", "json", capture=True, check=False)
        if not raw.strip():
            continue
        names = [ns.get("metadata", {}).get("name", "") for ns in json.loads(raw).get("items", [])]
        orphans = [n for n in names if n.startswith(f"{NAMESPACE_PREFIX}-")]
        for name in orphans:
            _kubectl(kubeconfig, "delete", "namespace", name, "--wait=false", check=False)


@pytest.fixture
def cluster_for_case(request: pytest.FixtureRequest, kubeconfigs: list[str | None]) -> Iterator[Cluster]:
    """Match the case's manifest to a cluster, serialize on scarce hardware, clean up.

    The case's manifest path is supplied via indirect parametrize, e.g.:

        @pytest.mark.live
        @pytest.mark.parametrize(
            "cluster_for_case", sorted(CASES_DIR.glob("*/vllm.yaml")), indirect=True
        )
        def test_case(cluster_for_case): ...
    """
    manifest_path = request.param
    requirement = requirements.infer_node_affinity(manifest_path)

    # Live-nodes-only matching. capacity is the contention-class size.
    candidates = [(kc, requirements.matching_node_count(requirement, kc)) for kc in kubeconfigs]
    usable = [(kc, n) for kc, n in candidates if n > 0]
    if not usable:
        pytest.skip(
            f"no supplied cluster has a live node satisfying "
            f"{requirements.describe(requirement)} (scaled-to-zero pools are not detected)"
        )

    kubeconfig, capacity = usable[0]
    key = slots.class_key(kubeconfig, repr(sorted(requirement)))
    namespace = f"{NAMESPACE_PREFIX}-{uuid.uuid4().hex[:8]}"

    with slots.acquire_slot(key, capacity):
        _kubectl(kubeconfig, "create", "namespace", namespace)
        try:
            yield Cluster(kubeconfig=kubeconfig, namespace=namespace, manifest_path=Path(manifest_path))
        finally:
            # Release == delete the namespace so the next queued test in this
            # class can schedule onto the freed node. Non-negotiable for reuse.
            _kubectl(kubeconfig, "delete", "namespace", namespace, "--wait=true", check=False)


def _kubectl(kubeconfig: str | None, *args: str, capture: bool = False, check: bool = True) -> str:
    cmd = ["kubectl", *args]
    if kubeconfig:
        cmd[1:1] = ["--kubeconfig", kubeconfig]
    result = subprocess.run(cmd, capture_output=capture, text=True, check=check)
    return result.stdout if capture else ""
