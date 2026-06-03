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
"""Live multimodal end-to-end tests, one per case under manual/multimodal/cases.

Each case is parametrized by its vllm.yaml. The cluster_for_case fixture (see
conftest.py) infers the case's nodeSelector from that manifest, skips when no
supplied --kubeconfigs cluster has a matching live node, and otherwise serializes
contending cases onto the available nodes. Run with:

    pytest tests/optional/test_multimodal.py -m live --kubeconfigs=/path/to/kubeconfig
"""
from __future__ import annotations

from pathlib import Path

import pytest

from harness import runner
from harness.runner import Cluster

CASES_DIR = Path(__file__).parent / "manual" / "multimodal" / "cases"
CASE_MANIFESTS = sorted(CASES_DIR.glob("*/vllm.yaml"))


@pytest.mark.live
@pytest.mark.parametrize(
    "cluster_for_case",
    CASE_MANIFESTS,
    ids=[m.parent.name for m in CASE_MANIFESTS],
    indirect=True,
)
def test_multimodal_case(cluster_for_case: Cluster) -> None:
    case_dir = cluster_for_case.manifest_path.parent
    runner.run_case(cluster_for_case.kubeconfig, cluster_for_case.namespace, case_dir)
