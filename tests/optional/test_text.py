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
"""Live text-only end-to-end test: a worked example of a non-multimodal suite.

This file is a near-copy of test_multimodal.py and exists to show that the live
harness is not multimodal-specific. Adding a new live suite is exactly this:

  1. drop cases under <suite>/cases/<case>/ (a vllm.yaml + a config.yml), and
  2. parametrize the cluster_for_case fixture by those manifests.

Everything else (nodeSelector inference, cluster matching, the slot semaphore,
namespace lifecycle, report extraction) is shared and untouched. The case here
deploys a differently named Deployment (picked up from the manifest, not from a
hardcoded constant) onto the same GPU class as the multimodal suite, so it
co-schedules on the same nodes instead of skipping for want of a different one.

    pytest tests/optional/test_text.py -m live --kubeconfigs=/path/to/kubeconfig
"""

from __future__ import annotations

from pathlib import Path

import pytest

from harness import runner
from harness.runner import Cluster

CASES_DIR = Path(__file__).parent / "text" / "cases"
CASE_MANIFESTS = sorted(CASES_DIR.glob("*/vllm.yaml"))


@pytest.mark.live
@pytest.mark.parametrize(
    "cluster_for_case",
    CASE_MANIFESTS,
    ids=[m.parent.name for m in CASE_MANIFESTS],
    indirect=True,
)
def test_text_case(cluster_for_case: Cluster, image: str) -> None:
    case_dir = cluster_for_case.manifest_path.parent
    runner.run_case(cluster_for_case.kubeconfig, cluster_for_case.namespace, case_dir, image)
