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
"""The deploy artifacts name the runtime metrics port the tool actually serves.

The Helm chart and the plain manifest declare the metrics port as literals,
since neither can import Python. These tests pin those literals to
DEFAULT_PORT so changing the tool's default cannot silently leave in-cluster
scraping pointed at a dead port.
"""

import re
import unittest
from pathlib import Path
from typing import Any

import yaml

from inference_perf.observability.metrics.prometheus import DEFAULT_PORT

REPO_ROOT = Path(__file__).resolve().parents[3]
CHART_DIR = REPO_ROOT / "deploy" / "inference-perf"


class TestDeployMetricsPort(unittest.TestCase):
    # Input: deploy/manifests.yaml, the plain Job manifest.
    # Expects the container's "metrics" port and the prometheus.io/port pod
    # annotation to both equal DEFAULT_PORT (9464), and the path annotation
    # to be /metrics.
    def test_plain_manifest_declares_default_port(self) -> None:
        manifest = yaml.safe_load((REPO_ROOT / "deploy" / "manifests.yaml").read_text())
        pod: Any = manifest["spec"]["template"]
        ports = {p["name"]: p["containerPort"] for p in pod["spec"]["containers"][0]["ports"]}
        self.assertEqual(ports["metrics"], DEFAULT_PORT)
        annotations = pod["metadata"]["annotations"]
        self.assertEqual(annotations["prometheus.io/port"], str(DEFAULT_PORT))
        self.assertEqual(annotations["prometheus.io/path"], "/metrics")

    # Input: the chart's _helpers.tpl, which falls back to a literal port when
    # config.observability.metrics.port is unset.
    # Expects exactly one such fallback, equal to DEFAULT_PORT (9464).
    def test_chart_fallback_port_is_default_port(self) -> None:
        helpers = (CHART_DIR / "templates" / "_helpers.tpl").read_text()
        fallbacks = re.findall(r'dig "observability" "metrics" "port" (\d+)', helpers)
        self.assertEqual(fallbacks, [str(DEFAULT_PORT)])

    # Input: the chart's default values.yaml.
    # Expects it not to set config.observability.metrics at all, so a default
    # install serves on the tool's default port and the chart's fallback is
    # what applies.
    def test_chart_values_leave_metrics_config_to_the_tool(self) -> None:
        values = yaml.safe_load((CHART_DIR / "values.yaml").read_text())
        self.assertNotIn("observability", values["config"])


if __name__ == "__main__":
    unittest.main()
