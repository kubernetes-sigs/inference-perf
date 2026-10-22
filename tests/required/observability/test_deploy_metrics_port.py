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
scraping pointed at a dead port. The rendering tests check that the chart
declares a port exactly when the config it ships would make the tool serve one,
and that its monitors select the Job's pod.
"""

import re
import shutil
import subprocess
import unittest
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from inference_perf.config import Config
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
    # Expects config.observability.metrics to be exactly {port: 9464}, so a
    # default install turns the endpoint on at DEFAULT_PORT and sets nothing else.
    def test_chart_values_turn_metrics_on_at_default_port(self) -> None:
        values = yaml.safe_load((CHART_DIR / "values.yaml").read_text())
        self.assertEqual(values["config"]["observability"]["metrics"], {"port": DEFAULT_PORT})


# Runs helm template on the chart with each --set override and returns its
# documents keyed by kind (Job, ConfigMap, PodMonitor, ...). Fails the test with
# helm's stderr if the render fails.
def _render(test: unittest.TestCase, *overrides: str) -> Dict[str, Any]:
    args = ["helm", "template", "t", str(CHART_DIR)]
    for override in overrides:
        args += ["--set", override]
    result = subprocess.run(args, capture_output=True, text=True, timeout=60)
    test.assertEqual(result.returncode, 0, result.stderr)
    return {doc["kind"]: doc for doc in yaml.safe_load_all(result.stdout) if doc}


# The port the tool would serve on, read the way it reads the rendered ConfigMap.
# None when the config leaves observability.metrics out.
def _served_port(docs: Dict[str, Any]) -> Optional[int]:
    config = Config.model_validate(yaml.safe_load(docs["ConfigMap"]["data"]["config.yml"]))
    metrics = config.observability.metrics
    return None if metrics is None else metrics.port


# The Job pod's metrics container port, or None when the chart declares none.
def _declared_port(docs: Dict[str, Any]) -> Optional[int]:
    container = docs["Job"]["spec"]["template"]["spec"]["containers"][0]
    ports = {p["name"]: p["containerPort"] for p in container.get("ports", [])}
    return ports.get("metrics")


# The Job pod's annotations, empty when the chart sets none.
def _annotations(docs: Dict[str, Any]) -> Dict[str, str]:
    annotations: Dict[str, str] = docs["Job"]["spec"]["template"]["metadata"].get("annotations") or {}
    return annotations


@unittest.skipUnless(shutil.which("helm"), "helm is not installed")
class TestChartMetricsRendering(unittest.TestCase):
    # Default values, which set config.observability.metrics.port: 9464.
    # Expects the tool to serve 9464, the Job to declare it as a container port
    # and in prometheus.io/port, and no PodMonitor or PodMonitoring.
    def test_default_install_declares_the_port_it_serves(self) -> None:
        docs = _render(self)
        self.assertEqual(_served_port(docs), DEFAULT_PORT)
        self.assertEqual(_declared_port(docs), DEFAULT_PORT)
        self.assertEqual(_annotations(docs)["prometheus.io/port"], str(DEFAULT_PORT))
        self.assertNotIn("PodMonitor", docs)
        self.assertNotIn("PodMonitoring", docs)

    # --set config.observability.metrics=null with both monitors turned on.
    # Expects the tool to serve nothing, and the chart to render no metrics port,
    # no prometheus.io annotations and neither monitor.
    def test_null_block_declares_nothing(self) -> None:
        docs = _render(
            self,
            "config.observability.metrics=null",
            "metrics.podMonitor.enabled=true",
            "metrics.podMonitoring.enabled=true",
        )
        self.assertIsNone(_served_port(docs))
        self.assertIsNone(_declared_port(docs))
        self.assertEqual(_annotations(docs), {})
        self.assertNotIn("PodMonitor", docs)
        self.assertNotIn("PodMonitoring", docs)

    # Port 9500 with annotations, a PodMonitor (release label prom) and a PodMonitoring.
    # Expects the container port and annotation at 9500, and both monitors to
    # select the Job's pod by its template labels and scrape the named
    # "metrics" port at /metrics.
    def test_custom_port_reaches_every_scrape_surface(self) -> None:
        docs = _render(
            self,
            "config.observability.metrics.port=9500",
            "metrics.podMonitor.enabled=true",
            "metrics.podMonitor.labels.release=prom",
            "metrics.podMonitoring.enabled=true",
        )
        self.assertEqual(_served_port(docs), 9500)
        self.assertEqual(_declared_port(docs), 9500)
        self.assertEqual(_annotations(docs)["prometheus.io/port"], "9500")
        self.assertEqual(docs["PodMonitor"]["metadata"]["labels"]["release"], "prom")
        pod_labels = docs["Job"]["spec"]["template"]["metadata"]["labels"]
        for monitor, endpoint in (
            (docs["PodMonitor"], docs["PodMonitor"]["spec"]["podMetricsEndpoints"][0]),
            (docs["PodMonitoring"], docs["PodMonitoring"]["spec"]["endpoints"][0]),
        ):
            # A selector the pod's labels do not satisfy renders fine and scrapes nothing.
            selector = monitor["spec"]["selector"]["matchLabels"]
            self.assertEqual({key: pod_labels.get(key) for key in selector}, selector)
            self.assertEqual((endpoint["port"], endpoint["path"]), ("metrics", "/metrics"))


if __name__ == "__main__":
    unittest.main()
