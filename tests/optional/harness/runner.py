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
"""Run one multimodal case end-to-end against a cluster.

Ports steps 1-4 of tests/optional/manual/multimodal/e2e_test.sh into Python so the
pytest tier owns deploy, run, and verification. The namespace lifecycle and
concurrency control live in the fixture (conftest.py); this module assumes the
namespace already exists and does not delete it.
"""
from __future__ import annotations

import json
import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Cluster:
    """A matched cluster plus the namespace and manifest for one case."""

    kubeconfig: str | None
    namespace: str
    manifest_path: Path


_IMAGE = "quay.io/inference-perf/inference-perf:latest"
_TEMPLATE_IMAGE = "quay.io/inference-perf/inference-perf:latest"
_VLLM_DEPLOYMENT = "qwen3-32b-vllm-deployment"
_CONFIG_MAP = "inference-perf-config"
_JOB = "inference-perf"
_JOB_TIMEOUT = "600s"
_ROLLOUT_TIMEOUT = "900s"
# deploy/manifests.yaml relative to the repo root (this file is 3 levels deep
# under tests/optional/harness/).
_MANIFEST_TEMPLATE = Path(__file__).resolve().parents[3] / "deploy" / "manifests.yaml"


def _kubectl(
    kubeconfig: str | None,
    namespace: str | None,
    *args: str,
    stdin: str | None = None,
    capture: bool = False,
    check: bool = True,
) -> str:
    cmd = ["kubectl", *args]
    if namespace:
        cmd[1:1] = ["-n", namespace]
    if kubeconfig:
        cmd[1:1] = ["--kubeconfig", kubeconfig]
    result = subprocess.run(
        cmd, input=stdin, capture_output=capture, text=True, check=check
    )
    return result.stdout if capture else ""


def _copy_hf_secret(kubeconfig: str | None, namespace: str) -> None:
    """Copy hf-secret from the default namespace into the case namespace."""
    raw = _kubectl(
        kubeconfig, "default", "get", "secret", "hf-secret", "-o", "json",
        capture=True, check=False,
    )
    if not raw.strip():
        # Mirrors the bash warning: gated models will fail to start without it.
        print("Warning: hf-secret not found in 'default' namespace.")
        return
    secret = json.loads(raw)
    meta = secret.get("metadata", {})
    for field in ("namespace", "resourceVersion", "uid", "creationTimestamp"):
        meta.pop(field, None)
    secret["metadata"] = meta
    _kubectl(kubeconfig, namespace, "apply", "-f", "-", stdin=json.dumps(secret))


def run_case(kubeconfig: str | None, namespace: str, case_dir: Path, image: str = _IMAGE) -> None:
    """Deploy the model server, run inference-perf, and assert all requests succeeded.

    Raises AssertionError on zero successes or any failures; lets subprocess errors
    (rollout/job timeouts) propagate so the test fails loudly rather than hanging.
    """
    case_dir = Path(case_dir)

    _copy_hf_secret(kubeconfig, namespace)

    # Step 1: model server.
    vllm = case_dir / "vllm.yaml"
    if vllm.is_file():
        _kubectl(kubeconfig, namespace, "apply", "-f", str(vllm))
    _kubectl(
        kubeconfig, namespace, "rollout", "status",
        f"deployment/{_VLLM_DEPLOYMENT}", f"--timeout={_ROLLOUT_TIMEOUT}",
    )

    # Step 2: config.
    cm = _kubectl(
        kubeconfig, namespace, "create", "configmap", _CONFIG_MAP,
        f"--from-file={case_dir / 'config.yml'}", "--dry-run=client", "-o", "yaml",
        capture=True,
    )
    _kubectl(kubeconfig, namespace, "apply", "-f", "-", stdin=cm)

    # Step 3: inference-perf job (case override, else templated default).
    _kubectl(kubeconfig, namespace, "delete", "job", _JOB, "--ignore-not-found")
    case_manifest = case_dir / "manifest.yaml"
    if case_manifest.is_file():
        _kubectl(kubeconfig, namespace, "apply", "-f", str(case_manifest))
    else:
        rendered = _MANIFEST_TEMPLATE.read_text().replace(_TEMPLATE_IMAGE, image)
        _kubectl(kubeconfig, namespace, "apply", "-f", "-", stdin=rendered)

    # Step 4: wait, extract, verify.
    _kubectl(
        kubeconfig, namespace, "wait", "--for=condition=complete",
        f"job/{_JOB}", f"--timeout={_JOB_TIMEOUT}",
    )
    logs = _kubectl(kubeconfig, namespace, "logs", f"job/{_JOB}", capture=True)
    summary = _extract_block(logs, "=== START_SUMMARY ===", "=== END_SUMMARY ===")
    assert summary, "no summary block found in job logs"
    metrics = json.loads(summary)
    successes = metrics.get("successes", {}).get("count", 0)
    failures = metrics.get("failures", {}).get("count", 0)
    assert successes, f"zero successful requests for case {case_dir.name}"
    assert not failures, f"case {case_dir.name} had {failures} failed requests"


def _extract_block(text: str, start: str, end: str) -> str:
    lines = text.splitlines()
    try:
        i = lines.index(start)
        j = lines.index(end, i + 1)
    except ValueError:
        return ""
    return "\n".join(lines[i + 1 : j])
