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
"""Keep the YAML checked into this repo honest.

Two separate properties, because "is this YAML OK" means two different things:

1. Every tracked YAML file parses as YAML (`test_every_tracked_yaml_parses`). Cheap, and it
   covers workflow files and manifests that nothing else in the unit tier loads.
2. Every file that is an inference-perf *config* still satisfies the current schema
   (`test_config_yaml_matches_schema`). This is the anti-staleness gate: as the schema
   evolves, examples and docs would otherwise desync from the code with no signal.

The third test is the one that keeps this honest over time. Classifying files by directory
means a new directory of configs would simply never be validated, which is the exact silent
failure this file exists to prevent, so `test_every_tracked_yaml_is_classified` fails until
someone puts each new YAML in one bucket or the other.

Note that (2) only has teeth because the config models set `extra="forbid"` (see
`StrictBaseModel`). With Pydantic's default an obsolete key is discarded silently, so a
stale config would validate cleanly while quietly losing the setting it meant to apply.
"""

import subprocess
from pathlib import Path
from typing import List

import pytest
import yaml

from inference_perf.config import read_config

REPO_ROOT = Path(__file__).resolve().parents[3]

# Directories and files that are inference-perf configs, i.e. loadable by `read_config`.
CONFIG_DIRS = ("examples", "workload-catalog", "e2e")
CONFIG_FILES = ("config.yml", "config-weka.yaml", "demos/kubecon-na-2025/config.yaml")

# YAML that is deliberately not an inference-perf config. Each entry is a directory prefix or
# an exact path, with the reason it cannot be schema-validated.
NON_CONFIG_PREFIXES = {
    ".github/": "GitHub Actions workflows",
    "deploy/": "Helm templates; Go templating is not parseable YAML by design",
    "tests/": "test fixtures and server manifests, not inference-perf configs",
    "demos/kubecon-na-2025/vllm": "multi-document Kubernetes manifests",
}
NON_CONFIG_FILES = {
    "cloudbuild.yaml": "Google Cloud Build config",
    ".pre-push-config.yaml": "pre-commit config",
    "examples/vllm/docker-compose.yml": "docker-compose service definition",
    "examples/vllm/prometheus.yml": "Prometheus scrape config",
    "examples/sglang/docker-compose.yml": "docker-compose service definition",
    "examples/sglang/prometheus.yml": "Prometheus scrape config",
    "examples/tgi/docker-compose.yml": "docker-compose service definition",
    "examples/tgi/prometheus.yml": "Prometheus scrape config",
}


def _tracked_yaml_files() -> List[str]:
    out = subprocess.check_output(
        ["git", "ls-files", "*.yaml", "*.yml"],
        cwd=REPO_ROOT,
        text=True,
    )
    return sorted(p for p in out.split("\n") if p)


def _is_config(path: str) -> bool:
    if path in CONFIG_FILES:
        return True
    return any(path.startswith(f"{d}/") for d in CONFIG_DIRS) and path not in NON_CONFIG_FILES


def _is_declared_non_config(path: str) -> bool:
    if path in NON_CONFIG_FILES:
        return True
    return any(path.startswith(prefix) for prefix in NON_CONFIG_PREFIXES)


ALL_YAML = _tracked_yaml_files()
CONFIG_YAML = [p for p in ALL_YAML if _is_config(p)]


def test_repo_has_tracked_yaml() -> None:
    """Guard against the discovery itself silently returning nothing."""
    assert ALL_YAML, "no tracked YAML found; is git ls-files working from the repo root?"
    assert CONFIG_YAML, "no config YAML discovered; CONFIG_DIRS/CONFIG_FILES are likely stale"


@pytest.mark.parametrize("rel_path", ALL_YAML)
def test_every_tracked_yaml_parses(rel_path: str) -> None:
    """Every tracked YAML is syntactically valid, including multi-document files."""
    if rel_path.startswith("deploy/"):
        pytest.skip("Helm template, not parseable as plain YAML")
    with open(REPO_ROOT / rel_path) as f:
        list(yaml.safe_load_all(f))


@pytest.mark.parametrize("rel_path", CONFIG_YAML)
def test_config_yaml_matches_schema(rel_path: str) -> None:
    """Every config YAML in the repo still satisfies the current schema.

    Uses `read_config` rather than `Config(**data)` so the test exercises the same path the
    CLI does, including default merging and load-stage type conversion.
    """
    read_config(str(REPO_ROOT / rel_path))


@pytest.mark.parametrize("rel_path", ALL_YAML)
def test_every_tracked_yaml_is_classified(rel_path: str) -> None:
    """Every tracked YAML is either schema-validated or explicitly declared a non-config.

    If this fails for a file you just added, add it to CONFIG_DIRS/CONFIG_FILES if it is an
    inference-perf config, or to NON_CONFIG_PREFIXES/NON_CONFIG_FILES with a reason if it is
    not. Silently unvalidated config YAML is what this whole module exists to prevent.
    """
    assert _is_config(rel_path) or _is_declared_non_config(rel_path), (
        f"{rel_path} is neither validated as a config nor declared a non-config. "
        "Add it to CONFIG_DIRS/CONFIG_FILES or NON_CONFIG_PREFIXES/NON_CONFIG_FILES."
    )
