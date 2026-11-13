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
"""Every workflow step runs under bash with -eo pipefail.

GitHub only adds pipefail when a step's shell is named explicitly. With no
shell set it runs `bash -e`, and a step ending in `| tee` returns tee's status.
"""

from pathlib import Path

import pytest
import yaml

WORKFLOWS = sorted((Path(__file__).resolve().parents[1] / ".github" / "workflows").glob("*.y*ml"))


# Walks nested mappings by key. dig({"a": {"b": 1}}, "a", "b") is 1, and any missing key gives None.
def dig(node: object, *keys: str) -> object:
    for key in keys:
        node = node.get(key) if isinstance(node, dict) else None
    return node


# The shell every `run:` step in one workflow resolves to: the step's own
# `shell`, else the job default, else the workflow default, else None.
def resolved_shells(workflow: object) -> dict[str, object]:
    top = dig(workflow, "defaults", "run", "shell")
    jobs = dig(workflow, "jobs")
    shells: dict[str, object] = {}
    for job_name, job in jobs.items() if isinstance(jobs, dict) else ():
        job_shell = dig(job, "defaults", "run", "shell") or top
        steps = dig(job, "steps")
        for index, step in enumerate(steps if isinstance(steps, list) else ()):
            if dig(step, "run") is not None:
                shells[f"{job_name}[{index}]"] = dig(step, "shell") or job_shell
    return shells


# The repo has workflow files to check. An empty glob would pass the tests below vacuously.
def test_workflows_are_found() -> None:
    assert WORKFLOWS


# Each workflow file sets defaults.run.shell to bash at the top level, even one with no run steps yet.
@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_workflow_defaults_to_bash(path: Path) -> None:
    assert dig(yaml.safe_load(path.read_text()), "defaults", "run", "shell") == "bash", (
        f"{path.name} must set `defaults.run.shell: bash` so every step runs with -eo pipefail"
    )


# No job or step in a workflow file overrides the default with another shell such as sh or python.
@pytest.mark.parametrize("path", WORKFLOWS, ids=lambda p: p.name)
def test_no_step_leaves_bash(path: Path) -> None:
    shells = resolved_shells(yaml.safe_load(path.read_text()))
    other = {step: shell for step, shell in shells.items() if shell != "bash"}
    assert not other, f"{path.name} runs steps outside bash, which drops pipefail: {other}"


# A workflow with a bash default, one job overriding to sh and one step overriding to python.
# Pins the precedence: step over job over workflow.
def test_resolved_shells_precedence() -> None:
    workflow = {
        "defaults": {"run": {"shell": "bash"}},
        "jobs": {
            "a": {"steps": [{"run": "true"}, {"run": "true", "shell": "python"}, {"uses": "actions/checkout"}]},
            "b": {"defaults": {"run": {"shell": "sh"}}, "steps": [{"run": "true"}]},
        },
    }
    assert resolved_shells(workflow) == {"a[0]": "bash", "a[1]": "python", "b[0]": "sh"}


# A workflow with no defaults at all. Its one run step resolves to None, which the bash check rejects.
def test_resolved_shells_without_defaults() -> None:
    assert resolved_shells({"jobs": {"a": {"steps": [{"run": "true"}]}}}) == {"a[0]": None}
