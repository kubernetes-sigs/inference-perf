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
import tomllib
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest


def load_pdm_scripts() -> dict[str, Any]:
    pyproject = Path(__file__).resolve().parents[1] / "pyproject.toml"
    with pyproject.open("rb") as f:
        data = tomllib.load(f)
    scripts: dict[str, Any] = data["tool"]["pdm"]["scripts"]
    return scripts


def test_every_pdm_script_has_help() -> None:
    """`pdm run --list` is the reference for dev commands, so every script must
    carry a `help` text. Scripts prefixed with `_` are hidden from the listing
    and exempt."""
    scripts = load_pdm_scripts()
    missing = [
        name
        for name, spec in scripts.items()
        if not name.startswith("_") and not (isinstance(spec, dict) and spec.get("help"))
    ]
    assert not missing, f"pdm scripts missing a `help` text (shown by `pdm run --list`): {missing}"


# A temporary index contains a valid, missing, or incorrect header, or no Python files.
# Running outside a Git repository must also fail rather than skip the check.
@pytest.mark.parametrize(
    ("scenario", "expected_output"),
    [
        ("valid", "All files passed license check."),
        ("missing", "File missing or incorrect license header: sample with spaces.py"),
        ("incorrect", "File missing or incorrect license header: sample with spaces.py"),
        ("empty", "No tracked Python files found."),
        ("git_failure", "CalledProcessError"),
    ],
)
def test_license_script(scenario: str, expected_output: str, tmp_path: Path) -> None:
    source = Path(__file__).resolve().parents[1] / "scripts" / "check_license.py"
    checker = tmp_path / "scripts" / "check_license.py"
    checker.parent.mkdir()
    checker.write_bytes(source.read_bytes())
    if scenario != "git_failure":
        subprocess.run(["git", "init", "-q"], cwd=tmp_path, check=True, capture_output=True)
    if scenario in {"valid", "missing", "incorrect"}:
        content = source.read_bytes()
        if scenario == "missing":
            content = b"pass\n"
        elif scenario == "incorrect":
            content = content.replace(b"The Kubernetes Authors.", b"The Wrong Authors.", 1)
        sample = tmp_path / "sample with spaces.py"
        sample.write_bytes(content)
        subprocess.run(["git", "add", "--", sample.name], cwd=tmp_path, check=True, capture_output=True)

    command = load_pdm_scripts()["check:license"]["cmd"]
    result = subprocess.run([sys.executable, *command[1:]], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == (0 if scenario == "valid" else 1), result.stdout + result.stderr
    assert expected_output in result.stdout + result.stderr
