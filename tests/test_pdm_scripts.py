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
from pathlib import Path
from typing import Any


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
