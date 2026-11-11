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
"""Unit tests for the license header check in scripts/check_license.py."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_license.py"

BODY = """#
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
"""


def header(year: int = 2026) -> str:
    return f"# Copyright {year} The Kubernetes Authors.\n{BODY}"


# Loads scripts/check_license.py by path, since scripts/ is not a package.
@pytest.fixture
def cl() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_license", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# The header as new files are expected to write it.
def test_accepts_current_year(cl: ModuleType) -> None:
    assert cl.has_license_header(header(2026) + "\nimport os\n")


# A file keeps the year it was written in. Pinning the template to the year the
# check runs in rejected every 2025 header the moment the calendar turned over.
@pytest.mark.parametrize("year", [2014, 2025, 2026, 2031])
def test_accepts_any_four_digit_year(cl: ModuleType, year: int) -> None:
    assert cl.has_license_header(header(year))


# The year has to be four digits, so prose in its place is still a failure.
@pytest.mark.parametrize("year_text", ["", "YEAR", "202", "20266", "two thousand"])
def test_rejects_non_year_in_copyright_line(cl: ModuleType, year_text: str) -> None:
    content = f"# Copyright {year_text} The Kubernetes Authors.\n{BODY}"
    assert not cl.has_license_header(content)


# A shebang has to be the first line, so an executable script carries the
# header on line 2. The check used to reject every such file, itself included.
def test_accepts_header_after_shebang(cl: ModuleType) -> None:
    assert cl.has_license_header(f"#!/usr/bin/env python3\n{header()}")


# Some scripts leave a blank line between the shebang and the header.
def test_accepts_blank_line_between_shebang_and_header(cl: ModuleType) -> None:
    assert cl.has_license_header(f"#!/usr/bin/env python3\n\n{header(2025)}")


# Both URL indentations are the Apache boilerplate and both are in the tree.
def test_accepts_indented_license_url(cl: ModuleType) -> None:
    content = header().replace(
        "# http://www.apache.org/licenses/LICENSE-2.0",
        "#     http://www.apache.org/licenses/LICENSE-2.0",
    )
    assert cl.has_license_header(content)


# The check still has teeth: altered license wording is not a pass.
def test_rejects_altered_license_text(cl: ModuleType) -> None:
    content = header().replace("Apache License", "Apache Licence")
    assert not cl.has_license_header(content)


# A module docstring ahead of the header does not count as a header.
def test_rejects_header_below_a_docstring(cl: ModuleType) -> None:
    assert not cl.has_license_header(f'"""Module."""\n\n{header()}')


def test_rejects_file_without_header(cl: ModuleType) -> None:
    assert not cl.has_license_header("import os\n")


# An empty or whitespace-only file has no code to license, so it is not failed.
@pytest.mark.parametrize("content", ["", "\n", "   \n\n"])
def test_empty_file_passes(cl: ModuleType, content: str, tmp_path: Path) -> None:
    empty = tmp_path / "__init__.py"
    empty.write_text(content, encoding="utf-8")
    assert cl.check_file(str(empty))


def test_unreadable_file_fails(cl: ModuleType, tmp_path: Path) -> None:
    assert not cl.check_file(str(tmp_path / "does_not_exist.py"))


# The checker carries a shebang and so has to pass its own check.
def test_script_passes_its_own_check() -> None:
    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(SCRIPT)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_reports_and_exits_nonzero_on_a_bad_file(tmp_path: Path) -> None:
    good = tmp_path / "good.py"
    good.write_text(header(2025), encoding="utf-8")
    bad = tmp_path / "bad.py"
    bad.write_text("import os\n", encoding="utf-8")

    result = subprocess.run(
        [sys.executable, str(SCRIPT), str(good), str(bad)],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 1
    assert "bad.py" in result.stdout
    assert "good.py" not in result.stdout
