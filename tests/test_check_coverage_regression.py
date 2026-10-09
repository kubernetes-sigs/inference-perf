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
import importlib.util
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_coverage_regression.py"


# Loads scripts/check_coverage_regression.py by path, since scripts/ is not a package.
# Importing it sets two env vars. monkeypatch puts them back after each test.
@pytest.fixture
def ccr(monkeypatch: pytest.MonkeyPatch) -> ModuleType:
    monkeypatch.delenv("COVERAGE_PROCESS_START", raising=False)
    monkeypatch.delenv("PDM_IGNORE_ACTIVE_VENV", raising=False)
    spec = importlib.util.spec_from_file_location("check_coverage_regression", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# coverage.py JSON totals from covered and total statement counts.
# (8, 10) gives percent_covered=80.0 and missing_lines=2.
def totals(covered: int, statements: int) -> dict[str, Any]:
    return {"percent_covered": 100 * covered / statements, "missing_lines": statements - covered}


# main at 11340 of 13930 statements covered (81.41%, 2590 uncovered).
MAIN = totals(11340, 13930)


# #871 deletes 794 statements, 60 of them uncovered. Total falls 81.41% -> 80.74%
# but uncovered falls 2590 -> 2530, so it passes.
def test_deleting_well_covered_code_passes(ccr: ModuleType) -> None:
    assert not ccr.coverage_regressed(totals(10606, 13136), MAIN)


# Deletes a 164/164 file. Total falls to 81.19% and uncovered stays at 2590, so it passes.
def test_deleting_fully_covered_file_passes(ccr: ModuleType) -> None:
    assert not ccr.coverage_regressed(totals(11176, 13766), MAIN)


# Moves a 164/164 file (neither count changes) and adds 39 untested statements.
# Total falls to 81.18% and uncovered rises 2590 -> 2629, so it fails.
def test_untested_code_beside_a_move_fails(ccr: ModuleType) -> None:
    assert ccr.coverage_regressed(totals(11340, 13969), MAIN)
