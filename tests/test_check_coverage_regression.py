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
import subprocess
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest
import yaml

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_coverage_regression.py"
COVERAGE_WORKFLOW = Path(__file__).resolve().parents[1] / ".github" / "workflows" / "coverage.yml"


# Loads scripts/check_coverage_regression.py by path, since scripts/ is not a package.
# Importing it has no side effects, the env vars it needs are set in main().
@pytest.fixture
def ccr() -> ModuleType:
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


# Adds 100 statements, 90 covered (90% > 81.41% baseline). Uncovered rises
# 2590 -> 2600 but total rises 81.41% -> 81.47%, so it passes.
def test_adding_above_average_covered_code_passes(ccr: ModuleType) -> None:
    assert not ccr.coverage_regressed(totals(11430, 14030), MAIN)


# --- generate_baseline: the two sides must differ only by the change under test (#724) ---


# Replaces the module's run_command with a recorder that succeeds every call (except the one
# named by `fail`) and answers `git merge-base ...` with `sha`, so generate_baseline runs its
# whole flow without touching git, venvs or pytest. Returns the list the commands land in.
def record_commands(ccr: ModuleType, monkeypatch: pytest.MonkeyPatch, sha: str, fail: str | None = None) -> list[str]:
    commands: list[str] = []

    def fake_run_command(cmd: str, cwd: str | Path | None = None) -> subprocess.CompletedProcess[str]:
        if cmd == fail:
            raise subprocess.CalledProcessError(returncode=1, cmd=cmd)
        commands.append(cmd)
        stdout = f"{sha}\n" if cmd.startswith("git merge-base") else ""
        return subprocess.CompletedProcess(args=cmd, returncode=0, stdout=stdout, stderr="")

    monkeypatch.setattr(ccr, "run_command", fake_run_command)
    return commands


# generate_baseline on a branch behind main must measure the fork point, not main's tip: the
# worktree is created at the sha `git merge-base origin/main HEAD` prints, never at origin/main.
def test_generate_baseline_checks_out_the_merge_base(ccr: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.chdir(tmp_path)
    commands = record_commands(ccr, monkeypatch, sha="f00dfeed")

    ccr.generate_baseline(tmp_path / "coverage_main.json")

    assert "git merge-base origin/main HEAD" in commands
    worktree_adds = [c for c in commands if c.startswith("git worktree add")]
    assert worktree_adds == [f"git worktree add -d {Path.cwd() / '.baseline_temp'} f00dfeed"]


# The baseline install must be the exact install command of the coverage job in
# .github/workflows/coverage.yml, read from the workflow itself, so the baseline run and the
# current run install the same extras and no test skips on one side only.
def test_generate_baseline_installs_like_the_coverage_workflow(
    ccr: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    commands = record_commands(ccr, monkeypatch, sha="f00dfeed")

    ccr.generate_baseline(tmp_path / "coverage_main.json")

    workflow = yaml.safe_load(COVERAGE_WORKFLOW.read_text())
    steps = workflow["jobs"]["coverage"]["steps"]
    install = next(step for step in steps if step.get("name") == "Install dependencies")["run"].strip()
    assert install in commands, f"baseline must install with the workflow's own command: {install}"


# When fetching origin/main fails, the baseline still measures the fork point, against the
# local main branch instead: merge-base with main, worktree at the sha it prints.
def test_generate_baseline_falls_back_to_the_local_main_merge_base(
    ccr: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.chdir(tmp_path)
    commands = record_commands(ccr, monkeypatch, sha="cafebabe", fail="git fetch origin main")

    ccr.generate_baseline(tmp_path / "coverage_main.json")

    assert "git merge-base main HEAD" in commands
    assert f"git worktree add -d {Path.cwd() / '.baseline_temp'} cafebabe" in commands
