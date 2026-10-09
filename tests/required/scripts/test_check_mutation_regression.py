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
import tomllib
from pathlib import Path
from types import ModuleType
from typing import Any

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "check_mutation_regression.py"


# Loads scripts/check_mutation_regression.py as a module. scripts/ is not a package.
@pytest.fixture(scope="module")
def cmr() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_mutation_regression", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# Runs git in `repo` with a fixed identity, so commits work on a bare CI runner.
def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "init.defaultBranch=main", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


# A module function, an async function, a method and a decorated method.
# Each maps to mutmut's mangled name, and the decorated one's span starts at its decorator line.
def test_function_spans_use_mutmut_names(cmr: Any) -> None:
    source = (
        "def f():\n"  # 1
        "    def inner():\n"  # 2
        "        pass\n"  # 3
        "async def g():\n"  # 4
        "    pass\n"  # 5
        "class C:\n"  # 6
        "    def m(self):\n"  # 7
        "        pass\n"  # 8
        "    @property\n"  # 9
        "    def p(self):\n"  # 10
        "        return 1\n"  # 11
    )
    assert cmr.function_spans(source) == {"x_f": (1, 3), "x_g": (4, 5), "xǁCǁm": (7, 8), "xǁCǁp": (9, 11)}


# inference_perf/a/b.py is inference_perf.a.b. A package's __init__.py is the package itself,
# the way mutmut names its mutants.
def test_module_name_folds_init_into_package(cmr: Any) -> None:
    assert cmr.module_name("inference_perf/a/b.py") == "inference_perf.a.b"
    assert cmr.module_name("inference_perf/a/__init__.py") == "inference_perf.a"


# A two-line change at old 3-4 / new 3-5, two lines inserted after old line 7, old line 10 deleted.
# The insertion marks only new lines 9-10 and the deletion only old line 10.
def test_parse_hunks_marks_only_the_side_with_lines(cmr: Any) -> None:
    diff = "@@ -3,2 +3,3 @@\n-a\n+b\n@@ -7,0 +9,2 @@\n+c\n@@ -10 +12,0 @@\n-d\n"
    old, new = cmr.parse_hunks(diff)
    assert old == {3, 4, 10}
    assert new == {3, 4, 5, 9, 10}


# Base has f, g and C.m. The working tree edits g and appends h right after C.m.
# Only g is an existing touched function, and h is new. Appending after C.m does not touch it.
def test_touched_functions_splits_existing_and_new(cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    pkg = tmp_path / "inference_perf"
    pkg.mkdir()
    (pkg / "mod.py").write_text(
        "def f():\n    return 1\n\n\ndef g():\n    return 2\n\n\nclass C:\n    def m(self):\n        return 3\n"
    )
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "base")
    (pkg / "mod.py").write_text(
        "def f():\n    return 1\n\n\ndef g():\n    return 20\n\n\nclass C:\n    def m(self):\n        return 3\n\n\n"
        "def h():\n    return 4\n"
    )
    monkeypatch.chdir(tmp_path)

    touched = cmr.touched_functions("HEAD", [])

    assert touched.existing == [("inference_perf.mod.x_g", "inference_perf.mod.x_g")]
    assert touched.new == ["inference_perf.mod.x_h"]


# A committed file moved from old.py to new.py with one function edited, an untracked new file,
# and an edited file that do_not_mutate excludes. The moved function pairs its old and new module
# names, every function in the untracked file is new, and the excluded file contributes nothing.
def test_touched_functions_follows_renames_and_untracked_files(
    cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = tmp_path / "inference_perf"
    pkg.mkdir()
    body = "".join(f"def f{i}():\n    return {i}\n\n\n" for i in range(10))
    (pkg / "old.py").write_text(body)
    (pkg / "skip.py").write_text("def s():\n    return 1\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "base")
    git(tmp_path, "mv", "inference_perf/old.py", "inference_perf/new.py")
    (pkg / "new.py").write_text(body.replace("return 3", "return 30"))
    (pkg / "skip.py").write_text("def s():\n    return 2\n")
    (pkg / "fresh.py").write_text("def a():\n    return 1\n\n\nclass K:\n    def b(self):\n        return 2\n")
    monkeypatch.chdir(tmp_path)

    touched = cmr.touched_functions("HEAD", ["inference_perf/skip.py"])

    assert touched.existing == [("inference_perf.old.x_f3", "inference_perf.new.x_f3")]
    assert touched.new == ["inference_perf.fresh.x_a", "inference_perf.fresh.xǁKǁb"]


# Tests that import the module directly or through a re-exporting package are picked, plus any
# changed test. Unrelated tests, tests/optional and the excluded git-dependent test are not.
def test_select_tests_follows_imports_and_reexports(cmr: Any, tmp_path: Path) -> None:
    (tmp_path / "inference_perf" / "utils").mkdir(parents=True)
    (tmp_path / "inference_perf" / "utils" / "__init__.py").write_text("from .cpu import count\n")
    files = {
        "tests/test_direct.py": "from inference_perf.utils.cpu import count\n",
        "tests/test_package.py": "from inference_perf.utils import count\n",
        "tests/test_other.py": "from inference_perf.other import x\n",
        "tests/test_changed.py": "import json\n",
        "tests/optional/test_opt.py": "from inference_perf.utils.cpu import count\n",
        "tests/required/config/test_yaml_configs.py": "from inference_perf.utils.cpu import count\n",
    }
    for rel, text in files.items():
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text(text)

    picked = cmr.select_tests(tmp_path, {"inference_perf.utils.cpu"}, ["tests/test_changed.py", "tests/test_gone.py"])

    assert picked == ["tests/test_changed.py", "tests/test_direct.py", "tests/test_package.py"]


# A pyproject with [tool.mutmut] between two other tables. The rewrite drops the old table, keeps
# the others, and appends the shared config plus this run's files and tests. Regex backslashes survive.
def test_write_config_replaces_the_mutmut_table(cmr: Any, tmp_path: Path) -> None:
    (tmp_path / "pyproject.toml").write_text(
        '[project]\nname = "x"\n\n[tool.mutmut]\nsource_paths = ["old"]\ndebug = true\n\n[tool.ruff]\nline-length = 127\n'
    )
    config = {"source_paths": ["inference_perf"], "do_not_mutate_patterns": ["logger\\.", "raise \\w+\\("]}

    cmr.write_config(tmp_path, config, ["inference_perf/a.py"], ["tests/test_a.py"])

    data = tomllib.loads((tmp_path / "pyproject.toml").read_text())
    assert data["project"] == {"name": "x"}
    assert data["tool"]["ruff"] == {"line-length": 127}
    assert data["tool"]["mutmut"] == {
        "source_paths": ["inference_perf"],
        "do_not_mutate_patterns": ["logger\\.", "raise \\w+\\("],
        "only_mutate": ["inference_perf/a.py"],
        "pytest_add_cli_args_test_selection": ["tests/test_a.py"],
    }


# Builds a Side whose results map `count` mutants of `name` to each status given.
def side(cmr: Any, **by_name: dict[str, int]) -> Any:
    results = {}
    for name, statuses in by_name.items():
        i = 0
        for status, count in statuses.items():
            for _ in range(count):
                i += 1
                results[f"inference_perf.m.{name}__mutmut_{i}"] = status
    return cmr.Side(tree=Path("."), tests=[], results=results)


# Existing function g: 3 survivors on the base, 3 on the head. Passes, and nothing is rerun.
def test_judge_passes_when_survivors_hold(cmr: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cmr, "still_survives", lambda *a: pytest.fail("rerun without a rise"))
    touched = cmr.Touched(existing=[("inference_perf.m.x_g", "inference_perf.m.x_g")])
    base = side(cmr, x_g={"killed": 5, "survived": 3})
    head = side(cmr, x_g={"killed": 6, "survived": 2, "no tests": 1})

    verdict = cmr.judge(touched, base, head, [])

    assert (verdict.base_survivors, verdict.head_survivors, verdict.failures) == (3, 3, [])


# g goes from 1 survivor to 3 while f drops from 2 to 1: total 3 to 4, so it fails.
# Only g's survivors are rerun, because only g's count rose.
def test_judge_fails_on_a_rise_and_reruns_only_rising_functions(cmr: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    rerun: list[str] = []

    def record(s: Any, mutant: str, a: Any) -> bool:
        rerun.append(mutant)
        return True

    monkeypatch.setattr(cmr, "still_survives", record)
    touched = cmr.Touched(
        existing=[("inference_perf.m.x_f", "inference_perf.m.x_f"), ("inference_perf.m.x_g", "inference_perf.m.x_g")]
    )
    base = side(cmr, x_f={"survived": 2}, x_g={"survived": 1})
    head = side(cmr, x_f={"survived": 1}, x_g={"survived": 3})

    verdict = cmr.judge(touched, base, head, [])

    assert all(m.startswith("inference_perf.m.x_g__") for m in rerun) and len(rerun) == 3
    assert verdict.failures == ["Survivors in existing functions rose from 3 to 4"]


# Same rise of one, but one of g's survivors is killed on rerun. It is reported as flaky and the
# total falls back to 3, so the change passes.
def test_judge_drops_flaky_survivors(cmr: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    flaky = "inference_perf.m.x_g__mutmut_2"
    monkeypatch.setattr(cmr, "still_survives", lambda s, m, a: m != flaky)
    touched = cmr.Touched(existing=[("inference_perf.m.x_g", "inference_perf.m.x_g")])
    base = side(cmr, x_g={"survived": 3})
    head = side(cmr, x_g={"survived": 4})

    verdict = cmr.judge(touched, base, head, [])

    assert verdict.flaky == [flaky]
    assert (verdict.head_survivors, verdict.failures) == (3, [])


# New function h: 6 killed or timed out, 4 survived or untested, 60.0%, which meets the floor.
# One more survivor puts it at 54.5% and fails. A new function with no mutants is not scored.
def test_judge_scores_new_functions_against_the_floor(cmr: Any) -> None:
    touched = cmr.Touched(new=["inference_perf.m.x_h"])
    at_floor = cmr.judge(touched, None, side(cmr, x_h={"killed": 5, "timeout": 1, "survived": 3, "no tests": 1}), [])
    under = cmr.judge(touched, None, side(cmr, x_h={"killed": 6, "survived": 5}), [])
    empty = cmr.judge(touched, None, side(cmr), [])

    assert (at_floor.new_score, at_floor.failures) == (60.0, [])
    assert under.failures == ["New functions score 54.5%, under the 60% floor"]
    assert (empty.new_score, empty.failures) == (None, [])
