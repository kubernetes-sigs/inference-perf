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
"""Tests for scripts/check_mutation_regression.py, the mutation ratchet CI runs on every change."""

import importlib.util
import json
import os
import subprocess
import sys
import time
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


# The base has old.py with a and b, and gone.py with c. The working tree deletes b from old.py and
# deletes gone.py. b and c are deleted functions named as they were on the base, and a is untouched.
def test_touched_functions_records_deleted_functions_and_files(
    cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = tmp_path / "inference_perf"
    pkg.mkdir()
    (pkg / "old.py").write_text("def a():\n    return 1\n\n\ndef b():\n    return 2\n")
    (pkg / "gone.py").write_text("def c():\n    return 3\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "base")
    (pkg / "old.py").write_text("def a():\n    return 1\n")
    (pkg / "gone.py").unlink()
    monkeypatch.chdir(tmp_path)

    touched = cmr.touched_functions("HEAD", [])

    assert (touched.existing, touched.new) == ([], [])
    assert touched.deleted == ["inference_perf.gone.x_c", "inference_perf.old.x_b"]


# Every test_*.py under tests/, nested or not, except tests/optional and the git-dependent YAML
# test. Helpers and conftest files are not test files.
def test_unit_suite_is_every_unit_test_file(cmr: Any, tmp_path: Path) -> None:
    for rel in [
        "tests/test_a.py",
        "tests/required/utils/test_b.py",
        "tests/required/utils/helpers.py",
        "tests/conftest.py",
        "tests/optional/test_opt.py",
        "tests/required/config/test_yaml_configs.py",
    ]:
        (tmp_path / rel).parent.mkdir(parents=True, exist_ok=True)
        (tmp_path / rel).write_text("")

    assert cmr.unit_suite(tmp_path) == ["tests/required/utils/test_b.py", "tests/test_a.py"]


# The change edits tests/optional/test_live.py and the YAML test and adds tests/test_new.py. Only
# the new file is a changed test, so the live tests never run under coverage.
def test_changed_test_files_skip_what_the_suite_skips(cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_tree(tmp_path, {"tests/optional/test_live.py": "", "tests/required/config/test_yaml_configs.py": ""})
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "base")
    write_tree(
        tmp_path, {"tests/optional/test_live.py": "# edited\n", "tests/required/config/test_yaml_configs.py": "# edited\n"}
    )
    write_tree(tmp_path, {"tests/test_new.py": ""})
    monkeypatch.chdir(tmp_path)

    assert cmr.changed_test_files("HEAD") == ["tests/test_new.py"]


# A module with f, called by the test, g, called at import time, and h, never called.
# Only f counts as run by the test. Import-time calls do not.
def test_functions_run_by_ignores_import_time(cmr: Any, tmp_path: Path) -> None:
    (tmp_path / "inference_perf").mkdir()
    (tmp_path / "inference_perf" / "__init__.py").write_text("")
    (tmp_path / "inference_perf" / "m.py").write_text(
        "def f():\n    return 1\n\n\ndef g():\n    return 2\n\n\ndef h():\n    return 3\n\n\nVALUE = g()\n"
    )
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_m.py").write_text("from inference_perf.m import f\n\n\ndef test_f():\n    assert f() == 1\n")

    assert cmr.functions_run_by(tmp_path, ["tests/test_m.py"]) == {"inference_perf.m.x_f"}


# The same module, run from inside mutmut's stats run, which is how this file runs once the script is
# on main. The child plugin has set MUTATION_CHILD_CONTEXT=child and mutmut MUTANT_UNDER_TEST=stats.
# Neither reaches the coverage run, so g, called at import, still does not count.
def test_functions_run_by_ignores_an_outer_harness(cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "inference_perf").mkdir()
    (tmp_path / "inference_perf" / "__init__.py").write_text("")
    (tmp_path / "inference_perf" / "m.py").write_text("def f():\n    return 1\n\n\ndef g():\n    return 2\n\n\nVALUE = g()\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_m.py").write_text("from inference_perf.m import f\n\n\ndef test_f():\n    assert f() == 1\n")
    monkeypatch.setenv("MUTATION_CHILD_CONTEXT", "child")
    monkeypatch.setenv("MUTANT_UNDER_TEST", "stats")

    assert cmr.functions_run_by(tmp_path, ["tests/test_m.py"]) == {"inference_perf.m.x_f"}


# A pyproject with [tool.mutmut] between two other tables. The rewrite drops the old table, keeps
# the others, and appends the shared config plus this run's files and tests, with the child plugin
# added to pytest's arguments. Regex backslashes survive.
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
        "pytest_add_cli_args": ["-p", "mutation_child_hits"],
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

    def record(s: Any, mutant: str, a: Any) -> Any:
        rerun.append(mutant)
        return cmr.Survival(True, [])

    monkeypatch.setattr(cmr, "still_survives", record)
    monkeypatch.setattr(cmr, "mutation_lines", lambda s, m: (str(id(s)), m))
    touched = cmr.Touched(
        existing=[("inference_perf.m.x_f", "inference_perf.m.x_f"), ("inference_perf.m.x_g", "inference_perf.m.x_g")]
    )
    base = side(cmr, x_f={"survived": 2}, x_g={"survived": 1})
    head = side(cmr, x_f={"survived": 1}, x_g={"survived": 3})

    verdict = cmr.judge(touched, base, head, [])

    assert all(m.startswith("inference_perf.m.x_g__") for m in rerun) and len(rerun) == 3
    assert verdict.failures == ["Surviving mutants rose from 3 to 4"]


# Same rise of one. Every rerun kills g's survivor 2, so the first run missed it and it does not
# count. The total falls back to 3 and the change passes.
def test_judge_drops_survivors_every_rerun_kills(cmr: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    fluke = "inference_perf.m.x_g__mutmut_2"
    monkeypatch.setattr(cmr, "still_survives", lambda s, m, a: cmr.Survival(m != fluke, ["t"] if m == fluke else []))
    monkeypatch.setattr(cmr, "mutation_lines", lambda s, m: (str(id(s)), m))
    touched = cmr.Touched(existing=[("inference_perf.m.x_g", "inference_perf.m.x_g")])
    base = side(cmr, x_g={"survived": 3})
    head = side(cmr, x_g={"survived": 4})

    verdict = cmr.judge(touched, base, head, [])

    assert (verdict.flukes, verdict.flaky) == ([fluke], {})
    assert (verdict.head_survivors, verdict.failures) == (3, [])


# Code moved from deleted function d into new function h keeps its 2 survivors, so it passes.
# A third survivor in h fails, and only h's survivors are rerun.
def test_judge_counts_new_and_deleted_functions(cmr: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    rerun: list[str] = []

    def record(s: Any, mutant: str, a: Any) -> Any:
        rerun.append(mutant)
        return cmr.Survival(True, [])

    monkeypatch.setattr(cmr, "still_survives", record)
    touched = cmr.Touched(new=["inference_perf.m.x_h"], deleted=["inference_perf.m.x_d"])
    base = side(cmr, x_d={"killed": 4, "survived": 2})

    moved = cmr.judge(touched, base, side(cmr, x_h={"killed": 4, "survived": 2}), [])
    worse = cmr.judge(touched, base, side(cmr, x_h={"killed": 3, "survived": 3}), [])

    assert (moved.base_survivors, moved.head_survivors, moved.failures) == (2, 2, [])
    assert worse.failures == ["Surviving mutants rose from 2 to 3"]
    assert len(rerun) == 3 and all(m.startswith("inference_perf.m.x_h__") for m in rerun)


# New function h has no test running it. The deleted function's 10 survivors would cover the count,
# but code no test runs cannot be paid for, so it fails.
def test_judge_fails_on_a_new_function_no_test_runs(cmr: Any) -> None:
    touched = cmr.Touched(new=["inference_perf.m.x_h"], deleted=["inference_perf.m.x_d"])
    base = side(cmr, x_d={"survived": 10})

    verdict = cmr.judge(touched, base, side(cmr, x_h={"no tests": 4}), [])

    assert verdict.failures == ["No test runs `inference_perf.m.x_h`"]


# Undecorated and lone-staticmethod functions are mutated, so they add nothing. Skipped bodies that
# only delegate add nothing either: a property returning an attribute, a setter assigning one, a
# validator calling a helper, an abstract stub. Anything else adds every statement: one's cached
# return at line 25, fat's if, assignment and return at 28-30, and both's two lines at 34-35.
def test_unchecked_statements_counts_skipped_functions_that_do_work(cmr: Any) -> None:
    source = (
        "import functools\n"
        "def plain(x):\n    y = x\n    return y\n"
        "class C:\n"
        "    @staticmethod\n    def s(x):\n        y = x\n        return y\n"
        "    @property\n    def p(self):\n        return self._p\n"
        "    @p.setter\n    def q(self, v):\n        self._p = v\n"
        '    @model_validator(mode="after")\n    def thin(self):\n        """doc"""\n        return _check(self, strict=True)\n'
        "    @abstractmethod\n    def stub(self):\n        ...\n"
        "    @functools.cache\n    def one(x):\n        return max(1, round(x / 1024))\n"
        '    @model_validator(mode="after")\n'
        "    def fat(self):\n        if self.x:\n            self.y = 1\n        return self\n"
        "    @staticmethod\n    @functools.cache\n    def both(x):\n        y = x\n        return y\n"
    )

    found = [(u.line, u.why) for u in cmr.unchecked_statements(source, [])]

    assert found == [
        (25, "decorated function"),
        (28, "decorated function"),
        (29, "decorated function"),
        (30, "decorated function"),
        (34, "decorated function"),
        (35, "decorated function"),
    ]


# Three pragma comments, one trailing code and a start/end pair on their own lines, plus a string
# that only looks like one. The comments go, the string stays, the line count holds, and it returns 3.
def test_strip_no_mutate_pragmas_removes_comments_only(cmr: Any, tmp_path: Path) -> None:
    src = tmp_path / "inference_perf" / "m.py"
    src.parent.mkdir()
    src.write_text(
        "x = 1  # pragma: no mutate, int() raises anyway\n"
        "# pragma: no mutate start\n"
        "s = '# pragma: no mutate'\n"
        "# pragma: no mutate end\n"
    )

    assert cmr.strip_no_mutate_pragmas(tmp_path) == 3
    assert src.read_text() == "x = 1\n\ns = '# pragma: no mutate'\n\n"


# The base commit's pyproject excludes a.py. The working tree swaps that for b.py.
# The config comes from the base, so the change cannot exclude its own file.
def test_load_config_reads_the_base_branch(cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "pyproject.toml").write_text('[tool.mutmut]\ndo_not_mutate = ["inference_perf/a.py"]\n')
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "base")
    (tmp_path / "pyproject.toml").write_text('[tool.mutmut]\ndo_not_mutate = ["inference_perf/b.py"]\n')
    monkeypatch.chdir(tmp_path)

    assert cmr.load_config("HEAD") == {"do_not_mutate": ["inference_perf/a.py"]}


# The base commit predates [tool.mutmut]. Main's table excludes a.py and the working tree's
# excludes b.py. Main's wins, so a PR into an older branch cannot write its own config either.
def test_load_config_falls_back_to_main(cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    (tmp_path / "pyproject.toml").write_text("[tool.other]\n")
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "old base")
    git(tmp_path, "checkout", "-qb", "main-tip")
    (tmp_path / "pyproject.toml").write_text('[tool.mutmut]\ndo_not_mutate = ["inference_perf/a.py"]\n')
    git(tmp_path, "commit", "-qam", "table lands")
    git(tmp_path, "checkout", "-q", "HEAD~1")
    (tmp_path / "pyproject.toml").write_text('[tool.mutmut]\ndo_not_mutate = ["inference_perf/b.py"]\n')
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cmr, "DEFAULT_BRANCH", "main-tip")

    assert cmr.load_config("HEAD") == {"do_not_mutate": ["inference_perf/a.py"]}


# g had 4 mutants on the base and none with the change, as when a decorator hides it. That fails
# even though its survivors fell from 1 to 0.
def test_judge_fails_when_a_function_loses_all_its_mutants(cmr: Any) -> None:
    touched = cmr.Touched(existing=[("inference_perf.m.x_g", "inference_perf.m.x_g")])
    base = side(cmr, x_g={"killed": 3, "survived": 1})

    verdict = cmr.judge(touched, base, side(cmr), [])

    assert verdict.failures == ["`inference_perf.m.x_g` had 4 mutants on the base and none with this change"]


# The head's tests failed the canary run, so they react to the mutation harness. That fails on its
# own, with no survivors anywhere.
def test_judge_fails_when_the_canary_fails(cmr: Any) -> None:
    head = side(cmr, x_h={"killed": 5})
    head.canary_failed = True

    verdict = cmr.judge(cmr.Touched(new=["inference_perf.m.x_h"]), None, head, [])

    assert verdict.failures == ["Tests fail with an active mutant id that changes no code, so they react to the harness"]


# A mutation tree whose tests/test_a.py has one passing and one failing test. The plain rerun names
# only the failing one and keeps the pytest summary line in the tail.
def test_diagnose_setup_failure_names_failing_tests(cmr: Any, tmp_path: Path) -> None:
    tests = tmp_path / "mutants" / "tests"
    tests.mkdir(parents=True)
    (tests / "test_a.py").write_text("def test_ok():\n    pass\n\n\ndef test_bad():\n    assert 1 == 2\n")

    failure = cmr.diagnose_setup_failure(cmr.Side(tree=tmp_path, tests=["tests/test_a.py"]), [])

    assert failure.failing == ["tests/test_a.py::test_bad"]
    assert "1 failed, 1 passed" in failure.tail


# test_harness fails only where the child plugin has set MUTATION_CHILD_CONTEXT, as it does inside
# mutmut's run. The diagnosis loads the plugin too, so it names that test instead of finding nothing.
def test_diagnose_setup_failure_reproduces_the_harness(cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cmr, "HARNESS_DIR", tmp_path / "harness")
    tests = tmp_path / "mutants" / "tests"
    tests.mkdir(parents=True)
    (tests / "test_a.py").write_text(
        "import os\n\n\ndef test_harness():\n    assert 'MUTATION_CHILD_CONTEXT' not in os.environ\n"
    )

    failure = cmr.diagnose_setup_failure(cmr.Side(tree=tmp_path, tests=["tests/test_a.py"]), [])

    assert failure.failing == ["tests/test_a.py::test_harness"]


# mutmut's stats run fails twice in a row, so run_mutmut raises the diagnosis after exactly two runs.
# Failing once and passing on retry returns normally and leaves a note.
def test_run_mutmut_diagnoses_a_setup_failure_that_repeats(cmr: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    failed = subprocess.CompletedProcess([], 1, "failed to collect stats. runner returned 1\n", "")
    passed = subprocess.CompletedProcess([], 0, "done\n", "")
    diagnosis = cmr.SetupFailure(Path("tree"), ["tests/test_a.py::test_bad"], "1 failed")
    monkeypatch.setattr(cmr, "diagnose_setup_failure", lambda side, args: diagnosis)

    runs = iter([failed, failed])
    monkeypatch.setattr(cmr, "mutmut", lambda *a: next(runs))
    with pytest.raises(cmr.SetupFailure) as raised:
        cmr.run_mutmut(cmr.Side(tree=Path("tree")), [], 1, [])
    assert raised.value is diagnosis and next(runs, None) is None

    runs = iter([failed, passed])
    flaky = cmr.Side(tree=Path("tree"))
    assert cmr.run_mutmut(flaky, [], 1, []) == ("done\n", False)
    assert flaky.note == "mutmut's setup test run failed once and passed on retry"


# Inside GitHub Actions an error prints one ::error annotation with the first line, and the whole
# message goes to the job summary in a code block.
def test_report_error_annotates_and_summarises(
    cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    summary = tmp_path / "summary.md"
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))

    cmr.report_error("The base branch is red.\nFailing tests:\n  tests/test_a.py::test_bad")

    assert capsys.readouterr().out == "::error title=Mutation ratchet::The base branch is red.\n"
    assert "```\nThe base branch is red.\nFailing tests:\n  tests/test_a.py::test_bad\n```" in summary.read_text()


# do_not_mutate = ["inference_perf/datagen/*", "skip.py"]. The glob excludes a file two levels below
# datagen/, since `*` spans directories, and a bare file name excludes nothing outside the repo root.
def test_mutable_matches_do_not_mutate_like_mutmut(cmr: Any) -> None:
    patterns = ["inference_perf/datagen/*", "skip.py"]

    assert not cmr.mutable("inference_perf/datagen/sub/b.py", patterns)
    assert cmr.mutable("inference_perf/loadgen/skip.py", patterns)
    assert cmr.mutable("inference_perf/loadgen/a.py", patterns)


# Stats measured test a at 2 s and b at 10 s. Rerunning a plus a test with no measurement gets
# 60 s of grace plus 5 x 2 s, so 70 s.
def test_rerun_timeout_scales_the_measured_durations(cmr: Any, tmp_path: Path) -> None:
    (tmp_path / "mutants").mkdir()
    (tmp_path / "mutants" / "mutmut-stats.json").write_text('{"duration_by_test": {"a": 2.0, "b": 10.0}}')

    assert cmr.rerun_timeout(tmp_path, ["a", "missing"]) == 70.0


# With a 1 s rerun limit, test_slow_fail sleeps 20 s and then fails, and test_slow_pass sleeps 20 s
# and then passes. Both hang. The survivor still counts, since a hang is not a kill, and the canary
# fails, since a hang is not a pass. Without the limit each answer flips.
def test_hung_reruns_fail_closed(cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tests = tmp_path / "mutants" / "tests"
    tests.mkdir(parents=True)
    (tests / "test_a.py").write_text(
        "import time\n\n\n"
        "def test_slow_fail():\n    time.sleep(20)\n    assert False\n\n\n"
        "def test_slow_pass():\n    time.sleep(20)\n"
    )
    (tmp_path / "mutants" / "mutmut-stats.json").write_text(
        '{"tests_by_mangled_function_name": {"m.x_f": ["tests/test_a.py::test_slow_pass"]}, "duration_by_test": {}}'
    )
    monkeypatch.setattr(cmr, "RERUN_GRACE_S", 1)
    listed = subprocess.CompletedProcess([], 0, "tests/test_a.py::test_slow_fail\n", "")
    monkeypatch.setattr(cmr, "mutmut", lambda *a: listed)
    side = cmr.Side(tree=tmp_path, results={"m.x_f__mutmut_1": "survived"})

    assert cmr.still_survives(side, "m.x_f__mutmut_1", []).counts
    assert not cmr.canary_passes(side, ["m.x_f"], [])


# Faked reruns. Exit 1 naming FAILED tests/test_a.py::t three times is a kill, so the survivor stops
# counting. Exit 4 (usage error), exit 5 (nothing collected) and exit 1 with no test named each
# stop the run with an error that quotes the output, since none of them ran a test that failed.
# The canary accepts 0 and 5, fails on 1, and stops the run on 4 the same way.
def test_reruns_count_only_a_named_failing_test_as_a_kill(cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    listed = subprocess.CompletedProcess([], 0, "tests/test_a.py::t\n", "")
    monkeypatch.setattr(cmr, "mutmut", lambda *a: listed)
    monkeypatch.setattr(cmr, "rerun_timeout", lambda *a: 1.0)
    side = cmr.Side(tree=tmp_path, results={"m.x_f__mutmut_1": "survived"})
    (tmp_path / "mutants").mkdir()
    (tmp_path / "mutants" / "mutmut-stats.json").write_text(
        '{"tests_by_mangled_function_name": {"m.x_f": ["tests/test_a.py::t"]}, "duration_by_test": {}}'
    )

    def fake(code: int, output: str) -> None:
        monkeypatch.setattr(cmr, "rerun", lambda *a: (code, output))

    fake(1, "FAILED tests/test_a.py::t - assert\n")
    assert not cmr.still_survives(side, "m.x_f__mutmut_1", []).counts
    assert not cmr.canary_passes(side, ["m.x_f"], [])
    for code, output in [(4, "ERROR: not found: tests/test_a.py::t\n"), (5, "no tests ran\n"), (1, "1 failed\n")]:
        fake(code, output)
        with pytest.raises(cmr.MutationRunError, match=f"exited {code} .*\n{output.strip()}"):
            cmr.still_survives(side, "m.x_f__mutmut_1", [])
    fake(4, "usage: pytest\n")
    with pytest.raises(cmr.MutationRunError, match="the canary exited 4"):
        cmr.canary_passes(side, ["m.x_f"], [])
    fake(5, "")
    assert cmr.canary_passes(side, ["m.x_f"], [])


# Commits inference_perf/m.py with f(x) = x > 1 and tests/test_m.py, whose test_f pins both sides of
# the comparison. Returns the repo root.
def tested_function_repo(tmp_path: Path) -> Path:
    (tmp_path / "inference_perf").mkdir(parents=True)
    (tmp_path / "inference_perf" / "__init__.py").write_text("")
    (tmp_path / "inference_perf" / "m.py").write_text("def f(x):\n    return x > 1\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_m.py").write_text(
        "from inference_perf.m import f\n\n\ndef test_f():\n    assert f(2)\n    assert not f(1)\n"
    )
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "base")
    return tmp_path


# Scope after a change: the functions the diff touches plus those the changed tests run, using an
# export of HEAD as the base tree and the working tree as the head.
def scope_after_change(cmr: Any, repo: Path, tmp_path: Path) -> Any:
    touched = cmr.touched_functions("HEAD", [])
    base_tree = tmp_path / "base-tree"
    cmr.export_base("HEAD", base_tree)
    base_tests, head_tests = cmr.tests_in_scope("HEAD", base_tree, repo)
    cmr.add_functions_run_by_changed_tests(touched, base_tree, repo, base_tests, head_tests)
    return touched


# The change deletes tests/test_m.py and touches nothing else. The deleted file still counts as a
# changed test, so f, which only it ran, comes into scope and its survivors can rise.
def test_deleting_a_test_file_brings_what_it_ran_into_scope(cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tested_function_repo(tmp_path / "repo")
    (repo / "tests" / "test_m.py").unlink()
    monkeypatch.chdir(repo)

    assert cmr.changed_test_files("HEAD") == ["tests/test_m.py"]
    assert scope_after_change(cmr, repo, tmp_path).via_tests == {"inference_perf.m.x_f"}


# The change renames tests/test_m.py to tests/checks_m.py, so it drops out of the unit suite. Both
# names count as changed tests, and f comes into scope.
def test_renaming_a_test_file_away_brings_what_it_ran_into_scope(
    cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tested_function_repo(tmp_path / "repo")
    git(repo, "mv", "tests/test_m.py", "tests/checks_m.py")
    monkeypatch.chdir(repo)

    assert cmr.changed_test_files("HEAD") == ["tests/checks_m.py", "tests/test_m.py"]
    assert scope_after_change(cmr, repo, tmp_path).via_tests == {"inference_perf.m.x_f"}


# tests/test_m.py has test_a running f and test_b running g. The change edits only test_b's assert,
# adds test_c, and reformats the import. Only test_b is selected on each side, so g joins the scope
# and f, which only the untouched test_a runs, does not.
def test_editing_one_test_scopes_only_what_it_runs(cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    repo = tmp_path / "repo"
    write_tree(
        repo,
        {
            "inference_perf/__init__.py": "",
            "inference_perf/m.py": "def f(x):\n    return x > 1\n\n\ndef g(x):\n    return x < 1\n",
            "tests/test_m.py": "from inference_perf.m import f, g\n\n\n"
            "def test_a():\n    assert f(2)\n\n\ndef test_b():\n    assert g(0)\n    assert not g(1)\n",
        },
    )
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    (repo / "tests" / "test_m.py").write_text(
        "from inference_perf.m import (\n    f,\n    g,\n)\n\n\n"
        "def test_a():\n    assert f(2)\n\n\ndef test_b():\n    assert g(0)\n\n\n"
        "# a new test only adds kills\ndef test_c():\n    assert f(3)\n"
    )
    monkeypatch.chdir(repo)
    base_tree = tmp_path / "base-tree"
    cmr.export_base("HEAD", base_tree)

    base_tests, head_tests = cmr.tests_in_scope("HEAD", base_tree, repo)

    assert (base_tests, head_tests) == (["tests/test_m.py::test_b"], ["tests/test_m.py::test_b"])
    touched = cmr.touched_functions("HEAD", [])
    cmr.add_functions_run_by_changed_tests(touched, base_tree, repo, base_tests, head_tests)
    assert touched.via_tests == {"inference_perf.m.x_g"}


# A file with helper check(), fixture value, constant N, class TestC (limit = N) with test_c, module-level
# tests test_a (uses check), test_b (uses value and N), test_d (uses os), test_f (holds a string with a
# comment line), and a Helper class pytest never collects. Each kind of change reaches the tests that read
# it, a comment or blank line outside a test reaches none, a rebound or swapped import reaches its users,
# and a bare statement, a pytestmark or a star import reaches the whole file.
def test_changed_tests_in_follows_names_to_the_tests_that_use_them(cmr: Any) -> None:
    base = (
        "import os\n"  # 1
        "import pytest\n"  # 2
        "\n"  # 3
        "N = 1\n"  # 4
        "\n"  # 5
        "\n"  # 6
        "def check(x):\n"  # 7
        "    assert x\n"  # 8
        "\n"  # 9
        "\n"  # 10
        "@pytest.fixture\n"  # 11
        "def value():\n"  # 12
        "    return 1\n"  # 13
        "\n"  # 14
        "\n"  # 15
        "class TestC:\n"  # 16
        "    limit = N\n"  # 17
        "\n"  # 18
        "    def test_c(self):\n"  # 19
        "        assert self.limit\n"  # 20
        "\n"  # 21
        "\n"  # 22
        "# about test_a\n"  # 23
        "def test_a():\n"  # 24
        "    check(True)\n"  # 25
        "\n"  # 26
        "\n"  # 27
        "def test_b(value):\n"  # 28
        "    assert value == N\n"  # 29
        "\n"  # 30
        "\n"  # 31
        "def test_d():\n"  # 32
        "    assert os.sep\n"  # 33
        "\n"  # 34
        "\n"  # 35
        "def test_f():\n"  # 36
        '    s = """\n'  # 37
        "    # inside a string\n"  # 38
        "\n"  # 39
        '    """\n'  # 40
        "    assert s\n"  # 41
        "\n"  # 42
        "\n"  # 43
        "class Helper:\n"  # 44
        "    def test_x(self):\n"  # 45
        "        pass\n"  # 46
    )

    def selected(head: str, old: set[int], new: set[int]) -> set[str] | None:
        result: set[str] | None = cmr.changed_tests_in(base, head, old, new, "tests/test_x.py")
        return result

    assert "Helper::test_x" not in cmr.test_ids(base)  # pytest collects Test* classes only
    assert selected(base, {8}, {8}) == {"test_a"}  # helper body
    assert selected(base, {13}, {13}) == {"test_b"}  # fixture body
    assert selected(base, {4}, {4}) == {"test_b", "TestC::test_c"}  # constant, read by a class attribute too
    assert selected(base, {17}, {17}) == {"TestC::test_c"}  # class attribute
    assert selected(base, {25}, {25}) == {"test_a"}  # a test's own line
    assert selected(base, {23, 3}, {23, 3}) == set()  # comment and blank
    assert selected(base, {38}, {38}) == {"test_f"}  # a comment line inside a test's string
    assert selected(base, {39}, {39}) == {"test_f"}  # a blank line inside a test's string
    assert selected(base, {45}, {45}) == set()  # a class pytest does not collect is a helper
    decorated = base.replace("@pytest.fixture\n", '@pytest.fixture(scope="module")\n')
    assert selected(decorated, {11}, {11}) == {"test_b"}  # a fixture's decorator
    assert selected(base.replace("import os\n", "import sys\n"), set(), {1}) == {"test_d"}  # import rebound
    assert selected(base.replace("import os\n", "import posixpath as os\n"), {1}, {1}) == {"test_d"}  # source swapped
    assert selected(base, set(), {30}) == set()  # nothing added where only blanks sit
    assert selected(base + "print(N)\n", set(), {47}) is None  # module-level statement
    assert selected(base + "pytestmark = pytest.mark.skip\n", set(), {47}) is None  # pytest reads it for the file
    assert selected(base + "from os import *\n", set(), {47}) is None  # can rebind anything
    assert selected(base + "def broken(:\n", set(), {47}) is None  # does not parse
    # a test added on the head is not selected; a test deleted from the base is
    added = base + "\n\ndef test_e():\n    check(1)\n"
    assert selected(added, set(), {49, 50}) == set()
    assert selected(base.replace("def test_d():\n    assert os.sep\n", ""), {32, 33}, set()) == {"test_d"}


# tests/test_x.py has constant N, helper check(), and class TestC with limit = N, test_c reading
# self.limit and test_g calling check. Editing N reaches both tests through the class attribute.
# Editing check's body reaches test_g only, not its sibling test_c.
def test_a_symbol_read_by_one_method_reaches_only_that_method(cmr: Any) -> None:
    base = (
        "N = 1\n"  # 1
        "\n"  # 2
        "\n"  # 3
        "def check(x):\n"  # 4
        "    assert x\n"  # 5
        "\n"  # 6
        "\n"  # 7
        "class TestC:\n"  # 8
        "    limit = N\n"  # 9
        "\n"  # 10
        "    def test_c(self):\n"  # 11
        "        assert self.limit\n"  # 12
        "\n"  # 13
        "    def test_g(self):\n"  # 14
        "        check(1)\n"  # 15
    )

    def selected(old: set[int], new: set[int]) -> set[str] | None:
        result: set[str] | None = cmr.changed_tests_in(base, base, old, new, "tests/test_x.py")
        return result

    assert selected({1}, {1}) == {"TestC::test_c", "TestC::test_g"}
    assert selected({5}, {5}) == {"TestC::test_g"}


# tests/test_x.py has constant TOL, constant LIMIT, and class TestC with helper _close reading TOL,
# fixture thing reading LIMIT, test_a calling self._close, test_b requesting thing, and test_c using
# neither. Editing TOL reaches test_a only. Editing LIMIT reaches test_b only. Editing _close's body
# or thing's body is a class line and reaches all three.
def test_a_symbol_reaches_a_test_through_a_sibling_helper_or_fixture(cmr: Any) -> None:
    base = (
        "import pytest\n"  # 1
        "\n"  # 2
        "TOL = 1\n"  # 3
        "LIMIT = 2\n"  # 4
        "\n"  # 5
        "\n"  # 6
        "class TestC:\n"  # 7
        "    def _close(self, x):\n"  # 8
        "        return x < TOL\n"  # 9
        "\n"  # 10
        "    @pytest.fixture\n"  # 11
        "    def thing(self):\n"  # 12
        "        return LIMIT\n"  # 13
        "\n"  # 14
        "    def test_a(self):\n"  # 15
        "        assert self._close(0)\n"  # 16
        "\n"  # 17
        "    def test_b(self, thing):\n"  # 18
        "        assert thing\n"  # 19
        "\n"  # 20
        "    def test_c(self):\n"  # 21
        "        assert True\n"  # 22
    )

    def selected(old: set[int], new: set[int]) -> set[str] | None:
        result: set[str] | None = cmr.changed_tests_in(base, base, old, new, "tests/test_x.py")
        return result

    everything = {"TestC::test_a", "TestC::test_b", "TestC::test_c"}
    assert selected({3}, {3}) == {"TestC::test_a"}
    assert selected({4}, {4}) == {"TestC::test_b"}
    assert selected({9}, {9}) == everything
    assert selected({13}, {13}) == everything


# tests/test_x.py imports Suite and has class TestV(Suite) with attribute url and own test_a, plus class
# TestO with test_o. TestV inherits its other tests, so pytest's id for the class itself is listed after
# TestV::test_a. Editing url or adding an override to TestV selects the class id, the test_a line
# selects test_a, and swapping the import's source selects both. Editing test_o selects test_o only.
def test_a_class_with_inherited_tests_is_selected_by_its_own_node_id(cmr: Any) -> None:
    base = (
        "from suite import Suite\n"  # 1
        "\n"  # 2
        "\n"  # 3
        "class TestV(Suite):\n"  # 4
        '    url = "http://x"\n'  # 5
        "\n"  # 6
        "    def test_a(self):\n"  # 7
        "        assert self.url\n"  # 8
        "\n"  # 9
        "\n"  # 10
        "class TestO:\n"  # 11
        "    def test_o(self):\n"  # 12
        "        assert True\n"  # 13
    )

    def selected(head: str, old: set[int], new: set[int]) -> set[str] | None:
        result: set[str] | None = cmr.changed_tests_in(base, head, old, new, "tests/test_x.py")
        return result

    assert list(cmr.test_ids(base)) == ["TestV::test_a", "TestV", "TestO::test_o"]
    assert "TestO" not in cmr.test_ids(base)  # no bases, so every test it runs has its own id
    assert selected(base, {5}, {5}) == {"TestV"}
    assert selected(base, {8}, {8}) == {"TestV::test_a"}
    assert selected(base, {13}, {13}) == {"TestO::test_o"}
    assert selected(base.replace("from suite import", "from other import"), {1}, {1}) == {"TestV", "TestV::test_a"}
    override = base.replace("        assert self.url\n", "        assert self.url\n\n    def test_z(self):\n        pass\n")
    assert selected(override, set(), {10, 11}) == {"TestV"}


# tests/test_x.py has an autouse fixture clock, a plain fixture value, test_a naming no fixture,
# test_b(value), and class TestC marked usefixtures("value") holding test_c. Editing clock's body
# reaches all three tests. Editing value's body reaches test_b and TestC::test_c, not test_a.
def test_changed_tests_in_reaches_tests_through_autouse_and_usefixtures(cmr: Any) -> None:
    base = (
        "import pytest\n"  # 1
        "\n"  # 2
        "\n"  # 3
        "@pytest.fixture(autouse=True)\n"  # 4
        "def clock():\n"  # 5
        "    return 1\n"  # 6
        "\n"  # 7
        "\n"  # 8
        "@pytest.fixture\n"  # 9
        "def value():\n"  # 10
        "    return 2\n"  # 11
        "\n"  # 12
        "\n"  # 13
        "def test_a():\n"  # 14
        "    assert True\n"  # 15
        "\n"  # 16
        "\n"  # 17
        "def test_b(value):\n"  # 18
        "    assert value\n"  # 19
        "\n"  # 20
        "\n"  # 21
        '@pytest.mark.usefixtures("value")\n'  # 22
        "class TestC:\n"  # 23
        "    def test_c(self):\n"  # 24
        "        assert True\n"  # 25
    )

    def selected(old: set[int], new: set[int]) -> set[str] | None:
        result: set[str] | None = cmr.changed_tests_in(base, base, old, new, "tests/test_x.py")
        return result

    assert selected({6}, {6}) == {"test_a", "test_b", "TestC::test_c"}  # autouse fixture body
    assert selected({11}, {11}) == {"test_b", "TestC::test_c"}  # fixture named by arg and by usefixtures


# A survivor's tests exit 4 under the mutant with an ERROR line naming the file, because the mutant
# broke the module at import. The same tests collect with the mutant off (the canary exits 1 with a
# failure), so it counts as a kill by that file. The canary runs once, not once per rerun. When the
# canary cannot collect either, the rerun is unusable.
def test_still_survives_counts_a_collection_failure_the_mutant_caused(
    cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    side = cmr.Side(tree=tmp_path, results={"m.x_f__mutmut_1": "survived"})
    (tmp_path / "mutants").mkdir()
    (tmp_path / "mutants" / "mutmut-stats.json").write_text(
        '{"tests_by_mangled_function_name": {"m.x_f": ["tests/test_a.py::t"]}, "duration_by_test": {}}'
    )
    monkeypatch.setattr(cmr, "mutmut", lambda *a: subprocess.CompletedProcess(a, 0, "tests/test_a.py::t\n", ""))
    broken = "ERROR tests/test_a.py - ValueError: buckets are only valid for Histogram metrics\n"
    calls: list[str] = []

    def rerun(side: Any, tests: list[str], mutant: str, args: list[str]) -> tuple[int, str]:
        calls.append(mutant)
        if mutant == "m.x_f__mutmut_1":
            return 4, broken
        return 1, "FAILED tests/test_a.py::t - flaky\n"

    monkeypatch.setattr(cmr, "rerun", rerun)
    survival = cmr.still_survives(side, "m.x_f__mutmut_1", [])
    assert not survival.counts
    assert survival.killed_by == ["tests/test_a.py"]
    assert len(calls) == cmr.RERUNS + 1
    assert len([m for m in calls if m != "m.x_f__mutmut_1"]) == 1

    monkeypatch.setattr(cmr, "rerun", lambda *a: (4, broken))
    with pytest.raises(cmr.MutationRunError, match="exited 4"):
        cmr.still_survives(side, "m.x_f__mutmut_1", [])


# Writes each {path: source} under `root`, creating directories.
def write_tree(root: Path, files: dict[str, str]) -> None:
    for rel, source in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(source)


# tests/test_m.py imports the sibling helper check from tests/helpers.py. The change loosens check to
# assert nothing and adds tests/test_new.py, which imports check too. test_m.py counts as changed on
# both sides, so f comes into scope; the added file only adds kills and is selected on neither side.
def test_loosening_a_test_helper_brings_what_its_tests_run_into_scope(
    cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    repo = tmp_path / "repo"
    write_tree(
        repo,
        {
            "inference_perf/__init__.py": "",
            "inference_perf/m.py": "def f(x):\n    return x > 1\n",
            "tests/helpers.py": "def check(value, expected):\n    assert value == expected\n",
            "tests/test_m.py": "from helpers import check\n\nfrom inference_perf.m import f\n\n\n"
            "def test_f():\n    check(f(2), True)\n    check(f(1), False)\n",
        },
    )
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    (repo / "tests" / "helpers.py").write_text("def check(value, expected):\n    pass\n")
    (repo / "tests" / "test_new.py").write_text("from helpers import check\n\n\ndef test_new():\n    check(1, 1)\n")
    monkeypatch.chdir(repo)
    base_tree = tmp_path / "base-tree"
    cmr.export_base("HEAD", base_tree)

    assert cmr.changed_test_files("HEAD") == ["tests/test_new.py"]
    base_tests, head_tests = cmr.tests_in_scope("HEAD", base_tree, repo)
    assert (base_tests, head_tests) == (["tests/test_m.py"], ["tests/test_m.py"])
    touched = cmr.touched_functions("HEAD", [])
    cmr.add_functions_run_by_changed_tests(touched, base_tree, repo, base_tests, head_tests)
    assert touched.via_tests == {"inference_perf.m.x_f"}


# One tree, four ways a test reaches changed support code:
# a/test_a.py imports a/helpers.py as a sibling, which imports a/base.py.
# pkg/test_p.py imports pkg/helpers.py relatively, inside a package.
# deep/x/test_x.py imports tests.shared.util by its full name.
# sub/test_s.py only shares sub/conftest.py, and sub/inner/test_i.py sits below it.
# tests/optional never counts. A change to a/base.py reaches only a/test_a.py, through helpers.py.
def test_tests_using_follows_imports_and_conftests(cmr: Any, tmp_path: Path) -> None:
    write_tree(
        tmp_path,
        {
            "tests/a/base.py": "X = 1\n",
            "tests/a/helpers.py": "from base import X\n",
            "tests/a/test_a.py": "from helpers import X\n",
            "tests/a/test_other.py": "import os\n",
            "tests/pkg/__init__.py": "",
            "tests/pkg/helpers.py": "Y = 1\n",
            "tests/pkg/test_p.py": "from .helpers import Y\n",
            "tests/shared/util.py": "Z = 1\n",
            "tests/deep/x/test_x.py": "from tests.shared import util\n",
            "tests/sub/conftest.py": "",
            "tests/sub/test_s.py": "",
            "tests/sub/inner/test_i.py": "",
            "tests/optional/test_opt.py": "from tests.shared.util import Z\n",
        },
    )

    assert cmr.tests_using(tmp_path, ["tests/a/base.py"]) == {"tests/a/test_a.py"}
    assert cmr.tests_using(tmp_path, ["tests/pkg/helpers.py"]) == {"tests/pkg/test_p.py"}
    assert cmr.tests_using(tmp_path, ["tests/shared/util.py"]) == {"tests/deep/x/test_x.py"}
    assert cmr.tests_using(tmp_path, ["tests/sub/conftest.py"]) == {"tests/sub/test_s.py", "tests/sub/inner/test_i.py"}
    assert cmr.tests_using(tmp_path, ["tests/pkg/__init__.py"]) == {"tests/pkg/test_p.py"}
    assert cmr.tests_using(tmp_path, ["tests/gone.py"]) == set()


# Repo roots: inference_perf, tests, tools (with a nested .py) and a root helper.py. docs/ holds no
# Python. Every root holding Python except inference_perf itself is first-party code outside it.
def test_first_party_roots_are_every_python_root_but_the_package(cmr: Any) -> None:
    paths = ["inference_perf/a.py", "tests/test_a.py", "tools/x/y.py", "helper.py", "docs/a.md", "README.md"]

    assert cmr.first_party_roots(paths) == {"tests", "tools", "helper"}


# inference_perf/a.py imports tests.util, inference_perf/b.py imports tools and loads e2e.run by name,
# and inference_perf/c.py only imports yaml, its own package and a sibling. a and b fail, c does not.
def test_outside_imports_flags_first_party_code_outside_the_package(cmr: Any, tmp_path: Path) -> None:
    write_tree(
        tmp_path,
        {
            "inference_perf/a.py": "from tests.util import check\n",
            "inference_perf/b.py": "import importlib\nimport tools.gen\n\nrun = importlib.import_module('e2e.run')\n",
            "inference_perf/c.py": "import yaml\nfrom inference_perf import a\nfrom . import b\n",
        },
    )

    hits = cmr.outside_imports(
        tmp_path, ["inference_perf/a.py", "inference_perf/b.py", "inference_perf/c.py"], {"tests", "tools", "e2e"}
    )

    assert hits == [
        "inference_perf/a.py:1 imports `tests`",
        "inference_perf/b.py:2 imports `tools`",
        "inference_perf/b.py:4 imports `e2e`",
    ]


# The change adds yaml.py, a json package and mine.py at the repo root, and keeps the existing tests/.
# yaml is installed and json is in the standard library, so both shadow. mine.py shadows nothing.
def test_shadowing_roots_flags_new_root_modules_with_taken_names(cmr: Any) -> None:
    base = ["inference_perf/a.py", "tests/__init__.py"]
    head = base + ["yaml.py", "json/__init__.py", "mine.py"]

    assert cmr.shadowing_roots(base, head) == ["json", "yaml"]


# The change only adds a root yaml.py, so no function or test is in scope. The check still fails on
# the shadowing module instead of passing with nothing to mutate.
def test_main_fails_on_a_shadowing_module_with_nothing_to_mutate(
    cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_tree(
        tmp_path,
        {
            "pyproject.toml": '[tool.mutmut]\nsource_paths = ["inference_perf"]\n',
            "inference_perf/__init__.py": "",
            "inference_perf/m.py": "import yaml\n",
        },
    )
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "base")
    (tmp_path / "yaml.py").write_text("def safe_load(s):\n    return {}\n")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr("sys.argv", ["check", "--base", "HEAD"])

    with pytest.raises(SystemExit) as exited:
        cmr.main()

    assert exited.value.code == 1
    assert "`yaml`" in capsys.readouterr().out


MUTMUT_CONFIG: dict[str, Any] = {
    "do_not_mutate_patterns": ["logger\\.", "raise \\w+\\("],
    "do_not_mutate": ["inference_perf/skip.py"],
}


# Commits `base` files plus a pdm.lock pinning mutmut, writes `head` over them in the working tree, and
# returns the checks that need no mutants, run with MUTMUT_CONFIG as the base's [tool.mutmut].
def static_result(cmr: Any, repo: Path, monkeypatch: pytest.MonkeyPatch, base: dict[str, str], head: dict[str, str]) -> Any:
    write_tree(repo, {"inference_perf/__init__.py": "", "pdm.lock": MUTMUT_LOCK_ENTRY, **base})
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    write_tree(repo, head)
    monkeypatch.chdir(repo)
    monkeypatch.setattr(cmr, "installed_mutmut", lambda: FakeDistribution("3.8.0"), raising=False)
    return cmr.static_checks("HEAD", MUTMUT_CONFIG)


# static_result's failures alone.
def static_failures(
    cmr: Any, repo: Path, monkeypatch: pytest.MonkeyPatch, base: dict[str, str], head: dict[str, str]
) -> list[str]:
    failures: list[str] = static_result(cmr, repo, monkeypatch, base, head).failures
    return failures


GT = "def f(x):\n    return x > 1\n"

# Each change moves f's `x > 1` somewhere mutmut 3.8 never mutates, keeping behaviour the same.
HIDING_SPOTS = {
    "module-level lambda": "_gt = lambda x: x > 1\n\n\ndef f(x):\n    return _gt(x)\n",
    "class body": "class C:\n    BIG = [i for i in range(5) if i > 1]\n\n\ndef f(x):\n    return x in C.BIG\n",
    "nested class method": "class C:\n    class Inner:\n        def gt(self, x):\n            return x > 1\n\n\n"
    "def f(x):\n    return C.Inner().gt(x)\n",
    "nested decorated function": "import functools\n\n\ndef f(x):\n    @functools.cache\n    def gt(y):\n"
    "        return y > 1\n\n    return gt(x)\n",
    "function inside a module-level if": "import sys\n\nif sys.platform:\n\n    def gt(x):\n        return x > 1\n\n\n"
    "def f(x):\n    return gt(x)\n",
    "logger line": "import logging\n\nlogger = logging.getLogger(__name__)\n\n\n"
    "def f(x):\n    return x > 1 or logger.disabled\n",
    "comment matching a pattern": "def f(x):\n    return x > 1  # see logger.debug\n",
    "len() argument": "def f(x):\n    return len(range(x - 1)) > 0\n",
    "exec": "def f(x):\n    scope = {'x': x}\n    exec('r = x > 1', scope)\n    return scope['r']\n",
    "default argument": "def f(x, big=lambda y: y > 1):\n    return big(x)\n",
    "decorated function": "import functools\n\n\n@functools.cache\ndef f(x):\n    y = x\n    return y > 1\n",
}


# Base: inference_perf/m.py with f(x) = x > 1, fully mutated. Head: the same logic moved to one of
# HIDING_SPOTS. Unchecked statements rise from 0, so the change fails.
@pytest.mark.parametrize("spot", sorted(HIDING_SPOTS))
def test_logic_moved_where_mutmut_never_looks_fails(
    cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, spot: str
) -> None:
    failures = static_failures(
        cmr, tmp_path, monkeypatch, {"inference_perf/m.py": GT}, {"inference_perf/m.py": HIDING_SPOTS[spot]}
    )

    assert any(f.startswith("Statements mutmut never mutates rose from 0 to ") for f in failures), failures


# Base: skip.py, which do_not_mutate excludes, holds g, which only returns 1. Head adds gt(x) = x > 1
# to it and f calls it. gt's return is unchecked and g's plain return is not, so 0 rises to 1.
def test_logic_moved_into_an_excluded_file_fails(cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    base = {"inference_perf/m.py": GT, "inference_perf/skip.py": "def g():\n    return 1\n"}
    head = {
        "inference_perf/m.py": "from inference_perf.skip import gt\n\n\ndef f(x):\n    return gt(x)\n",
        "inference_perf/skip.py": "def g():\n    return 1\n\n\ndef gt(x):\n    return x > 1\n",
    }

    checks = static_result(cmr, tmp_path, monkeypatch, base, head)

    assert "Statements mutmut never mutates rose from 0 to 1" in checks.failures
    # moving the logic within skip.py cannot help, so the hint says to leave the file
    assert "Move the logic into a file do_not_mutate does not exclude" in checks.sections[-1]
    assert "undecorated" not in checks.sections[-1]
    assert [(a.line, a.message.split(". ")[-1]) for a in checks.annotations] == [
        (6, "Move the logic into a file do_not_mutate does not exclude and call it from here.")
    ]


# The head's m.py is missing a closing parenthesis. The run fails on "does not parse" with the file
# named, writes the summary and the record, and exits 1 instead of raising a SyntaxError.
def test_main_fails_on_a_file_that_does_not_parse(
    cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    write_tree(tmp_path, {"inference_perf/__init__.py": "", "inference_perf/m.py": GT, "pdm.lock": MUTMUT_LOCK_ENTRY})
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "base")
    write_tree(tmp_path, {"inference_perf/m.py": "def f(x:\n    return x > 1\n"})
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(cmr, "installed_mutmut", lambda: FakeDistribution("3.8.0"), raising=False)
    monkeypatch.setattr(cmr, "load_config", lambda base: MUTMUT_CONFIG)
    monkeypatch.setattr("sys.argv", ["check", "--base", "HEAD", "--output", str(tmp_path / "mutation.json")])

    with pytest.raises(SystemExit) as exited:
        cmr.main()

    assert exited.value.code == 1
    assert "FAIL: inference_perf/m.py does not parse" in capsys.readouterr().out
    assert json.loads((tmp_path / "mutation.json").read_text())["failures"] == ["inference_perf/m.py does not parse"]


# Head adds only declarations next to f: constants, constant arithmetic, __all__, a type alias, a
# TYPE_CHECKING import block, a docstring, a pydantic field with Field(...), a delegating validator,
# a staticmethod with logic, and a log call and a raise inside f. None of it is unchecked logic.
def test_declarations_and_mutated_code_pass(cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    head = (
        '"""Module docstring."""\n'
        "import logging\n"
        "from typing import TYPE_CHECKING\n\n"
        "from pydantic import BaseModel, Field, field_validator\n\n"
        "if TYPE_CHECKING:\n    import os\n\n"
        "__all__ = ['f']\n"
        "TIMEOUT_S = 60 * 5\n"
        "Number = int | float\n"
        "logger = logging.getLogger(__name__)\n\n\n"
        "class Config(BaseModel):\n"
        '    """Config."""\n\n'
        "    limit: int = Field(default=3, ge=0)\n\n"
        "    @field_validator('limit')\n    @classmethod\n    def check(cls, v):\n        return _check(v)\n\n"
        "    @staticmethod\n    def half(x):\n        return x / 2\n\n\n"
        "def _check(v):\n    if v > 10:\n        raise ValueError(f'limit {v} is over 10')\n    return v\n\n\n"
        "def f(x):\n    logger.info('f(%s)', x)\n    return x > 1\n"
    )

    assert static_failures(cmr, tmp_path, monkeypatch, {"inference_perf/m.py": GT}, {"inference_perf/m.py": head}) == []


# g survives 2 mutants on the base, `x > 1` to `x >= 1` and `x > 1` to `x > 2`. The head keeps both
# and adds a third, `y + 1` to `y - 1`. Only the third is rerun, and it still survives, so 2 rises to 3.
# The two the base already had count on both sides, so a rerun that killed one by chance cannot lower the head.
def test_judge_reruns_only_survivors_new_on_the_head(cmr: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    base = side(cmr, x_g={"survived": 2})
    head = side(cmr, x_g={"survived": 3})
    diffs = {
        (id(base), "inference_perf.m.x_g__mutmut_1"): ("-    return x > 1", "+    return x >= 1"),
        (id(base), "inference_perf.m.x_g__mutmut_2"): ("-    return x > 1", "+    return x > 2"),
        (id(head), "inference_perf.m.x_g__mutmut_1"): ("-    y = y + 1", "+    y = y - 1"),
        (id(head), "inference_perf.m.x_g__mutmut_2"): ("-    return x > 1", "+    return x > 2"),
        (id(head), "inference_perf.m.x_g__mutmut_3"): ("-    return x > 1", "+    return x >= 1"),
    }
    monkeypatch.setattr(cmr, "mutation_lines", lambda s, m: diffs[(id(s), m)])
    rerun: list[str] = []

    def record(s: Any, mutant: str, a: Any) -> Any:
        rerun.append(mutant)
        return cmr.Survival(True, [])

    monkeypatch.setattr(cmr, "still_survives", record)
    touched = cmr.Touched(existing=[("inference_perf.m.x_g", "inference_perf.m.x_g")])

    verdict = cmr.judge(touched, base, head, [])

    assert rerun == ["inference_perf.m.x_g__mutmut_1"]
    assert verdict.carried == ["inference_perf.m.x_g__mutmut_2", "inference_perf.m.x_g__mutmut_3"]
    assert (verdict.base_survivors, verdict.head_survivors) == (2, 3)


# g goes from 1 survivor to 2, and survivor 1 is the base's own. The new one is rerun: test_r kills it
# on the first rerun and misses it on the second, the way a test that fails at random would. Killed in
# only some reruns, it still counts, so the change fails and the report names test_r.
def test_judge_counts_a_survivor_only_some_reruns_kill(cmr: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    outcomes = iter([(1, "FAILED tests/test_a.py::test_r - assert 0\n"), (0, "1 passed\n")])
    monkeypatch.setattr(cmr, "rerun", lambda *a: next(outcomes))
    listed = subprocess.CompletedProcess([], 0, "tests/test_a.py::test_r\n", "")
    monkeypatch.setattr(cmr, "mutmut", lambda *a: listed)
    monkeypatch.setattr(cmr, "mutation_lines", lambda s, m: (m,))
    touched = cmr.Touched(existing=[("inference_perf.m.x_g", "inference_perf.m.x_g")])

    verdict = cmr.judge(touched, side(cmr, x_g={"survived": 1}), side(cmr, x_g={"survived": 2}), [])

    assert verdict.failures == ["Surviving mutants rose from 1 to 2"]
    assert verdict.flaky == {"inference_perf.m.x_g__mutmut_2": ["tests/test_a.py::test_r"]}


# A 0.5 s budget and a command whose shell starts a background sleep and then sleeps itself, both for
# 30 s. The run stops at the budget with BudgetExhausted, and the background sleep, a grandchild, is
# killed with it instead of holding the runner.
def test_run_bounded_stops_the_whole_process_group_at_the_budget(
    cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(cmr, "BUDGET", cmr.Budget(0.5 / 60))
    pid_file = tmp_path / "pid"

    with pytest.raises(cmr.BudgetExhausted):
        cmr.run_bounded(["sh", "-c", f"sleep 30 & echo $! > {pid_file}; sleep 30"], cwd=tmp_path)

    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        for _ in range(50):
            os.kill(pid, 0)
            time.sleep(0.1)


# A command's own 0.2 s limit binds before a 10 minute budget, so it raises TimeoutExpired, which the
# caller treats as a hang, not BudgetExhausted. With no budget at all, a quick command just runs.
def test_run_bounded_keeps_the_callers_own_timeout(cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cmr, "BUDGET", cmr.Budget(10))
    with pytest.raises(subprocess.TimeoutExpired):
        cmr.run_bounded(["sleep", "5"], cwd=tmp_path, timeout=0.2)

    monkeypatch.setattr(cmr, "BUDGET", cmr.Budget(None))
    assert cmr.run_bounded(["echo", "hi"], cwd=tmp_path).stdout == "hi\n"


# f changes, the base side finishes, and the budget runs out while mutating the head. The run exits 1,
# and mutation.json and the summary name the budget, the stage, and f as not reached.
def test_main_reports_what_the_budget_did_not_reach(
    cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    write_tree(
        repo,
        {
            "pyproject.toml": '[tool.mutmut]\nsource_paths = ["inference_perf"]\n',
            "inference_perf/__init__.py": "",
            "inference_perf/m.py": GT,
        },
    )
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    (repo / "inference_perf" / "m.py").write_text("def f(x):\n    return x > 2\n")
    monkeypatch.chdir(repo)
    output = tmp_path / "mutation.json"
    monkeypatch.setattr("sys.argv", ["check", "--base", "HEAD", "--budget-minutes", "1", "--output", str(output)])
    monkeypatch.setattr(cmr, "static_checks", lambda *a: cmr.Checks())
    sides = iter([cmr.Side(tree=repo, results={"inference_perf.m.x_f__mutmut_1": "killed"})])

    def run_side(*a: Any, **k: Any) -> Any:
        side = next(sides, None)
        if side is None:
            raise cmr.BudgetExhausted()
        return side

    monkeypatch.setattr(cmr, "run_side", run_side)

    with pytest.raises(SystemExit) as exited:
        cmr.main()

    failure = "Ran out of the 1 minute budget while mutating the head. Not reached: `inference_perf.m.x_f`"
    assert exited.value.code == 1
    assert json.loads(output.read_text())["failures"] == [failure]
    assert failure in capsys.readouterr().out


# A test that reads the active mutant id and fails whenever one is set, except the id ending in
# 999999999 that a fixed canary would use. It would kill every real mutant and still pass that
# canary. A canary id picked at random each run fails it.
def test_canary_id_cannot_be_special_cased(cmr: Any, tmp_path: Path) -> None:
    tests = tmp_path / "mutants" / "tests"
    tests.mkdir(parents=True)
    (tests / "test_a.py").write_text(
        "import os\n\n\n"
        "def test_rigged():\n"
        "    active = os.environ.get('MUTANT' + '_UNDER_TEST', '')\n"
        "    assert not active or active.endswith('999999999')\n"
    )
    (tmp_path / "mutants" / "mutmut-stats.json").write_text(
        '{"tests_by_mangled_function_name": {"m.x_f": ["tests/test_a.py::test_rigged"]}, "duration_by_test": {}}'
    )
    side = cmr.Side(tree=tmp_path, results={"m.x_f__mutmut_1": "killed"})

    assert not cmr.canary_passes(side, ["m.x_f"], [])


# The change adds a conftest.py that fails every test while a real mutant is active, and a source
# file that reads the active mutant id. Both name the harness, so the change fails and the report
# points at each line. The script's own test file may name it and is not reported.
def test_tripwire_fails_code_that_names_the_harness(cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    head = {
        "tests/conftest.py": "import os\n\nIN_HARNESS = 'mutants' in __file__\n",
        "inference_perf/m.py": GT + "\n\nACTIVE = __import__('os').environ.get('MUTANT_UNDER_TEST')\n",
        "tests/required/scripts/test_check_mutation_regression.py": "# runs mutmut\n",
    }

    failures = static_failures(cmr, tmp_path, monkeypatch, {"inference_perf/m.py": GT}, head)

    assert "Code under inference_perf/ or tests/ names the mutation harness" in failures
    assert cmr.harness_references(
        ["tests/conftest.py", "inference_perf/m.py", "tests/required/scripts/test_check_mutation_regression.py"]
    ) == [
        "inference_perf/m.py:5",
        "tests/conftest.py:3",
    ]


MUTMUT_LOCK_ENTRY = """[[package]]
name = "mutmut"
version = "3.8.0"
requires_python = ">=3.10"
summary = "mutation testing for Python 3"
groups = ["mutation"]
files = [
    {file = "mutmut-3.8.0-py3-none-any.whl", hash = "sha256:4c0abcef6b8d1116f2adfaad19df12e1c621e4a402d4367ef15bf8cba497b91d"},
    {file = "mutmut-3.8.0.tar.gz", hash = "sha256:bc670276518f893ab356f1d28826c49b7e606e7cf89c300e3888c0b52fc1d2ed"},
]
"""


# Stands in for importlib.metadata's view of the installed mutmut.
class FakeDistribution:
    def __init__(self, version: str, direct_url: str | None = None) -> None:
        self.version, self.direct_url = version, direct_url

    def read_text(self, name: str) -> str | None:
        return self.direct_url if name == "direct_url.json" else None


# The locked and installed mutmut are the pinned 3.8.0 from the index: no problems. Each other case
# swaps one thing: the locked version, a file hash, a git source, the installed version, or an
# install from a direct URL. Each one is reported.
def test_mutmut_pin_problems(cmr: Any) -> None:
    good = FakeDistribution("3.8.0")
    forked = MUTMUT_LOCK_ENTRY.replace(
        'groups = ["mutation"]', 'groups = ["mutation"]\ngit = "https://example.com/mutmut.git"'
    )

    assert cmr.mutmut_pin_problems(MUTMUT_LOCK_ENTRY, good) == []
    assert cmr.mutmut_pin_problems(MUTMUT_LOCK_ENTRY.replace('version = "3.8.0"', 'version = "3.9.0"'), good) == [
        "pdm.lock pins mutmut 3.9.0, not 3.8.0"
    ]
    assert cmr.mutmut_pin_problems(MUTMUT_LOCK_ENTRY.replace("4c0abcef", "00000000"), good) == [
        "pdm.lock's mutmut file hashes are not the pinned release's"
    ]
    assert cmr.mutmut_pin_problems(forked, good) == ["pdm.lock installs mutmut from git"]
    assert cmr.mutmut_pin_problems(MUTMUT_LOCK_ENTRY, FakeDistribution("3.9.0")) == [
        "The installed mutmut is 3.9.0, not 3.8.0"
    ]
    assert cmr.mutmut_pin_problems(MUTMUT_LOCK_ENTRY, FakeDistribution("3.8.0", "{}")) == [
        "The installed mutmut came from a direct URL, not the package index"
    ]
    assert cmr.mutmut_pin_problems("", good) == ["pdm.lock has 0 mutmut entries, not 1"]


# The change repoints pdm.lock's mutmut at a git fork and keeps everything else. The check fails
# before mutating anything.
def test_a_swapped_mutmut_fails(cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    forked = MUTMUT_LOCK_ENTRY.replace(
        'groups = ["mutation"]', 'groups = ["mutation"]\ngit = "https://example.com/mutmut.git"'
    )

    failures = static_failures(cmr, tmp_path, monkeypatch, {"inference_perf/m.py": GT}, {"pdm.lock": forked})

    assert "mutmut is not the pinned release" in failures


# The command's shell starts a 30 s background sleep with its output closed and exits at once, the way
# a test can leave a Worker process behind. The run returns normally and the leftover sleep is killed.
def test_run_bounded_kills_what_a_finished_command_left_behind(cmr: Any, tmp_path: Path) -> None:
    pid_file = tmp_path / "pid"

    cmr.run_bounded(["sh", "-c", f"sleep 30 > /dev/null 2>&1 & echo $! > {pid_file}"], cwd=tmp_path)

    pid = int(pid_file.read_text())
    with pytest.raises(ProcessLookupError):
        for _ in range(50):
            os.kill(pid, 0)
            time.sleep(0.1)


# Two more ways a test reaches support code: test_req.py puts tests/optional on sys.path itself and
# imports harness.slots by that name, and sub/conftest.py loads tests/plugins/fix.py through
# pytest_plugins, which every test under sub/ then runs. A change to either file reaches those tests.
def test_tests_using_follows_sys_path_names_and_pytest_plugins(cmr: Any, tmp_path: Path) -> None:
    write_tree(
        tmp_path,
        {
            "tests/optional/harness/__init__.py": "",
            "tests/optional/harness/slots.py": "X = 1\n",
            "tests/test_req.py": "import sys\n\nsys.path.insert(0, 'tests/optional')\nfrom harness.slots import X\n",
            "tests/plugins/fix.py": "",
            "tests/sub/conftest.py": "pytest_plugins = ['tests.plugins.fix']\n",
            "tests/sub/test_s.py": "",
            "tests/test_other.py": "",
        },
    )

    assert cmr.tests_using(tmp_path, ["tests/optional/harness/slots.py"]) == {"tests/test_req.py"}
    assert cmr.tests_using(tmp_path, ["tests/plugins/fix.py"]) == {"tests/sub/test_s.py"}


# Runs pytest on `tests` in `project` with the child plugin loaded and `env` on top of a clean
# environment. Returns the exit code.
def pytest_with_plugin(cmr: Any, project: Path, tests: list[str], env: dict[str, str]) -> int:
    base = {k: v for k, v in os.environ.items() if k not in ("MUTANT_UNDER_TEST", "PYTHONPATH")}
    full = cmr.harness_env(project, {**base, **env})
    command = [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-p", cmr.CHILD_PLUGIN, *tests]
    return subprocess.run(command, cwd=project, env=full, capture_output=True).returncode


# One test forks a child that records a trampoline hit the way mutmut does, and the next test
# asserts the hit reached the parent's set. Under mutmut's stats run the plugin merges it at
# teardown and leaves no hits file behind, so the pair passes. Under any other run it does not.
def test_child_plugin_carries_child_hits_to_the_test(cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cmr, "HARNESS_DIR", tmp_path / "harness")
    project = tmp_path / "project"
    (project / "tests").mkdir(parents=True)
    (project / "tests" / "test_fork.py").write_text(
        "import multiprocessing as mp\n"
        "from mutmut.state import state\n\n\n"
        "def work():\n"
        "    state()._stats.add('inference_perf.m.x_child')\n\n\n"
        "def test_child_records_a_hit():\n"
        "    p = mp.get_context('fork').Process(target=work)\n"
        "    p.start()\n"
        "    p.join()\n"
        "    assert 'inference_perf.m.x_child' not in state()._stats\n\n\n"
        "def test_the_hit_reached_the_parent():\n"
        "    assert 'inference_perf.m.x_child' in state()._stats\n"
    )

    assert pytest_with_plugin(cmr, project, ["tests/test_fork.py"], {"MUTANT_UNDER_TEST": "stats"}) == 0
    assert list((tmp_path / "harness" / "hits" / "project").iterdir()) == []
    assert pytest_with_plugin(cmr, project, ["tests/test_fork.py"], {}) == 1


# inference_perf/m.py has in_child, which the test only runs in a forked process, and in_parent,
# run directly. Both count as run by the test.
def test_functions_run_by_sees_code_run_in_a_forked_child(cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cmr, "HARNESS_DIR", tmp_path / "harness")
    (tmp_path / "inference_perf").mkdir()
    (tmp_path / "inference_perf" / "__init__.py").write_text("")
    (tmp_path / "inference_perf" / "m.py").write_text("def in_child():\n    return 1\n\n\ndef in_parent():\n    return 2\n")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_m.py").write_text(
        "import multiprocessing as mp\n"
        "from inference_perf.m import in_child, in_parent\n\n\n"
        "def test_both():\n"
        "    p = mp.get_context('fork').Process(target=in_child)\n"
        "    p.start()\n"
        "    p.join()\n"
        "    assert in_parent() == 2\n"
    )

    assert cmr.functions_run_by(tmp_path, ["tests/test_m.py"]) == {
        "inference_perf.m.x_in_child",
        "inference_perf.m.x_in_parent",
    }


# tests/test_m.py imports a module that does not exist, so the node id tests/test_m.py::test_f
# cannot be collected and pytest exits 4 with no report. functions_run_by raises with the exit code
# and the ModuleNotFoundError line instead of the bare "produced no report".
def test_functions_run_by_names_a_collection_failure(cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cmr, "HARNESS_DIR", tmp_path / "harness")
    write_tree(
        tmp_path,
        {
            "inference_perf/__init__.py": "",
            "inference_perf/m.py": "def f():\n    return 1\n",
            "tests/test_m.py": "import nothere_xyz\n\n\ndef test_f():\n    assert nothere_xyz\n",
        },
    )

    with pytest.raises(cmr.MutationRunError, match=r"exited 4\n(.*\n)*.*ModuleNotFoundError.*nothere_xyz"):
        cmr.functions_run_by(tmp_path, ["tests/test_m.py::test_f"])


F = "inference_perf.m.x_f__mutmut_"


# f has 4 mutants. The changed tests catch 1 on both sides and miss 2 on both, so 2 are settled.
# They catch 3 only on the base and 4 only on the head: 3 is undecided on the head, 4 on the base.
# g has mutants 1-2 on the base and 1-3 on the head, so g runs in full like a changed function.
def test_split_by_changed_tests_leaves_only_disagreements(cmr: Any) -> None:
    g = "inference_perf.m.x_g__mutmut_"
    base = {f"{F}1": "killed", f"{F}2": "survived", f"{F}3": "killed", f"{F}4": "no tests", f"{g}1": "killed"}
    head = {f"{F}1": "timeout", f"{F}2": "no tests", f"{F}3": "survived", f"{F}4": "killed", f"{g}1": "killed"}
    base[f"{g}2"] = head[f"{g}2"] = head[f"{g}3"] = "killed"

    shortcut = cmr.split_by_changed_tests(
        ["inference_perf.m.x_f", "inference_perf.m.x_g"],
        cmr.Side(tree=Path("."), results=base),
        cmr.Side(tree=Path("."), results=head),
    )

    assert shortcut.settled == 2
    assert shortcut.undecided == {f"{F}3": "head", f"{F}4": "base"}
    assert shortcut.in_full == ["inference_perf.m.x_g"]
    assert (shortcut.on("base"), shortcut.on("head")) == ([f"{F}4"], [f"{F}3"])


# Every changed test is new, so the base has none to run. The head's changed tests catch f's
# mutant 1 and miss 2. 1 is undecided on the base, where nothing caught it, and 2 is settled.
def test_split_by_changed_tests_with_no_changed_tests_on_the_base(cmr: Any) -> None:
    head = cmr.Side(tree=Path("."), results={f"{F}1": "killed", f"{F}2": "survived"})

    shortcut = cmr.split_by_changed_tests(["inference_perf.m.x_f"], None, head)

    assert (shortcut.settled, shortcut.undecided, shortcut.in_full) == (1, {f"{F}1": "base"}, [])


# Mutant 1 was missed on the base and mutant 2 on the head. Each is recorded as killed on the side
# whose changed tests caught it, and each side keeps its own run's result for the one it missed.
def test_settle_undecided_marks_the_other_side_caught(cmr: Any) -> None:
    shortcut = cmr.Shortcut(undecided={f"{F}1": "base", f"{F}2": "head"})
    base = cmr.Side(tree=Path("."), results={f"{F}1": "survived"})
    head = cmr.Side(tree=Path("."), results={f"{F}2": "survived"})

    cmr.settle_undecided(shortcut, base, head)

    assert base.results == {f"{F}1": "survived", f"{F}2": "killed"}
    assert head.results == {f"{F}2": "survived", f"{F}1": "killed"}


# mutmut given no mutant names runs every mutant in the tree, and given no tests runs the whole tests
# directory. run_side with either returns an empty side and never writes config or starts mutmut.
def test_run_side_runs_nothing_without_names_or_tests(cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cmr, "mutmut", lambda *a: pytest.fail("mutmut started"))
    monkeypatch.setattr(cmr, "write_config", lambda *a: pytest.fail("config written"))

    assert cmr.run_side(tmp_path, [], {}, 1, 0).results == {}
    assert cmr.run_side(tmp_path, ["inference_perf.m.x_f"], {}, 1, 0, tests=[]).results == {}


# The changed tests are tests/test_new.py, which this side lacks, and tests/optional/test_o.py, which
# is never in the unit suite. The side has nothing to run, so it gets None and mutmut never starts.
def test_run_changed_tests_with_none_on_the_side(cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_tree(tmp_path, {"tests/test_old.py": "", "tests/optional/test_o.py": ""})
    monkeypatch.setattr(cmr, "run_side", lambda *a, **k: pytest.fail("mutmut started"))

    tests = ["tests/test_new.py", "tests/optional/test_o.py"]
    assert cmr.run_changed_tests(tmp_path, ["inference_perf.m.x_f"], tests, {}, 1, 0, False) is None


# Builds `mutmut show` output for a mutant of f whose diff has `body` after the header.
def shown(mutant: str, body: str) -> str:
    return f"# {mutant}: survived\n--- inference_perf/m.py\n+++ inference_perf/m.py\n{body}"


# f starts on line 3 and holds `return x > 1` on lines 5 and 6. mutmut show numbers from f's def, so
# a mutant of the line-6 copy (diff position 4) maps to 6 and one of the line-5 copy maps to 5.
# A removed line the source lacks falls back to the def line, 3.
def test_mutant_line_finds_the_mutated_copy(cmr: Any) -> None:
    source = "import os\n\ndef f(x):\n    if x:\n        return x > 1\n    return x > 1\n"
    second = "@@ -1,4 +1,4 @@\n def f(x):\n     if x:\n         return x > 1\n-    return x > 1\n+    return x >= 1"
    first = "@@ -1,4 +1,4 @@\n def f(x):\n     if x:\n-        return x > 1\n+        return x >= 1\n     return x > 1"
    missing = "@@ -1,2 +1,2 @@\n def f(x):\n-    return y\n+    return None"

    assert cmr.mutant_line(source, "x_f", shown(f"{F}2", second)) == 6
    assert cmr.mutant_line(source, "x_f", shown(f"{F}1", first)) == 5
    assert cmr.mutant_line(source, "x_f", shown(f"{F}3", missing)) == 3


# Three asks for the same mutant's diff on one tree start one `mutmut show`. A second tree starts
# its own.
def test_mutant_diff_runs_mutmut_show_once_per_tree(cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[tuple[Path, str]] = []

    def fake(tree: Path, *args: str) -> subprocess.CompletedProcess[str]:
        calls.append((tree, args[1]))
        return subprocess.CompletedProcess([], 0, shown(args[1], f"# from {tree}"), "")

    monkeypatch.setattr(cmr, "mutmut", fake)
    side = cmr.Side(tree=tmp_path / "a")

    diffs = [
        cmr.mutant_diff(tmp_path / "a", f"{F}1"),
        cmr.mutation_lines(side, f"{F}1"),
        cmr.mutant_diff(tmp_path / "a", f"{F}1"),
    ]
    other = cmr.mutant_diff(tmp_path / "b", f"{F}1")

    assert diffs[0] == diffs[2] and diffs[1] == ()
    assert "from " + str(tmp_path / "b") in other
    assert calls == [(tmp_path / "a", f"{F}1"), (tmp_path / "b", f"{F}1")]


# Path, title and message each hold characters GitHub's workflow commands reserve. The message
# encodes '%', CR and LF and keeps ':' and ','. The path and title encode all five.
def test_workflow_command_escapes_reserved_characters(cmr: Any) -> None:
    annotation = cmr.Annotation("a,b:c.py", 7, "T: 50%, done", "line 1\nx > 1, 100%\r")

    assert cmr.workflow_command(annotation) == (
        "::error file=a%2Cb%3Ac.py,line=7,title=T%3A 50%25%2C done::line 1%0Ax > 1, 100%25%0D"
    )


# m.py holds f (line 1) and g (line 5). Two new survivors in f become annotations on line 2 that quote
# the mutated line, then g, which no test runs, gets one on its def line, then 20 from the checks.
# The list stops at MAX_ANNOTATIONS, the most GitHub shows from one step.
def test_annotations_put_each_failure_on_its_line(cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    write_tree(tmp_path, {"inference_perf/__init__.py": "", "inference_perf/m.py": GT + "\n\ndef g():\n    return 2\n"})
    diffs = {
        f"{F}1": shown(f"{F}1", "@@ -1,2 +1,2 @@\n def f(x):\n-    return x > 1\n+    return x >= 1"),
        f"{F}2": shown(f"{F}2", "@@ -1,2 +1,2 @@\n def f(x):\n-    return x > 1\n+    return x > 2"),
    }
    monkeypatch.setattr(cmr, "mutmut", lambda tree, *a: subprocess.CompletedProcess([], 0, diffs[a[1]], ""))
    verdict = cmr.Verdict(0, 2, {}, ["inference_perf.m.x_g"], [], new=list(diffs))
    checks = cmr.Checks(annotations=[cmr.Annotation("inference_perf/m.py", 6, "Statement mutmut never mutates", "")] * 20)

    found = cmr.annotations(cmr.Side(tree=tmp_path), verdict, checks)

    assert [(a.line, a.title) for a in found[:3]] == [
        (2, "Surviving mutant"),
        (2, "Surviving mutant"),
        (5, "No test runs this function"),
    ]
    assert "return x >= 1" in found[0].message and "return x > 2" in found[1].message
    assert len(found) == cmr.MAX_ANNOTATIONS


# The base has a module-level lambda on line 1, which mutmut never mutates. The head keeps it and
# appends a cached function whose body is logic. Only the appended statements, lines the change
# added, are annotated. Line 1 is not, though it counts toward the rise.
def test_unchecked_statements_are_annotated_only_where_the_change_added_them(
    cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    base = "_gt = lambda x: x > 1\n\n\ndef f(x):\n    return _gt(x)\n"
    head = base + "\n\nimport functools\n\n\n@functools.cache\ndef g(x):\n    y = x\n    return y > 1\n"

    checks = static_result(cmr, tmp_path, monkeypatch, {"inference_perf/m.py": base}, {"inference_perf/m.py": head})

    assert "Statements mutmut never mutates rose from 1 to 3" in checks.failures
    assert sorted(a.line for a in checks.annotations) == [13, 14]
    assert {a.path for a in checks.annotations} == {"inference_perf/m.py"}


# Base: f(x) = x > 1, pinned by tests/test_m.py. The change loosens that test to only call f, so f is
# in scope through it. Faked mutmut: the changed tests catch both of f's mutants on the base, and on
# the head catch 2 and miss 1. Only mutant 1 reaches the head's full run, which misses it too, so the
# run fails 0 to 1, with mutant 1 annotated on f's line and nothing else sent to a full run.
def test_main_runs_only_undecided_mutants_in_full(
    cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    repo = tmp_path / "repo"
    write_tree(
        repo,
        {
            "pyproject.toml": '[tool.mutmut]\nsource_paths = ["inference_perf"]\n',
            "inference_perf/__init__.py": "",
            "inference_perf/m.py": GT,
            "tests/test_m.py": "from inference_perf.m import f\n\n\ndef test_f():\n    assert f(2)\n    assert not f(1)\n",
        },
    )
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    (repo / "tests" / "test_m.py").write_text("from inference_perf.m import f\n\n\ndef test_f():\n    f(2)\n")
    monkeypatch.chdir(repo)
    output = tmp_path / "mutation.json"
    monkeypatch.setattr("sys.argv", ["check", "--base", "HEAD", "--output", str(output)])
    monkeypatch.setattr(cmr, "static_checks", lambda *a: cmr.Checks())
    monkeypatch.setenv("GITHUB_ACTIONS", "true")
    monkeypatch.delenv("GITHUB_STEP_SUMMARY", raising=False)
    calls: list[tuple[str, list[str], list[str], bool]] = []
    changed_tests = {"base": {f"{F}1": "killed", f"{F}2": "killed"}, "head": {f"{F}1": "survived", f"{F}2": "killed"}}

    def run_side(tree: Path, names: list[str], *a: Any, mutants: Any = None, tests: Any = None, **k: Any) -> Any:
        side_name = tree.name.removesuffix("-changed-tests")
        calls.append((tree.name, names, mutants or [], tests is not None))
        results = dict(changed_tests[side_name]) if tests is not None else {m: "survived" for m in mutants or []}
        return cmr.Side(tree=tree, results=results)

    monkeypatch.setattr(cmr, "run_side", run_side)
    monkeypatch.setattr(cmr, "still_survives", lambda *a: cmr.Survival(True, []))
    monkeypatch.setattr(cmr, "mutation_lines", lambda s, m: (str(id(s)), m))
    diff = shown(f"{F}1", "@@ -1,2 +1,2 @@\n def f(x):\n-    return x > 1\n+    return x >= 1")
    monkeypatch.setattr(cmr, "mutmut", lambda tree, *a: subprocess.CompletedProcess([], 0, diff, ""))

    with pytest.raises(SystemExit) as exited:
        cmr.main()

    assert exited.value.code == 1
    assert calls == [
        ("base-changed-tests", ["inference_perf.m.x_f"], [], True),
        ("head-changed-tests", ["inference_perf.m.x_f"], [], True),
        ("base", [], [], False),
        ("head", [], [f"{F}1"], False),
    ]
    record = json.loads(output.read_text())
    assert record["failures"] == ["Surviving mutants rose from 0 to 1"]
    assert (record["settled_by_changed_tests"], record["undecided_by_changed_tests"]) == (1, {f"{F}1": "head"})
    assert [(a["path"], a["line"]) for a in record["annotations"]] == [("inference_perf/m.py", 2)]
    assert "::error file=inference_perf/m.py,line=2,title=Surviving mutant::" in capsys.readouterr().out
