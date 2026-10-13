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
    assert verdict.failures == ["Surviving mutants rose from 3 to 4"]


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


# Code moved from deleted function d into new function h keeps its 2 survivors, so it passes.
# A third survivor in h fails, and only h's survivors are rerun.
def test_judge_counts_new_and_deleted_functions(cmr: Any, monkeypatch: pytest.MonkeyPatch) -> None:
    rerun: list[str] = []

    def record(s: Any, mutant: str, a: Any) -> bool:
        rerun.append(mutant)
        return True

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


# Undecorated and lone-staticmethod functions are mutated, so they are absent. Bodies that only
# delegate count 0: a property returning an attribute, a setter assigning one, a validator calling
# a helper, an abstract stub. Anything else counts every statement: a one-line return with
# arithmetic is 1, a validator with an if, an assignment and a return is 3.
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
    assert cmr.unchecked_statements(source) == {
        "xǁCǁp": 0,
        "xǁCǁq": 0,
        "xǁCǁthin": 0,
        "xǁCǁstub": 0,
        "xǁCǁone": 1,
        "xǁCǁfat": 3,
        "xǁCǁboth": 2,
    }


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


# The base has a 3-statement function f. The working tree wraps it in functools.cache, which mutmut
# skips. f stays an existing function and its unchecked statements go from 0 to 3.
def test_touched_functions_records_a_new_decorator_as_unchecked(
    cmr: Any, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pkg = tmp_path / "inference_perf"
    pkg.mkdir()
    body = "def f(x):\n    y = x + 1\n    z = y * 2\n    return z\n"
    (pkg / "mod.py").write_text(body)
    git(tmp_path, "init", "-q")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-qm", "base")
    (pkg / "mod.py").write_text("import functools\n\n\n@functools.cache\n" + body)
    monkeypatch.chdir(tmp_path)

    touched = cmr.touched_functions("HEAD", [])

    assert touched.existing == [("inference_perf.mod.x_f", "inference_perf.mod.x_f")]
    assert touched.unchecked == {"inference_perf.mod.x_f": (0, 3)}


# g had 4 mutants on the base and none with the change, as when a decorator hides it. That fails
# even though its survivors fell from 1 to 0.
def test_judge_fails_when_a_function_loses_all_its_mutants(cmr: Any) -> None:
    touched = cmr.Touched(existing=[("inference_perf.m.x_g", "inference_perf.m.x_g")])
    base = side(cmr, x_g={"killed": 3, "survived": 1})

    verdict = cmr.judge(touched, base, side(cmr), [])

    assert verdict.failures == ["`inference_perf.m.x_g` had 4 mutants on the base and none with this change"]


# Unchecked statements across touched functions go from 2 to 3, so it fails. Equal counts pass.
def test_judge_fails_when_unchecked_statements_rise(cmr: Any) -> None:
    worse = cmr.Touched(
        new=["inference_perf.m.x_v"], unchecked={"inference_perf.m.x_w": (2, 2), "inference_perf.m.x_v": (0, 1)}
    )
    held = cmr.Touched(unchecked={"inference_perf.m.x_w": (2, 2)})

    assert cmr.judge(worse, None, side(cmr), []).failures == [
        "Statements mutmut cannot check rose from 2 to 3. Decorated functions should delegate to an undecorated one"
    ]
    assert cmr.judge(held, None, side(cmr), []).failures == []


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
