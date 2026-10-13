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
import json
import os
import subprocess
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
    cmr.add_functions_run_by_changed_tests(touched, base_tree, repo, cmr.changed_test_files("HEAD"))
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


# Writes each {path: source} under `root`, creating directories.
def write_tree(root: Path, files: dict[str, str]) -> None:
    for rel, source in files.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_text(source)


# tests/test_m.py imports the sibling helper check from tests/helpers.py. The change loosens check to
# assert nothing and touches no test file. test_m.py still counts as changed, so f comes into scope.
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
    monkeypatch.chdir(repo)
    base_tree = tmp_path / "base-tree"
    cmr.export_base("HEAD", base_tree)

    assert cmr.changed_test_files("HEAD") == []
    tests = cmr.tests_in_scope("HEAD", base_tree, repo)
    assert tests == ["tests/test_m.py"]
    touched = cmr.touched_functions("HEAD", [])
    cmr.add_functions_run_by_changed_tests(touched, base_tree, repo, tests)
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


# Commits `base` files, writes `head` over them in the working tree, and returns the failures of the
# checks that need no mutants, with MUTMUT_CONFIG as the base's [tool.mutmut].
def static_failures(
    cmr: Any, repo: Path, monkeypatch: pytest.MonkeyPatch, base: dict[str, str], head: dict[str, str]
) -> list[str]:
    write_tree(repo, {"inference_perf/__init__.py": "", **base})
    git(repo, "init", "-q")
    git(repo, "add", ".")
    git(repo, "commit", "-qm", "base")
    write_tree(repo, head)
    monkeypatch.chdir(repo)
    failures: list[str] = cmr.static_checks("HEAD", MUTMUT_CONFIG).failures
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

    failures = static_failures(cmr, tmp_path, monkeypatch, base, head)

    assert "Statements mutmut never mutates rose from 0 to 1" in failures


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
