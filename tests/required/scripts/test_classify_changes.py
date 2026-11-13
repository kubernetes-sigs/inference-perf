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
"""Tests for scripts/classify_changes.py, which labels a diff by what it does to tests, code and gates."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "scripts" / "classify_changes.py"

TEST_FILE = "tests/required/test_thing.py"
SOURCE_FILE = "inference_perf/utils/thing.py"


# Loads scripts/classify_changes.py as a module. scripts/ is not a package.
@pytest.fixture(scope="module")
def cc() -> ModuleType:
    spec = importlib.util.spec_from_file_location("classify_changes", SCRIPT)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# Builds one changed file from text. None for `base` means the file is new, None for `head` means it was removed.
# `deletions` is the number of lines the diff removes, which the classifier takes from git and does not recount.
def change(cc: ModuleType, path: str, base: str | None, head: str | None, deletions: int = 0) -> object:
    return cc.FileChange(path, None if base is None else base.encode(), None if head is None else head.encode(), deletions)


# Runs git in `repo` with a fixed identity, so commits work on a bare CI runner.
def git(repo: Path, *args: str) -> None:
    subprocess.run(
        ["git", "-c", "user.name=t", "-c", "user.email=t@t", "-c", "init.defaultBranch=main", *args],
        cwd=repo,
        check=True,
        capture_output=True,
    )


# A test gains a docstring and a comment and nothing else. Nothing is altered and the diff is purely additive.
def test_docstring_and_comment_edits_change_nothing(cc: ModuleType) -> None:
    base = "def test_a():\n    assert f() == 1\n"
    head = 'def test_a():\n    """Checks f."""\n    # the answer\n    assert f() == 1\n'
    c = cc.classify([change(cc, TEST_FILE, base, head)])
    assert c.tests == cc.UnitDiff()
    assert c.labels == ["purely-additive"]


# test_a's expected value goes from 1 to 2. It is listed as altered and the diff changes pinned behavior.
def test_changed_test_body_is_altered(cc: ModuleType) -> None:
    c = cc.classify([change(cc, TEST_FILE, "def test_a():\n    assert f() == 1\n", "def test_a():\n    assert f() == 2\n")])
    assert c.tests.altered == [f"{TEST_FILE}::test_a"]
    assert c.changes_pinned_behavior
    assert not c.purely_additive


# test_b is deleted and test_a is untouched. test_b is listed as removed.
def test_deleted_test_is_removed(cc: ModuleType) -> None:
    base = "def test_a():\n    assert 1\n\ndef test_b():\n    assert 2\n"
    c = cc.classify([change(cc, TEST_FILE, base, "def test_a():\n    assert 1\n")])
    assert c.tests.removed == [f"{TEST_FILE}::test_b"]
    assert c.changes_pinned_behavior


# test_a becomes test_renamed with the same body. It is a rename, not a removal, and nothing pinned changed.
def test_same_body_under_a_new_test_name_is_a_rename(cc: ModuleType) -> None:
    c = cc.classify([change(cc, TEST_FILE, "def test_a():\n    assert 1\n", "def test_renamed():\n    assert 1\n")])
    assert c.tests.renamed == [(f"{TEST_FILE}::test_a", f"{TEST_FILE}::test_renamed")]
    assert c.tests.removed == [] and c.tests.added == []
    assert not c.changes_pinned_behavior


# test_a becomes _test_a with the same body. pytest no longer collects it, so it counts as removed.
def test_rename_out_of_collection_is_a_removal(cc: ModuleType) -> None:
    c = cc.classify([change(cc, TEST_FILE, "def test_a():\n    assert 1\n", "def _test_a():\n    assert 1\n")])
    assert c.tests.removed == [f"{TEST_FILE}::test_a"]
    assert c.tests.renamed == []


# test_a moves unchanged from test_thing.py to thing_cases.py, a file pytest does not collect. It counts as removed.
def test_move_to_an_uncollected_file_is_a_removal(cc: ModuleType) -> None:
    body = "def test_a():\n    assert 1\n"
    c = cc.classify([change(cc, TEST_FILE, body, None), change(cc, "tests/required/thing_cases.py", None, body)])
    assert c.tests.removed == [f"{TEST_FILE}::test_a"]


# test_a gains @pytest.mark.skip with its body unchanged. The decorator counts, so the test is altered.
def test_new_decorator_alters_the_test(cc: ModuleType) -> None:
    base = "def test_a():\n    assert 1\n"
    c = cc.classify([change(cc, TEST_FILE, base, "@pytest.mark.skip\n" + base)])
    assert c.tests.altered == [f"{TEST_FILE}::test_a"]


# A test class gains a skip decorator. Both of its tests are altered though neither body changed.
def test_class_decorator_alters_every_test_in_the_class(cc: ModuleType) -> None:
    base = "class TestThing:\n    def test_a(self):\n        assert 1\n    def test_b(self):\n        assert 2\n"
    c = cc.classify([change(cc, TEST_FILE, base, "@pytest.mark.skip\n" + base)])
    assert c.tests.altered == [f"{TEST_FILE}::TestThing::test_a", f"{TEST_FILE}::TestThing::test_b"]


# A class named Suite holds test_a. pytest only collects Test* classes, so test_a is a helper and not a test.
def test_methods_of_an_uncollected_class_are_not_tests(cc: ModuleType) -> None:
    c = cc.classify([change(cc, TEST_FILE, None, "class Suite:\n    def test_a(self):\n        assert 1\n")])
    assert c.tests.added == []
    assert f"{TEST_FILE}::Suite::test_a" in c.code.added


# An existing test module gains `pytestmark = pytest.mark.skip` at top level. No test body changed,
# but every test in the file is now skipped, so the new statement is a changed helper.
def test_statement_added_to_an_existing_test_module(cc: ModuleType) -> None:
    base = "def test_a():\n    assert 1\n"
    c = cc.classify([change(cc, TEST_FILE, base, "pytestmark = pytest.mark.skip\n" + base)])
    assert c.helpers_changed == [f"{TEST_FILE}::pytestmark"]
    assert c.changes_pinned_behavior


# An existing test class gains a class attribute, and an existing test module gains a bare top-level call.
# Both run for tests that were already there, so both are changed helpers.
def test_statement_added_to_an_existing_test_class_or_module_body(cc: ModuleType) -> None:
    base = "class TestThing:\n    def test_a(self):\n        assert 1\n"
    c = cc.classify([change(cc, TEST_FILE, base, base + "    timeout = 1\n\nsetup()\n")])
    assert c.helpers_changed == [f"{TEST_FILE}::TestThing::timeout", f"{TEST_FILE}::<module>"]


# The same `pytestmark` in a brand-new test file, a new helper function in an existing test file, and a new
# constant in an existing source file. None of them changes a test that was already there.
def test_additions_that_pin_nothing(cc: ModuleType) -> None:
    base = "def test_a():\n    assert 1\n"
    changes = [
        change(cc, "tests/required/test_new.py", None, "pytestmark = pytest.mark.skip\n" + base),
        change(cc, TEST_FILE, base, base + "\ndef make():\n    return 1\n"),
        change(cc, SOURCE_FILE, "A = 1\n", "A = 1\nB = 2\n"),
    ]
    c = cc.classify(changes)
    assert c.helpers_changed == []
    assert c.labels == ["purely-additive"]


# EXPECTED goes from 1 to 2 at the top of a test module. The constant is a changed helper, which pins new behavior.
def test_changed_module_constant_is_a_changed_helper(cc: ModuleType) -> None:
    test = "def test_a():\n    assert f() == EXPECTED\n"
    c = cc.classify([change(cc, TEST_FILE, "EXPECTED = 1\n" + test, "EXPECTED = 2\n" + test)])
    assert c.helpers_changed == [f"{TEST_FILE}::EXPECTED"]
    assert c.changes_pinned_behavior
    assert c.tests == cc.UnitDiff()


# A bare `if` block at the top of a test module changes its condition. It is reported as the module's own unit.
def test_changed_loose_module_code_is_a_changed_helper(cc: ModuleType) -> None:
    c = cc.classify([change(cc, TEST_FILE, "if A:\n    setup()\n", "if B:\n    setup()\n")])
    assert c.helpers_changed == [f"{TEST_FILE}::<module>"]


# make_client() in a test module returns a different value. It is a changed helper and pins new behavior.
def test_changed_helper_function(cc: ModuleType) -> None:
    c = cc.classify([change(cc, TEST_FILE, "def make_client():\n    return 1\n", "def make_client():\n    return 2\n")])
    assert c.helpers_changed == [f"{TEST_FILE}::make_client"]
    assert c.changes_pinned_behavior


# The same function body changes under inference_perf/. That is altered code but not a test helper.
def test_changed_source_function_is_not_a_helper(cc: ModuleType) -> None:
    c = cc.classify([change(cc, SOURCE_FILE, "def f():\n    return 1\n", "def f():\n    return 2\n")])
    assert c.code.altered == [f"{SOURCE_FILE}::f"]
    assert c.helpers_changed == []
    assert not c.changes_pinned_behavior
    assert not c.purely_additive


# A module gains function g and method C.m beside an unchanged f. Only additions, so the diff is purely additive.
def test_new_function_and_method_are_additive(cc: ModuleType) -> None:
    base = "def f():\n    return 1\n\nclass C:\n    def n(self):\n        return 1\n"
    head = base + "    def m(self):\n        return 2\n\ndef g():\n    return 3\n"
    c = cc.classify([change(cc, SOURCE_FILE, base, head)])
    assert c.code.added == [f"{SOURCE_FILE}::C::m", f"{SOURCE_FILE}::g"]
    assert c.purely_additive


# Class C gains a base class. Its methods are untouched but the class itself is altered.
def test_new_base_class_alters_the_class(cc: ModuleType) -> None:
    body = "    def m(self):\n        return 1\n"
    c = cc.classify([change(cc, SOURCE_FILE, "class C:\n" + body, "class C(Base):\n" + body)])
    assert c.code.altered == [f"{SOURCE_FILE}::C::<class>"]


# f moves unchanged from one module to another. It is reported as moved, and nothing counts as removed.
def test_moved_function_is_not_removed(cc: ModuleType) -> None:
    body = "def f():\n    return 1\n"
    other = "inference_perf/utils/other.py"
    c = cc.classify([change(cc, SOURCE_FILE, body, None), change(cc, other, None, body)])
    assert c.code.renamed == [(f"{SOURCE_FILE}::f", f"{other}::f")]
    assert c.purely_additive


# Only an import line changes. Imports are not units, so nothing is altered.
def test_import_changes_are_ignored(cc: ModuleType) -> None:
    body = "def f():\n    return 1\n"
    c = cc.classify([change(cc, SOURCE_FILE, "import a\n" + body, "import b\n" + body)])
    assert c.code == cc.UnitDiff()


# Two old keys share one fingerprint and one new key holds it. Exactly one pairs as a rename, the other is removed.
def test_rename_matching_is_one_to_one(cc: ModuleType) -> None:
    diff = cc.diff_units({"a": "x", "b": "x"}, {"c": "x"})
    assert diff.renamed == [("a", "c")]
    assert diff.removed == ["b"]


# The same two sides with move matching off. Both old keys are removed and the new key is added.
def test_diff_units_without_move_matching(cc: ModuleType) -> None:
    diff = cc.diff_units({"a": "x", "b": "x"}, {"c": "x"}, match_moves=False)
    assert (diff.removed, diff.added, diff.renamed) == (["a", "b"], ["c"], [])


# A function is defined twice under one name, as overloads are. Changing either definition alters the one unit.
@pytest.mark.parametrize("edited", ["return 1", "return 2"])
def test_repeated_name_is_one_unit(cc: ModuleType, edited: str) -> None:
    base = "def f():\n    return 1\n\ndef f():\n    return 2\n"
    c = cc.classify([change(cc, SOURCE_FILE, base, base.replace(edited, "return 3"))])
    assert c.code.altered == [f"{SOURCE_FILE}::f"]


# test_a lives in inference_perf/utils/test_helpers.py. The file name matches test_*.py but it is outside
# tests/ and e2e/, so test_a is code and altering it changes no pinned behavior.
def test_test_named_file_outside_the_test_roots(cc: ModuleType) -> None:
    path = "inference_perf/utils/test_helpers.py"
    c = cc.classify([change(cc, path, "def test_a():\n    assert 1\n", "def test_a():\n    assert 2\n")])
    assert c.tests == cc.UnitDiff()
    assert c.code.altered == [f"{path}::test_a"]
    assert not c.changes_pinned_behavior


# A golden under tests/ changes one byte, and one under an e2e/ subdirectory is deleted. Both pin new behavior.
def test_changed_and_removed_goldens(cc: ModuleType) -> None:
    changes = [
        change(cc, "tests/required/goldens/a.txt", "1\n", "2\n"),
        change(cc, "e2e/testdata/b.json", "{}\n", None),
    ]
    c = cc.classify(changes)
    assert c.data_changed == ["tests/required/goldens/a.txt", "e2e/testdata/b.json"]
    assert c.changes_pinned_behavior


# A golden is added, and another moves to a new path with the same bytes. Neither changes pinned behavior.
def test_new_and_moved_goldens_do_not_count(cc: ModuleType) -> None:
    changes = [
        change(cc, "tests/goldens/new.txt", None, "1\n"),
        change(cc, "tests/goldens/old_name.txt", "same\n", None),
        change(cc, "tests/goldens/new_name.txt", None, "same\n"),
    ]
    assert cc.classify(changes).data_changed == []


# Files at the e2e/ root, Markdown under tests/ and files outside the test roots are not goldens.
@pytest.mark.parametrize("path", ["e2e/vllm_cpu_server.sh", "tests/README.md", "docs/config.md", "examples/a.yaml"])
def test_paths_that_are_not_data_files(cc: ModuleType, path: str) -> None:
    assert cc.classify([change(cc, path, "1\n", "2\n")]).data_changed == []


# A doc swaps one line for another, so git counts one deleted line. The diff is not purely additive.
def test_deleted_line_in_a_non_python_file(cc: ModuleType) -> None:
    c = cc.classify([change(cc, "docs/config.md", "a\nb\n", "a\nc\n", deletions=1)])
    assert c.lines_deleted == ["docs/config.md"]
    assert not c.purely_additive


# A doc only gains a line and a new doc appears. git counts no deleted line in either. The new doc carries a
# stray deletions count to show an added file can never have lost a line.
def test_added_lines_are_additive(cc: ModuleType) -> None:
    changes = [
        change(cc, "docs/a.md", "a\n", "a\nb\n"),
        change(cc, "docs/new.md", None, "a\n", deletions=3),
    ]
    c = cc.classify(changes)
    assert c.lines_deleted == []
    assert c.purely_additive


# A doc is deleted outright. Every one of its lines is lost.
def test_removed_non_python_file_lost_lines(cc: ModuleType) -> None:
    assert cc.classify([change(cc, "docs/a.md", "a\n", None)]).lines_deleted == ["docs/a.md"]


# A source file's head is not valid Python. It is listed as unparsed and the diff is not additive, but no test is pinned.
def test_unparsed_source_file(cc: ModuleType) -> None:
    c = cc.classify([change(cc, SOURCE_FILE, "def f():\n    return 1\n", "def f(:\n")])
    assert c.unparsed == [SOURCE_FILE]
    assert not c.purely_additive
    assert not c.changes_pinned_behavior


# A source file's head is one expression nested 100,000 operators deep, which overflows the parser.
# It is listed as unparsed like any other file that can't be read, and the run does not crash.
def test_source_too_deep_to_parse(cc: ModuleType) -> None:
    c = cc.classify([change(cc, SOURCE_FILE, "X = 1\n", "X = " + "1+" * 100_000 + "1\n")])
    assert c.unparsed == [SOURCE_FILE]


# The classifier reads the bytes of Python files, pyproject.toml, pdm.lock and the three vLLM lists.
# Docs and goldens are only compared for equality.
@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("inference_perf/x.py", True),
        ("pyproject.toml", True),
        ("pdm.lock", True),
        ("e2e/vllm_known_bad.txt", True),
        ("e2e/vllm_releases.txt", True),
        ("e2e/vllm_multimodal_releases.txt", True),
        ("docs/config.md", False),
        ("tests/goldens/a.txt", False),
    ],
)
def test_needs_content(cc: ModuleType, path: str, expected: bool) -> None:
    assert cc.needs_content(path) is expected


# A workflow and a source file change together. Only the workflow is listed as a gate path.
def test_gate_paths_lists_only_gate_files(cc: ModuleType) -> None:
    changes = [change(cc, ".github/workflows/x.yml", "a\n", "b\n"), change(cc, SOURCE_FILE, None, "X = 1\n")]
    c = cc.classify(changes)
    assert c.gate_paths == [".github/workflows/x.yml"]
    assert c.paths == [".github/workflows/x.yml", SOURCE_FILE]


# A test file's head is not valid Python. Its tests can't be compared, so pinned behavior counts as changed.
def test_unparsed_test_file_changes_pinned_behavior(cc: ModuleType) -> None:
    c = cc.classify([change(cc, TEST_FILE, "def test_a():\n    assert 1\n", "def test_a(:\n")])
    assert c.changes_pinned_behavior


# A test is altered with no change under inference_perf/, then again beside a source change.
# Only the first is a tests-only change.
def test_tests_only_change_needs_no_source_change(cc: ModuleType) -> None:
    altered = change(cc, TEST_FILE, "def test_a():\n    assert 1\n", "def test_a():\n    assert 2\n")
    source = change(cc, SOURCE_FILE, None, "def f():\n    return 1\n")
    assert cc.classify([altered]).tests_only_change
    assert not cc.classify([altered, source]).tests_only_change


# A PR that only adds a test changes no pinned behavior, so it is not a tests-only change.
def test_test_backfill_is_not_a_tests_only_change(cc: ModuleType) -> None:
    assert not cc.classify([change(cc, TEST_FILE, None, "def test_a():\n    assert 1\n")]).tests_only_change


# A change under each of datagen, loadgen, client and reportgen with no e2e test gets no-oracle-test.
@pytest.mark.parametrize("package", ["datagen", "loadgen", "client", "reportgen"])
def test_oracle_package_without_e2e_test(cc: ModuleType, package: str) -> None:
    c = cc.classify([change(cc, f"inference_perf/{package}/x.py", None, "def f():\n    return 1\n")])
    assert c.no_oracle_test


# The same datagen change with a new e2e test, then with an altered e2e test. Either one clears no-oracle-test.
@pytest.mark.parametrize("base", [None, "def test_a():\n    assert 1\n"])
def test_e2e_test_clears_no_oracle_test(cc: ModuleType, base: str | None) -> None:
    source = change(cc, "inference_perf/datagen/x.py", None, "def f():\n    return 1\n")
    e2e = change(cc, "e2e/tests/test_x.py", base, "def test_a():\n    assert 2\n")
    assert not cc.classify([source, e2e]).no_oracle_test


# A datagen change with a new unit test under tests/ still has no oracle. A config-only change needs none.
def test_no_oracle_test_boundaries(cc: ModuleType) -> None:
    source = change(cc, "inference_perf/datagen/x.py", None, "def f():\n    return 1\n")
    unit = change(cc, TEST_FILE, None, "def test_a():\n    assert 1\n")
    assert cc.classify([source, unit]).no_oracle_test
    assert not cc.classify([change(cc, "inference_perf/config/x.py", None, "X = 1\n")]).no_oracle_test


CONFIG_FILE = "inference_perf/config/loadgen/config.py"
REPORT_FILE = "inference_perf/reportgen/br/v0_2/schema.py"


# A config model gains field b, changes a's default and drops c. Each lands in its own list under touches/config-schema.
def test_config_fields_added_altered_removed(cc: ModuleType) -> None:
    base = "class LoadConfig(BaseModel):\n    a: int = 1\n    c: int = 3\n"
    head = "class LoadConfig(BaseModel):\n    a: int = 2\n    b: int = 0\n"
    c = cc.classify([change(cc, CONFIG_FILE, base, head)])
    prefix = f"{CONFIG_FILE}::LoadConfig::"
    assert (c.config_fields.added, c.config_fields.altered, c.config_fields.removed) == (
        [prefix + "b"],
        [prefix + "a"],
        [prefix + "c"],
    )
    assert "touches/config-schema" in c.labels
    assert "touches/report-schema" not in c.labels


# An enum under config/ gains a member. Enum members are config values, so it counts as an added field.
def test_enum_member_is_a_config_field(cc: ModuleType) -> None:
    base = 'class LoadType(Enum):\n    CONSTANT = "constant"\n'
    c = cc.classify([change(cc, CONFIG_FILE, base, base + '    POISSON = "poisson"\n')])
    assert c.config_fields.added == [f"{CONFIG_FILE}::LoadType::POISSON"]


# Field a moves unchanged from class A to class B. For the schema that is a removal and an addition, not a move.
def test_field_moved_between_classes_is_removed_and_added(cc: ModuleType) -> None:
    base = "class A(BaseModel):\n    a: int = 1\n\nclass B(BaseModel):\n    pass\n"
    head = "class A(BaseModel):\n    pass\n\nclass B(BaseModel):\n    a: int = 1\n"
    c = cc.classify([change(cc, CONFIG_FILE, base, head)])
    assert c.config_fields.removed == [f"{CONFIG_FILE}::A::a"]
    assert c.config_fields.added == [f"{CONFIG_FILE}::B::a"]


# A validator method and a module constant change in a config file. Neither is a field, so the schema label stays off.
def test_methods_and_module_constants_are_not_fields(cc: ModuleType) -> None:
    base = "LIMIT = 1\n\nclass C(BaseModel):\n    a: int = 1\n    def check(self):\n        return 1\n"
    c = cc.classify([change(cc, CONFIG_FILE, base, base.replace("LIMIT = 1", "LIMIT = 2").replace("return 1", "return 2"))])
    assert c.config_fields == cc.UnitDiff()
    assert "touches/config-schema" not in c.labels


# A report model under reportgen/ gains a field. It gets touches/report-schema and not the config label.
def test_report_field_added(cc: ModuleType) -> None:
    base = "class Report(BaseModel):\n    a: int\n"
    c = cc.classify([change(cc, REPORT_FILE, base, base + "    b: int\n")])
    assert c.report_fields.added == [f"{REPORT_FILE}::Report::b"]
    assert "touches/report-schema" in c.labels
    assert "touches/config-schema" not in c.labels


# The same class field outside config/ and reportgen/ touches neither schema.
def test_fields_outside_schema_roots_do_not_count(cc: ModuleType) -> None:
    base = "class C(BaseModel):\n    a: int\n"
    labels = cc.classify([change(cc, "inference_perf/apis/base.py", base, base + "    b: int\n")]).labels
    assert not [label for label in labels if label.startswith("touches/")]


# A file under each metrics root gets touches/metrics. A file beside them does not.
@pytest.mark.parametrize(
    ("path", "expected"),
    [
        ("inference_perf/metrics/session_collector.py", True),
        ("inference_perf/observability/metrics/registry.py", True),
        ("inference_perf/client/server_metrics/x.py", True),
        ("inference_perf/config/metrics/config.py", True),
        ("inference_perf/observability/context.py", False),
    ],
)
def test_metrics_paths(cc: ModuleType, path: str, expected: bool) -> None:
    c = cc.classify([change(cc, path, None, "X = 1\n")])
    assert ("touches/metrics" in c.labels) is expected


# Each path that is a gate file by where it lives or what it is called.
@pytest.mark.parametrize(
    "path",
    [
        ".github/workflows/unit_test.yml",
        "scripts/check_license.py",
        "tests/required/scripts/test_x.py",
        "tests/test_check_license.py",
        "conftest.py",
        "tests/optional/conftest.py",
        "OWNERS",
        "scripts/OWNERS",
        "OWNERS_ALIASES",
        "AGENTS.md",
        "CLAUDE.md",
        "flake.nix",
        "e2e/known_bad.py",
    ],
)
def test_gate_paths(cc: ModuleType, path: str) -> None:
    assert cc.gate_reasons(change(cc, path, "a\n", "b\n")) == [path]


# Paths that look close to a gate file and are not one.
@pytest.mark.parametrize(
    "path",
    ["docs/AGENTS.md", "tests/test_config.py", "inference_perf/scripts/x.py", "e2e/tests/test_known_bad.py", "README.md"],
)
def test_paths_that_are_not_gates(cc: ModuleType, path: str) -> None:
    assert cc.gate_reasons(change(cc, path, "a\n", "b\n")) == []


# One value changes inside each gated pyproject table in turn. Each is reported by its table name.
@pytest.mark.parametrize(
    "table", ["tool.pdm.scripts", "tool.ruff", "tool.mypy", "tool.pytest", "tool.coverage", "tool.mutmut"]
)
def test_each_gated_pyproject_table(cc: ModuleType, table: str) -> None:
    base, head = f"[{table}]\nx = 1\n", f"[{table}]\nx = 2\n"
    assert cc.gate_reasons(change(cc, "pyproject.toml", base, head)) == [f"pyproject.toml [{table}]"]


# A runtime dependency and an ungated table change in pyproject.toml. Neither is a gate change.
def test_dependency_bump_is_not_a_gate_change(cc: ModuleType) -> None:
    base = '[project]\ndependencies = ["aiohttp>=3.11"]\n[tool.setuptools]\nx = 1\n'
    head = '[project]\ndependencies = ["aiohttp>=3.12"]\n[tool.setuptools]\nx = 2\n'
    c = cc.classify([change(cc, "pyproject.toml", base, head)])
    assert c.gate_reasons == []
    assert "touches/ci-gates" not in c.labels


# The requirement on each test tool changes in a dependency group. Each one is a gate change.
@pytest.mark.parametrize("package", ["pytest", "pytest-cov", "coverage", "mutmut", "Pytest_Cov"])
def test_test_tool_requirement_is_a_gate_change(cc: ModuleType, package: str) -> None:
    base = f'[dependency-groups]\ntest = ["{package}>=1.0", "hypothesis>=6"]\n'
    reasons = cc.gate_reasons(change(cc, "pyproject.toml", base, base.replace(">=1.0", ">=2.0")))
    assert reasons == ["pyproject.toml (coverage, mutmut, pytest, pytest-cov requirement)"]


# pytest-asyncio and pytest-xdist share a prefix with pytest and are not gated. Bumping them is not a gate change.
def test_similarly_named_packages_are_not_gated(cc: ModuleType) -> None:
    base = '[dependency-groups]\ntest = ["pytest-asyncio>=1.0", "pytest-xdist>=3.0"]\n'
    assert cc.gate_reasons(change(cc, "pyproject.toml", base, base.replace(">=1.0", ">=2.0"))) == []


# The requirement lives under project.dependencies and under an optional extra. Both places are read.
@pytest.mark.parametrize(
    "layout", ['[project]\ndependencies = ["mutmut==1"]\n', '[project.optional-dependencies]\nx = ["mutmut==1"]\n']
)
def test_test_tool_requirement_in_other_tables(cc: ModuleType, layout: str) -> None:
    assert cc.gate_reasons(change(cc, "pyproject.toml", layout, layout.replace("==1", "==2")))


# pyproject.toml is added, removed, or stops being valid TOML. Each is a gate change.
@pytest.mark.parametrize(
    ("base", "head"), [(None, "[tool.ruff]\n"), ("[tool.ruff]\n", None), ("[tool.ruff]\n", "[tool.ruff\n")]
)
def test_pyproject_added_removed_or_broken(cc: ModuleType, base: str | None, head: str | None) -> None:
    assert cc.gate_reasons(change(cc, "pyproject.toml", base, head))


# pdm.lock moves pytest from 8.3.4 to 9.0.0 and aiohttp from 3.11 to 3.12. Only the pytest move is a gate change.
def test_lockfile_gates_only_test_tool_versions(cc: ModuleType) -> None:
    lock = '[[package]]\nname = "pytest"\nversion = "8.3.4"\n\n[[package]]\nname = "aiohttp"\nversion = "3.11"\n'
    bumped = lock.replace("8.3.4", "9.0.0").replace("3.11", "3.12")
    assert cc.gate_reasons(change(cc, "pdm.lock", lock, bumped)) == ["pdm.lock pytest 8.3.4 -> 9.0.0"]
    assert cc.gate_reasons(change(cc, "pdm.lock", lock, lock.replace("3.11", "3.12"))) == []


# The known-bad list gains 0.33.0, so CI skips one more release. Pruning an entry or rewording a reason does not.
def test_known_bad_list_only_gates_new_entries(cc: ModuleType) -> None:
    path = "e2e/vllm_known_bad.txt"
    base = "# header\n0.32.0 upstream bug\n"
    assert cc.gate_reasons(change(cc, path, base, base + "0.33.0 another bug\n")) == [f"{path} skips 0.33.0"]
    assert cc.gate_reasons(change(cc, path, base, "# header\n")) == []
    assert cc.gate_reasons(change(cc, path, base, "# header\n0.32.0 reworded reason\n# new comment\n")) == []


# A release list loses v0.26.0, so CI tests one fewer release. Swapping it for v0.33.0 or adding a row does not.
@pytest.mark.parametrize("path", ["e2e/vllm_releases.txt", "e2e/vllm_multimodal_releases.txt"])
def test_release_list_only_gates_a_shorter_list(cc: ModuleType, path: str) -> None:
    base = "# header\nv0.26.0\nlatest\n"
    assert cc.gate_reasons(change(cc, path, base, "# header\nlatest\n")) == [f"{path} drops v0.26.0"]
    assert cc.gate_reasons(change(cc, path, base, "# header\nv0.33.0\nlatest\n")) == []
    assert cc.gate_reasons(change(cc, path, base, base + "v0.33.0\n")) == []
    assert cc.gate_reasons(change(cc, path, base, None)) == [f"{path} drops latest", f"{path} drops v0.26.0"]


# A conftest fixture changes. conftest.py is both a test helper and a gate file, so both labels apply.
def test_conftest_is_a_helper_and_a_gate(cc: ModuleType) -> None:
    c = cc.classify([change(cc, "tests/conftest.py", "def fx():\n    return 1\n", "def fx():\n    return 2\n")])
    assert c.labels == ["changes-pinned-behavior", "tests-only-change", "touches/ci-gates"]


# A Co-Authored-By trailer naming an assistant, a "Generated with" line, and a bot author each mark a commit.
@pytest.mark.parametrize(
    ("author", "message"),
    [
        ("Dev <d@x>", "Fix it\n\nCo-Authored-By: Claude <noreply@anthropic.com>"),
        ("Dev <d@x>", "Fix it\n\nassisted-by: GitHub Copilot"),
        ("Dev <d@x>", "Fix it\n\nGenerated with [Claude Code](https://claude.com/claude-code)"),
        ("dependabot[bot] <bot@x>", "Bump aiohttp"),
    ],
)
def test_ai_assisted_commits(cc: ModuleType, author: str, message: str) -> None:
    c = cc.classify([], [cc.Commit(author, message)])
    assert c.ai_commits == [message.splitlines()[0]]
    assert "ai-assisted" in c.labels


# A human co-author trailer and a body that only mentions an assistant by name do not mark the commit.
@pytest.mark.parametrize("message", ["Fix it\n\nCo-Authored-By: Jo <jo@x>", "Fix the claude client timeout"])
def test_plain_commits_are_not_ai_assisted(cc: ModuleType, message: str) -> None:
    assert cc.classify([], [cc.Commit("Dev <d@x>", message)]).ai_commits == []


# A diff that turns every label on at once. The labels come back in the fixed LABELS order, minus purely-additive.
def test_labels_follow_the_declared_order(cc: ModuleType) -> None:
    changes = [
        change(cc, "tests/conftest.py", "def fx():\n    return 1\n", "def fx():\n    return 2\n"),
        change(cc, CONFIG_FILE, "class C(BaseModel):\n    a: int\n", "class C(BaseModel):\n    a: str\n"),
        change(cc, REPORT_FILE, None, "class R(BaseModel):\n    a: int\n"),
        change(cc, "inference_perf/metrics/x.py", None, "X = 1\n"),
    ]
    c = cc.classify(changes, [cc.Commit("x[bot]", "m")])
    assert c.labels == [label for label in cc.LABELS if label not in ("purely-additive", "tests-only-change")]


# An empty diff gets purely-additive and nothing else.
def test_empty_diff(cc: ModuleType) -> None:
    assert cc.classify([]).labels == ["purely-additive"]


# A repo where main has one test and one doc, and a branch alters the test, deletes the doc and adds a source file
# in a commit with an assistant trailer. Returns the repo path with the branch checked out.
@pytest.fixture
def repo(tmp_path: Path) -> Path:
    git(tmp_path, "init", "-q")
    (tmp_path / "tests").mkdir()
    (tmp_path / "tests" / "test_a.py").write_text("def test_a():\n    assert 1\n")
    (tmp_path / "doc.md").write_text("a\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "base")
    git(tmp_path, "switch", "-q", "-c", "feature")
    (tmp_path / "tests" / "test_a.py").write_text("def test_a():\n    assert 2\n")
    (tmp_path / "doc.md").unlink()
    (tmp_path / "inference_perf").mkdir()
    (tmp_path / "inference_perf" / "x.py").write_text("def f():\n    return 1\n")
    git(tmp_path, "add", "-A")
    git(tmp_path, "commit", "-q", "-m", "Change things\n\nCo-Authored-By: Claude <noreply@anthropic.com>")
    return tmp_path


# load_git on that repo returns the three files with the right sides present and the one branch commit.
def test_load_git_reads_both_sides(cc: ModuleType, repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(repo)
    files, commits = cc.load_git("main", "HEAD")
    assert {f.path: (f.base, f.head, f.deletions) for f in files} == {
        "doc.md": (b"a\n", None, 1),
        "inference_perf/x.py": (None, b"def f():\n    return 1\n", 0),
        "tests/test_a.py": (b"def test_a():\n    assert 1\n", b"def test_a():\n    assert 2\n", 1),
    }
    assert [(c.author, c.message.splitlines()[0]) for c in commits] == [("t <t@t>", "Change things")]


# One more commit renames tests/test_a.py to tests/test_b.py with the same bytes. load_git reports the old
# path as removed and the new path as added, never as one renamed entry.
def test_load_git_splits_a_rename(cc: ModuleType, repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    git(repo, "mv", "tests/test_a.py", "tests/test_b.py")
    git(repo, "commit", "-q", "-m", "rename")
    monkeypatch.chdir(repo)
    files, _ = cc.load_git("main", "HEAD")
    sides = {f.path: (f.base is None, f.head is None) for f in files}
    assert sides["tests/test_a.py"] == (False, True)
    assert sides["tests/test_b.py"] == (True, False)


# One more commit rewrites a binary file. git prints no line counts for it, so it counts as one deletion.
def test_load_git_counts_a_changed_binary_as_a_deletion(cc: ModuleType, repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    git(repo, "switch", "-q", "main")
    (repo / "blob.bin").write_bytes(b"\x00\x01")
    git(repo, "add", "-A")
    git(repo, "commit", "-q", "-m", "binary")
    git(repo, "switch", "-q", "-c", "binary-change")
    (repo / "blob.bin").write_bytes(b"\x00\x02")
    git(repo, "commit", "-q", "-am", "change binary")
    monkeypatch.chdir(repo)
    files, _ = cc.load_git("main", "HEAD")
    assert [(f.path, f.deletions) for f in files] == [("blob.bin", 1)]


# main() on that repo prints the label line first, then the altered test and the deleted doc, and exits 0.
def test_main_prints_the_report(
    cc: ModuleType, repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(repo)
    assert cc.main(["--base", "main"]) == 0
    out = capsys.readouterr().out
    assert out.splitlines()[0] == "labels: changes-pinned-behavior, ai-assisted"
    assert "tests altered:\n  tests/test_a.py::test_a" in out
    assert "files that lost a line:\n  doc.md" in out
    assert "ai-assisted commits:\n  Change things" in out


# main(--json) on that repo prints one JSON object carrying the labels and the unit lists.
def test_main_prints_json(
    cc: ModuleType, repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(repo)
    assert cc.main(["--base", "main", "--json"]) == 0
    data = json.loads(capsys.readouterr().out)
    assert data["labels"] == ["changes-pinned-behavior", "ai-assisted"]
    assert data["tests"]["altered"] == ["tests/test_a.py::test_a"]
    assert data["code"]["added"] == ["inference_perf/x.py::f"]


# main() with a base that does not exist exits 2 and says git failed, instead of printing an empty report.
def test_main_fails_on_a_missing_base(
    cc: ModuleType, repo: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.chdir(repo)
    assert cc.main(["--base", "no-such-branch"]) == 2
    assert "git failed" in capsys.readouterr().err


# With no --base, the default is upstream/main when an upstream remote exists and origin/main when it doesn't.
def test_default_base_prefers_upstream(cc: ModuleType, repo: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.chdir(repo)
    git(repo, "remote", "add", "origin", "https://example.invalid/fork.git")
    assert cc.default_base() == "origin/main"
    git(repo, "remote", "add", "upstream", "https://example.invalid/upstream.git")
    assert cc.default_base() == "upstream/main"
