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
"""Mutation ratchet over the functions a change touches (#872).

The number of surviving mutants in the repo must never go up. Only the
functions in the change's scope can move that number, so they are mutated
twice, on the merge base and on the working tree, with the base branch's
[tool.mutmut] config and each side's full unit suite. The scope is every
function the diff touches or deletes, plus every function a changed test runs.
A test counts as changed when it is added, edited, deleted or renamed, or when
a helper, conftest.py or package __init__.py it runs is.

There is no in-code way to opt out. `# pragma: no mutate` is stripped before
mutating, and hiding code from mutmut behind a decorator fails on its own.

`--all` mutates the whole package instead and reports survivors per module.
"""

import argparse
import ast
import fnmatch
import io
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tokenize
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# A survivor only counts against the change once it survives this many
# reruns of its tests. Timing-dependent tests kill some mutants on one run
# and miss them on the next.
RERUNS = 3

# Where [tool.mutmut] comes from when the base commit has none.
DEFAULT_BRANCH = "origin/main"

SOURCE_PREFIX = "inference_perf/"
TESTS_PREFIX = "tests/"
EXCLUDED_TEST_PREFIXES = ("tests/optional/",)
EXCLUDED_TESTS = {
    # lists tracked YAML with `git ls-files`, and the mutation trees are not checkouts
    "tests/required/config/test_yaml_configs.py",
}

SURVIVED = {"survived", "no tests", "suspicious"}
NO_COVERING_TESTS = "could not find any test case for any mutant"
NOTHING_MATCHES = "nothing matches"
# a failing test during mutmut's own setup runs, which one retry absorbs when it is flaky
SETUP_FAILURES = ("failed to collect stats", "Failed to run clean test")
SPINNER = re.compile(r"^\s*[⠀-⣿] ")

# Survivor diffs printed for the head, so a large change keeps a readable summary.
MAX_SHOWN_SURVIVORS = 30

# The canary sets the active mutant to an id mutmut never generates, so every
# function runs unmutated. Tests that still fail are reacting to the harness.
CANARY_SUFFIX = "__mutmut_999999999"

# A rerun gets its tests' measured time, scaled for a busier runner, plus room for pytest to start.
# Past that it is hung, and a hang must not run into the workflow's 60 minute limit.
RERUN_SLOWDOWN = 5
RERUN_GRACE_S = 60

# Limits for the plain suite run that explains a failed setup. mutmut's own run
# has neither, so one hung test would otherwise hold the job to the workflow timeout.
DIAGNOSIS_TEST_TIMEOUT_S = 300
DIAGNOSIS_TIMEOUT_S = 20 * 60
DIAGNOSIS_TAIL_LINES = 40


@dataclass
class Touched:
    """Functions in a change's scope, named the way mutmut names them."""

    # (name on the base, name on the head); they differ when the file moved
    existing: list[tuple[str, str]] = field(default_factory=list)
    new: list[str] = field(default_factory=list)
    deleted: list[str] = field(default_factory=list)
    # head names of existing functions in scope only because a changed test, or a test using a changed helper, runs them
    via_tests: set[str] = field(default_factory=set)
    # name -> (base, head) statements mutmut cannot check, for functions it skips
    unchecked: dict[str, tuple[int, int]] = field(default_factory=dict)

    @property
    def base_names(self) -> list[str]:
        return [b for b, _ in self.existing] + self.deleted

    @property
    def head_names(self) -> list[str]:
        return [h for _, h in self.existing] + self.new


@dataclass
class Side:
    tree: Path
    tests: list[str] = field(default_factory=list)
    results: dict[str, str] = field(default_factory=dict)
    note: str = ""
    pragmas_stripped: int = 0
    canary_failed: bool = False


class MutationRunError(Exception):
    pass


class SetupFailure(MutationRunError):
    """mutmut's unmutated setup run failed twice. Carries what a plain run of the same suite shows."""

    def __init__(self, tree: Path, failing: list[str], tail: str) -> None:
        self.failing, self.tail = failing, tail
        super().__init__(f"Unit tests fail in the mutation tree {tree} before any mutant runs.\n{self.details()}")

    def details(self) -> str:
        shown = (
            "\n".join(f"  {test}" for test in self.failing)
            or "  none in a plain rerun, so it is flaky or specific to mutmut's run"
        )
        return f"Failing tests:\n{shown}\nLast lines of the pytest run:\n{self.tail}"


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def module_name(path: str) -> str:
    """mutmut's module name for a source path: a package's __init__.py is the package."""
    return path.removesuffix(".py").replace("/", ".").removesuffix(".__init__")


FunctionNode = ast.FunctionDef | ast.AsyncFunctionDef


def functions(source: str) -> dict[str, FunctionNode]:
    """Each module-level function and method, keyed by mutmut's mangled name."""
    found: dict[str, FunctionNode] = {}
    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            found[f"x_{node.name}"] = node
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    found[f"xǁ{node.name}ǁ{sub.name}"] = sub
    return found


def function_spans(source: str) -> dict[str, tuple[int, int]]:
    """Line span of each module-level function and method, keyed by mutmut's mangled name."""
    # decorators belong to the function, so a change to one touches it
    return {
        name: (min([d.lineno for d in node.decorator_list] + [node.lineno]), node.end_lineno or node.lineno)
        for name, node in functions(source).items()
    }


def mutmut_skips(node: FunctionNode) -> bool:
    """mutmut 3.8 leaves decorated functions unmutated, unless the one decorator is a bare staticmethod or classmethod."""
    if not node.decorator_list:
        return False
    only = node.decorator_list[0] if len(node.decorator_list) == 1 else None
    return not (isinstance(only, ast.Name) and only.id in ("staticmethod", "classmethod"))


def _plain(expr: ast.expr | None) -> bool:
    """Names, attributes, constants and calls built only from those: nothing a mutation could flip."""
    if expr is None or isinstance(expr, (ast.Name, ast.Constant)):
        return True
    if isinstance(expr, (ast.Attribute, ast.Starred, ast.Await)):
        return _plain(expr.value)
    if isinstance(expr, ast.Call):
        return _plain(expr.func) and all(_plain(a) for a in expr.args) and all(_plain(k.value) for k in expr.keywords)
    return False


def _delegates(stmt: ast.stmt) -> bool:
    if isinstance(stmt, ast.Pass):
        return True
    if isinstance(stmt, (ast.Expr, ast.Return)):
        return _plain(stmt.value)
    if isinstance(stmt, ast.Raise):
        return _plain(stmt.exc)
    if isinstance(stmt, ast.Assign):
        return all(isinstance(t, (ast.Name, ast.Attribute)) and _plain(t) for t in stmt.targets) and _plain(stmt.value)
    return False


def unchecked_statements(source: str) -> dict[str, int]:
    """Statements in each function mutmut skips. A body that only delegates, like `return helper(self)`, counts 0."""
    counts: dict[str, int] = {}
    for name, node in functions(source).items():
        if not mutmut_skips(node):
            continue
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            if isinstance(body[0].value.value, str):
                body = body[1:]  # docstring
        if len(body) <= 1 and all(_delegates(stmt) for stmt in body):
            counts[name] = 0
        else:
            counts[name] = sum(1 for stmt in body for sub in ast.walk(stmt) if isinstance(sub, ast.stmt))
    return counts


def touched_names(spans: dict[str, tuple[int, int]], lines: set[int]) -> set[str]:
    return {name for name, (start, end) in spans.items() if any(start <= line <= end for line in lines)}


def parse_hunks(diff: str) -> tuple[set[int], set[int]]:
    """Lines a zero-context diff touches, numbered on the old side and on the new side.

    A pure insertion marks no old line. The inserted lines already sit inside
    whichever function they extend on the new side, and the old neighbour may
    be the end of an unrelated function. Deletions mirror this.
    """
    old: set[int] = set()
    new: set[int] = set()
    for m in re.finditer(r"^@@ -(\d+)(?:,(\d+))? \+(\d+)(?:,(\d+))? @@", diff, re.M):
        old_start, old_len = int(m.group(1)), int(m.group(2) or 1)
        new_start, new_len = int(m.group(3)), int(m.group(4) or 1)
        old.update(range(old_start, old_start + old_len))
        new.update(range(new_start, new_start + new_len))
    return old, new


def mutable(path: str | None, do_not_mutate: list[str]) -> bool:
    # fnmatch on the repo-relative path, as mutmut matches do_not_mutate, so `*` spans directories
    return path is not None and path.endswith(".py") and not any(fnmatch.fnmatch(path, p) for p in do_not_mutate)


def changed_source_files(base: str, do_not_mutate: list[str]) -> list[tuple[str | None, str | None]]:
    """(base path, head path) for each changed source file, following renames. None marks a side without the file.

    The head is the working tree, so uncommitted and untracked files count.
    """
    pairs: list[tuple[str | None, str | None]] = []
    for line in git("diff", "--name-status", "-M", base, "--", SOURCE_PREFIX).splitlines():
        status, *paths = line.split("\t")
        if status.startswith("R"):
            pairs.append((paths[0], paths[1]))
        elif status == "D":
            pairs.append((paths[0], None))
        elif status == "A":
            pairs.append((None, paths[0]))
        else:
            pairs.append((paths[0], paths[0]))
    for path in git("ls-files", "--others", "--exclude-standard", "--", SOURCE_PREFIX).splitlines():
        pairs.append((None, path))
    kept = [(b if mutable(b, do_not_mutate) else None, h if mutable(h, do_not_mutate) else None) for b, h in pairs]
    return [(b, h) for b, h in kept if b or h]


def touched_functions(base: str, do_not_mutate: list[str]) -> Touched:
    touched = Touched()
    for base_path, head_path in changed_source_files(base, do_not_mutate):
        base_src = git("show", f"{base}:{base_path}") if base_path else ""
        head_src = Path(head_path).read_text() if head_path else ""
        base_spans, head_spans = function_spans(base_src), function_spans(head_src)
        base_unchecked, head_unchecked = unchecked_statements(base_src), unchecked_statements(head_src)
        if base_path and head_path:
            old_lines, new_lines = parse_hunks(git("diff", "-U0", "-M", base, "--", base_path, head_path))
            hit = touched_names(base_spans, old_lines) | touched_names(head_spans, new_lines)
        else:
            hit = set(base_spans) | set(head_spans)
        for name in sorted(hit):
            base_name = f"{module_name(base_path)}.{name}" if base_path and name in base_spans else None
            head_name = f"{module_name(head_path)}.{name}" if head_path and name in head_spans else None
            if base_name and head_name:
                touched.existing.append((base_name, head_name))
            elif head_name:
                touched.new.append(head_name)
            elif base_name:
                touched.deleted.append(base_name)
            counts = (base_unchecked.get(name, 0), head_unchecked.get(name, 0))
            if any(counts):
                touched.unchecked[head_name or base_name or name] = counts
    return touched


def is_test_file(path: str) -> bool:
    return Path(path).name.startswith("test_") and path.endswith(".py")


def changed_test_files(base: str) -> list[str]:
    """Test files the change adds, edits, deletes or renames, under their names on both sides.

    A deleted test, or one renamed out of the suite, still exists on the base, and what it ran there
    is where the change can let survivors through.
    """
    changed: set[str] = set()
    for line in git("diff", "--name-status", "-M", base, "--", TESTS_PREFIX).splitlines():
        _, *paths = line.split("\t")
        if any(is_test_file(p) for p in paths):
            changed.update(paths)
    changed.update(
        p for p in git("ls-files", "--others", "--exclude-standard", "--", TESTS_PREFIX).splitlines() if is_test_file(p)
    )
    return sorted(changed)


def changed_test_support(base: str) -> list[str]:
    """Python files under tests/ the change touches that are not test files: helpers, conftests, package inits."""
    changed: set[str] = set()
    for line in git("diff", "--name-status", "-M", base, "--", TESTS_PREFIX).splitlines():
        _, *paths = line.split("\t")
        changed.update(paths)
    changed.update(git("ls-files", "--others", "--exclude-standard", "--", TESTS_PREFIX).splitlines())
    return sorted(p for p in changed if p.endswith(".py") and not is_test_file(p))


def _dotted_file(tree: Path, root: Path, dotted: str) -> str | None:
    for candidate in (root / (dotted.replace(".", "/") + ".py"), root / dotted.replace(".", "/") / "__init__.py"):
        if candidate.is_file():
            return str(candidate.relative_to(tree))
    return None


def imports_of(tree: Path, rel: str) -> set[str]:
    """Files under tests/ that `rel` imports, resolved every way pytest's default import mode makes them importable."""
    path = tree / rel
    try:
        module = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return set()
    # pytest puts the first directory above the test's package on sys.path, and runs from the repo root
    basedir = path.parent
    while (basedir / "__init__.py").is_file() and basedir != tree:
        basedir = basedir.parent
    roots = {tree, path.parent, basedir}
    found: set[str] = set()
    for node in ast.walk(module):
        wanted: list[tuple[set[Path], str]]
        if isinstance(node, ast.Import):
            wanted = [(roots, ".".join(a.name.split(".")[: i + 1])) for a in node.names for i in range(a.name.count(".") + 1)]
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            wanted = [(roots, node.module)] + [(roots, f"{node.module}.{a.name}") for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            package = path.parent
            for _ in range(node.level - 1):
                package = package.parent
            prefix = f"{node.module}." if node.module else ""
            wanted = [({package}, f"{prefix}{a.name}") for a in node.names]
            if node.module:
                wanted.append(({package}, node.module))
        else:
            continue
        for search, dotted in wanted:
            for root in search:
                hit = _dotted_file(tree, root, dotted)
                if hit and hit.startswith(TESTS_PREFIX):
                    found.add(hit)
    return found


def tests_using(tree: Path, support: list[str]) -> set[str]:
    """Unit test files that run any of `support`.

    A test runs a helper it imports, directly or through other helpers, and every conftest.py and
    package __init__.py in the directories above it.
    """
    importers: dict[str, set[str]] = {}
    for path in (tree / TESTS_PREFIX).rglob("*.py"):
        rel = str(path.relative_to(tree))
        for imported in imports_of(tree, rel):
            importers.setdefault(imported, set()).add(rel)
    reached = {p for p in support if (tree / p).is_file()}
    pending = list(reached)
    while pending:
        for importer in importers.get(pending.pop(), set()) - reached:
            reached.add(importer)
            pending.append(importer)
    suite = unit_suite(tree)
    found = {p for p in reached if p in suite}
    for rel in reached:
        if Path(rel).name in ("conftest.py", "__init__.py"):
            directory = str(Path(rel).parent) + "/"
            found.update(t for t in suite if t.startswith(directory))
    return found


def tests_in_scope(base: str, base_tree: Path, head_tree: Path) -> list[str]:
    """Changed test files plus the unit tests that run changed helpers, conftests or package inits, on either side."""
    support = changed_test_support(base)
    return sorted(set(changed_test_files(base)) | tests_using(base_tree, support) | tests_using(head_tree, support))


def unit_suite(tree: Path) -> list[str]:
    """Every unit test file in the tree. mutmut's own stats then decide which tests run against which function."""
    found = (str(p.relative_to(tree)) for p in (tree / TESTS_PREFIX).rglob("test_*.py"))
    return sorted(p for p in found if p not in EXCLUDED_TESTS and not p.startswith(EXCLUDED_TEST_PREFIXES))


def functions_run_by(tree: Path, tests: list[str]) -> set[str]:
    """Functions whose body runs while one of `tests` runs, not at import or collection time."""
    tests = [t for t in tests if (tree / t).exists() and t not in EXCLUDED_TESTS]
    if not tests:
        return set()
    data, report = tree / ".mutation-coverage", tree / ".mutation-coverage.json"
    # the tree's own copy must win over the editable install of the checkout, and an inherited
    # COVERAGE_PROCESS_START from an outer coverage run would start a second tracer
    env = {k: v for k, v in os.environ.items() if k != "COVERAGE_PROCESS_START"}
    env.update(COVERAGE_FILE=str(data), PYTHONPATH=str(tree))
    cov = ["--cov=inference_perf", "--cov-context=test", "--cov-report="]
    subprocess.run(
        [sys.executable, "-m", "pytest", "-q", "-p", "no:cacheprovider", "-o", "log_cli=false", *cov, *tests],
        cwd=tree,
        env=env,
        capture_output=True,
    )
    subprocess.run(
        [sys.executable, "-m", "coverage", "json", "--show-contexts", "-o", str(report)],
        cwd=tree,
        env=env,
        capture_output=True,
    )
    if not report.exists():
        raise MutationRunError(f"coverage over the changed tests produced no report in {tree}")
    found: set[str] = set()
    for path, info in json.loads(report.read_text())["files"].items():
        rel = os.path.relpath(path, tree) if os.path.isabs(path) else path
        if not rel.startswith(SOURCE_PREFIX):
            continue
        # an empty context is import or collection time
        lines = {int(n) for n, contexts in info.get("contexts", {}).items() if any(contexts)}
        for name, (start, end) in function_spans((tree / rel).read_text()).items():
            # the def line itself runs at import, so a function counts only once its body runs
            if any(start < line <= end for line in lines):
                found.add(f"{module_name(rel)}.{name}")
    return found


def add_functions_run_by_changed_tests(touched: Touched, base_tree: Path, head_tree: Path, tests: list[str]) -> None:
    """Bring in unchanged functions a changed test runs. That is where a loosened assert or a backfill test shows."""
    in_scope = set(touched.base_names) | set(touched.head_names)
    for name in sorted(functions_run_by(base_tree, tests) | functions_run_by(head_tree, tests)):
        if name in in_scope or not _exists(base_tree, name) or not _exists(head_tree, name):
            continue
        touched.existing.append((name, name))
        touched.via_tests.add(name)


def _exists(tree: Path, name: str) -> bool:
    module, _, function = name.rpartition(".")
    path = _source_path(tree, module)
    return path is not None and function in function_spans((tree / path).read_text())


def write_config(tree: Path, config: dict[str, Any], files: list[str], tests: list[str]) -> None:
    """Replace the tree's [tool.mutmut] with the shared config plus this run's scope."""
    pyproject = tree / "pyproject.toml"
    text = re.sub(r"^\[tool\.mutmut\]\n(?:(?!\[).*\n?)*", "", pyproject.read_text(), flags=re.M)
    merged = {**config, "only_mutate": files, "pytest_add_cli_args_test_selection": tests}
    # JSON scalars and arrays are valid TOML inline values
    table = "\n".join(["[tool.mutmut]", *(f"{k} = {json.dumps(v)}" for k, v in merged.items())])
    pyproject.write_text(text.rstrip("\n") + "\n\n" + table + "\n")


def export_base(base: str, dest: Path) -> None:
    dest.mkdir(parents=True)
    archive = subprocess.Popen(["git", "archive", base], stdout=subprocess.PIPE)
    subprocess.run(["tar", "-x", "-C", str(dest)], stdin=archive.stdout, check=True)
    if archive.wait() != 0:
        raise MutationRunError(f"git archive {base} failed")


def export_working_tree(dest: Path) -> None:
    for rel in git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split("\0"):
        src = Path(rel)
        if rel and src.is_file():
            (dest / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dest / rel)


def mutmut(tree: Path, *args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, "-m", "mutmut", *args], cwd=tree, capture_output=True, text=True)


def pytest_command(tests: list[str], pytest_args: list[str]) -> list[str]:
    # mirrors how mutmut's own runner invokes pytest for a mutant
    head = [sys.executable, "-m", "pytest", "--rootdir=.", "--tb=native", "-x", "-q"]
    return head + ["-p", "no:randomly", "-p", "no:random-order", *tests, *pytest_args]


def strip_no_mutate_pragmas(tree: Path) -> int:
    """Delete every `# pragma: no mutate` comment under the source tree and return how many there were."""
    stripped = 0
    for path in sorted((tree / SOURCE_PREFIX).rglob("*.py")):
        text = path.read_text()
        if "no mutate" not in text:
            continue
        lines = text.splitlines(keepends=True)
        tokens = tokenize.generate_tokens(io.StringIO(text).readline)
        # same test mutmut uses to recognise one
        hits = [t for t in tokens if t.type == tokenize.COMMENT and "# pragma:" in t.string and "no mutate" in t.string]
        for token in reversed(hits):
            row, col = token.start
            line = lines[row - 1]
            ending = line[len(line.rstrip("\r\n")) :]
            lines[row - 1] = line[:col].rstrip() + ending
        path.write_text("".join(lines))
        stripped += len(hits)
    return stripped


def tests_covering(tree: Path, names: list[str]) -> list[str]:
    """Test ids mutmut's stats map to any of `names`."""
    stats = tree / "mutants" / "mutmut-stats.json"
    if not stats.exists():
        return []
    by_function: dict[str, list[str]] = json.loads(stats.read_text())["tests_by_mangled_function_name"]
    return sorted({test for name in names for test in by_function.get(name, [])})


def rerun_timeout(tree: Path, tests: list[str]) -> float:
    """Wall-clock limit for one rerun of `tests`, from the durations mutmut's stats run measured."""
    stats = tree / "mutants" / "mutmut-stats.json"
    durations: dict[str, float] = json.loads(stats.read_text()).get("duration_by_test", {}) if stats.exists() else {}
    return RERUN_GRACE_S + RERUN_SLOWDOWN * sum(durations.get(test, 0.0) for test in tests)


def rerun(side: Side, tests: list[str], mutant: str, pytest_args: list[str]) -> int | None:
    """Run `tests` with `mutant` active in the side's mutation tree. Returns pytest's exit code, or None if it hung."""
    try:
        return subprocess.run(
            pytest_command(tests, pytest_args),
            cwd=side.tree / "mutants",
            env={**os.environ, "MUTANT_UNDER_TEST": mutant},
            capture_output=True,
            timeout=rerun_timeout(side.tree, tests),
        ).returncode
    except subprocess.TimeoutExpired:
        return None


def canary_passes(side: Side, names: list[str], pytest_args: list[str]) -> bool:
    """Run the tests covering `names` with an active mutant id that matches no mutant. One pass clears a flaky test."""
    tests = tests_covering(side.tree, names)
    if not tests:
        return True
    for _ in range(RERUNS):
        # a hang counts as a failed run, so tests that only hang under the harness still fail the canary
        if rerun(side, tests, names[0] + CANARY_SUFFIX, pytest_args) in (0, 5):  # 5: nothing collected
            return True
    return False


def diagnose_setup_failure(side: Side, pytest_args: list[str]) -> SetupFailure:
    """Run the side's suite unmutated to the end and name what fails. mutmut's setup run swallows pytest's output."""
    command = [sys.executable, "-m", "pytest", "--rootdir=.", "-rfE", "--tb=short", "--disable-warnings", "-p", "no:randomly"]
    command += ["-p", "no:random-order", f"--timeout={DIAGNOSIS_TEST_TIMEOUT_S}", *side.tests, *pytest_args]
    env = {**os.environ, "MUTANT_UNDER_TEST": "setup" + CANARY_SUFFIX}
    try:
        run = subprocess.run(
            command, cwd=side.tree / "mutants", env=env, capture_output=True, text=True, timeout=DIAGNOSIS_TIMEOUT_S
        )
        output = run.stdout + run.stderr
    except subprocess.TimeoutExpired as e:
        # captured output is bytes here even with text=True
        partial = e.stdout.decode(errors="replace") if isinstance(e.stdout, bytes) else ""
        output = partial + f"\nThe suite was still running after {DIAGNOSIS_TIMEOUT_S} s and was stopped."
    lines = output.splitlines()
    failing = sorted({line.split(" ")[1] for line in lines if line.startswith(("FAILED ", "ERROR ")) and " " in line})
    return SetupFailure(side.tree, failing, "\n".join(lines[-DIAGNOSIS_TAIL_LINES:]))


def run_mutmut(side: Side, globs: list[str], max_children: int, pytest_args: list[str]) -> tuple[str, bool]:
    """Run mutmut, retrying once when its setup test runs fail. Returns the output and whether no test covered anything."""
    for attempt in range(2):
        run = mutmut(side.tree, "run", "--max-children", str(max_children), *globs)
        output = (run.stdout + run.stderr).replace("\r", "\n")
        uncovered = NO_COVERING_TESTS in output
        if run.returncode == 0 or uncovered or NOTHING_MATCHES in output:
            return output, uncovered
        if not any(marker in output for marker in SETUP_FAILURES):
            break
        if attempt == 1:
            raise diagnose_setup_failure(side, pytest_args)
        side.note = "mutmut's setup test run failed once and passed on retry"
    # drop mutmut's spinner frames so the real error is what's left
    detail = "\n".join(line for line in output.splitlines() if line.strip() and not SPINNER.match(line))
    raise MutationRunError(f"mutmut run failed in {side.tree} (exit {run.returncode}):\n{detail[-3000:]}")


def run_side(
    tree: Path, names: list[str], config: dict[str, Any], max_children: int, pragmas_stripped: int, canary: bool = False
) -> Side:
    side = Side(tree=tree, tests=unit_suite(tree), pragmas_stripped=pragmas_stripped)
    files = sorted({p for p in (_source_path(tree, name.rpartition(".")[0]) for name in names) if p is not None})
    write_config(tree, config, files, side.tests)
    pytest_args = [str(a) for a in config.get("pytest_add_cli_args", [])]
    output, uncovered = run_mutmut(side, [f"{n}__mutmut_*" for n in names], max_children, pytest_args)
    if NOTHING_MATCHES in output:
        side.note = "no mutants (decorated or nothing to mutate)"
        return side
    wanted = set(names)
    for line in mutmut(tree, "results", "--all", "true").stdout.splitlines():
        mutant, sep, status = line.strip().rpartition(": ")
        if sep and mutant.partition("__mutmut_")[0] in wanted:
            side.results[mutant] = "no tests" if uncovered and status == "not checked" else status
    unchecked = sorted(m for m, s in side.results.items() if s == "not checked")
    if unchecked:
        raise MutationRunError(f"mutmut left {len(unchecked)} mutants unchecked in {tree}, e.g. {unchecked[0]}")
    if uncovered:
        side.note = "no unit test runs these functions"
    elif canary and side.results:
        side.canary_failed = not canary_passes(side, names, pytest_args)
    return side


def _source_path(tree: Path, module: str) -> str | None:
    for candidate in (module.replace(".", "/") + ".py", module.replace(".", "/") + "/__init__.py"):
        if (tree / candidate).is_file():
            return candidate
    return None


def mutant_order(mutant: str) -> tuple[str, int]:
    name, _, number = mutant.partition("__mutmut_")
    return name, int(number) if number.isdigit() else 0


def survivors(side: Side | None, names: list[str]) -> list[str]:
    if side is None:
        return []
    wanted = set(names)
    found = (m for m, s in side.results.items() if s in SURVIVED and m.partition("__mutmut_")[0] in wanted)
    return sorted(found, key=mutant_order)


def still_survives(side: Side, mutant: str, pytest_args: list[str]) -> bool:
    """Rerun the tests that cover a surviving mutant. Killed in any rerun means flaky."""
    if side.results.get(mutant) == "no tests":
        return True
    tests = [t for t in mutmut(side.tree, "tests-for-mutant", mutant).stdout.splitlines() if "::" in t]
    for _ in range(RERUNS):
        code = rerun(side, tests, mutant, pytest_args)
        if code is None:
            # a hang is not a kill. Dropping the mutant as flaky would lower the count the change is held to
            return True
        if code != 0:
            return False
    return True


def by_function(mutants: list[str]) -> Counter[str]:
    return Counter(m.partition("__mutmut_")[0] for m in mutants)


@dataclass
class Verdict:
    base_survivors: int
    head_survivors: int
    flaky: list[str]
    unexecuted: list[str]
    failures: list[str]


def judge(touched: Touched, base: Side | None, head: Side, pytest_args: list[str]) -> Verdict:
    base_surv = survivors(base, touched.base_names)
    head_surv = survivors(head, touched.head_names)
    flaky: list[str] = []
    if len(head_surv) > len(base_surv):
        # only a function whose own count rose can hold a survivor the change added
        before, after = by_function(base_surv), by_function(head_surv)
        rose = {h for b, h in touched.existing if after[h] > before[b]} | {n for n in touched.new if after[n]}
        flaky = [m for m in head_surv if m.partition("__mutmut_")[0] in rose and not still_survives(head, m, pytest_args)]
        head_surv = [m for m in head_surv if m not in flaky]
    # unlike a surviving mutant, code no test runs cannot be an equivalent mutant, so nothing can pay for it
    unexecuted = sorted({m.partition("__mutmut_")[0] for m, s in head.results.items() if s == "no tests"} & set(touched.new))
    verdict = Verdict(len(base_surv), len(head_surv), flaky, unexecuted, [])
    if verdict.head_survivors > verdict.base_survivors:
        verdict.failures.append(f"Surviving mutants rose from {verdict.base_survivors} to {verdict.head_survivors}")
    if unexecuted:
        verdict.failures.append(f"No test runs {', '.join(f'`{n}`' for n in unexecuted)}")
    # a decorator, or anything else that leaves mutmut nothing to mutate, must not hide a checked function
    base_mutants, head_mutants = by_function(list(base.results) if base else []), by_function(list(head.results))
    for b, h in touched.existing:
        if base_mutants[b] and not head_mutants[h]:
            verdict.failures.append(f"`{h}` had {base_mutants[b]} mutants on the base and none with this change")
    unchecked_base = sum(b for b, _ in touched.unchecked.values())
    unchecked_head = sum(h for _, h in touched.unchecked.values())
    if unchecked_head > unchecked_base:
        verdict.failures.append(
            f"Statements mutmut cannot check rose from {unchecked_base} to {unchecked_head}. "
            "Decorated functions should delegate to an undecorated one"
        )
    if head.canary_failed:
        verdict.failures.append("Tests fail with an active mutant id that changes no code, so they react to the harness")
    return verdict


def report(base_ref: str, touched: Touched, base: Side | None, head: Side, verdict: Verdict) -> str:
    before = by_function(survivors(base, touched.base_names))
    after = by_function([m for m in survivors(head, touched.head_names) if m not in verdict.flaky])
    lines = [f"## Mutation ratchet against `{base_ref}`", "", "| Function | Why | Base | Head |", "|---|---|---|---|"]
    for b, h in touched.existing:
        why = "a changed test or test helper runs it" if h in touched.via_tests else "changed"
        lines.append(f"| `{h}` | {why} | {before[b]} | {after[h]} |")
    lines += [f"| `{n}` | new | | {after[n]} |" for n in touched.new]
    lines += [f"| `{b}` | deleted | {before[b]} | |" for b in touched.deleted]
    lines += ["", f"Surviving mutants: {verdict.base_survivors} on the base, {verdict.head_survivors} with this change.", ""]
    for side, label in ((base, "Base"), (head, "Head")):
        if side is not None and side.note:
            lines.append(f"{label}: {side.note}.")
    if touched.unchecked:
        lines += ["", "| Function mutmut skips | Unchecked statements, base | Head |", "|---|---|---|"]
        lines += [f"| `{name}` | {b} | {h} |" for name, (b, h) in touched.unchecked.items()]
    if head.pragmas_stripped:
        lines += ["", f"Stripped {head.pragmas_stripped} `# pragma: no mutate` comments before mutating. They have no effect."]
    if verdict.flaky:
        lines += ["", f"Flaky, killed on rerun and not counted: {', '.join(f'`{m}`' for m in verdict.flaky)}"]
    shown = [m for m in survivors(head, touched.head_names) if m not in verdict.flaky]
    if shown:
        lines += ["", f"<details><summary>{len(shown)} surviving mutants on the head</summary>", ""]
        for mutant in shown[:MAX_SHOWN_SURVIVORS]:
            # mutmut show already heads the diff with the mutant name and status
            lines += ["```diff", mutmut(head.tree, "show", mutant).stdout.strip(), "```", ""]
        if len(shown) > MAX_SHOWN_SURVIVORS:
            lines.append(f"{len(shown) - MAX_SHOWN_SURVIVORS} more not shown.")
        lines.append("</details>")
    lines += ["", "FAIL: " + "; ".join(verdict.failures) if verdict.failures else "PASS"]
    return "\n".join(lines) + "\n"


def report_error(message: str) -> None:
    """Put an error that stops the check where the PR author sees it: an annotation on the check and the job summary."""
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::error title=Mutation ratchet::{message.splitlines()[0]}")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(f"## Mutation ratchet: error\n\n```\n{message}\n```\n")


def load_config(merge_base: str) -> dict[str, Any]:
    """[tool.mutmut] as the base branch has it, so a change cannot loosen its own check.

    A base cut before the table landed, like an older feature branch, gets main's instead.
    """
    for ref in (merge_base, DEFAULT_BRANCH):
        try:
            config: dict[str, Any] = tomllib.loads(git("show", f"{ref}:pyproject.toml"))["tool"]["mutmut"]
            return config
        except (KeyError, subprocess.CalledProcessError):
            continue
    # reachable only until [tool.mutmut] first lands on main
    print(f"Neither the base branch nor {DEFAULT_BRANCH} has [tool.mutmut]. Using the working tree's.")
    fallback: dict[str, Any] = tomllib.loads(Path("pyproject.toml").read_text())["tool"]["mutmut"]
    return fallback


def whole_package(args: argparse.Namespace) -> None:
    """Mutate every source file against the full unit suite and print survivors per module. Takes hours."""
    config: dict[str, Any] = tomllib.loads(Path("pyproject.toml").read_text())["tool"]["mutmut"]
    workdir = Path(tempfile.mkdtemp(prefix="mutation-all-"))
    try:
        export_working_tree(workdir)
        side = Side(tree=workdir, tests=unit_suite(workdir), pragmas_stripped=strip_no_mutate_pragmas(workdir))
        files = sorted(str(p.relative_to(workdir)) for p in (workdir / SOURCE_PREFIX).rglob("*.py"))
        write_config(workdir, config, [f for f in files if mutable(f, config.get("do_not_mutate", []))], side.tests)
        run_mutmut(side, [], args.max_children, [str(a) for a in config.get("pytest_add_cli_args", [])])
        results: dict[str, str] = {}
        for line in mutmut(workdir, "results", "--all", "true").stdout.splitlines():
            mutant, sep, status = line.strip().rpartition(": ")
            if sep and "__mutmut_" in mutant:
                results[mutant] = status
    except MutationRunError as e:
        print(f"ERROR: {e}")
        report_error(str(e))
        sys.exit(1)
    finally:
        if args.keep:
            print(f"Mutation tree kept in {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)
    totals: dict[str, Counter[str]] = {}
    for mutant, status in results.items():
        module = mutant.partition("__mutmut_")[0].rpartition(".")[0]
        totals.setdefault(module, Counter())["survived" if status in SURVIVED else "other"] += 1
    print("| Module | Mutants | Surviving |\n|---|---|---|")
    for module, counts in sorted(totals.items(), key=lambda kv: -kv[1]["survived"]):
        print(f"| `{module}` | {sum(counts.values())} | {counts['survived']} |")
    surviving = sum(c["survived"] for c in totals.values())
    print(f"\n{len(results)} mutants, {surviving} surviving.")
    Path(args.output).write_text(json.dumps(results, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="Mutation ratchet over the functions changed since the base branch.")
    parser.add_argument("--base", default="origin/main", help="Branch the change merges into (default: origin/main)")
    parser.add_argument("--all", action="store_true", help="Mutate the whole package and report survivors per module")
    parser.add_argument("--max-children", type=int, default=os.cpu_count() or 4, help="Parallel mutmut workers")
    parser.add_argument("--keep", action="store_true", help="Keep the mutation trees for debugging")
    parser.add_argument("--output", default="mutation.json", help="Where to write the full results")
    args = parser.parse_args()
    if args.all:
        whole_package(args)
        return

    merge_base = git("merge-base", args.base, "HEAD").strip()
    config = load_config(merge_base)
    do_not_mutate = [str(p) for p in config.get("do_not_mutate", [])]
    touched = touched_functions(merge_base, do_not_mutate)
    changed_tests = changed_test_files(merge_base) + changed_test_support(merge_base)
    if not (touched.existing or touched.new or touched.deleted or changed_tests):
        print(f"No functions under {SOURCE_PREFIX} and no tests changed since {args.base}. Nothing to mutate.")
        return

    pytest_args = [str(a) for a in config.get("pytest_add_cli_args", [])]
    workdir = Path(tempfile.mkdtemp(prefix="mutation-"))
    try:
        export_base(merge_base, workdir / "base")
        (workdir / "head").mkdir()
        export_working_tree(workdir / "head")
        stripped = {side: strip_no_mutate_pragmas(workdir / side) for side in ("base", "head")}
        tests = tests_in_scope(merge_base, workdir / "base", workdir / "head")
        if tests:
            add_functions_run_by_changed_tests(touched, workdir / "base", workdir / "head", tests)
        if not (touched.base_names or touched.head_names):
            print(f"No function in scope since {args.base}. Nothing to mutate.")
            return
        print(
            f"Mutating {len(touched.existing)} existing ({len(touched.via_tests)} via changed tests), "
            f"{len(touched.new)} new and {len(touched.deleted)} deleted functions against {args.base} ({merge_base[:8]})."
        )
        base = None
        if touched.base_names:
            try:
                base = run_side(workdir / "base", touched.base_names, config, args.max_children, stripped["base"])
            except SetupFailure as e:
                raise MutationRunError(
                    f"The base branch {args.base} at {merge_base[:8]} fails its own unit tests in the mutation tree, "
                    f"so there is nothing to compare this change against. Fix {args.base} first. "
                    f"Every PR fails this check until then.\n{e.details()}"
                ) from e
        head = Side(tree=workdir / "head", pragmas_stripped=stripped["head"])
        if touched.head_names:
            head = run_side(workdir / "head", touched.head_names, config, args.max_children, stripped["head"], canary=True)
        verdict = judge(touched, base, head, pytest_args)
        summary = report(args.base, touched, base, head, verdict)
    except MutationRunError as e:
        print(f"ERROR: {e}")
        report_error(str(e))
        sys.exit(1)
    finally:
        if args.keep:
            print(f"Mutation trees kept in {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(summary)
    record = {
        "base": args.base,
        "merge_base": merge_base,
        "existing": touched.existing,
        "new": touched.new,
        "deleted": touched.deleted,
        "via_tests": sorted(touched.via_tests),
        "base_results": base.results if base else {},
        "head_results": head.results,
        "flaky": verdict.flaky,
        "unchecked": touched.unchecked,
        "pragmas_stripped": head.pragmas_stripped,
        "canary_failed": head.canary_failed,
        "failures": verdict.failures,
    }
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n")
    if verdict.failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
