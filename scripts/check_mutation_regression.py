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

Code that only runs in a multiprocessing child counts too. A pytest plugin written
outside the tree carries the child's trampoline hits and coverage back to the test.

There is no in-code way to opt out. `# pragma: no mutate` is stripped before
mutating. Some checks need no mutants and fail a change even when nothing is
in scope: statements mutmut never mutates cannot rise across the changed files,
inference_perf/ cannot import repo code from outside it, a new root module
cannot shadow an installed one, no source or test may name the harness, and
mutmut must be the pinned release.

`--all` mutates the whole package instead and reports survivors per module.
"""

import argparse
import ast
import atexit
import fnmatch
import importlib.metadata
import io
import json
import os
import re
import secrets
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import tokenize
import tomllib
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, NamedTuple, Protocol

# A new survivor is rerun up to this many times. It stops counting only when every rerun kills it,
# which means the first run missed. A test that kills it only sometimes is flaky, and a flaky
# kill is not evidence the tests check the code.
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

# Code that can tell it runs under mutmut can kill every mutant without checking anything, and no
# test needs to know. Matched case-insensitively: "mutant" also covers MUTANT_UNDER_TEST and mutants/.
HARNESS_WORDS = ("mutmut", "mutant")
# The script's own tests have to name the harness. #876's gate list covers this file.
HARNESS_WORDS_ALLOWED_IN = {"tests/required/scripts/test_check_mutation_regression.py"}

# The mutmut release every run uses and its files' hashes in pdm.lock. A PR may only change them by
# changing this script, which #876 lands alone in front of an owner, so a swapped mutmut cannot grade itself.
MUTMUT_VERSION = "3.8.0"
MUTMUT_HASHES = {
    "sha256:4c0abcef6b8d1116f2adfaad19df12e1c621e4a402d4367ef15bf8cba497b91d",
    "sha256:bc670276518f893ab356f1d28826c49b7e606e7cf89c300e3888c0b52fc1d2ed",
}
# pdm.lock keys that install a package from somewhere other than the package index
NON_INDEX_SOURCES = ("git", "url", "path", "editable")

# Survivor diffs printed for the head, so a large change keeps a readable summary.
MAX_SHOWN_SURVIVORS = 30

# What each failure means and how to pass. Every failing summary and error ends by pointing here.
DOC_URL = "https://github.com/kubernetes-sigs/inference-perf/blob/main/docs/mutation_testing.md"
DOC_LINE = f"What each failure means and how to pass: {DOC_URL}"

# Code that only runs in a multiprocessing child is invisible twice over: mutmut records trampoline
# hits in the pytest process, and coverage in a child starts from the config file with the empty
# context that marks import time. This pytest plugin, loaded into every test run, carries both back
# to the test that started the child. It is written outside the tree, so a change cannot edit it.
CHILD_PLUGIN = "mutation_child_hits"
CHILD_PLUGIN_SOURCE = '''"""Carry what a forked child does back to the test that started it."""

import os
from pathlib import Path

import pytest

HITS_DIR = Path(os.environ.get("MUTATION_HITS_DIR", "."))
_current = None
_count = 0
_registered = False


def pytest_configure(config):
    global _registered
    # pytest.main can run more than once in a process, and a forked child inherits the handler
    if not _registered:
        _registered = True
        os.register_at_fork(after_in_child=_in_child)


def pytest_sessionstart(session):
    # coverage in this process has read its config by now. A child reads the config again and
    # expands this into its static context, so its lines do not read as import time.
    os.environ["MUTATION_CHILD_CONTEXT"] = "child"


def pytest_runtest_setup(item):
    global _current, _count
    _count += 1
    _current = HITS_DIR / f"{os.getpid()}-{_count}.hits"


def _in_child():
    # only mutmut's stats run records hits. A mutant run reaches the child through the environment
    if os.environ.get("MUTANT_UNDER_TEST") != "stats" or _current is None:
        return
    from mutmut.state import state

    path = _current

    class Recorded(set):
        def add(self, name):
            if name not in self:
                super().add(name)
                # one line per new name, so a child killed at teardown still leaves its hits
                with open(path, "a") as f:
                    f.write(name + "\\n")

    state()._stats = Recorded(state()._stats)


@pytest.hookimpl(tryfirst=True)
def pytest_runtest_teardown(item, nextitem):
    # before mutmut's own collector reads the set and clears it
    if os.environ.get("MUTANT_UNDER_TEST") != "stats" or _current is None or not _current.exists():
        return
    from mutmut.state import state

    state()._stats.update(_current.read_text().split())
    _current.unlink()
'''
CHILD_COVERAGE_CONFIG = """[tool.coverage.run]
source = ["inference_perf"]
concurrency = ["multiprocessing"]
# empty in the pytest process, which reads this before the plugin sets the variable; set in a child
context = "${MUTATION_CHILD_CONTEXT-}"
"""

# Where the plugin and its coverage config are written. main() puts it next to the trees.
HARNESS_DIR: Path | None = None

# The canary sets the active mutant to an id mutmut never generated, so every
# function runs unmutated. Tests that still fail are reacting to the harness. The
# id is random per run, so a test cannot recognise the canary and pass only for it.
CANARY_RANGE = 10**9

# A rerun gets its tests' measured time, scaled for a busier runner, plus room for pytest to start.
# Past that it is hung, and a hang must not run into the workflow's timeout.
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


class BudgetExhausted(Exception):
    """The run used up --budget-minutes. The workflow's own timeout would otherwise kill it without a summary."""


class Budget:
    def __init__(self, minutes: float | None) -> None:
        self.minutes = minutes
        self.deadline = None if minutes is None else time.monotonic() + minutes * 60

    def limit(self, timeout: float | None) -> float | None:
        """The tighter of `timeout` and what is left. Raises BudgetExhausted when nothing is."""
        if self.deadline is None:
            return timeout
        left = self.deadline - time.monotonic()
        if left <= 0:
            raise BudgetExhausted()
        return left if timeout is None else min(timeout, left)

    def spent(self) -> bool:
        return self.deadline is not None and time.monotonic() >= self.deadline


BUDGET = Budget(None)


@dataclass
class Progress:
    """Where a run is, so running out of budget can say what was left."""

    stage: str = "starting"
    # functions whose verdict the current stage was still working out
    pending: list[str] = field(default_factory=list)


PROGRESS = Progress()


def run_bounded(
    command: list[str], cwd: Path, env: dict[str, str] | None = None, timeout: float | None = None
) -> subprocess.CompletedProcess[str]:
    """Run a command within `timeout` and the budget, killing its whole process group when either runs out.

    mutmut's workers and the Worker processes tests start are grandchildren, which a plain timeout leaves running.
    Raises BudgetExhausted when the budget ran out, and TimeoutExpired when `timeout` did.
    """
    limit = BUDGET.limit(timeout)
    proc = subprocess.Popen(
        command, cwd=cwd, env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, start_new_session=True
    )
    try:
        out, err = proc.communicate(timeout=limit)
    except subprocess.TimeoutExpired:
        _kill_group(proc.pid)
        out, err = proc.communicate()
        if BUDGET.spent():
            raise BudgetExhausted() from None
        raise subprocess.TimeoutExpired(command, limit or 0, output=out, stderr=err) from None
    except BaseException:
        # the new session keeps a Ctrl-C from reaching the group, so pass it on
        _kill_group(proc.pid)
        raise
    # a test's Worker processes can outlive the pytest run that started them
    _kill_group(proc.pid)
    return subprocess.CompletedProcess(command, proc.returncode, out, err)


def _kill_group(pid: int) -> None:
    try:
        os.killpg(pid, signal.SIGKILL)
    except ProcessLookupError:
        pass


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def harness_dir() -> Path:
    """The directory holding the child plugin and its coverage config, written on first use."""
    global HARNESS_DIR
    if HARNESS_DIR is None:
        HARNESS_DIR = Path(tempfile.mkdtemp(prefix="mutation-harness-"))
        atexit.register(shutil.rmtree, HARNESS_DIR, ignore_errors=True)
    HARNESS_DIR.mkdir(parents=True, exist_ok=True)
    plugin = HARNESS_DIR / f"{CHILD_PLUGIN}.py"
    if not plugin.exists():
        plugin.write_text(CHILD_PLUGIN_SOURCE)
        (HARNESS_DIR / "coverage.toml").write_text(CHILD_COVERAGE_CONFIG)
    return HARNESS_DIR


def harness_env(tree: Path, env: dict[str, str] | None = None) -> dict[str, str]:
    """`env` (default os.environ) with the child plugin importable and a hits directory for `tree`."""
    env = dict(os.environ if env is None else env)
    hits = harness_dir() / "hits" / tree.name
    hits.mkdir(parents=True, exist_ok=True)
    env["MUTATION_HITS_DIR"] = str(hits)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (env.get("PYTHONPATH"), str(harness_dir())) if p)
    return env


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


# mutmut 3.8 never mutates functions with these names (file_mutation.NEVER_MUTATE_FUNCTION_NAMES)
NEVER_MUTATED_FUNCTIONS = {"__getattribute__", "__setattr__", "__new__"}
# nor a call to these, arguments included (file_mutation.NEVER_MUTATE_FUNCTION_CALLS)
NEVER_MUTATED_CALLS = {"len", "isinstance"}
# code passed as a string runs without ever being mutated
DYNAMIC_CODE = {"exec", "eval", "compile", "__import__"}


def _called_name(func: ast.expr) -> str:
    return func.id if isinstance(func, ast.Name) else func.attr if isinstance(func, ast.Attribute) else ""


def _runs_code_from_text(call: ast.Call) -> bool:
    """A call to the exec, eval, compile or __import__ builtin, bare or as builtins.<name>."""
    func = call.func
    if isinstance(func, ast.Attribute) and not (isinstance(func.value, ast.Name) and func.value.id == "builtins"):
        return False
    return _called_name(func) in DYNAMIC_CODE


def _plain(expr: ast.AST | None, data: bool = False) -> bool:
    """Names, attributes, constants, containers and calls built only from those: nothing a mutation could flip.

    With `data`, also what a declaration holds: subscripts, f-strings, arithmetic on constants, and
    lambdas that only build such data, like a pydantic default_factory.
    """
    if expr is None or isinstance(expr, (ast.Name, ast.Constant)):
        return True
    if isinstance(expr, (ast.Attribute, ast.Starred, ast.Await)):
        return _plain(expr.value, data)
    if isinstance(expr, ast.Call):
        args = [expr.func, *expr.args, *(k.value for k in expr.keywords)]
        return not _runs_code_from_text(expr) and all(_plain(a, data) for a in args)
    if isinstance(expr, (ast.Tuple, ast.List, ast.Set)):
        return all(_plain(e, data) for e in expr.elts)
    if isinstance(expr, ast.Dict):
        return all(_plain(k, data) for k in expr.keys) and all(_plain(v, data) for v in expr.values)
    if isinstance(expr, ast.BinOp) and isinstance(expr.op, ast.BitOr):
        # a type union like `int | None`
        return _plain(expr.left, data) and _plain(expr.right, data)
    if not data:
        return False
    if isinstance(expr, ast.Subscript):
        return _plain(expr.value, data) and _plain(expr.slice, data)
    if isinstance(expr, ast.Slice):
        return all(_plain(part, data) for part in (expr.lower, expr.upper, expr.step))
    if isinstance(expr, (ast.JoinedStr, ast.FormattedValue)):
        return all(_plain(part, data) for part in ast.iter_child_nodes(expr))
    if isinstance(expr, (ast.BinOp, ast.UnaryOp)):
        return all(isinstance(n, (ast.Constant, ast.operator, ast.unaryop, ast.BinOp, ast.UnaryOp)) for n in ast.walk(expr))
    if isinstance(expr, ast.Lambda):
        return not (expr.args.defaults or any(expr.args.kw_defaults)) and _plain(expr.body, data)
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


def _declares(stmt: ast.stmt) -> bool:
    """A module or class level statement that only declares: imports, docstrings, names bound to plain data."""
    if isinstance(stmt, (ast.Import, ast.ImportFrom, ast.Pass, ast.Delete, ast.Global, ast.Nonlocal)):
        return True
    if isinstance(stmt, getattr(ast, "TypeAlias", ())):
        return True
    if isinstance(stmt, ast.Expr):
        return _plain(stmt.value, data=True)
    if isinstance(stmt, ast.Assign):
        return all(_plain(t, data=True) for t in stmt.targets) and _plain(stmt.value, data=True)
    if isinstance(stmt, (ast.AnnAssign, ast.AugAssign)):
        return _plain(stmt.target, data=True) and _plain(stmt.value, data=True)
    return False


def _type_checking(stmt: ast.stmt) -> bool:
    return isinstance(stmt, ast.If) and _called_name(stmt.test) == "TYPE_CHECKING"


def _log_or_raise(stmt: ast.stmt) -> bool:
    """`logger.<method>(...)` or `raise X(...)`: the text do_not_mutate_patterns exempts, carrying no logic out."""
    if any(isinstance(n, (ast.NamedExpr, ast.Lambda, ast.Await, ast.Yield, ast.YieldFrom)) for n in ast.walk(stmt)):
        return False
    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call) and isinstance(stmt.value.func, ast.Attribute):
        return _called_name(stmt.value.func.value).endswith("logger")
    return isinstance(stmt, ast.Raise) and isinstance(stmt.exc, ast.Call) and isinstance(stmt.exc.func, ast.Name)


def _blocks(stmt: ast.stmt) -> list[list[ast.stmt]]:
    """The statement lists nested in a compound statement: bodies, else branches, handlers, match cases."""
    blocks: list[list[ast.stmt]] = []
    for _, value in ast.iter_fields(stmt):
        if isinstance(value, list):
            blocks += [value] if value and isinstance(value[0], ast.stmt) else []
            blocks += [item.body for item in value if isinstance(item, (ast.ExceptHandler, ast.match_case))]
    return blocks


def _own_nodes(stmt: ast.stmt) -> list[ast.AST]:
    """A statement's expressions, without the statements nested in it."""
    found: list[ast.AST] = []
    pending: list[ast.AST] = [stmt]
    while pending:
        for child in ast.iter_child_nodes(pending.pop()):
            if not isinstance(child, ast.stmt):
                found.append(child)
                pending.append(child)
    return found


@dataclass(frozen=True)
class Unchecked:
    line: int
    why: str


class _UncheckedScan:
    """Walks a module the way mutmut 3.8 decides what to mutate (file_mutation.py) and records what it never will."""

    def __init__(self, source: str, patterns: list[str], excluded: bool) -> None:
        self.pattern_lines = {n for n, text in enumerate(source.splitlines(), 1) if any(re.search(p, text) for p in patterns)}
        self.excluded = excluded
        self.found: list[Unchecked] = []

    def declarations(self, body: list[ast.stmt], where: str) -> None:
        """A module, class or module-level block body. `where` is "module", "class" for a module-level class, or "other"."""
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.header(stmt)
                why = self.skipped_because(stmt, where)
                if why:
                    self.skipped_function(stmt, why)
                else:
                    self.mutated(stmt.body)
            elif isinstance(stmt, ast.ClassDef):
                self.header(stmt)
                self.declarations(stmt.body, "class" if where == "module" else "other")
            elif _type_checking(stmt):
                for block in _blocks(stmt):
                    self.declarations(block, "other")
            elif not _declares(stmt):
                self.found.append(Unchecked(stmt.lineno, "module or class level code"))
                for block in _blocks(stmt):
                    self.declarations(block, "other")

    def skipped_because(self, node: FunctionNode, where: str) -> str:
        if self.excluded:
            return "in a file do_not_mutate excludes"
        if where == "other":
            return "function outside module level and module-level classes"
        if mutmut_skips(node):
            return "decorated function"
        if node.name in NEVER_MUTATED_FUNCTIONS:
            return f"{node.name} is never mutated"
        return ""

    def header(self, node: FunctionNode | ast.ClassDef) -> None:
        """Decorators, base classes and default arguments run at definition time, and mutmut leaves them alone."""
        parts: list[ast.expr] = list(node.decorator_list)
        if isinstance(node, ast.ClassDef):
            parts += node.bases + [k.value for k in node.keywords]
        else:
            parts += node.args.defaults + [d for d in node.args.kw_defaults if d is not None]
        if not all(_plain(part, data=True) for part in parts):
            self.found.append(Unchecked(node.lineno, "decorator, base class or default argument"))

    def skipped_function(self, node: FunctionNode, why: str) -> None:
        body = node.body
        if body and isinstance(body[0], ast.Expr) and isinstance(body[0].value, ast.Constant):
            body = body[1:]  # docstring
        if len(body) <= 1 and all(_delegates(stmt) for stmt in body):
            return
        self.found += [Unchecked(sub.lineno, why) for stmt in body for sub in ast.walk(stmt) if isinstance(sub, ast.stmt)]

    def mutated(self, body: list[ast.stmt]) -> None:
        """A body mutmut mutates, minus what it skips inside one."""
        for stmt in body:
            if isinstance(stmt, (ast.FunctionDef, ast.AsyncFunctionDef)):
                self.header(stmt)
                if mutmut_skips(stmt):
                    self.skipped_function(stmt, "decorated function")
                elif stmt.name in NEVER_MUTATED_FUNCTIONS:
                    self.skipped_function(stmt, f"{stmt.name} is never mutated")
                else:
                    self.mutated(stmt.body)
                continue
            if isinstance(stmt, ast.ClassDef):
                self.header(stmt)
                self.mutated(stmt.body)
                continue
            why = self.skipped_inside(stmt)
            if why:
                self.found.append(Unchecked(stmt.lineno, why))
            for block in _blocks(stmt):
                self.mutated(block)

    def skipped_inside(self, stmt: ast.stmt) -> str:
        own = _own_nodes(stmt)
        calls = [n for n in own if isinstance(n, ast.Call)]
        if any(_runs_code_from_text(c) for c in calls):
            return "exec, eval, compile or __import__"
        hidden = [c for c in calls if isinstance(c.func, ast.Name) and c.func.id in NEVER_MUTATED_CALLS]
        if any(not all(_plain(a, data=True) for a in c.args) for c in hidden):
            return "len() or isinstance() argument"
        lambdas = [n for n in own if isinstance(n, ast.Lambda)]
        if any(not all(_plain(d, data=True) for d in [*n.args.defaults, *n.args.kw_defaults]) for n in lambdas):
            return "decorator, base class or default argument"
        blocks = _blocks(stmt)
        header_end = max(stmt.lineno, blocks[0][0].lineno - 1) if blocks else stmt.end_lineno or stmt.lineno
        on_pattern = any(n in self.pattern_lines for n in range(stmt.lineno, header_end + 1))
        if on_pattern and not (_log_or_raise(stmt) or _delegates(stmt)):
            return "line matching do_not_mutate_patterns"
        return ""


def unchecked_statements(source: str, patterns: list[str], excluded: bool = False) -> list[Unchecked]:
    """Statements mutmut never mutates that do more than declare or delegate, one entry per statement.

    mutmut mutates the bodies of module-level functions and of methods of module-level classes. Inside
    those it skips decorated functions, a few names, len() and isinstance() calls, defaults and
    decorators, and expressions starting on a line that matches do_not_mutate_patterns. A decorated
    function whose body is one plain delegation counts 0. `excluded` marks a file do_not_mutate skips.
    """
    scan = _UncheckedScan(source, patterns, excluded)
    scan.declarations(ast.parse(source).body, "module")
    return scan.found


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


def changed_source_pairs(base: str) -> list[tuple[str | None, str | None]]:
    """(base path, head path) for each changed Python file under the source tree, following renames.

    None marks a side without the file. The head is the working tree, so uncommitted and untracked files count.
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
    return [(b, h) for b, h in pairs if (b or "").endswith(".py") or (h or "").endswith(".py")]


def changed_source_files(base: str, do_not_mutate: list[str]) -> list[tuple[str | None, str | None]]:
    """changed_source_pairs, minus the files do_not_mutate excludes."""
    pairs = changed_source_pairs(base)
    kept = [(b if mutable(b, do_not_mutate) else None, h if mutable(h, do_not_mutate) else None) for b, h in pairs]
    return [(b, h) for b, h in kept if b or h]


def touched_functions(base: str, do_not_mutate: list[str]) -> Touched:
    touched = Touched()
    for base_path, head_path in changed_source_files(base, do_not_mutate):
        base_src = git("show", f"{base}:{base_path}") if base_path else ""
        head_src = Path(head_path).read_text() if head_path else ""
        base_spans, head_spans = function_spans(base_src), function_spans(head_src)
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


def imports_of(tree: Path, rel: str, test_dirs: set[Path]) -> set[str]:
    """Files under tests/ that `rel` imports or loads as a pytest plugin.

    An absolute name resolves where pytest's default import mode puts it: the repo root, the file's own
    directory, or the first directory above its package. A name none of those has resolves from every
    directory under tests/, since a test can put any of them on sys.path itself.
    """
    path = tree / rel
    try:
        module = ast.parse(path.read_text())
    except (SyntaxError, UnicodeDecodeError):
        return set()
    basedir = path.parent
    while (basedir / "__init__.py").is_file() and basedir != tree:
        basedir = basedir.parent
    roots = {tree, path.parent, basedir}
    found: set[str] = set()
    for node in ast.walk(module):
        wanted: list[tuple[set[Path], str]]
        if isinstance(node, ast.Assign) and any(isinstance(t, ast.Name) and t.id == "pytest_plugins" for t in node.targets):
            plugins = [node.value] if isinstance(node.value, ast.Constant) else getattr(node.value, "elts", [])
            wanted = [(roots, p.value) for p in plugins if isinstance(p, ast.Constant) and isinstance(p.value, str)]
        elif isinstance(node, ast.Import):
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
            hits = {_dotted_file(tree, root, dotted) for root in search} - {None}
            if not hits and search is roots:
                hits = {_dotted_file(tree, root, dotted) for root in test_dirs} - {None}
            found.update(h for h in hits if h and h.startswith(TESTS_PREFIX))
    return found


def tests_using(tree: Path, support: list[str]) -> set[str]:
    """Unit test files that run any of `support`.

    A test runs a helper it imports, directly or through other helpers, and every conftest.py and
    package __init__.py in the directories above it.
    """
    sources = sorted((tree / TESTS_PREFIX).rglob("*.py"))
    test_dirs = {d for path in sources for d in path.parents if d.is_relative_to(tree / TESTS_PREFIX)}
    importers: dict[str, set[str]] = {}
    for path in sources:
        rel = str(path.relative_to(tree))
        for imported in imports_of(tree, rel, test_dirs):
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
    env = harness_env(tree, env)
    cov = ["--cov=inference_perf", "--cov-context=test", "--cov-report=", f"--cov-config={harness_dir() / 'coverage.toml'}"]
    plugins = ["-p", "no:cacheprovider", "-p", CHILD_PLUGIN]
    run_bounded([sys.executable, "-m", "pytest", "-q", *plugins, "-o", "log_cli=false", *cov, *tests], tree, env)
    run_bounded([sys.executable, "-m", "coverage", "json", "--show-contexts", "-o", str(report)], tree, env)
    if not report.exists():
        raise MutationRunError(f"coverage over the changed tests produced no report in {tree}")
    found: set[str] = set()
    for path, info in json.loads(report.read_text())["files"].items():
        rel = os.path.relpath(path, tree) if os.path.isabs(path) else path
        if not rel.startswith(SOURCE_PREFIX):
            continue
        # an empty context is import or collection time; a child's lines carry the plugin's static context
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
    pytest_args = [str(a) for a in config.get("pytest_add_cli_args", [])] + ["-p", CHILD_PLUGIN]
    merged = {**config, "pytest_add_cli_args": pytest_args, "only_mutate": files, "pytest_add_cli_args_test_selection": tests}
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
    return run_bounded([sys.executable, "-m", "mutmut", *args], tree, harness_env(tree))


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


def rerun(side: Side, tests: list[str], mutant: str, pytest_args: list[str]) -> tuple[int | None, str]:
    """Run `tests` with `mutant` active in the side's mutation tree.

    Returns pytest's exit code, or None if it hung, and its output.
    """
    env = {**os.environ, "MUTANT_UNDER_TEST": mutant}
    try:
        run = run_bounded(pytest_command(tests, pytest_args), side.tree / "mutants", env, rerun_timeout(side.tree, tests))
        return run.returncode, run.stdout + run.stderr
    except subprocess.TimeoutExpired:
        return None, ""


def failed_tests(output: str) -> list[str]:
    """Test ids pytest's short summary reports as failed or errored."""
    lines = output.splitlines()
    return sorted({line.split(" ")[1] for line in lines if line.startswith(("FAILED ", "ERROR ")) and " " in line})


def canary_id(name: str, taken: set[str]) -> str:
    """A mutant id for `name` in mutmut's format that is not in `taken`, picked at random."""
    while True:
        candidate = f"{name}__mutmut_{secrets.randbelow(CANARY_RANGE) + 1}"
        if candidate not in taken:
            return candidate


def canary_passes(side: Side, names: list[str], pytest_args: list[str]) -> bool:
    """Run the tests covering `names` with an active mutant id that matches no mutant. One pass clears a flaky test."""
    tests = tests_covering(side.tree, names)
    if not tests:
        return True
    canary = canary_id(names[0], set(side.results))
    for _ in range(RERUNS):
        # a hang counts as a failed run, so tests that only hang under the harness still fail the canary
        if rerun(side, tests, canary, pytest_args)[0] in (0, 5):  # 5: nothing collected
            return True
    return False


def diagnose_setup_failure(side: Side, pytest_args: list[str]) -> SetupFailure:
    """Run the side's suite unmutated to the end and name what fails. mutmut's setup run swallows pytest's output."""
    command = [sys.executable, "-m", "pytest", "--rootdir=.", "-rfE", "--tb=short", "--disable-warnings", "-p", "no:randomly"]
    command += ["-p", "no:random-order", f"--timeout={DIAGNOSIS_TEST_TIMEOUT_S}", *side.tests, *pytest_args]
    env = {**os.environ, "MUTANT_UNDER_TEST": canary_id("setup", set())}
    try:
        run = run_bounded(command, side.tree / "mutants", env, DIAGNOSIS_TIMEOUT_S)
        output = run.stdout + run.stderr
    except subprocess.TimeoutExpired as e:
        partial = e.stdout if isinstance(e.stdout, str) else ""
        output = partial + f"\nThe suite was still running after {DIAGNOSIS_TIMEOUT_S} s and was stopped."
    return SetupFailure(side.tree, failed_tests(output), "\n".join(output.splitlines()[-DIAGNOSIS_TAIL_LINES:]))


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


class Survival(NamedTuple):
    counts: bool
    # tests that killed it in some rerun
    killed_by: list[str]


def still_survives(side: Side, mutant: str, pytest_args: list[str]) -> Survival:
    """Rerun the tests that cover a surviving mutant. It stops counting only when every rerun kills it."""
    if side.results.get(mutant) == "no tests":
        return Survival(True, [])
    tests = [t for t in mutmut(side.tree, "tests-for-mutant", mutant).stdout.splitlines() if "::" in t]
    killed_by: set[str] = set()
    for _ in range(RERUNS):
        code, output = rerun(side, tests, mutant, pytest_args)
        # a hang is not a kill
        if code is None or code == 0:
            return Survival(True, sorted(killed_by))
        killed_by.update(failed_tests(output) or ["(no failing test named)"])
    return Survival(False, sorted(killed_by))


def by_function(mutants: list[str]) -> Counter[str]:
    return Counter(m.partition("__mutmut_")[0] for m in mutants)


@dataclass
class Verdict:
    base_survivors: int
    head_survivors: int
    # survivor -> tests that killed it in some reruns but not all. It counts
    flaky: dict[str, list[str]]
    unexecuted: list[str]
    failures: list[str]
    # head survivors whose mutation also survived on the base, counted on both sides and never rerun
    carried: list[str] = field(default_factory=list)
    # survivors every rerun killed, so the first run missed. They do not count
    flukes: list[str] = field(default_factory=list)


def mutation_lines(side: Side, mutant: str) -> tuple[str, ...]:
    """The lines a mutant changes, without diff headers or context, so the same mutation matches across versions."""
    shown = mutmut(side.tree, "show", mutant).stdout.splitlines()
    return tuple(sorted(line.rstrip() for line in shown if line[:1] in "+-" and not line.startswith(("+++", "---"))))


def carried_over(touched: Touched, base: Side | None, head: Side, candidates: list[str]) -> list[str]:
    """Candidates whose function survived the identical mutation on the base, matched one for one."""
    if base is None:
        return []
    base_name = {h: b for b, h in touched.existing}
    pools: dict[str, Counter[tuple[str, ...]]] = {}
    carried: list[str] = []
    for mutant in candidates:
        function = mutant.partition("__mutmut_")[0]
        if function not in base_name:
            continue
        if function not in pools:
            pools[function] = Counter(mutation_lines(base, m) for m in survivors(base, [base_name[function]]))
        lines = mutation_lines(head, mutant)
        if pools[function][lines]:
            pools[function][lines] -= 1
            carried.append(mutant)
    return carried


def judge(touched: Touched, base: Side | None, head: Side, pytest_args: list[str]) -> Verdict:
    base_surv = survivors(base, touched.base_names)
    head_surv = survivors(head, touched.head_names)
    flaky: dict[str, list[str]] = {}
    flukes: list[str] = []
    carried: list[str] = []
    if len(head_surv) > len(base_surv):
        # only a function whose own count rose can hold a survivor the change added
        before, after = by_function(base_surv), by_function(head_surv)
        rose = {h for b, h in touched.existing if after[h] > before[b]} | {n for n in touched.new if after[n]}
        candidates = [m for m in head_surv if m.partition("__mutmut_")[0] in rose]
        PROGRESS.stage, PROGRESS.pending = "rerunning new survivors", sorted(rose)
        carried = carried_over(touched, base, head, candidates)
        outcomes = {m: still_survives(head, m, pytest_args) for m in candidates if m not in carried}
        flukes = [m for m, survival in outcomes.items() if not survival.counts]
        flaky = {m: survival.killed_by for m, survival in outcomes.items() if survival.counts and survival.killed_by}
        head_surv = [m for m in head_surv if m not in flukes]
    # unlike a surviving mutant, code no test runs cannot be an equivalent mutant, so nothing can pay for it
    unexecuted = sorted({m.partition("__mutmut_")[0] for m, s in head.results.items() if s == "no tests"} & set(touched.new))
    verdict = Verdict(len(base_surv), len(head_surv), flaky, unexecuted, [], carried, flukes)
    if verdict.head_survivors > verdict.base_survivors:
        verdict.failures.append(f"Surviving mutants rose from {verdict.base_survivors} to {verdict.head_survivors}")
    if unexecuted:
        verdict.failures.append(f"No test runs {', '.join(f'`{n}`' for n in unexecuted)}")
    # a decorator, or anything else that leaves mutmut nothing to mutate, must not hide a checked function
    base_mutants, head_mutants = by_function(list(base.results) if base else []), by_function(list(head.results))
    for b, h in touched.existing:
        if base_mutants[b] and not head_mutants[h]:
            verdict.failures.append(f"`{h}` had {base_mutants[b]} mutants on the base and none with this change")
    if head.canary_failed:
        verdict.failures.append("Tests fail with an active mutant id that changes no code, so they react to the harness")
    return verdict


@dataclass
class Checks:
    """Findings that need no mutants. They fail a change even when nothing is in scope to mutate."""

    failures: list[str] = field(default_factory=list)
    # one markdown block per failure, for the summary
    sections: list[str] = field(default_factory=list)
    # markdown blocks shown whether or not anything failed
    notes: list[str] = field(default_factory=list)
    # changed file -> unchecked statements on the base and with the change
    unchecked: dict[str, tuple[int, int]] = field(default_factory=dict)

    def fail(self, failure: str, title: str, lines: list[str]) -> None:
        self.failures.append(failure)
        self.sections.append("\n".join([f"{title}:", "", *(f"- {line}" for line in lines)]))


def head_files() -> list[str]:
    """Every file in the working tree that git tracks or would track."""
    listed = git("ls-files", "-z", "--cached", "--others", "--exclude-standard").split("\0")
    return [p for p in listed if p and Path(p).is_file()]


def base_files(base: str) -> list[str]:
    return [p for p in git("ls-tree", "-r", "-z", "--name-only", base).split("\0") if p]


def first_party_roots(paths: list[str]) -> set[str]:
    """Top-level import names that reach repo code outside the package: root modules and root directories holding Python."""
    roots = set()
    for path in paths:
        if path.endswith(".py"):
            top, sep, _ = path.partition("/")
            roots.add(top if sep else top.removesuffix(".py"))
    return roots - {SOURCE_PREFIX.rstrip("/")}


def outside_imports(tree: Path, files: list[str], roots: set[str]) -> list[str]:
    """file:line of each import in `files` of a name in `roots`, including a module loaded by its name as a string.

    mutmut only mutates inference_perf/, so logic moved outside it and imported back is never checked.
    """
    hits: list[tuple[str, int, str]] = []
    for rel in files:
        try:
            module = ast.parse((tree / rel).read_text())
        except (SyntaxError, UnicodeDecodeError):
            continue
        for node in ast.walk(module):
            if not isinstance(node, (ast.Import, ast.ImportFrom, ast.Call)):
                continue
            names: list[str] = []
            if isinstance(node, ast.Import):
                names = [a.name for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            elif isinstance(node, ast.Call) and _called_name(node.func) in ("import_module", "__import__") and node.args:
                first = node.args[0]
                if isinstance(first, ast.Constant) and isinstance(first.value, str):
                    names = [first.value]
            hits += [(rel, node.lineno, top) for top in (n.partition(".")[0] for n in names) if top in roots]
    return [f"{rel}:{line} imports `{top}`" for rel, line, top in sorted(hits)]


def root_modules(paths: list[str]) -> set[str]:
    """Modules and regular packages at the repo root. Running from the root, they win over anything installed."""
    modules = {p.removesuffix(".py") for p in paths if "/" not in p and p.endswith(".py")}
    return modules | {p.split("/")[0] for p in paths if p.count("/") == 1 and p.endswith("/__init__.py")}


def shadowing_roots(base_paths: list[str], head_paths: list[str]) -> list[str]:
    """Root modules the change adds under a name the standard library or an installed distribution already uses."""
    taken = set(sys.stdlib_module_names) | set(importlib.metadata.packages_distributions())
    return sorted((root_modules(head_paths) - root_modules(base_paths)) & taken)


def unchecked_by_file(merge_base: str, config: dict[str, Any]) -> list[tuple[str, list[Unchecked], list[Unchecked]]]:
    """(path, base findings, head findings) of unchecked statements for each changed source file."""
    patterns = [str(p) for p in config.get("do_not_mutate_patterns", [])]
    do_not_mutate = [str(p) for p in config.get("do_not_mutate", [])]
    rows = []
    for base_path, head_path in changed_source_pairs(merge_base):
        before, after = [], []
        if base_path and base_path.endswith(".py"):
            source = git("show", f"{merge_base}:{base_path}")
            before = unchecked_statements(source, patterns, excluded=not mutable(base_path, do_not_mutate))
        if head_path and head_path.endswith(".py"):
            source = Path(head_path).read_text()
            after = unchecked_statements(source, patterns, excluded=not mutable(head_path, do_not_mutate))
        rows.append((head_path or base_path or "", before, after))
    return rows


def check_unchecked(checks: Checks, merge_base: str, config: dict[str, Any]) -> None:
    """Statements mutmut never mutates, summed over the changed files, cannot rise."""
    try:
        rows = unchecked_by_file(merge_base, config)
    except SyntaxError as e:
        checks.fail(f"{e.filename or 'A changed file'} does not parse", "Unparseable source", [str(e)])
        return
    checks.unchecked = {path: (len(b), len(a)) for path, b, a in rows if b or a}
    if checks.unchecked:
        table = ["| File | Statements mutmut never mutates, base | Head |", "|---|---|---|"]
        checks.notes.append("\n".join(table + [f"| `{p}` | {b} | {a} |" for p, (b, a) in checks.unchecked.items()]))
    before, after = sum(b for b, _ in checks.unchecked.values()), sum(a for _, a in checks.unchecked.values())
    if after > before:
        found = [f"{path}:{u.line} {u.why}" for path, b, a in rows if len(a) > len(b) for u in a]
        shown = found[:MAX_SHOWN_SURVIVORS] + (
            [f"{len(found) - MAX_SHOWN_SURVIVORS} more"] if len(found) > MAX_SHOWN_SURVIVORS else []
        )
        checks.fail(
            f"Statements mutmut never mutates rose from {before} to {after}",
            "Statements mutmut never mutates, in the files where they rose. "
            "Move the logic into an undecorated module-level function or method",
            shown,
        )


def harness_references(paths: list[str]) -> list[str]:
    """file:line of each line under the source or test tree that names the mutation harness."""
    hits = []
    for rel in paths:
        if not rel.startswith((SOURCE_PREFIX, TESTS_PREFIX)) or rel in HARNESS_WORDS_ALLOWED_IN:
            continue
        try:
            text = Path(rel).read_text()
        except (UnicodeDecodeError, OSError):
            continue
        hits += [f"{rel}:{n}" for n, line in enumerate(text.splitlines(), 1) if any(w in line.lower() for w in HARNESS_WORDS)]
    return sorted(hits)


class Distribution(Protocol):
    @property
    def version(self) -> str: ...

    def read_text(self, filename: str) -> str | None: ...


def installed_mutmut() -> Distribution | None:
    try:
        return importlib.metadata.distribution("mutmut")
    except importlib.metadata.PackageNotFoundError:
        return None


def mutmut_pin_problems(lock_text: str, installed: Distribution | None) -> list[str]:
    """Ways the locked or installed mutmut differs from the pinned release on the package index."""
    entries = [p for p in tomllib.loads(lock_text).get("package", []) if p.get("name") == "mutmut"]
    problems = [] if len(entries) == 1 else [f"pdm.lock has {len(entries)} mutmut entries, not 1"]
    for entry in entries:
        if entry.get("version") != MUTMUT_VERSION:
            problems.append(f"pdm.lock pins mutmut {entry.get('version')}, not {MUTMUT_VERSION}")
        if {f.get("hash") for f in entry.get("files", [])} != MUTMUT_HASHES:
            problems.append("pdm.lock's mutmut file hashes are not the pinned release's")
        problems += [f"pdm.lock installs mutmut from {key}" for key in NON_INDEX_SOURCES if key in entry]
    if installed is None:
        problems.append("mutmut is not installed")
    elif installed.version != MUTMUT_VERSION:
        problems.append(f"The installed mutmut is {installed.version}, not {MUTMUT_VERSION}")
    elif installed.read_text("direct_url.json") is not None:
        # PEP 610: pip and pdm write this file only for an install from a URL, a path or a VCS
        problems.append("The installed mutmut came from a direct URL, not the package index")
    return problems


def static_checks(merge_base: str, config: dict[str, Any]) -> Checks:
    checks = Checks()
    check_unchecked(checks, merge_base, config)
    head, base = head_files(), base_files(merge_base)
    lock = Path("pdm.lock")
    pin = mutmut_pin_problems(lock.read_text() if lock.is_file() else "", installed_mutmut())
    if pin:
        checks.fail("mutmut is not the pinned release", "mutmut pin", pin)
    references = harness_references(head)
    if references:
        checks.fail(
            "Code under inference_perf/ or tests/ names the mutation harness",
            "Lines that name the mutation harness",
            references,
        )
    sources = [p for p in head if p.startswith(SOURCE_PREFIX) and p.endswith(".py")]
    hits = outside_imports(Path("."), sources, first_party_roots(head))
    if hits:
        checks.fail(
            "Code under inference_perf/ imports repo code outside it, which is never mutated",
            "Imports from outside inference_perf/",
            hits,
        )
    shadowing = shadowing_roots(base, head)
    if shadowing:
        names = ", ".join(f"`{n}`" for n in shadowing)
        checks.fail(
            f"New root modules shadow an installed or standard library module: {names}",
            "Root modules that shadow an import",
            [f"`{n}`" for n in shadowing],
        )
    return checks


def publish(summary: str, record: dict[str, Any], output: str) -> None:
    """Print the summary, add it to the job summary, write the record, and exit 1 if the record holds failures."""
    print(summary)
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(summary)
    Path(output).write_text(json.dumps(record, indent=2) + "\n")
    if record["failures"]:
        sys.exit(1)


def out_of_budget(
    args: argparse.Namespace, merge_base: str, touched: Touched, base: Side | None, head: Side | None, checks: Checks
) -> None:
    """Fail with what the run knew when the budget ran out, so the author sees more than a cancelled job."""
    pending = ", ".join(f"`{n}`" for n in PROGRESS.pending)
    failure = f"Ran out of the {args.budget_minutes:g} minute budget while {PROGRESS.stage}."
    failure += f" Not reached: {pending}" if pending else ""
    failures = [failure, *checks.failures]
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::error title=Mutation ratchet::{failure}")
    scope = f"In scope: {len(touched.existing)} existing, {len(touched.new)} new and {len(touched.deleted)} deleted functions."
    lines = [f"## Mutation ratchet against `{args.base}`", "", scope, *(f"\n{s}" for s in checks.notes + checks.sections)]
    lines += ["", "FAIL: " + "; ".join(failures), "", DOC_LINE]
    record: dict[str, Any] = {
        "base": args.base,
        "merge_base": merge_base,
        "existing": touched.existing,
        "new": touched.new,
        "deleted": touched.deleted,
        "base_results": base.results if base else {},
        "head_results": head.results if head else {},
        "budget_minutes": args.budget_minutes,
        "stage": PROGRESS.stage,
        "not_reached": PROGRESS.pending,
        "failures": failures,
    }
    publish("\n".join(lines) + "\n", record, args.output)


def nothing_to_mutate(message: str, args: argparse.Namespace, merge_base: str, checks: Checks) -> None:
    """End a run with no function in scope. Passes unless a check that needs no mutants failed."""
    if not checks.failures:
        print(message)
        return
    lines = [f"## Mutation ratchet against `{args.base}`", "", message, *(f"\n{s}" for s in checks.notes + checks.sections)]
    lines += ["", "FAIL: " + "; ".join(checks.failures), "", DOC_LINE]
    publish("\n".join(lines) + "\n", {"base": args.base, "merge_base": merge_base, "failures": checks.failures}, args.output)


def report(base_ref: str, touched: Touched, base: Side | None, head: Side, verdict: Verdict, checks: Checks) -> str:
    before = by_function(survivors(base, touched.base_names))
    after = by_function([m for m in survivors(head, touched.head_names) if m not in verdict.flukes])
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
    if head.pragmas_stripped:
        lines += ["", f"Stripped {head.pragmas_stripped} `# pragma: no mutate` comments before mutating. They have no effect."]
    if verdict.carried:
        lines += ["", f"{len(verdict.carried)} surviving mutants on the head also survived on the base. Not rerun."]
    if verdict.flukes:
        lines += ["", f"Killed in every rerun, so not counted: {', '.join(f'`{m}`' for m in verdict.flukes)}"]
    if verdict.flaky:
        lines += ["", "Killed in some reruns but not all. They count, and these tests are flaky:"]
        lines += [f"- `{m}`: {', '.join(f'`{t}`' for t in tests)}" for m, tests in verdict.flaky.items()]
    shown = [m for m in survivors(head, touched.head_names) if m not in verdict.flukes]
    if shown:
        lines += ["", f"<details><summary>{len(shown)} surviving mutants on the head</summary>", ""]
        for mutant in shown[:MAX_SHOWN_SURVIVORS]:
            try:
                diff = mutmut(head.tree, "show", mutant).stdout.strip()
            except BudgetExhausted:
                lines.append("Out of time for the remaining diffs.")
                break
            # mutmut show already heads the diff with the mutant name and status
            lines += ["```diff", diff, "```", ""]
        if len(shown) > MAX_SHOWN_SURVIVORS:
            lines.append(f"{len(shown) - MAX_SHOWN_SURVIVORS} more not shown.")
        lines.append("</details>")
    lines += [f"\n{section}" for section in checks.notes + checks.sections]
    lines += ["", "FAIL: " + "; ".join(verdict.failures), "", DOC_LINE] if verdict.failures else ["", "PASS"]
    return "\n".join(lines) + "\n"


def report_error(message: str) -> None:
    """Put an error that stops the check where the PR author sees it: an annotation on the check and the job summary."""
    if os.environ.get("GITHUB_ACTIONS"):
        print(f"::error title=Mutation ratchet::{message.splitlines()[0]}")
    if os.environ.get("GITHUB_STEP_SUMMARY"):
        with open(os.environ["GITHUB_STEP_SUMMARY"], "a") as f:
            f.write(f"## Mutation ratchet: error\n\n```\n{message}\n```\n\n{DOC_LINE}\n")


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
    global HARNESS_DIR
    HARNESS_DIR = workdir.parent / (workdir.name + "-harness")
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
            shutil.rmtree(harness_dir(), ignore_errors=True)
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
    parser.add_argument(
        "--budget-minutes",
        type=float,
        help="Stop and fail with a partial report after this long, inside the CI job's own timeout",
    )
    args = parser.parse_args()
    global BUDGET
    BUDGET = Budget(args.budget_minutes)
    if args.all:
        whole_package(args)
        return

    merge_base = git("merge-base", args.base, "HEAD").strip()
    config = load_config(merge_base)
    do_not_mutate = [str(p) for p in config.get("do_not_mutate", [])]
    checks = static_checks(merge_base, config)
    touched = touched_functions(merge_base, do_not_mutate)
    changed_tests = changed_test_files(merge_base) + changed_test_support(merge_base)
    if not (touched.existing or touched.new or touched.deleted or changed_tests):
        message = f"No functions under {SOURCE_PREFIX} and no tests changed since {args.base}. Nothing to mutate."
        nothing_to_mutate(message, args, merge_base, checks)
        return

    pytest_args = [str(a) for a in config.get("pytest_add_cli_args", [])]
    workdir = Path(tempfile.mkdtemp(prefix="mutation-"))
    global HARNESS_DIR
    HARNESS_DIR = workdir / "harness"
    base: Side | None = None
    head: Side | None = None
    try:
        export_base(merge_base, workdir / "base")
        (workdir / "head").mkdir()
        export_working_tree(workdir / "head")
        stripped = {side: strip_no_mutate_pragmas(workdir / side) for side in ("base", "head")}
        PROGRESS.stage = "finding the functions the changed tests run"
        tests = tests_in_scope(merge_base, workdir / "base", workdir / "head")
        if tests:
            add_functions_run_by_changed_tests(touched, workdir / "base", workdir / "head", tests)
        if not (touched.base_names or touched.head_names):
            nothing_to_mutate(f"No function in scope since {args.base}. Nothing to mutate.", args, merge_base, checks)
            return
        print(
            f"Mutating {len(touched.existing)} existing ({len(touched.via_tests)} via changed tests), "
            f"{len(touched.new)} new and {len(touched.deleted)} deleted functions against {args.base} ({merge_base[:8]})."
        )
        if touched.base_names:
            PROGRESS.stage, PROGRESS.pending = "mutating the base", sorted(set(touched.base_names) | set(touched.head_names))
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
            PROGRESS.stage, PROGRESS.pending = "mutating the head", sorted(touched.head_names)
            head = run_side(workdir / "head", touched.head_names, config, args.max_children, stripped["head"], canary=True)
        verdict = judge(touched, base, head, pytest_args)
        verdict.failures += checks.failures
        PROGRESS.stage, PROGRESS.pending = "writing the report", []
        summary = report(args.base, touched, base, head, verdict, checks)
    except BudgetExhausted:
        out_of_budget(args, merge_base, touched, base, head, checks)
    except MutationRunError as e:
        print(f"ERROR: {e}\n{DOC_LINE}")
        report_error(str(e))
        sys.exit(1)
    finally:
        if args.keep:
            print(f"Mutation trees kept in {workdir}")
        else:
            shutil.rmtree(workdir, ignore_errors=True)

    # every path that leaves head unset has already exited
    assert head is not None
    record: dict[str, Any] = {
        "base": args.base,
        "merge_base": merge_base,
        "existing": touched.existing,
        "new": touched.new,
        "deleted": touched.deleted,
        "via_tests": sorted(touched.via_tests),
        "base_results": base.results if base else {},
        "head_results": head.results,
        "flaky": verdict.flaky,
        "flukes": verdict.flukes,
        "carried": verdict.carried,
        "unchecked": checks.unchecked,
        "pragmas_stripped": head.pragmas_stripped,
        "canary_failed": head.canary_failed,
        "failures": verdict.failures,
    }
    publish(summary, record, args.output)


if __name__ == "__main__":
    main()
