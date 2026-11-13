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
"""Say what a diff does, without running any of it.

Each changed Python file is parsed with `ast` at base and at head and split into
units: tests, functions, methods, class fields and whatever is left at the top
of a module or class. Comparing the two sides gives the units that were added,
altered, removed or renamed, and from those the labels a reviewer reads before
opening the diff. `classify` takes file contents, so the same code runs on a
local checkout and on files fetched through the GitHub API.
"""

import argparse
import ast
import json
import re
import subprocess
import sys
import tomllib
from collections.abc import Iterable, Mapping
from dataclasses import asdict, dataclass, field

# Mirror python_files, python_classes and python_functions in pyproject.toml.
TEST_FILE_PREFIX = "test_"
TEST_CLASS_PREFIX = "Test"
TEST_FUNCTION_PREFIX = "test_"

TEST_ROOTS = ("tests/", "e2e/")
SOURCE_ROOT = "inference_perf/"
E2E_ROOT = "e2e/"
ORACLE_ROOTS = tuple(f"{SOURCE_ROOT}{name}/" for name in ("datagen", "loadgen", "client", "reportgen"))
CONFIG_ROOT = f"{SOURCE_ROOT}config/"
REPORT_ROOT = f"{SOURCE_ROOT}reportgen/"
METRICS_ROOTS = (
    f"{SOURCE_ROOT}metrics/",
    f"{SOURCE_ROOT}observability/metrics/",
    f"{SOURCE_ROOT}client/server_metrics/",
    f"{SOURCE_ROOT}config/metrics/",
)

# Files that decide whether CI passes. See "Gate files" in #868.
GATE_DIRS = (".github/", "scripts/", "tests/required/scripts/")
GATE_PATHS = frozenset({"OWNERS_ALIASES", "AGENTS.md", "CLAUDE.md", "flake.nix", "e2e/known_bad.py"})
GATE_BASENAMES = frozenset({"conftest.py", "OWNERS"})
GATE_TEST_PREFIX = "tests/test_check_"
PYPROJECT = "pyproject.toml"
LOCKFILE = "pdm.lock"
PYPROJECT_GATE_TABLES = ("tool.pdm.scripts", "tool.ruff", "tool.mypy", "tool.pytest", "tool.coverage", "tool.mutmut")
GATE_PACKAGES = frozenset({"pytest", "pytest-cov", "coverage", "mutmut"})
KNOWN_BAD_LIST = "e2e/vllm_known_bad.txt"
RELEASE_LISTS = frozenset({"e2e/vllm_releases.txt", "e2e/vllm_multimodal_releases.txt"})

# Every label this script can apply. One of these on a PR that `labels` no longer returns is stale.
LABELS = (
    "purely-additive",
    "changes-pinned-behavior",
    "tests-only-change",
    "no-oracle-test",
    "touches/config-schema",
    "touches/report-schema",
    "touches/metrics",
    "touches/ci-gates",
    "ai-assisted",
)

AI_TRAILER = re.compile(
    r"^(co-authored-by|assisted-by|generated-by):.*\b(claude|copilot|codex|gemini|cursor|devin|anthropic|openai)\b"
    r"|generated with \[?claude code",
    re.IGNORECASE | re.MULTILINE,
)


@dataclass(frozen=True)
class FileChange:
    """One changed file. `base` is None for an added file, `head` is None for a removed one.

    Both sides hold the file's bytes when `needs_content(path)`. For any other
    file they are only compared for equality, so a loader may pass blob ids.
    """

    path: str
    base: bytes | None
    head: bytes | None
    # lines the diff removes from the file, as `git diff --numstat` counts them
    deletions: int = 0


@dataclass(frozen=True)
class Commit:
    author: str
    message: str


@dataclass
class Module:
    # unit key -> fingerprint. A key is `path::Class::name`, the way pytest spells a node id.
    units: dict[str, str] = field(default_factory=dict)
    tests: set[str] = field(default_factory=set)
    fields: set[str] = field(default_factory=set)
    # units that are not a def: assignments and loose statements at the top of a module or class
    statements: set[str] = field(default_factory=set)


@dataclass
class UnitDiff:
    added: list[str] = field(default_factory=list)
    altered: list[str] = field(default_factory=list)
    removed: list[str] = field(default_factory=list)
    renamed: list[tuple[str, str]] = field(default_factory=list)


@dataclass
class Classification:
    paths: list[str] = field(default_factory=list)
    tests: UnitDiff = field(default_factory=UnitDiff)
    code: UnitDiff = field(default_factory=UnitDiff)
    # Existing helpers and data files under tests/ and e2e/ that changed or went away.
    # Tests that read them can change outcome without their own bodies changing.
    helpers_changed: list[str] = field(default_factory=list)
    data_changed: list[str] = field(default_factory=list)
    # Python files that don't parse on one side. Every unit in them counts as altered.
    unparsed: list[str] = field(default_factory=list)
    # Files other than Python that lost a line.
    lines_deleted: list[str] = field(default_factory=list)
    config_fields: UnitDiff = field(default_factory=UnitDiff)
    report_fields: UnitDiff = field(default_factory=UnitDiff)
    metrics_paths: list[str] = field(default_factory=list)
    gate_reasons: list[str] = field(default_factory=list)
    gate_paths: list[str] = field(default_factory=list)
    ai_commits: list[str] = field(default_factory=list)

    @property
    def source_changed(self) -> bool:
        return any(p.startswith(SOURCE_ROOT) for p in self.paths)

    @property
    def purely_additive(self) -> bool:
        existing = [self.tests.altered, self.tests.removed, self.code.altered, self.code.removed]
        return not (any(existing) or self.unparsed or self.lines_deleted)

    @property
    def changes_pinned_behavior(self) -> bool:
        unparsed_tests = [p for p in self.unparsed if p.startswith(TEST_ROOTS)]
        return bool(self.tests.altered or self.tests.removed or self.helpers_changed or self.data_changed or unparsed_tests)

    @property
    def tests_only_change(self) -> bool:
        return self.changes_pinned_behavior and not self.source_changed

    @property
    def no_oracle_test(self) -> bool:
        oracle_code = any(p.startswith(ORACLE_ROOTS) for p in self.paths)
        e2e_tests = [t for t in (*self.tests.added, *self.tests.altered) if t.startswith(E2E_ROOT)]
        return oracle_code and not e2e_tests

    @property
    def labels(self) -> list[str]:
        flags = (
            self.purely_additive,
            self.changes_pinned_behavior,
            self.tests_only_change,
            self.no_oracle_test,
            any(asdict(self.config_fields).values()),
            any(asdict(self.report_fields).values()),
            bool(self.metrics_paths),
            bool(self.gate_reasons),
            bool(self.ai_commits),
        )
        return [label for label, on in zip(LABELS, flags, strict=True) if on]


class _DropDocstrings(ast.NodeTransformer):
    def _strip(self, node: ast.Module | ast.ClassDef | ast.FunctionDef | ast.AsyncFunctionDef) -> ast.AST:
        self.generic_visit(node)
        first = node.body[0] if node.body else None
        if isinstance(first, ast.Expr) and isinstance(first.value, ast.Constant) and isinstance(first.value.value, str):
            node.body = node.body[1:] or [ast.Pass()]
        return node

    visit_Module = visit_ClassDef = visit_FunctionDef = visit_AsyncFunctionDef = _strip


def fingerprint(node: ast.FunctionDef | ast.AsyncFunctionDef) -> str:
    # the name is blanked so a renamed test compares equal to its original.
    # comments never reach the AST and line numbers are left out of the dump.
    name, node.name = node.name, ""
    try:
        return ast.dump(node)
    finally:
        node.name = name


def is_test_file(path: str) -> bool:
    return path.startswith(TEST_ROOTS) and path.rsplit("/", 1)[-1].startswith(TEST_FILE_PREFIX)


def parse_module(path: str, source: bytes) -> Module:
    module = Module()
    tree = _DropDocstrings().visit(ast.parse(source, filename=path))
    _collect(path, tree.body, "", "", is_test_file(path), module)
    return module


def _collect(path: str, body: list[ast.stmt], scope: str, header: str, collects_tests: bool, module: Module) -> None:
    def put(name: str, value: str) -> str:
        key = f"{path}::{scope}{name}"
        # a name bound twice (overloads, if/else definitions) is one unit made of both
        module.units[key] = module.units.get(key, "") + value
        return key

    loose: list[ast.stmt] = []
    for node in body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            continue
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if collects_tests and node.name.startswith(TEST_FUNCTION_PREFIX):
                # a class decorator or base applies to every test in the class
                module.tests.add(put(node.name, header + fingerprint(node)))
            else:
                put(node.name, fingerprint(node))
        elif isinstance(node, ast.ClassDef):
            class_header = "".join(ast.dump(n) for n in (*node.decorator_list, *node.bases, *node.keywords))
            inner = f"{scope}{node.name}::"
            module.units[f"{path}::{inner}<class>"] = "class" + class_header
            is_test_class = collects_tests and node.name.startswith(TEST_CLASS_PREFIX)
            _collect(path, node.body, inner, header + class_header, is_test_class, module)
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            targets = node.targets if isinstance(node, ast.Assign) else [node.target]
            key = put(" = ".join(ast.unparse(t) for t in targets), ast.dump(node))
            module.statements.add(key)
            if scope:
                module.fields.add(key)
        elif not isinstance(node, ast.Pass):
            loose.append(node)
    if loose:
        module.statements.add(put("<body>" if scope else "<module>", "".join(ast.dump(n) for n in loose)))


def diff_units(old: Mapping[str, str], new: Mapping[str, str], match_moves: bool = True) -> UnitDiff:
    """Compare two sides of `key -> fingerprint`.

    With `match_moves`, a key that went away pairs one-to-one with a new key
    holding the same fingerprint and is reported as renamed, not removed.
    """
    diff = UnitDiff()
    added = {k: v for k, v in new.items() if k not in old}
    for key, value in old.items():
        if key in new:
            if new[key] != value:
                diff.altered.append(key)
            continue
        match = next((k for k, v in added.items() if v == value), None) if match_moves else None
        if match is None:
            diff.removed.append(key)
        else:
            diff.renamed.append((key, match))
            del added[match]
    diff.added = list(added)
    return diff


def is_data_file(path: str) -> bool:
    # files at the e2e/ root are environment (server versions, launch scripts), not pinned data
    if path.endswith((".py", ".md")):
        return False
    return path.startswith("tests/") or (path.startswith(E2E_ROOT) and "/" in path[len(E2E_ROOT) :])


def needs_content(path: str) -> bool:
    """Whether `classify` reads this file's bytes, not just whether they changed."""
    return path.endswith(".py") or path in (PYPROJECT, LOCKFILE, KNOWN_BAD_LIST) or path in RELEASE_LISTS


def lost_a_line(change: FileChange) -> bool:
    return change.base is not None and (change.head is None or change.deletions > 0)


def _dig(table: object, dotted: str) -> object:
    for key in dotted.split("."):
        table = table.get(key) if isinstance(table, dict) else None
    return table


def _package_name(requirement: str) -> str:
    match = re.match(r"[A-Za-z0-9._-]+", requirement.strip())
    return re.sub(r"[._]+", "-", match.group(0)).lower() if match else ""


def _gate_requirements(pyproject: object) -> list[str]:
    groups: list[object] = [_dig(pyproject, "project.dependencies")]
    for table in ("project.optional-dependencies", "dependency-groups"):
        found = _dig(pyproject, table)
        groups.extend(found.values() if isinstance(found, dict) else [])
    requirements = [r for group in groups if isinstance(group, list) for r in group if isinstance(r, str)]
    return sorted(r for r in requirements if _package_name(r) in GATE_PACKAGES)


def _locked_versions(lock: object) -> dict[str, object]:
    packages = _dig(lock, "package")
    return {
        _package_name(str(p.get("name"))): p.get("version")
        for p in (packages if isinstance(packages, list) else [])
        if isinstance(p, dict) and _package_name(str(p.get("name"))) in GATE_PACKAGES
    }


def _entries(content: bytes | None) -> set[str]:
    # first field of every line that is not blank or a comment: the release or version it names
    lines = (content or b"").decode(errors="replace").splitlines()
    return {line.split()[0] for line in lines if line.strip() and not line.lstrip().startswith("#")}


def gate_reasons(change: FileChange) -> list[str]:
    """Why this file counts as a gate change. Empty when it doesn't."""
    path = change.path
    if path in (PYPROJECT, LOCKFILE):
        if change.base is None or change.head is None:
            return [path]
        try:
            base, head = tomllib.loads(change.base.decode()), tomllib.loads(change.head.decode())
        except (tomllib.TOMLDecodeError, UnicodeDecodeError):
            return [f"{path} (not valid TOML)"]
        if path == LOCKFILE:
            was, now = _locked_versions(base), _locked_versions(head)
            return [
                f"{path} {name} {was.get(name)} -> {now.get(name)}"
                for name in sorted({*was, *now})
                if was.get(name) != now.get(name)
            ]
        reasons = [f"{path} [{table}]" for table in PYPROJECT_GATE_TABLES if _dig(base, table) != _dig(head, table)]
        if _gate_requirements(base) != _gate_requirements(head):
            reasons.append(f"{path} ({', '.join(sorted(GATE_PACKAGES))} requirement)")
        return reasons
    if path == KNOWN_BAD_LIST:
        skipped = sorted(_entries(change.head) - _entries(change.base))
        return [f"{path} skips {entry}" for entry in skipped]
    if path in RELEASE_LISTS:
        # a bump swaps one release for another and tests as much as before. only a shorter list tests less.
        before, after = _entries(change.base), _entries(change.head)
        return [f"{path} drops {entry}" for entry in sorted(before - after)] if len(after) < len(before) else []
    is_gate = (
        path.startswith(GATE_DIRS)
        or path in GATE_PATHS
        or path.rsplit("/", 1)[-1] in GATE_BASENAMES
        or path.startswith(GATE_TEST_PREFIX)
    )
    return [path] if is_gate else []


def classify(files: Iterable[FileChange], commits: Iterable[Commit] = ()) -> Classification:
    result = Classification()
    old, new = Module(), Module()
    old_data: dict[str, str] = {}
    new_data: dict[str, str] = {}
    files = sorted(files, key=lambda c: c.path)
    for change in files:
        path = change.path
        result.paths.append(path)
        if reasons := gate_reasons(change):
            result.gate_reasons.extend(reasons)
            result.gate_paths.append(path)
        if path.startswith(METRICS_ROOTS):
            result.metrics_paths.append(path)
        if path.endswith(".py"):
            try:
                parsed = [
                    (side, parse_module(path, src))
                    for side, src in ((old, change.base), (new, change.head))
                    if src is not None
                ]
            except (SyntaxError, ValueError, RecursionError, MemoryError):
                result.unparsed.append(path)
                continue
            for side, module in parsed:
                side.units.update(module.units)
                side.tests.update(module.tests)
                side.fields.update(module.fields)
                side.statements.update(module.statements)
            continue
        if lost_a_line(change):
            result.lines_deleted.append(path)
        if is_data_file(path):
            for bucket, content in ((old_data, change.base), (new_data, change.head)):
                if content is not None:
                    bucket[path] = content.hex()

    def only(module: Module, keep: set[str]) -> dict[str, str]:
        return {k: v for k, v in module.units.items() if k in keep}

    result.tests = diff_units(only(old, old.tests), only(new, new.tests))
    result.code = diff_units(only(old, old.units.keys() - old.tests), only(new, new.units.keys() - new.tests))
    # a statement added to a test file that already existed runs for the tests already there:
    # a new `pytestmark` skips them all without touching one body
    existing = {c.path for c in files if c.base is not None}
    new_statements = [k for k in result.code.added if k in new.statements and k.split("::", 1)[0] in existing]
    changed = (*result.code.altered, *result.code.removed, *new_statements)
    result.helpers_changed = [k for k in changed if k.startswith(TEST_ROOTS)]
    result.data_changed = [*(data := diff_units(old_data, new_data)).altered, *data.removed]
    for root, name in ((CONFIG_ROOT, "config_fields"), (REPORT_ROOT, "report_fields")):
        # a field that moved to another class changed the schema, so moves are not matched here
        sides = [{k: v for k, v in only(m, m.fields).items() if k.startswith(root)} for m in (old, new)]
        setattr(result, name, diff_units(sides[0], sides[1], match_moves=False))
    result.ai_commits = [
        commit.message.splitlines()[0] if commit.message else commit.author
        for commit in commits
        if "[bot]" in commit.author or AI_TRAILER.search(commit.message)
    ]
    return result


def report(c: Classification) -> str:
    sections = {
        "tests added": c.tests.added,
        "tests altered": c.tests.altered,
        "tests removed": c.tests.removed,
        "tests renamed": [f"{a} -> {b}" for a, b in c.tests.renamed],
        "helpers changed": c.helpers_changed,
        "data files changed": c.data_changed,
        "code added": c.code.added,
        "code altered": c.code.altered,
        "code removed": c.code.removed,
        "code moved": [f"{a} -> {b}" for a, b in c.code.renamed],
        "config fields added": c.config_fields.added,
        "config fields changed": [*c.config_fields.altered, *c.config_fields.removed],
        "report fields added": c.report_fields.added,
        "report fields changed": [*c.report_fields.altered, *c.report_fields.removed],
        "metrics files": c.metrics_paths,
        "gate changes": c.gate_reasons,
        "files that lost a line": c.lines_deleted,
        "files that don't parse": c.unparsed,
        "ai-assisted commits": c.ai_commits,
    }
    lines = [f"labels: {', '.join(c.labels) or 'none'}"]
    lines += [f"{name}:\n" + "\n".join(f"  {item}" for item in items) for name, items in sections.items() if items]
    return "\n".join(lines)


def git(*args: str) -> bytes:
    return subprocess.run(["git", *args], check=True, capture_output=True).stdout


def default_base() -> str:
    # in a fork clone origin/main is the fork's copy and can be months behind
    remote = "upstream" if "upstream" in git("remote").decode().split() else "origin"
    return f"{remote}/main"


def load_git(base: str, head: str) -> tuple[list[FileChange], list[Commit]]:
    """Read the diff between `head` and where it left `base` from the local repository."""
    merge_base = git("merge-base", base, head).decode().strip()
    # --no-renames so a moved file shows as a removal plus an addition. Moves are matched by content.
    diff = ("diff", "--no-renames", "--no-color", "-z", merge_base, head)
    statuses = git(*diff, "--name-status").decode().split("\0")
    # numstat prints `added<TAB>deleted<TAB>path`, with `-` for both counts on a binary file
    numstat = [entry.split("\t", 2) for entry in git(*diff, "--numstat").decode().split("\0") if entry]
    deletions = {path: int(deleted) if deleted.isdigit() else 1 for _, deleted, path in numstat}
    files = [
        FileChange(
            path=path,
            base=None if status == "A" else git("cat-file", "blob", f"{merge_base}:{path}"),
            head=None if status == "D" else git("cat-file", "blob", f"{head}:{path}"),
            deletions=deletions.get(path, 0),
        )
        for status, path in zip(statuses[0::2], statuses[1::2], strict=False)
    ]
    log = git("log", "--format=%an <%ae>%x00%B%x01", f"{merge_base}..{head}").decode(errors="replace")
    commits = [Commit(*entry.strip("\n").split("\0", 1)) for entry in log.split("\x01") if entry.strip()]
    return files, commits


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Say what the diff between a base branch and a head does.")
    parser.add_argument("--base", help="Branch the change merges into. Default: upstream/main, else origin/main.")
    parser.add_argument("--head", default="HEAD", help="Commit to classify (default: HEAD)")
    parser.add_argument("--json", action="store_true", help="Print the full classification as JSON")
    args = parser.parse_args(argv)
    try:
        classification = classify(*load_git(args.base or default_base(), args.head))
    except subprocess.CalledProcessError as e:
        print(f"git failed: {e.stderr.decode().strip()}", file=sys.stderr)
        return 2
    if args.json:
        print(json.dumps({"labels": classification.labels, **asdict(classification)}, indent=2))
    else:
        print(report(classification))
    return 0


if __name__ == "__main__":
    sys.exit(main())
