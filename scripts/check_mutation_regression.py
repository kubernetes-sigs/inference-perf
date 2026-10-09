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

Mutates every touched function twice, on the merge base and on the working
tree, with the same [tool.mutmut] config and each side's own tests. Fails when
survivors in existing functions rise, or when the functions the change adds
score under MIN_NEW_FUNCTION_SCORE.
"""

import argparse
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

# Share of mutants the tests must kill across the functions a change adds.
# Replaying 17 merged PRs put new functions at 67-92% (#872). Raise it as
# more history comes in.
MIN_NEW_FUNCTION_SCORE = 60.0

# A survivor only counts against the change once it survives this many
# reruns of its tests. Timing-dependent tests kill some mutants on one run
# and miss them on the next.
RERUNS = 3

SOURCE_PREFIX = "inference_perf/"
TESTS_PREFIX = "tests/"
EXCLUDED_TEST_PREFIXES = ("tests/optional/",)
EXCLUDED_TESTS = {
    # lists tracked YAML with `git ls-files`, and the mutation trees are not checkouts
    "tests/required/config/test_yaml_configs.py",
}

SURVIVED = {"survived", "no tests", "suspicious"}
KILLED = {"killed", "timeout", "segfault", "caught by type check"}

# Selected when a side has no tests in scope. With an empty selection mutmut
# runs the whole suite, and the side should count every mutant as untested.
PLACEHOLDER_TEST = "tests/test_mutation_placeholder.py"
NO_COVERING_TESTS = "could not find any test case for any mutant"
NOTHING_MATCHES = "nothing matches"

# Survivor diffs printed per side, so a large change keeps a readable summary.
MAX_SHOWN_SURVIVORS = 30


@dataclass
class Touched:
    """Functions a change touches, named the way mutmut names them."""

    # (name on the base, name on the head); they differ when the file moved
    existing: list[tuple[str, str]] = field(default_factory=list)
    new: list[str] = field(default_factory=list)


@dataclass
class Side:
    tree: Path
    tests: list[str]
    results: dict[str, str] = field(default_factory=dict)
    note: str = ""


class MutationRunError(Exception):
    pass


def git(*args: str) -> str:
    return subprocess.run(["git", *args], check=True, capture_output=True, text=True).stdout


def module_name(path: str) -> str:
    """mutmut's module name for a source path: a package's __init__.py is the package."""
    return path.removesuffix(".py").replace("/", ".").removesuffix(".__init__")


def function_spans(source: str) -> dict[str, tuple[int, int]]:
    """Line span of each module-level function and method, keyed by mutmut's mangled name."""
    spans: dict[str, tuple[int, int]] = {}

    def span(node: ast.FunctionDef | ast.AsyncFunctionDef) -> tuple[int, int]:
        # decorators belong to the function, so a change to one touches it
        return min([d.lineno for d in node.decorator_list] + [node.lineno]), node.end_lineno or node.lineno

    for node in ast.parse(source).body:
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            spans[f"x_{node.name}"] = span(node)
        elif isinstance(node, ast.ClassDef):
            for sub in node.body:
                if isinstance(sub, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    spans[f"xǁ{node.name}ǁ{sub.name}"] = span(sub)
    return spans


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


def changed_source_files(base: str, do_not_mutate: list[str]) -> list[tuple[str | None, str | None]]:
    """(base path, head path) for each changed source file, following renames.

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

    def mutable(path: str | None) -> bool:
        return path is not None and path.endswith(".py") and not any(Path(path).match(p) for p in do_not_mutate)

    return [(b if mutable(b) else None, h) for b, h in pairs if mutable(h)]


def touched_functions(base: str, do_not_mutate: list[str]) -> Touched:
    touched = Touched()
    for base_path, head_path in changed_source_files(base, do_not_mutate):
        assert head_path is not None
        head_src = Path(head_path).read_text()
        head_spans = function_spans(head_src)
        if base_path is None:
            for name in sorted(head_spans):
                touched.new.append(f"{module_name(head_path)}.{name}")
            continue
        base_spans = function_spans(git("show", f"{base}:{base_path}"))
        old_lines, new_lines = parse_hunks(git("diff", "-U0", "-M", base, "--", base_path, head_path))
        hit = touched_names(base_spans, old_lines) | touched_names(head_spans, new_lines)
        for name in sorted(hit):
            if name in base_spans and name in head_spans:
                touched.existing.append((f"{module_name(base_path)}.{name}", f"{module_name(head_path)}.{name}"))
            elif name in head_spans:
                touched.new.append(f"{module_name(head_path)}.{name}")
    return touched


def changed_test_files(base: str) -> list[str]:
    changed = git("diff", "--name-only", "--diff-filter=d", base, "--", TESTS_PREFIX).splitlines()
    changed += git("ls-files", "--others", "--exclude-standard", "--", TESTS_PREFIX).splitlines()
    return sorted({p for p in changed if Path(p).name.startswith("test_") and p.endswith(".py")})


def select_tests(tree: Path, modules: set[str], changed_tests: list[str]) -> list[str]:
    """Test files that import a mutated module, directly or through its package, plus the changed tests."""
    patterns = []
    for module in sorted(modules):
        patterns.append(re.escape(module) + r"\b")
        package, _, name = module.rpartition(".")
        init = tree / package.replace(".", "/") / "__init__.py"
        if init.exists() and re.search(rf"from \.{re.escape(name)} import", init.read_text()):
            patterns.append(rf"from {re.escape(package)} import")
    imports = re.compile("|".join(patterns)) if patterns else None
    picked = {p for p in changed_tests if (tree / p).exists()}
    for path in (tree / TESTS_PREFIX).rglob("test_*.py"):
        if imports and imports.search(path.read_text(errors="ignore")):
            picked.add(str(path.relative_to(tree)))
    return sorted(p for p in picked if p not in EXCLUDED_TESTS and not p.startswith(EXCLUDED_TEST_PREFIXES))


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


def run_side(tree: Path, names: list[str], config: dict[str, Any], changed_tests: list[str], max_children: int) -> Side:
    files = sorted({name.rpartition(".")[0] for name in names})
    paths = [p for p in (_source_path(tree, m) for m in files) if p is not None]
    tests = select_tests(tree, set(files), changed_tests)
    side = Side(tree=tree, tests=tests)
    if not tests:
        (tree / PLACEHOLDER_TEST).write_text("def test_placeholder() -> None:\n    pass\n")
        tests = [PLACEHOLDER_TEST]
    write_config(tree, config, paths, tests)
    run = mutmut(tree, "run", "--max-children", str(max_children), *(f"{n}__mutmut_*" for n in names))
    output = (run.stdout + run.stderr).replace("\r", "\n")
    if NOTHING_MATCHES in output:
        side.note = "no mutants (decorated or nothing to mutate)"
        return side
    uncovered = NO_COVERING_TESTS in output
    if run.returncode != 0 and not uncovered:
        raise MutationRunError(f"mutmut run failed in {tree} (exit {run.returncode}):\n{output[-3000:]}")
    listing = mutmut(tree, "results", "--all", "true").stdout
    for line in listing.splitlines():
        mutant, sep, status = line.strip().rpartition(": ")
        if sep and mutant.partition("__mutmut_")[0] in names:
            side.results[mutant] = "no tests" if uncovered and status == "not checked" else status
    unchecked = sorted(m for m, s in side.results.items() if s == "not checked")
    if unchecked:
        raise MutationRunError(f"mutmut left {len(unchecked)} mutants unchecked in {tree}, e.g. {unchecked[0]}")
    if uncovered:
        side.note = "no selected test covers these functions"
    return side


def _source_path(tree: Path, module: str) -> str | None:
    for candidate in (module.replace(".", "/") + ".py", module.replace(".", "/") + "/__init__.py"):
        if (tree / candidate).is_file():
            return candidate
    return None


def mutant_order(mutant: str) -> tuple[str, int]:
    name, _, number = mutant.partition("__mutmut_")
    return name, int(number) if number.isdigit() else 0


def survivors(side: Side, names: list[str]) -> list[str]:
    wanted = set(names)
    found = (m for m, s in side.results.items() if s in SURVIVED and m.partition("__mutmut_")[0] in wanted)
    return sorted(found, key=mutant_order)


def killed(side: Side, names: list[str]) -> list[str]:
    wanted = set(names)
    return sorted(m for m, s in side.results.items() if s in KILLED and m.partition("__mutmut_")[0] in wanted)


def still_survives(side: Side, mutant: str, pytest_args: list[str]) -> bool:
    """Rerun the tests that cover a surviving mutant. Killed in any rerun means flaky."""
    if side.results.get(mutant) == "no tests":
        return True
    tests = [t for t in mutmut(side.tree, "tests-for-mutant", mutant).stdout.splitlines() if "::" in t]
    for _ in range(RERUNS):
        rerun = subprocess.run(
            # mirrors how mutmut's own runner invokes pytest for a mutant
            [
                sys.executable,
                "-m",
                "pytest",
                "--rootdir=.",
                "--tb=native",
                "-x",
                "-q",
                "-p",
                "no:randomly",
                "-p",
                "no:random-order",
            ]
            + [*tests, *pytest_args],
            cwd=side.tree / "mutants",
            env={**os.environ, "MUTANT_UNDER_TEST": mutant},
            capture_output=True,
        )
        if rerun.returncode != 0:
            return False
    return True


@dataclass
class Verdict:
    base_survivors: int
    head_survivors: int
    flaky: list[str]
    new_killed: int
    new_survived: int
    failures: list[str]

    @property
    def new_score(self) -> float | None:
        total = self.new_killed + self.new_survived
        return 100.0 * self.new_killed / total if total else None


def judge(touched: Touched, base: Side | None, head: Side, pytest_args: list[str]) -> Verdict:
    base_names = [b for b, _ in touched.existing]
    head_names = [h for _, h in touched.existing]
    base_surv = survivors(base, base_names) if base else []
    head_surv = survivors(head, head_names)
    flaky: list[str] = []
    if len(head_surv) > len(base_surv):
        # only functions whose own count rose can hold a survivor the change added
        rose = {h for b, h in touched.existing if _count(head_surv, h) > _count(base_surv, b)}
        flaky = [m for m in head_surv if m.partition("__mutmut_")[0] in rose and not still_survives(head, m, pytest_args)]
        head_surv = [m for m in head_surv if m not in flaky]
    new_killed = len(killed(head, touched.new))
    new_survived = len(survivors(head, touched.new))
    verdict = Verdict(len(base_surv), len(head_surv), flaky, new_killed, new_survived, [])
    if verdict.head_survivors > verdict.base_survivors:
        verdict.failures.append(
            f"Survivors in existing functions rose from {verdict.base_survivors} to {verdict.head_survivors}"
        )
    score = verdict.new_score
    if score is not None and score < MIN_NEW_FUNCTION_SCORE - 0.01:
        verdict.failures.append(f"New functions score {score:.1f}%, under the {MIN_NEW_FUNCTION_SCORE:.0f}% floor")
    return verdict


def _count(mutants: list[str], name: str) -> int:
    return sum(1 for m in mutants if m.partition("__mutmut_")[0] == name)


def report(base_ref: str, touched: Touched, base: Side | None, head: Side, verdict: Verdict) -> str:
    lines = [f"## Mutation ratchet against `{base_ref}`", ""]
    if touched.existing:
        lines += ["| Existing function | Base survivors | Head survivors |", "|---|---|---|"]
        base_surv = survivors(base, [b for b, _ in touched.existing]) if base else []
        head_surv = survivors(head, [h for _, h in touched.existing])
        for b, h in touched.existing:
            lines.append(f"| `{h}` | {_count(base_surv, b)} | {_count(head_surv, h) - _count(verdict.flaky, h)} |")
        lines += ["", f"Total: {verdict.base_survivors} on the base, {verdict.head_survivors} with this change.", ""]
    if touched.new:
        score = verdict.new_score
        shown = (
            "no mutants" if score is None else f"{score:.1f}% ({verdict.new_killed} killed, {verdict.new_survived} survived)"
        )
        lines += [f"New functions: {len(touched.new)}, score {shown}, floor {MIN_NEW_FUNCTION_SCORE:.0f}%.", ""]
    for side, label in ((base, "Base"), (head, "Head")):
        if side is not None:
            note = f" ({side.note})" if side.note else ""
            lines.append(f"{label} tests: {len(side.tests)} files{note}.")
    if verdict.flaky:
        lines += ["", f"Flaky, killed on rerun and not counted: {', '.join(f'`{m}`' for m in verdict.flaky)}"]
    shown_survivors = survivors(head, [h for _, h in touched.existing] + touched.new)
    shown_survivors = [m for m in shown_survivors if m not in verdict.flaky]
    if shown_survivors:
        lines += ["", f"<details><summary>{len(shown_survivors)} surviving mutants on the head</summary>", ""]
        for mutant in shown_survivors[:MAX_SHOWN_SURVIVORS]:
            diff = mutmut(head.tree, "show", mutant).stdout.strip()
            # mutmut show already heads the diff with the mutant name and status
            lines += ["```diff", diff, "```", ""]
        if len(shown_survivors) > MAX_SHOWN_SURVIVORS:
            lines.append(f"{len(shown_survivors) - MAX_SHOWN_SURVIVORS} more not shown.")
        lines.append("</details>")
    lines += ["", "FAIL: " + "; ".join(verdict.failures) if verdict.failures else "PASS"]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser(description="Mutation ratchet over the functions changed since the base branch.")
    parser.add_argument("--base", default="origin/main", help="Branch the change merges into (default: origin/main)")
    parser.add_argument("--max-children", type=int, default=os.cpu_count() or 4, help="Parallel mutmut workers")
    parser.add_argument("--keep", action="store_true", help="Keep the mutation trees for debugging")
    parser.add_argument("--output", default="mutation.json", help="Where to write the full results")
    args = parser.parse_args()

    config: dict[str, Any] = tomllib.loads(Path("pyproject.toml").read_text())["tool"]["mutmut"]
    do_not_mutate = [str(p) for p in config.get("do_not_mutate", [])]
    merge_base = git("merge-base", args.base, "HEAD").strip()
    touched = touched_functions(merge_base, do_not_mutate)
    if not touched.existing and not touched.new:
        print(f"No functions under {SOURCE_PREFIX} changed since {args.base}. Nothing to mutate.")
        return
    print(
        f"Mutating {len(touched.existing)} existing and {len(touched.new)} new functions against {args.base} ({merge_base[:8]})."
    )

    changed_tests = changed_test_files(merge_base)
    pytest_args = [str(a) for a in config.get("pytest_add_cli_args", [])]
    workdir = Path(tempfile.mkdtemp(prefix="mutation-"))
    try:
        base: Side | None = None
        if touched.existing:
            export_base(merge_base, workdir / "base")
            base = run_side(workdir / "base", [b for b, _ in touched.existing], config, changed_tests, args.max_children)
        (workdir / "head").mkdir()
        export_working_tree(workdir / "head")
        head = run_side(
            workdir / "head", [h for _, h in touched.existing] + touched.new, config, changed_tests, args.max_children
        )
        verdict = judge(touched, base, head, pytest_args)
        summary = report(args.base, touched, base, head, verdict)
    except MutationRunError as e:
        print(f"ERROR: {e}")
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
        "base_results": base.results if base else {},
        "head_results": head.results,
        "flaky": verdict.flaky,
        "failures": verdict.failures,
    }
    Path(args.output).write_text(json.dumps(record, indent=2) + "\n")
    if verdict.failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
