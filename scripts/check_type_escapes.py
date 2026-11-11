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
"""Ratchet on the escapes that satisfy `mypy --strict` without fixing the type.

Counts `Any`, `cast(` and `# type: ignore` in every Python file under ROOTS, and
the modules that a `[[tool.mypy.overrides]]` block in pyproject.toml relaxes, in
the working tree and on main. Fails when the total of any one rises. Main is
read through git at check time, the way scripts/check_coverage_regression.py
reads its baseline, so no count is committed and a moved file costs nothing.
"""

import argparse
import io
import re
import subprocess
import sys
import tarfile
import tokenize
import tomllib
from pathlib import Path
from typing import NamedTuple

REPO_ROOT = Path(__file__).resolve().parents[1]
ROOTS = ("inference_perf", "tests")
PYPROJECT = "pyproject.toml"

TYPE_IGNORE = re.compile(r"#\s*type:\s*ignore")


class Counts(NamedTuple):
    any: int = 0
    cast: int = 0
    ignore: int = 0
    override: int = 0


FileCounts = dict[str, Counts]


def count_escapes(source: str) -> Counts:
    """Counts by token, so a string or comment that mentions Any or cast( is not an escape."""
    any_count = cast_count = ignore_count = 0
    previous = ""
    for token in tokenize.generate_tokens(io.StringIO(source).readline):
        if token.type == tokenize.COMMENT:
            if TYPE_IGNORE.match(token.string):
                ignore_count += 1
            continue
        if token.type == tokenize.NAME and token.string == "Any":
            any_count += 1
        elif token.type == tokenize.OP and token.string == "(" and previous == "cast":
            cast_count += 1
        previous = token.string
    return Counts(any_count, cast_count, ignore_count)


def count_overrides(pyproject: str) -> int:
    """Modules under a `[[tool.mypy.overrides]]` block that relaxes checking of the module itself."""
    count = 0
    for block in tomllib.loads(pyproject).get("tool", {}).get("mypy", {}).get("overrides", []):
        # an untyped dependency has no fix on our side, so a block that only sets this is free
        if block.keys() <= {"module", "ignore_missing_imports"}:
            continue
        modules = block.get("module", [])
        count += 1 if isinstance(modules, str) else len(modules)
    return count


def with_overrides(found: FileCounts, pyproject: str) -> FileCounts:
    overrides = count_overrides(pyproject)
    return {**found, PYPROJECT: Counts(override=overrides)} if overrides else found


def scan(repo_root: Path) -> FileCounts:
    """Working-tree counts for every file under ROOTS that has at least one escape."""
    found: FileCounts = {}
    for root in ROOTS:
        for path in sorted((repo_root / root).rglob("*.py")):
            counts = count_escapes(path.read_text(encoding="utf-8"))
            if any(counts):
                found[path.relative_to(repo_root).as_posix()] = counts
    pyproject = repo_root / PYPROJECT
    return with_overrides(found, pyproject.read_text(encoding="utf-8") if pyproject.exists() else "")


def git(repo_root: Path, *args: str) -> "subprocess.CompletedProcess[bytes]":
    return subprocess.run(["git", *args], cwd=repo_root, capture_output=True, check=False)


def resolve_base(repo_root: Path) -> str:
    """The commit to compare with: where HEAD left main, else the tip of main."""
    # in a fork clone origin/main is the fork's copy and can be months behind
    remote = "upstream" if "upstream" in git(repo_root, "remote").stdout.decode().split() else "origin"
    # a failed fetch (offline, no remote) leaves whatever main is already here
    git(repo_root, "fetch", remote, f"+refs/heads/main:refs/remotes/{remote}/main")
    for ref in (f"{remote}/main", "main"):
        tip = git(repo_root, "rev-parse", "--verify", "--quiet", f"{ref}^{{commit}}")
        if tip.returncode:
            continue
        # a shallow CI checkout has no merge base. it is the merge commit, so the tip is right
        merge_base = git(repo_root, "merge-base", "HEAD", ref)
        return (merge_base if merge_base.returncode == 0 else tip).stdout.decode().strip()
    raise SystemExit(f"check_type_escapes: neither {remote}/main nor main exists, pass --base")


def scan_ref(repo_root: Path, ref: str) -> FileCounts:
    """Counts at a commit for every file under ROOTS that has at least one escape."""
    found: FileCounts = {}
    roots = git(repo_root, "ls-tree", "-d", "--name-only", ref, "--", *ROOTS).stdout.decode().split()
    if roots:
        found = scan_archive(repo_root, ref, roots)
    # a commit with no pyproject.toml prints nothing, which counts as no overrides
    return with_overrides(found, git(repo_root, "show", f"{ref}:{PYPROJECT}").stdout.decode("utf-8"))


def scan_archive(repo_root: Path, ref: str, roots: list[str]) -> FileCounts:
    archive = git(repo_root, "archive", "--format=tar", ref, "--", *roots)
    if archive.returncode:
        raise SystemExit(f"check_type_escapes: git archive {ref} failed: {archive.stderr.decode().strip()}")
    found: FileCounts = {}
    with tarfile.open(fileobj=io.BytesIO(archive.stdout)) as tar:
        for member in tar:
            if not (member.isfile() and member.name.endswith(".py")):
                continue
            blob = tar.extractfile(member)
            assert blob is not None
            counts = count_escapes(blob.read().decode("utf-8"))
            if any(counts):
                found[member.name] = counts
    return found


def totals(files: FileCounts) -> Counts:
    return Counts(*(sum(column) for column in zip(*files.values(), strict=True))) if files else Counts()


def risen_kinds(current: Counts, base: Counts) -> list[str]:
    """The kinds whose total rose. A fall in one kind does not pay for a rise in another."""
    return [kind for kind in Counts._fields if getattr(current, kind) > getattr(base, kind)]


def file_changes(current: FileCounts, base: FileCounts, kind: str) -> list[str]:
    """Every file whose count of one kind differs from main, as report lines."""
    lines: list[str] = []
    for name in sorted(current.keys() | base.keys()):
        now, was = getattr(current.get(name, Counts()), kind), getattr(base.get(name, Counts()), kind)
        if now != was:
            lines.append(f"{name}: {was} -> {now}")
    return lines


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", help="Commit to compare with. Default: where HEAD left main.")
    args = parser.parse_args()

    base_ref = args.base or resolve_base(REPO_ROOT)
    current, base = scan(REPO_ROOT), scan_ref(REPO_ROOT, base_ref)
    now, was = totals(current), totals(base)

    labels = {"any": "Any", "cast": "cast(", "ignore": "type: ignore", "override": "mypy override"}
    print(f"Type escapes vs main ({base_ref[:12] if args.base is None else base_ref}):")
    for kind in Counts._fields:
        print(f"  {labels[kind]:<13} {getattr(was, kind)} -> {getattr(now, kind)}")

    risen = risen_kinds(now, was)
    if not risen:
        return 0
    for kind in risen:
        print(f"\n{labels[kind]} rose. Files that changed:")
        for line in file_changes(current, base, kind):
            print(f"  {line}")
    print(
        "\nFix the type instead of adding Any, cast(, type: ignore or a mypy override. A JSON boundary\n"
        "takes a TypedDict or a pydantic model. A new escape lands only with one of the same kind removed."
    )
    return 1


if __name__ == "__main__":
    sys.exit(main())
