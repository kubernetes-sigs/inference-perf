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
"""Unit tests for the type-escape ratchet in scripts/check_type_escapes.py."""

import importlib.util
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts" / "check_type_escapes.py"


# Loads scripts/check_type_escapes.py by path, since scripts/ is not a package.
# Registered in sys.modules first because its NamedTuples look their module up there.
@pytest.fixture
def cte() -> ModuleType:
    spec = importlib.util.spec_from_file_location("check_type_escapes", SCRIPT)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


# One import, one annotation and one typing.Any in a signature. Counts 3 Any, 0 cast, 0 ignore.
def test_counts_any_names(cte: ModuleType) -> None:
    source = "from typing import Any\nimport typing\nx: Any = 1\ndef f(y: typing.Any) -> None: ...\n"
    assert cte.count_escapes(source) == cte.Counts(any=3, cast=0, ignore=0)


# cast(int, x) and typing.cast(str, y) count. broadcast(x) and a bare `cast` name do not. 2 casts.
def test_counts_cast_calls(cte: ModuleType) -> None:
    source = "a = cast(int, x)\nb = typing.cast(str, y)\nc = broadcast(x)\nd = cast\n"
    assert cte.count_escapes(source).cast == 2


# A bare ignore, a coded ignore and one written without spaces all count. 3 ignores.
def test_counts_type_ignores_with_and_without_code(cte: ModuleType) -> None:
    source = "a = f()  # type: ignore\nb = g()  # type: ignore[arg-type]\nc = h()  #type:ignore\n"
    assert cte.count_escapes(source).ignore == 3


# Any, cast( and type: ignore inside a string, a docstring and an ordinary comment. All counts 0.
def test_strings_and_plain_comments_are_not_escapes(cte: ModuleType) -> None:
    source = '"""Any cast(x) # type: ignore"""\ns = "Any cast(int, x)"\n# Any value works, cast(x) too\n'
    assert cte.count_escapes(source) == cte.Counts()


# Runs git in a throwaway repo with a fixed identity, so commits work on a bare CI runner.
def git(repo: Path, *args: str) -> None:
    identity = ["-c", "user.name=t", "-c", "user.email=t@example.com", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", *identity, *args], cwd=repo, check=True, capture_output=True)


# A repo whose main has inference_perf/a.py with 2 Any and 1 cast, and tests/test_a.py with 1 ignore.
@pytest.fixture
def repo(tmp_path: Path) -> Path:
    (tmp_path / "inference_perf").mkdir()
    (tmp_path / "tests").mkdir()
    (tmp_path / "inference_perf" / "a.py").write_text("from typing import Any\nx: Any = cast(int, 1)\n")
    (tmp_path / "tests" / "test_a.py").write_text("y = f()  # type: ignore[arg-type]\n")
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "seed")
    return tmp_path


# The seeded main read back through git. a.py has 2 Any and 1 cast, test_a.py has 1 ignore.
def test_scan_ref_reads_counts_from_a_commit(cte: ModuleType, repo: Path) -> None:
    assert cte.scan_ref(repo, "main") == {
        "inference_perf/a.py": cte.Counts(any=2, cast=1),
        "tests/test_a.py": cte.Counts(ignore=1),
    }


# An uncommitted third Any in a.py. The tree totals 3 Any, main still totals 2, so any rose.
def test_new_escape_in_the_working_tree_rises(cte: ModuleType, repo: Path) -> None:
    with (repo / "inference_perf" / "a.py").open("a") as f:
        f.write("z: Any = 2\n")
    now, was = cte.totals(cte.scan(repo)), cte.totals(cte.scan_ref(repo, "main"))
    assert (was.any, now.any) == (2, 3)
    assert cte.risen_kinds(now, was) == ["any"]


# a.py renamed to b.py with no edit. Totals match main, so nothing rose.
def test_moved_file_does_not_rise(cte: ModuleType, repo: Path) -> None:
    (repo / "inference_perf" / "a.py").rename(repo / "inference_perf" / "b.py")
    now, was = cte.totals(cte.scan(repo)), cte.totals(cte.scan_ref(repo, "main"))
    assert cte.risen_kinds(now, was) == []


# A branch two commits past main, with no origin remote. resolve_base returns main's commit.
def test_resolve_base_is_where_head_left_main(cte: ModuleType, repo: Path) -> None:
    main_sha = subprocess.run(
        ["git", "rev-parse", "main"], cwd=repo, check=True, capture_output=True, text=True
    ).stdout.strip()
    git(repo, "checkout", "-q", "-b", "feature")
    git(repo, "commit", "-q", "--allow-empty", "-m", "one")
    git(repo, "commit", "-q", "--allow-empty", "-m", "two")
    assert cte.resolve_base(repo) == main_sha


# A commit with neither inference_perf/ nor tests/. scan_ref returns no files instead of failing.
def test_scan_ref_without_roots_is_empty(cte: ModuleType, tmp_path: Path) -> None:
    (tmp_path / "README").write_text("x\n")
    git(tmp_path, "init", "-q", "-b", "main")
    git(tmp_path, "add", ".")
    git(tmp_path, "commit", "-q", "-m", "seed")
    assert cte.scan_ref(tmp_path, "main") == {}


# Totals go 5 Any, 2 cast, 1 ignore -> 4 Any, 3 cast, 1 ignore. Only cast is reported: the Any drop does not pay for it.
def test_kinds_do_not_trade(cte: ModuleType) -> None:
    assert cte.risen_kinds(cte.Counts(any=4, cast=3, ignore=1), cte.Counts(any=5, cast=2, ignore=1)) == ["cast"]


# One Any leaves a.py and lands in b.py. Both files are listed for any, and the total holds at 2.
def test_file_changes_lists_both_sides_of_a_move(cte: ModuleType) -> None:
    base = {"a.py": cte.Counts(any=2)}
    current = {"a.py": cte.Counts(any=1), "b.py": cte.Counts(any=1)}
    assert cte.file_changes(current, base, "any") == ["a.py: 2 -> 1", "b.py: 0 -> 1"]
    assert cte.risen_kinds(cte.totals(current), cte.totals(base)) == []


# A clone whose origin main is one commit behind upstream main. resolve_base fetches and uses upstream's.
def test_resolve_base_prefers_upstream_over_a_stale_origin(
    cte: ModuleType, repo: Path, tmp_path_factory: pytest.TempPathFactory
) -> None:
    clone = tmp_path_factory.mktemp("clone")
    subprocess.run(["git", "clone", "-q", str(repo), str(clone)], check=True, capture_output=True)
    git(clone, "remote", "add", "upstream", str(repo))
    git(repo, "commit", "-q", "--allow-empty", "-m", "newer")
    newer = subprocess.run(["git", "rev-parse", "main"], cwd=repo, check=True, capture_output=True, text=True).stdout.strip()
    git(clone, "pull", "-q", "upstream", "main")
    assert cte.resolve_base(clone) == newer
