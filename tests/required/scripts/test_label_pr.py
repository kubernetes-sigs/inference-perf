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
"""Tests for scripts/label_pr.py, which labels a pull request and checks it against its kind."""

import importlib.util
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import ModuleType

import pytest
import yaml

ROOT = Path(__file__).resolve().parents[3]
REPO = "kubernetes-sigs/inference-perf"
BOT = "github-actions[bot]"


# Loads scripts/classify_changes.py and scripts/label_pr.py as modules. scripts/ is not a package, and
# label_pr imports classify_changes by its bare name, so that one is registered first.
@pytest.fixture(scope="module")
def lp() -> ModuleType:
    modules = []
    for name in ("classify_changes", "label_pr"):
        spec = importlib.util.spec_from_file_location(name, ROOT / "scripts" / f"{name}.py")
        assert spec and spec.loader
        module = importlib.util.module_from_spec(spec)
        sys.modules[name] = module
        spec.loader.exec_module(module)
        modules.append(module)
    return modules[-1]


# Stands in for the GitHub API. `routes` maps a GET path to its JSON answer, `blobs` maps a contents path to
# file bytes, and every write lands in `sent` as (method, path, body). A path with no route answers 404.
@dataclass
class FakeApi:
    lp: ModuleType
    routes: dict[str, object] = field(default_factory=dict)
    blobs: dict[str, bytes] = field(default_factory=dict)
    sent: list[tuple[str, str, object]] = field(default_factory=list)
    fail_delete: int = 0

    def get(self, path: str) -> object:
        if path not in self.routes:
            raise self.lp.ApiError(404, path, "Not Found")
        return self.routes[path]

    def get_all(self, path: str) -> list[object]:
        found = self.routes.get(path, [])
        return found if isinstance(found, list) else []

    def raw(self, path: str) -> bytes:
        if path not in self.blobs:
            raise self.lp.ApiError(404, path, "Not Found")
        return self.blobs[path]

    def send(self, method: str, path: str, body: object = None) -> None:
        if method == "DELETE" and self.fail_delete:
            raise self.lp.ApiError(self.fail_delete, path, "failed")
        self.sent.append((method, path, body))


# One file entry the way the pull request files endpoint returns it.
def file_entry(
    path: str, status: str = "modified", sha: str = "h", deletions: int = 0, previous: str = ""
) -> dict[str, object]:
    return {"filename": path, "status": status, "sha": sha, "deletions": deletions, "previous_filename": previous}


# A fake API holding PR #7 with the given description, labels and files. Its merge base is commit `mb`, whose
# tree holds `base_tree` (path -> blob id). `blobs` maps (path, ref) to bytes for files the classifier parses.
def api_with_pr(
    lp: ModuleType,
    body: str = "Part of #1",
    labels: tuple[str, ...] = ("kind/cleanup",),
    files: tuple[dict[str, object], ...] = (),
    base_tree: dict[str, str] | None = None,
    blobs: dict[tuple[str, str], str] | None = None,
    issues: dict[int, tuple[str, ...]] | None = None,
) -> FakeApi:
    api = FakeApi(lp)
    api.routes["/pulls/7"] = {
        "number": 7,
        "body": body,
        "labels": [{"name": name} for name in labels],
        "base": {"sha": "b"},
        "head": {"sha": "h"},
    }
    api.routes["/compare/b...h"] = {"merge_base_commit": {"sha": "mb"}}
    api.routes["/git/trees/mb?recursive=1"] = {
        "truncated": False,
        "tree": [{"path": path, "sha": sha} for path, sha in (base_tree or {}).items()],
    }
    api.routes["/pulls/7/files"] = list(files)
    api.routes["/pulls/7/commits"] = [{"author": {"login": "dev"}, "commit": {"author": {"name": "Dev"}, "message": "Change"}}]
    for number, issue_labels in ({1: ("kind/cleanup",)} if issues is None else issues).items():
        api.routes[f"/issues/{number}"] = {"number": number, "labels": [{"name": name} for name in issue_labels]}
    for (path, ref), content in (blobs or {}).items():
        api.blobs[f"/contents/{path}?ref={ref}"] = content.encode()
    return api


# A classification built from (path, base text, head text) triples, for the checks that take one directly.
def classified(lp: ModuleType, *changes: tuple[str, str | None, str | None]) -> object:
    cc = sys.modules["classify_changes"]
    sides = [(path, *(None if side is None else side.encode() for side in (base, head))) for path, base, head in changes]
    return cc.classify([cc.FileChange(*change) for change in sides])


SOURCE = ("inference_perf/utils/x.py", "def f():\n    return 1\n", "def f():\n    return 2\n")
NEW_SOURCE = ("inference_perf/utils/y.py", None, "def g():\n    return 1\n")
ALTERED_TEST = ("tests/required/test_x.py", "def test_a():\n    assert 1\n", "def test_a():\n    assert 2\n")
NEW_UNIT_TEST = ("tests/required/test_y.py", None, "def test_b():\n    assert 1\n")
NEW_E2E_TEST = ("e2e/tests/test_y.py", None, "def test_b():\n    assert 1\n")
CONFIG = "inference_perf/config/x.py"
NEW_FIELD = (CONFIG, "class C(BaseModel):\n    a: int\n", "class C(BaseModel):\n    a: int\n    b: int\n")
CHANGED_FIELD = (CONFIG, "class C(BaseModel):\n    a: int\n", "class C(BaseModel):\n    a: str\n")
REMOVED_FIELD = (CONFIG, "class C(BaseModel):\n    a: int\n    b: int\n", "class C(BaseModel):\n    a: int\n")
REPORT_FIELD = ("inference_perf/reportgen/x.py", "class R(BaseModel):\n    a: int\n", "class R(BaseModel):\n    a: str\n")


# `Fixes #12` closes the issue and `Part of #34` does not. Both are found, in the order written.
def test_find_links_reads_fixes_and_part_of(lp: ModuleType) -> None:
    links = lp.find_links("Fixes #12\nPart of #34\n", REPO)
    assert links == [lp.Link(12, True), lp.Link(34, False)]


# Every closing keyword GitHub honors counts as closing, in any case and with or without a colon.
@pytest.mark.parametrize("verb", ["fix", "Fixes", "fixed", "close", "Closes", "closed", "resolve", "Resolves:", "RESOLVED"])
def test_find_links_closing_keywords(lp: ModuleType, verb: str) -> None:
    assert lp.find_links(f"{verb} #5", REPO) == [lp.Link(5, True)]


# A full issue URL in this repository links the issue. The same URL in another repository does not.
def test_find_links_url_form_must_be_this_repo(lp: ModuleType) -> None:
    assert lp.find_links(f"Part of https://github.com/{REPO}/issues/9", REPO) == [lp.Link(9, False)]
    assert lp.find_links("Fixes https://github.com/other/repo/issues/9", REPO) == []


# A bare `#5`, `see #5`, `prefixes #5` and `Fixes other/repo#5` are not links to an issue here.
@pytest.mark.parametrize("body", ["#5", "see #5", "prefixes #5", "Fixes other/repo#5", ""])
def test_find_links_ignores_mentions(lp: ModuleType, body: str) -> None:
    assert lp.find_links(body, REPO) == []


# The same issue is named twice, once as `Part of` and once as `Fixes`, in either order. It is one link, and it closes.
@pytest.mark.parametrize("body", ["Part of #5\nFixes #5", "Fixes #5\nPart of #5"])
def test_find_links_merges_repeats(lp: ModuleType, body: str) -> None:
    assert lp.find_links(body, REPO) == [lp.Link(5, True)]


# A release-note block with text counts. An empty block, a block that says NONE and no block at all do not.
@pytest.mark.parametrize(
    ("body", "expected"),
    [
        ("```release-note\nThe `x` field is gone.\n```", True),
        ("```release-note\n\n```", False),
        ("```release-note\nnone\n```", False),
        ("```\nThe `x` field is gone.\n```", False),
        ("release-note: yes", False),
    ],
)
def test_has_release_note(lp: ModuleType, body: str, expected: bool) -> None:
    assert lp.has_release_note(body) is expected


# A PR with no link is told to add one. Nothing else about issues is checked.
def test_check_issue_without_a_link(lp: ModuleType) -> None:
    assert lp.check_issue(["kind/bug"], [], {}) == ["No linked issue. Put `Fixes #N` or `Part of #N` in the description."]


# The linked number does not exist, then is a pull request. Each gets its own message.
def test_check_issue_missing_or_a_pull_request(lp: ModuleType) -> None:
    link = [lp.Link(3, False)]
    assert lp.check_issue(["kind/bug"], link, {3: None}) == ["#3 is not an issue in this repository."]
    pull = lp.Issue(3, ("kind/bug",), True)
    assert lp.check_issue(["kind/bug"], link, {3: pull}) == ["#3 is a pull request. Link the issue it works on."]


# The linked issue has no kind, then two kinds, then only kind/support, which is not one of the five. Each fails.
@pytest.mark.parametrize("labels", [(), ("kind/bug", "kind/feature"), ("kind/support",)])
def test_check_issue_needs_exactly_one_kind_on_the_issue(lp: ModuleType, labels: tuple[str, ...]) -> None:
    problems = lp.check_issue(["kind/bug"], [lp.Link(3, False)], {3: lp.Issue(3, labels, False)})
    assert len(problems) == 1 and problems[0].startswith("#3 needs exactly one of kind/feature")


# A kind/cleanup PR against a kind/feature issue. `Fixes` must match and fails. `Part of` may differ and passes.
def test_check_issue_kind_must_match_only_when_fixing(lp: ModuleType) -> None:
    issue = {3: lp.Issue(3, ("kind/feature", "priority/backlog"), False)}
    fixes = lp.check_issue(["kind/cleanup"], [lp.Link(3, True)], issue)
    assert len(fixes) == 1 and "`kind/cleanup` and fixes #3, which is `kind/feature`" in fixes[0]
    assert lp.check_issue(["kind/cleanup"], [lp.Link(3, False)], issue) == []
    assert lp.check_issue(["kind/feature"], [lp.Link(3, True)], issue) == []


# A PR with no kind or two kinds that fixes an issue gets no mismatch message. Its kind problem is reported elsewhere.
@pytest.mark.parametrize("pr_kinds", [[], ["kind/bug", "kind/cleanup"]])
def test_check_issue_skips_the_match_without_one_pr_kind(lp: ModuleType, pr_kinds: list[str]) -> None:
    assert lp.check_issue(pr_kinds, [lp.Link(3, True)], {3: lp.Issue(3, ("kind/feature",), False)}) == []


# kind/feature with a new e2e test passes. With only a new unit test, or only an altered e2e test, it fails.
def test_feature_needs_a_new_e2e_test(lp: ModuleType) -> None:
    assert lp.check_kind("kind/feature", classified(lp, NEW_SOURCE, NEW_E2E_TEST)) == []
    assert lp.check_kind("kind/feature", classified(lp, NEW_SOURCE, NEW_UNIT_TEST)) != []
    altered_e2e = ("e2e/tests/test_x.py", "def test_a():\n    assert 1\n", "def test_a():\n    assert 2\n")
    assert lp.check_kind("kind/feature", classified(lp, NEW_SOURCE, altered_e2e)) != []


# kind/bug passes with a new test and with an altered test. With neither it fails.
def test_bug_needs_a_new_or_changed_test(lp: ModuleType) -> None:
    assert lp.check_kind("kind/bug", classified(lp, SOURCE, NEW_UNIT_TEST)) == []
    assert lp.check_kind("kind/bug", classified(lp, SOURCE, ALTERED_TEST)) == []
    assert lp.check_kind("kind/bug", classified(lp, SOURCE)) != []


# kind/cleanup that changes inference_perf/ and alters a test fails and names the test. Either change alone passes.
def test_cleanup_cannot_change_source_and_existing_tests(lp: ModuleType) -> None:
    problems = lp.check_kind("kind/cleanup", classified(lp, SOURCE, ALTERED_TEST))
    assert len(problems) == 1 and "`tests/required/test_x.py::test_a`" in problems[0]
    assert lp.check_kind("kind/cleanup", classified(lp, SOURCE)) == []
    assert lp.check_kind("kind/cleanup", classified(lp, ALTERED_TEST)) == []
    assert lp.check_kind("kind/cleanup", classified(lp, SOURCE, NEW_UNIT_TEST)) == []


# kind/cleanup fails when a config field is added, changed or removed, and when a report field changes.
@pytest.mark.parametrize("change", [NEW_FIELD, CHANGED_FIELD, REMOVED_FIELD, REPORT_FIELD])
def test_cleanup_cannot_touch_a_schema_field(lp: ModuleType, change: tuple[str, str, str]) -> None:
    problems = lp.check_kind("kind/cleanup", classified(lp, change))
    assert len(problems) == 1 and "can't add, change or remove a config or report field" in problems[0]


# kind/documentation passes on docs/ files and Markdown anywhere. A source file, AGENTS.md and CLAUDE.md fail,
# the last two because they are gate files.
def test_documentation_only_changes_docs(lp: ModuleType) -> None:
    docs = [("docs/config.md", "a\n", "b\n"), ("docs/images/x.png", None, "png"), ("README.md", "a\n", "b\n")]
    assert lp.check_kind("kind/documentation", classified(lp, *docs)) == []
    for path in ("inference_perf/x.py", "AGENTS.md", "CLAUDE.md"):
        problems = lp.check_kind("kind/documentation", classified(lp, *docs, (path, "a = 1\n", "a = 2\n")))
        assert len(problems) == 1 and f"`{path}`" in problems[0]


# kind/api-change passes when an existing config or report field is changed or removed. Adding a field is not enough.
def test_api_change_needs_an_existing_field_to_change(lp: ModuleType) -> None:
    for change in (CHANGED_FIELD, REMOVED_FIELD, REPORT_FIELD):
        assert lp.check_kind("kind/api-change", classified(lp, change)) == []
    assert lp.check_kind("kind/api-change", classified(lp, NEW_FIELD)) != []
    assert lp.check_kind("kind/api-change", classified(lp, SOURCE)) != []


# A config field change, a report field change and an altered test each need a release note, and the message says why.
@pytest.mark.parametrize(
    ("change", "reason"),
    [
        (NEW_FIELD, "adds, changes or removes a config or report field"),
        (REPORT_FIELD, "adds, changes or removes a config or report field"),
        (ALTERED_TEST, "changes a pinned behavior"),
    ],
)
def test_release_note_is_required(lp: ModuleType, change: tuple[str, str, str], reason: str) -> None:
    problems = lp.check_release_note("Part of #1", classified(lp, change))
    assert len(problems) == 1 and reason in problems[0]
    assert lp.check_release_note("```release-note\nWhat changed.\n```", classified(lp, change)) == []


# A source-only change with a new test needs no release note.
def test_release_note_is_not_required_for_other_changes(lp: ModuleType) -> None:
    assert lp.check_release_note("", classified(lp, SOURCE, NEW_UNIT_TEST)) == []


# A workflow change beside a source change fails the land-alone rule. The workflow alone, or the source alone, passes.
def test_gate_changes_land_alone(lp: ModuleType) -> None:
    workflow = (".github/workflows/x.yml", "a\n", "b\n")
    problems = lp.check_gates(classified(lp, workflow, SOURCE))
    assert len(problems) == 1 and "`.github/workflows/x.yml`" in problems[0] and "`inference_perf/utils/x.py`" in problems[0]
    assert lp.check_gates(classified(lp, workflow, ("scripts/x.py", None, "X = 1\n"))) == []
    assert lp.check_gates(classified(lp, SOURCE)) == []


# A PR with no kind, then two kinds, then only kind/flake. Each gets one kind message and no kind-specific check.
@pytest.mark.parametrize(
    ("labels", "message"),
    [((), "No kind."), (("kind/bug", "kind/cleanup"), "Pick one kind"), (("kind/flake", "lgtm"), "No kind.")],
)
def test_find_problems_wants_exactly_one_kind(lp: ModuleType, labels: tuple[str, ...], message: str) -> None:
    pr = lp.PullRequest(7, "Part of #1", labels, "b", "h")
    problems = lp.find_problems(pr, [lp.Link(1, False)], {1: lp.Issue(1, ("kind/bug",), False)}, classified(lp, SOURCE))
    assert len(problems) == 1 and problems[0].startswith(message)


# A kind/cleanup PR that changes a workflow and a doc together. Its kind and issue are fine, so the one problem
# find_problems returns is the land-alone rule.
def test_find_problems_includes_the_gate_rule(lp: ModuleType) -> None:
    pr = lp.PullRequest(7, "Part of #1", ("kind/cleanup",), "b", "h")
    c = classified(lp, (".github/workflows/x.yml", None, "a\n"), ("docs/a.md", None, "a\n"))
    problems = lp.find_problems(pr, [lp.Link(1, False)], {1: lp.Issue(1, ("kind/bug",), False)}, c)
    assert len(problems) == 1 and problems[0].startswith("Gate changes land alone.")


# Six names with a limit of five: the first five are shown in backticks and the rest is counted.
def test_shown_caps_the_list(lp: ModuleType) -> None:
    assert lp.shown(["a", "b"]) == "`a`, `b`"
    assert lp.shown(list("abcdef")) == "`a`, `b`, `c`, `d`, `e` and 1 more"


# The comment starts with the marker, lists problems as numbered steps and puts the names in a code block.
# With no problems it says so and has no numbered steps.
def test_render_comment(lp: ModuleType) -> None:
    c = classified(lp, ALTERED_TEST)
    failing = lp.render_comment(["first", "second"], c)
    assert failing.startswith(lp.COMMENT_MARKER)
    assert "To pass:\n1. first\n2. second" in failing
    assert (
        "```text\nlabels: changes-pinned-behavior, tests-only-change\ntests altered:\n  tests/required/test_x.py::test_a\n```"
        in failing
    )
    passing = lp.render_comment([], c)
    assert "All conditions are met." in passing and "To pass:" not in passing


# A file path carrying backticks and an escape character. Both are dropped, so the name can't close the code block.
def test_render_comment_strips_backticks_and_control_characters(lp: ModuleType) -> None:
    comment = lp.render_comment([], classified(lp, ("docs/a```\x1b[31m.md", "a\n", None)))
    assert comment.count("```") == 2
    assert "\x1b" not in comment
    assert "docs/a[31m.md" in comment


# A report longer than the limit is cut and says so, which keeps the comment under GitHub's size cap.
def test_render_comment_cuts_a_long_report(lp: ModuleType) -> None:
    many = [(f"docs/{'x' * 100}{i}.md", "a\n", None) for i in range(700)]
    comment = lp.render_comment([], classified(lp, *many))
    assert "(cut short)" in comment
    assert len(comment) < 65_536


# A PR with one parsed file, one doc and one golden. The Python file's bytes are fetched at the merge base and
# at head. The doc and the golden are never fetched: their blob ids stand in, and the doc's deletions carry over.
def test_load_changes_fetches_only_parsed_files(lp: ModuleType) -> None:
    api = api_with_pr(
        lp,
        files=(
            file_entry("inference_perf/x.py", sha="py-head"),
            file_entry("docs/a.md", sha="doc-head", deletions=2),
            file_entry("tests/goldens/g.txt", sha="g-head"),
        ),
        base_tree={"inference_perf/x.py": "py-base", "docs/a.md": "doc-base", "tests/goldens/g.txt": "g-base"},
        blobs={("inference_perf/x.py", "mb"): "A = 1\n", ("inference_perf/x.py", "h"): "A = 2\n"},
    )
    files, commits = lp.load_changes(api, lp.load_pr(api, 7))
    assert [(f.path, f.base, f.head, f.deletions) for f in files] == [
        ("inference_perf/x.py", b"A = 1\n", b"A = 2\n", 0),
        ("docs/a.md", b"doc-base", b"doc-head", 2),
        ("tests/goldens/g.txt", b"g-base", b"g-head", 0),
    ]
    assert [(c.author, c.message) for c in commits] == [("Dev (dev)", "Change")]


# An added file has no base side, a removed file has no head side, and a copied file counts as added.
def test_load_changes_added_removed_copied(lp: ModuleType) -> None:
    api = api_with_pr(
        lp,
        files=(
            file_entry("docs/new.md", "added", sha="n"),
            file_entry("docs/old.md", "removed", deletions=4),
            file_entry("docs/copy.md", "copied", sha="c"),
        ),
        base_tree={"docs/old.md": "o"},
    )
    files, _ = lp.load_changes(api, lp.load_pr(api, 7))
    assert [(f.path, f.base, f.head) for f in files] == [
        ("docs/new.md", None, b"n"),
        ("docs/old.md", b"o", None),
        ("docs/copy.md", None, b"c"),
    ]


# A golden renamed with the same blob id becomes a removal of the old path and an addition of the new one.
# The classifier then pairs them by content, so the rename pins nothing new.
def test_load_changes_splits_a_rename(lp: ModuleType) -> None:
    api = api_with_pr(
        lp,
        files=(file_entry("tests/goldens/new.txt", "renamed", sha="same", previous="tests/goldens/old.txt"),),
        base_tree={"tests/goldens/old.txt": "same"},
    )
    files, _ = lp.load_changes(api, lp.load_pr(api, 7))
    assert [(f.path, f.base, f.head) for f in files] == [
        ("tests/goldens/old.txt", b"same", None),
        ("tests/goldens/new.txt", None, b"same"),
    ]
    assert sys.modules["classify_changes"].classify(files).data_changed == []


# The compare call returns no merge base, so the tree lookup fails. Then the tree listing comes back cut short.
# Both stop the run with an error instead of classifying against a partial base.
def test_load_changes_refuses_a_partial_base(lp: ModuleType) -> None:
    api = api_with_pr(lp)
    api.routes["/compare/b...h"] = {}
    with pytest.raises(lp.ApiError):
        lp.load_changes(api, lp.load_pr(api, 7))
    api = api_with_pr(lp)
    api.routes["/git/trees/mb?recursive=1"] = {"truncated": True, "tree": []}
    with pytest.raises(lp.ApiError):
        lp.load_changes(api, lp.load_pr(api, 7))


# A linked number that answers 404 loads as None. A pull request number loads with is_pull_request set.
# Any other API failure is raised, not read as "no such issue".
def test_load_issue(lp: ModuleType) -> None:
    api = FakeApi(lp)
    assert lp.load_issue(api, 3) is None
    api.routes["/issues/4"] = {"labels": [{"name": "kind/bug"}], "pull_request": {"url": "x"}}
    assert lp.load_issue(api, 4) == lp.Issue(4, ("kind/bug",), True)
    api.routes["/issues/5"] = {"labels": []}
    assert lp.load_issue(api, 5) == lp.Issue(5, (), False)

    class Broken(FakeApi):
        def get(self, path: str) -> object:
            raise lp.ApiError(500, path, "boom")

    with pytest.raises(lp.ApiError):
        lp.load_issue(Broken(lp), 3)


# A kind/cleanup PR that alters a test and a source file, carrying one stale managed label and one prow label.
# decide adds the two labels the diff earns, removes the stale one, leaves prow's alone and reports two problems.
def test_decide_works_out_labels_and_problems(lp: ModuleType) -> None:
    path, base, head = ALTERED_TEST
    api = api_with_pr(
        lp,
        labels=("kind/cleanup", "touches/metrics", "lgtm"),
        files=(file_entry(path, deletions=1), file_entry(SOURCE[0], deletions=1)),
        blobs={(path, "mb"): base, (path, "h"): head, (SOURCE[0], "mb"): SOURCE[1], (SOURCE[0], "h"): SOURCE[2]},
    )
    outcome = lp.decide(api, REPO, 7)
    assert outcome.add == ["changes-pinned-behavior"]
    assert outcome.remove == ["touches/metrics"]
    assert [problem.split(",")[0].split(".")[0] for problem in outcome.problems] == [
        "`kind/cleanup` changes `inference_perf/`",
        "This PR changes a pinned behavior",
    ]
    assert api.sent == []


# A PR whose labels already match the diff and that meets its conditions. Nothing to add, remove or fix.
def test_decide_with_nothing_to_do(lp: ModuleType) -> None:
    api = api_with_pr(lp, labels=("kind/cleanup", "purely-additive"), files=(file_entry("docs/a.md", "added"),))
    outcome = lp.decide(api, REPO, 7)
    assert (outcome.add, outcome.remove, outcome.problems) == ([], [], [])


# An outcome that adds two labels, removes one and has no comment on the PR yet. apply posts the labels in one
# call, deletes the stale one with its slash escaped, and creates the comment.
def test_apply_writes_labels_and_a_new_comment(lp: ModuleType) -> None:
    api = FakeApi(lp)
    lp.apply(api, lp.Outcome(7, add=["purely-additive", "touches/metrics"], remove=["touches/ci-gates"], comment="body"))
    assert api.sent == [
        ("POST", "/issues/7/labels", {"labels": ["purely-additive", "touches/metrics"]}),
        ("DELETE", "/issues/7/labels/touches%2Fci-gates", None),
        ("POST", "/issues/7/comments", {"body": "body"}),
    ]


# The PR already has this workflow's comment with older text. apply edits that comment and posts no second one.
def test_apply_updates_its_own_comment(lp: ModuleType) -> None:
    api = FakeApi(lp)
    api.routes["/issues/7/comments"] = [{"id": 99, "user": {"login": BOT}, "body": f"{lp.COMMENT_MARKER}\nold"}]
    lp.apply(api, lp.Outcome(7, comment=f"{lp.COMMENT_MARKER}\nnew"))
    assert api.sent == [("PATCH", "/issues/comments/99", {"body": f"{lp.COMMENT_MARKER}\nnew"})]


# The existing comment already says exactly what the outcome says, and no label moves. apply writes nothing.
def test_apply_is_quiet_when_nothing_changed(lp: ModuleType) -> None:
    api = FakeApi(lp)
    api.routes["/issues/7/comments"] = [{"id": 99, "user": {"login": BOT}, "body": f"{lp.COMMENT_MARKER}\nsame"}]
    lp.apply(api, lp.Outcome(7, comment=f"{lp.COMMENT_MARKER}\nsame"))
    assert api.sent == []


# A person posted a comment that starts with the marker, and the bot posted one without it. Neither is this
# workflow's comment, so apply leaves both alone and posts its own.
def test_apply_never_edits_someone_elses_comment(lp: ModuleType) -> None:
    api = FakeApi(lp)
    api.routes["/issues/7/comments"] = [
        {"id": 1, "user": {"login": "someone"}, "body": f"{lp.COMMENT_MARKER}\nmine"},
        {"id": 2, "user": {"login": BOT}, "body": "coverage report"},
    ]
    lp.apply(api, lp.Outcome(7, comment="body"))
    assert api.sent == [("POST", "/issues/7/comments", {"body": "body"})]


# Removing a label that is already gone answers 404, which apply ignores. A 500 on the same call is raised.
def test_apply_tolerates_only_a_missing_label(lp: ModuleType) -> None:
    api = FakeApi(lp, fail_delete=404)
    lp.apply(api, lp.Outcome(7, remove=["touches/metrics"], comment="body"))
    assert api.sent == [("POST", "/issues/7/comments", {"body": "body"})]
    with pytest.raises(lp.ApiError):
        lp.apply(FakeApi(lp, fail_delete=500), lp.Outcome(7, remove=["touches/metrics"], comment="body"))


# Writes a GitHub event payload to a file and returns its path.
def event_file(tmp_path: Path, payload: dict[str, object]) -> str:
    path = tmp_path / "event.json"
    path.write_text(json.dumps(payload))
    return str(path)


# Two open PRs: #7 says `Part of #1` and #8 says `Fixes #2`. Returns a fake API that lists them.
def api_with_two_open_prs(lp: ModuleType) -> FakeApi:
    api = api_with_pr(lp, files=(file_entry("docs/a.md", "added"),), labels=("kind/cleanup", "purely-additive"))
    api.routes["/pulls?state=open"] = [{"number": 7, "body": "Part of #1"}, {"number": 8, "body": "Fixes #2"}]
    return api


# A pull request event picks that one PR and its problems fail the run. So does --pr.
def test_targets_for_a_pull_request(lp: ModuleType, tmp_path: Path) -> None:
    api = api_with_two_open_prs(lp)
    args = lp.argparse.Namespace(pr=None, issue=None, event=event_file(tmp_path, {"pull_request": {"number": 8}}))
    assert lp.targets(api, REPO, args) == ([8], True)
    assert lp.targets(api, REPO, lp.argparse.Namespace(pr=7, issue=None, event=None)) == ([7], True)


# An issue event for #2 picks only the open PR that links #2, and that sweep never fails the run. So does --issue.
def test_targets_for_an_issue(lp: ModuleType, tmp_path: Path) -> None:
    api = api_with_two_open_prs(lp)
    args = lp.argparse.Namespace(pr=None, issue=None, event=event_file(tmp_path, {"issue": {"number": 2}}))
    assert lp.targets(api, REPO, args) == ([8], False)
    assert lp.targets(api, REPO, lp.argparse.Namespace(pr=None, issue=1, event=None)) == ([7], False)


# A manual run has neither a pull request nor an issue in its payload. It sweeps every open PR and never fails.
def test_targets_for_a_manual_run(lp: ModuleType, tmp_path: Path) -> None:
    api = api_with_two_open_prs(lp)
    args = lp.argparse.Namespace(pr=None, issue=None, event=event_file(tmp_path, {"inputs": None}))
    assert lp.targets(api, REPO, args) == ([7, 8], False)


# main on a PR that meets its conditions exits 0, writes its comment and prints "ok".
def test_main_passes_a_clean_pr(lp: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    api = api_with_two_open_prs(lp)
    assert lp.main(["--repo", REPO, "--pr", "7"], api) == 0
    assert [method for method, _, _ in api.sent] == ["POST"]
    assert capsys.readouterr().out.splitlines()[0] == "#7: ok"


# main on a PR with no linked issue exits 1 and prints the problem. With --dry-run it still exits 1 and writes nothing.
def test_main_fails_a_pr_with_problems(lp: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    api = api_with_pr(lp, body="no link here", files=(file_entry("docs/a.md", "added"),))
    assert lp.main(["--repo", REPO, "--pr", "7", "--dry-run"], api) == 1
    assert api.sent == []
    out = capsys.readouterr().out
    assert "#7: conditions not met" in out and "add label purely-additive" in out and "problem: No linked issue." in out
    assert lp.main(["--repo", REPO, "--pr", "7"], api) == 1
    assert api.sent != []


# The same failing PR reached through an issue sweep. Its comment is refreshed but the run exits 0,
# because a sweep only fails on the PR that started it.
def test_main_sweep_does_not_fail_on_problems(lp: ModuleType) -> None:
    api = api_with_pr(lp, body="Part of #1", labels=(), files=(file_entry("docs/a.md", "added"),))
    api.routes["/pulls?state=open"] = [{"number": 7, "body": "Part of #1"}]
    assert lp.main(["--repo", REPO, "--issue", "1"], api) == 0
    assert ("POST", "/issues/7/labels", {"labels": ["purely-additive"]}) in api.sent


# A sweep over PR #7, which loads, and PR #8, whose API calls fail. #7 is still processed and the run exits 2.
def test_main_reports_an_api_error_and_carries_on(lp: ModuleType, capsys: pytest.CaptureFixture[str]) -> None:
    api = api_with_two_open_prs(lp)
    assert lp.main(["--repo", REPO, "--issue", "2"], api) == 2
    assert "#8: GitHub API 404" in capsys.readouterr().err
    api.routes["/pulls?state=open"] = [{"number": 8, "body": "Part of #1"}, {"number": 7, "body": "Part of #1"}]
    assert lp.main(["--repo", REPO, "--issue", "1"], api) == 2
    assert "#7: ok" in capsys.readouterr().out


# With no --repo and no GITHUB_REPOSITORY in the environment, main exits with a usage error.
def test_main_needs_a_repo(lp: ModuleType, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("GITHUB_REPOSITORY", raising=False)
    with pytest.raises(SystemExit) as exit_info:
        lp.main(["--pr", "7"], FakeApi(lp))
    assert exit_info.value.code == 2


# The workflow that runs this script. It triggers on pull_request_target, never checks out a PR ref, passes the
# event payload to the script, and is not one of the workflows the merge gate waits on.
def test_workflow_is_report_only_and_never_checks_out_pr_code() -> None:
    workflows = ROOT / ".github" / "workflows"
    text = (workflows / "pr_category.yml").read_text()
    workflow = yaml.safe_load(text)
    assert "pull_request_target" in workflow[True]
    checkouts = [step for step in workflow["jobs"]["categorize"]["steps"] if "actions/checkout" in step.get("uses", "")]
    assert len(checkouts) == 1 and "ref" not in checkouts[0].get("with", {})
    assert "github.event.pull_request.head" not in text
    assert 'python scripts/label_pr.py --event "$GITHUB_EVENT_PATH"' in text
    assert workflow["name"] not in (workflows / "merge_gate.yml").read_text()
