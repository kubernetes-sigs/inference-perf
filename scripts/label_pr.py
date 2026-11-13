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
"""Label a pull request by what its diff does and check it against its kind.

Reads the pull request through the GitHub API as data: its description, labels,
commits and the bytes of its changed files. Nothing from the pull request is
checked out, imported or run, so this is safe under `pull_request_target`.
scripts/classify_changes.py says what the diff does. This script applies those
labels, posts the list of names as one comment it keeps up to date, and exits 1
when the pull request does not meet the conditions in #868.
"""

import argparse
import json
import os
import re
import sys
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import classify_changes as cc

KINDS = ("kind/feature", "kind/bug", "kind/cleanup", "kind/documentation", "kind/api-change")
DOCS_ROOT = "docs/"

# `Fixes #N` and GitHub's other closing keywords finish the issue. `Part of #N` is one step of several.
LINK = re.compile(
    r"\b(?P<verb>close[sd]?|fix(?:e[sd])?|resolve[sd]?|part of):?\s+"
    r"(?:https://github\.com/(?P<repo>[\w.-]+/[\w.-]+)/issues/|#)(?P<number>\d+)\b",
    re.IGNORECASE,
)
RELEASE_NOTE = re.compile(r"```release-note[^\n]*\n(?P<text>.*?)```", re.DOTALL)

COMMENT_MARKER = "<!-- pr-category -->"
COMMENT_AUTHOR = "github-actions[bot]"
# GitHub rejects a comment over 65536 characters
REPORT_LIMIT = 60_000
NAMES_SHOWN = 5


class ApiError(Exception):
    def __init__(self, status: int, path: str, detail: str) -> None:
        super().__init__(f"GitHub API {status} on {path}: {detail}")
        self.status = status


class Api(Protocol):
    """The GitHub REST calls this script makes, relative to one repository."""

    def get(self, path: str) -> object: ...
    def get_all(self, path: str) -> list[object]: ...
    def raw(self, path: str) -> bytes: ...
    def send(self, method: str, path: str, body: object = None) -> None: ...


class GitHubApi:
    def __init__(self, repo: str, token: str | None) -> None:
        self._root = f"https://api.github.com/repos/{repo}"
        self._token = token

    def _open(self, method: str, path: str, accept: str, body: object = None) -> bytes:
        headers = {"Accept": accept, "X-GitHub-Api-Version": "2022-11-28", "User-Agent": "inference-perf-label-pr"}
        if self._token:
            headers["Authorization"] = f"Bearer {self._token}"
        data = None if body is None else json.dumps(body).encode()
        request = urllib.request.Request(self._root + path, data=data, headers=headers, method=method)
        try:
            with urllib.request.urlopen(request, timeout=60) as response:
                content: bytes = response.read()
                return content
        except urllib.error.HTTPError as e:
            raise ApiError(e.code, path, e.read().decode(errors="replace")[:300]) from e

    def get(self, path: str) -> object:
        result: object = json.loads(self._open("GET", path, "application/vnd.github+json"))
        return result

    def get_all(self, path: str) -> list[object]:
        items: list[object] = []
        page = 1
        while True:
            batch = self.get(f"{path}{'&' if '?' in path else '?'}per_page=100&page={page}")
            if not isinstance(batch, list):
                raise ApiError(200, path, "expected a list")
            items.extend(batch)
            if len(batch) < 100:
                return items
            page += 1

    def raw(self, path: str) -> bytes:
        return self._open("GET", path, "application/vnd.github.raw")

    def send(self, method: str, path: str, body: object = None) -> None:
        self._open(method, path, "application/vnd.github+json", body)


@dataclass(frozen=True)
class PullRequest:
    number: int
    body: str
    labels: tuple[str, ...]
    base_sha: str
    head_sha: str


@dataclass(frozen=True)
class Link:
    number: int
    closes: bool


@dataclass(frozen=True)
class Issue:
    number: int
    labels: tuple[str, ...]
    is_pull_request: bool


@dataclass
class Outcome:
    """What one run decided for one pull request."""

    number: int
    problems: list[str] = field(default_factory=list)
    add: list[str] = field(default_factory=list)
    remove: list[str] = field(default_factory=list)
    comment: str = ""


def dig(node: object, *keys: str) -> object:
    for key in keys:
        node = node.get(key) if isinstance(node, dict) else None
    return node


def text(node: object, *keys: str) -> str:
    value = dig(node, *keys)
    return value if isinstance(value, str) else ""


def label_names(node: object) -> tuple[str, ...]:
    labels = dig(node, "labels")
    return tuple(text(label, "name") for label in (labels if isinstance(labels, list) else []))


def kinds_of(labels: tuple[str, ...]) -> list[str]:
    return [label for label in labels if label in KINDS]


def find_links(body: str, repo: str) -> list[Link]:
    """The issues in this repository that a description says it fixes or is part of."""
    links: dict[int, bool] = {}
    for match in LINK.finditer(body):
        if (match["repo"] or repo).lower() != repo.lower():
            continue
        number = int(match["number"])
        links[number] = links.get(number, False) or match["verb"].lower() != "part of"
    return [Link(number, closes) for number, closes in links.items()]


def has_release_note(body: str) -> bool:
    match = RELEASE_NOTE.search(body)
    return match is not None and match["text"].strip().upper() not in ("", "NONE")


def shown(names: list[str]) -> str:
    more = f" and {len(names) - NAMES_SHOWN} more" if len(names) > NAMES_SHOWN else ""
    return ", ".join(f"`{name}`" for name in names[:NAMES_SHOWN]) + more


def check_issue(pr_kinds: list[str], links: list[Link], issues: dict[int, Issue | None]) -> list[str]:
    if not links:
        return ["No linked issue. Put `Fixes #N` or `Part of #N` in the description."]
    problems = []
    for link in links:
        issue = issues.get(link.number)
        if issue is None:
            problems.append(f"#{link.number} is not an issue in this repository.")
            continue
        if issue.is_pull_request:
            problems.append(f"#{link.number} is a pull request. Link the issue it works on.")
            continue
        issue_kinds = kinds_of(issue.labels)
        if len(issue_kinds) != 1:
            problems.append(f"#{link.number} needs exactly one of {', '.join(KINDS)}. Set it on the issue with `/kind`.")
        elif link.closes and len(pr_kinds) == 1 and pr_kinds != issue_kinds:
            problems.append(
                f"This PR is `{pr_kinds[0]}` and fixes #{link.number}, which is `{issue_kinds[0]}`. "
                "Relabel one of them, or say `Part of` if this PR is one step of several."
            )
    return problems


def check_kind(kind: str, c: cc.Classification) -> list[str]:
    fields_added = [*c.config_fields.added, *c.report_fields.added]
    fields_changed = [*c.config_fields.altered, *c.config_fields.removed, *c.report_fields.altered, *c.report_fields.removed]
    if kind == "kind/feature":
        if not any(test.startswith(cc.E2E_ROOT) for test in c.tests.added):
            return ["`kind/feature` needs a new test under `e2e/` that shows what the feature does."]
    elif kind == "kind/bug":
        if not (c.tests.added or c.tests.altered):
            return ["`kind/bug` needs a new or changed test that fails without the fix."]
    elif kind == "kind/cleanup":
        problems = []
        pinned = [*c.tests.altered, *c.tests.removed, *c.helpers_changed, *c.data_changed]
        if c.source_changed and c.changes_pinned_behavior:
            problems.append(f"`kind/cleanup` changes `inference_perf/`, so it can't change existing tests: {shown(pinned)}.")
        if fields_added or fields_changed:
            names = shown([*fields_added, *fields_changed])
            problems.append(f"`kind/cleanup` can't add, change or remove a config or report field: {names}.")
        return problems
    elif kind == "kind/documentation":
        other = [p for p in c.paths if p in c.gate_paths or not (p.startswith(DOCS_ROOT) or p.endswith(".md"))]
        if other:
            return [f"`kind/documentation` only changes `docs/` and Markdown. These are neither: {shown(other)}."]
    elif kind == "kind/api-change":
        if not fields_changed:
            return [
                "`kind/api-change` changes or removes an existing config or report field. This PR does neither. "
                "A PR that only adds a field is `kind/feature`."
            ]
    return []


def check_release_note(body: str, c: cc.Classification) -> list[str]:
    if has_release_note(body):
        return []
    reasons = []
    if any(label in c.labels for label in ("touches/config-schema", "touches/report-schema")):
        reasons.append("adds, changes or removes a config or report field")
    if c.changes_pinned_behavior:
        reasons.append("changes a pinned behavior")
    if not reasons:
        return []
    return [f"This PR {' and '.join(reasons)}, so its description needs a `release-note` block that says what changed."]


def check_gates(c: cc.Classification) -> list[str]:
    other = [p for p in c.paths if p not in c.gate_paths]
    if c.gate_paths and other:
        return [f"Gate changes land alone. Gate files: {shown(c.gate_paths)}. Other files: {shown(other)}."]
    return []


def find_problems(pr: PullRequest, links: list[Link], issues: dict[int, Issue | None], c: cc.Classification) -> list[str]:
    pr_kinds = kinds_of(pr.labels)
    problems = check_issue(pr_kinds, links, issues)
    if len(pr_kinds) == 1:
        problems += check_kind(pr_kinds[0], c)
    elif pr_kinds:
        problems.append(f"Pick one kind or split the PR. Found {', '.join(pr_kinds)}.")
    else:
        problems.append(f"No kind. Add one of {', '.join(KINDS)} with `/kind`.")
    return problems + check_release_note(pr.body, c) + check_gates(c)


def render_comment(problems: list[str], c: cc.Classification) -> str:
    # names come from the pull request. a backtick could close the code block and a control
    # character could hide text, so both are dropped before the names are shown.
    report = re.sub(r"[`\x00-\x08\x0b-\x1f\x7f]", "", cc.report(c))
    if len(report) > REPORT_LIMIT:
        report = report[:REPORT_LIMIT] + "\n(cut short)"
    lines = [COMMENT_MARKER, "PR category check. Report-only for now: nothing here blocks a merge.", ""]
    if problems:
        lines += ["To pass:", *(f"{i}. {problem}" for i, problem in enumerate(problems, 1))]
    else:
        lines.append("All conditions are met.")
    lines += ["", "<details>", "<summary>What the diff does</summary>", "", "```text", report, "```", "", "</details>"]
    return "\n".join(lines)


def load_pr(api: Api, number: int) -> PullRequest:
    data = api.get(f"/pulls/{number}")
    return PullRequest(number, text(data, "body"), label_names(data), text(data, "base", "sha"), text(data, "head", "sha"))


def load_issue(api: Api, number: int) -> Issue | None:
    try:
        data = api.get(f"/issues/{number}")
    except ApiError as e:
        if e.status in (404, 410):
            return None
        raise
    return Issue(number, label_names(data), dig(data, "pull_request") is not None)


def load_changes(api: Api, pr: PullRequest) -> tuple[list[cc.FileChange], list[cc.Commit]]:
    """The pull request's changed files and commits, read through the API."""
    merge_base = text(api.get(f"/compare/{pr.base_sha}...{pr.head_sha}"), "merge_base_commit", "sha")
    tree = api.get(f"/git/trees/{merge_base}?recursive=1")
    if dig(tree, "truncated"):
        raise ApiError(200, f"/git/trees/{merge_base}", "the tree listing was cut short")
    entries = dig(tree, "tree")
    base_blobs = {text(entry, "path"): text(entry, "sha") for entry in (entries if isinstance(entries, list) else [])}

    def side(path: str, ref: str, blob: str) -> bytes:
        # files the classifier parses are fetched. the rest are only compared, so the blob id stands in.
        if cc.needs_content(path):
            return api.raw(f"/contents/{urllib.parse.quote(path)}?ref={ref}")
        return blob.encode()

    files: list[cc.FileChange] = []
    for entry in api.get_all(f"/pulls/{pr.number}/files"):
        path, status = text(entry, "filename"), text(entry, "status")
        deleted = dig(entry, "deletions")
        deletions = deleted if isinstance(deleted, int) else 0
        head = None if status == "removed" else side(path, pr.head_sha, text(entry, "sha"))
        if status == "renamed":
            # split into a removal and an addition, the way the local loader's --no-renames does
            previous = text(entry, "previous_filename")
            files.append(cc.FileChange(previous, side(previous, merge_base, base_blobs.get(previous, "")), None))
            files.append(cc.FileChange(path, None, head))
        elif status in ("added", "copied"):
            files.append(cc.FileChange(path, None, head))
        else:
            files.append(cc.FileChange(path, side(path, merge_base, base_blobs.get(path, "")), head, deletions))

    commits = [
        cc.Commit(
            f"{text(entry, 'commit', 'author', 'name')} ({text(entry, 'author', 'login')})", text(entry, "commit", "message")
        )
        for entry in api.get_all(f"/pulls/{pr.number}/commits")
    ]
    return files, commits


def decide(api: Api, repo: str, number: int) -> Outcome:
    """Read one pull request and work out its labels, problems and comment. Writes nothing."""
    pr = load_pr(api, number)
    classification = cc.classify(*load_changes(api, pr))
    links = find_links(pr.body, repo)
    issues = {link.number: load_issue(api, link.number) for link in links}
    problems = find_problems(pr, links, issues, classification)
    wanted, current = set(classification.labels), set(pr.labels)
    return Outcome(
        number=number,
        problems=problems,
        add=[label for label in cc.LABELS if label in wanted - current],
        remove=[label for label in cc.LABELS if label in current - wanted],
        comment=render_comment(problems, classification),
    )


def apply(api: Api, outcome: Outcome) -> None:
    """Make the pull request's labels and comment match the outcome."""
    issue = f"/issues/{outcome.number}"
    if outcome.add:
        api.send("POST", f"{issue}/labels", {"labels": outcome.add})
    for label in outcome.remove:
        try:
            api.send("DELETE", f"{issue}/labels/{urllib.parse.quote(label, safe='')}")
        except ApiError as e:
            if e.status != 404:  # 404: someone already removed it
                raise
    ours = [
        comment
        for comment in api.get_all(f"{issue}/comments")
        # only ever edit this workflow's own comment. anyone can post the marker.
        if text(comment, "user", "login") == COMMENT_AUTHOR and text(comment, "body").startswith(COMMENT_MARKER)
    ]
    if not ours:
        api.send("POST", f"{issue}/comments", {"body": outcome.comment})
    elif text(ours[0], "body") != outcome.comment:
        api.send("PATCH", f"/issues/comments/{dig(ours[0], 'id')}", {"body": outcome.comment})


def summary(outcome: Outcome) -> str:
    lines = [f"#{outcome.number}: {'conditions not met' if outcome.problems else 'ok'}"]
    lines += [f"  add label {label}" for label in outcome.add]
    lines += [f"  remove label {label}" for label in outcome.remove]
    lines += [f"  problem: {problem}" for problem in outcome.problems]
    return "\n".join(lines)


def targets(api: Api, repo: str, args: argparse.Namespace) -> tuple[list[int], bool]:
    """The pull requests to process, and whether their problems fail this run.

    A run started by one pull request fails on that pull request's problems. A
    sweep over other pull requests only refreshes their labels and comments.
    """
    event: object = json.loads(Path(args.event).read_text()) if args.event else {}
    pr = args.pr or dig(event, "pull_request", "number")
    if isinstance(pr, int):
        return [pr], True
    numbered = [(n, text(p, "body")) for p in api.get_all("/pulls?state=open") if isinstance(n := dig(p, "number"), int)]
    issue = args.issue or dig(event, "issue", "number")
    if isinstance(issue, int):
        return [n for n, body in numbered if any(link.number == issue for link in find_links(body, repo))], False
    return [n for n, _ in numbered], False


def main(argv: list[str] | None = None, api: Api | None = None) -> int:
    parser = argparse.ArgumentParser(description="Label a pull request by what its diff does and check it against its kind.")
    parser.add_argument("--repo", default=os.environ.get("GITHUB_REPOSITORY"), help="owner/name. Default: $GITHUB_REPOSITORY")
    target = parser.add_mutually_exclusive_group()
    target.add_argument("--pr", type=int, help="Check this pull request. Exits 1 when its conditions are not met.")
    target.add_argument("--issue", type=int, help="Refresh every open pull request that links this issue")
    target.add_argument("--event", help="Path to a GitHub event payload. Picks the pull request or issue it is about.")
    parser.add_argument("--dry-run", action="store_true", help="Print what would change and write nothing")
    args = parser.parse_args(argv)
    if not args.repo:
        parser.error("--repo or $GITHUB_REPOSITORY is required")
    api = api or GitHubApi(args.repo, os.environ.get("GITHUB_TOKEN"))

    failed = errored = False
    try:
        numbers, fail_on_problems = targets(api, args.repo, args)
    except ApiError as e:
        print(e, file=sys.stderr)
        return 2
    for number in numbers:
        try:
            outcome = decide(api, args.repo, number)
            if not args.dry_run:
                apply(api, outcome)
        except ApiError as e:
            print(f"#{number}: {e}", file=sys.stderr)
            errored = True
            continue
        print(summary(outcome))
        failed = failed or (fail_on_problems and bool(outcome.problems))
    return 2 if errored else int(failed)


if __name__ == "__main__":
    sys.exit(main())
